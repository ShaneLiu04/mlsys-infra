"""GPT-2 with real HF weights in plain PyTorch: no-cache, static KV cache, CUDA-graph decode,
and batch-1 speculative decoding (Leviathan et al. 2023; Chen et al. 2023) on the GPU."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

MODEL_ROOT = os.environ.get("P2_MODEL_ROOT", "/root/autodl-tmp/hf/models")

CONFIGS = {  # name: (n_layer, n_head, n_embd)
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
    "gpt2-large": (36, 20, 1280),
    "gpt2-xl": (48, 25, 1600),
}


@dataclass(frozen=True)
class Cfg:
    n_layer: int
    n_head: int
    n_embd: int
    vocab: int = 50257
    block: int = 1024

    @property
    def hd(self) -> int:
        return self.n_embd // self.n_head


def load_state_dict(name: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODEL_ROOT, name, "model.safetensors"))
    return {k.removeprefix("transformer."): v for k, v in sd.items()}


class StaticKV:
    """Preallocated (layers, B, H, Tmax, hd) K/V; `length` = valid prefix, rollback = lower it."""

    def __init__(self, cfg: Cfg, batch: int, tmax: int, dtype: torch.dtype, device="cuda"):
        shape = (cfg.n_layer, batch, cfg.n_head, tmax, cfg.hd)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.length = 0
        self.tmax = tmax

    def nbytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()


def _ln(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


class GPT2:
    """nanoGPT-style forward over HF GPT-2 weights (Conv1D weights transposed to Linear layout)."""

    def __init__(self, name: str, dtype=torch.float16, device="cuda", sd=None):
        self.name, self.dtype, self.device = name, dtype, device
        self.cfg = Cfg(*CONFIGS[name])
        sd = load_state_dict(name) if sd is None else sd
        g = lambda k: sd[k].to(device=device, dtype=dtype).contiguous()  # noqa: E731
        gt = lambda k: sd[k].to(device=device, dtype=dtype).t().contiguous()  # noqa: E731
        self.wte, self.wpe = g("wte.weight"), g("wpe.weight")
        self.layers = []
        for i in range(self.cfg.n_layer):
            p = f"h.{i}."
            self.layers.append((g(p + "ln_1.weight"), g(p + "ln_1.bias"),
                                gt(p + "attn.c_attn.weight"), g(p + "attn.c_attn.bias"),
                                gt(p + "attn.c_proj.weight"), g(p + "attn.c_proj.bias"),
                                g(p + "ln_2.weight"), g(p + "ln_2.bias"),
                                gt(p + "mlp.c_fc.weight"), g(p + "mlp.c_fc.bias"),
                                gt(p + "mlp.c_proj.weight"), g(p + "mlp.c_proj.bias")))
        self.lnf_w, self.lnf_b = g("ln_f.weight"), g("ln_f.bias")
        self.ar = torch.arange(self.cfg.block + 64, device=device)
        self.attn_impl = "sdpa"  # static path only; "bmm" = explicit QK^T/softmax/PV (cuBLAS splits over T)

    def n_params(self) -> int:
        return sum(t.numel() for L in self.layers for t in L) + self.wte.numel() + self.lnf_w.numel() * 2

    def weight_bytes(self) -> int:
        return (self.n_params() + self.wpe.numel()) * self.wte.element_size()

    @torch.no_grad()
    def forward(self, idx: torch.Tensor, cache: StaticKV | None = None, pos: torch.Tensor | None = None,
                t_attn: int | None = None, n_last: int | None = None) -> torch.Tensor:
        """idx (B, t) new tokens -> logits (B, t or n_last, V).

        cache None            : full causal recompute (the no-cache baseline).
        cache, pos None       : eager path, tokens go to [cache.length, cache.length+t), attend to a slice.
        cache, pos (0-d long) : static, CUDA-graph-safe path; attends over cache[:t_attn] with a mask.
        """
        B, t = idx.shape
        c = self.cfg
        H, hd = c.n_head, c.hd
        if cache is None:
            positions = self.ar[:t]
        elif pos is None:
            s = cache.length
            if s + t > cache.tmax:
                raise ValueError("KV cache overflow")
            positions = self.ar[s:s + t]
        else:
            positions = pos + self.ar[:t]
        mask = None
        if cache is not None and pos is not None:
            mask = self.ar[:t_attn][None, :] <= positions[:, None]
        elif cache is not None and t > 1:
            mask = self.ar[:s + t][None, :] <= positions[:, None]
        x = self.wte[idx] + self.wpe[positions]
        for i, (l1w, l1b, qkvw, qkvb, ow, ob, l2w, l2b, fcw, fcb, prw, prb) in enumerate(self.layers):
            qkv = F.linear(_ln(x, l1w, l1b), qkvw, qkvb).view(B, t, 3, H, hd)
            q, k, v = (z.transpose(1, 2) for z in qkv.unbind(2))
            if cache is None:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif pos is None:
                cache.k[i, :, :, s:s + t] = k
                cache.v[i, :, :, s:s + t] = v
                kk, vv = cache.k[i, :, :, :s + t], cache.v[i, :, :, :s + t]
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            else:
                cache.k[i].index_copy_(2, positions, k)
                cache.v[i].index_copy_(2, positions, v)
                kk, vv = cache.k[i, :, :, :t_attn], cache.v[i, :, :, :t_attn]
                if self.attn_impl == "bmm":
                    s_ = (q @ kk.transpose(-1, -2)).float() * hd ** -0.5
                    y = torch.softmax(s_.masked_fill(~mask, float("-inf")), -1).to(q.dtype) @ vv
                else:
                    y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            x = x + F.linear(y.transpose(1, 2).reshape(B, t, c.n_embd), ow, ob)
            h = F.gelu(F.linear(_ln(x, l2w, l2b), fcw, fcb), approximate="tanh")
            x = x + F.linear(h, prw, prb)
        if cache is not None and pos is None:
            cache.length = s + t
        if n_last is not None:
            x = x[:, -n_last:]
        return F.linear(_ln(x, self.lnf_w, self.lnf_b), self.wte)


class GraphStep:
    """One CUDA graph for a fixed (B, t, t_attn) static forward. Capture BEFORE prefilling the cache:
    warm-up/capture writes junk K/V at positions [0, t), which the prefill overwrites."""

    def __init__(self, model: GPT2, cache: StaticKV, t: int, t_attn: int | None = None):
        B = cache.k.shape[1]
        self.t_attn = cache.tmax if t_attn is None else t_attn
        self.idx = torch.zeros(B, t, dtype=torch.long, device=model.device)
        self.pos = torch.zeros((), dtype=torch.long, device=model.device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                model.forward(self.idx, cache, self.pos, self.t_attn)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = model.forward(self.idx, cache, self.pos, self.t_attn)

    def __call__(self, idx: torch.Tensor, pos: int) -> torch.Tensor:
        self.idx.copy_(idx)
        self.pos.fill_(pos)
        self.graph.replay()
        return self.out


# --------------------------------------------------------------------------- plain generation

def _pick(logits: torch.Tensor, temperature: float, gen: torch.Generator | None) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)
    return torch.multinomial(torch.softmax(logits.float() / temperature, -1), 1, generator=gen)


@torch.no_grad()
def generate_nocache(m: GPT2, idx: torch.Tensor, n_new: int) -> torch.Tensor:
    for _ in range(n_new):
        idx = torch.cat([idx, m.forward(idx, n_last=1)[:, -1].argmax(-1, keepdim=True)], 1)
    return idx


@torch.no_grad()
def generate_kv(m: GPT2, idx: torch.Tensor, n_new: int) -> torch.Tensor:
    cache = StaticKV(m.cfg, idx.shape[0], idx.shape[1] + n_new, m.dtype)
    nxt = m.forward(idx, cache, n_last=1)[:, -1].argmax(-1, keepdim=True)
    out = [idx, nxt]
    for _ in range(n_new - 1):
        nxt = m.forward(nxt, cache)[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, 1)


@torch.no_grad()
def generate_graph(m: GPT2, idx: torch.Tensor, n_new: int, step: GraphStep | None = None,
                   cache: StaticKV | None = None) -> torch.Tensor:
    if cache is None:
        cache = StaticKV(m.cfg, idx.shape[0], idx.shape[1] + n_new, m.dtype)
        step = GraphStep(m, cache, 1)
    cache.length = 0
    P = idx.shape[1]
    nxt = m.forward(idx, cache, n_last=1)[:, -1].argmax(-1, keepdim=True)
    out = [idx, nxt]
    for i in range(n_new - 1):
        nxt = step(nxt, P + i)[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat(out, 1)


# --------------------------------------------------------------------------- speculative decoding

def accept_reject_torch(p: torch.Tensor, q: torch.Tensor, drafts: torch.Tensor,
                        gen: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """p (g+1, V) target probs, q (g, V) draft probs, drafts (g,) ~ q. All on device, no host sync.

    Accept draft i w.p. min(1, p_i/q_i); at the first rejection sample norm(max(p_i - q_i, 0)),
    else a bonus token from p_g. Returns (n_accepted, next_token, overlap) as 0-d tensors where
    overlap = sum over judged rows of sum_x min(p, q) (= 1 - TV).
    """
    g = drafts.shape[0]
    r = torch.arange(g, device=p.device)
    ratio = p[r, drafts] / q[r, drafts]
    u = torch.rand(g, device=p.device, generator=gen, dtype=p.dtype)
    n_acc = torch.cumprod((u < ratio).to(torch.long), 0).sum()
    resid = torch.cat([(p[:g] - q).clamp_min(0.0), p[g:]], 0)[n_acc]
    tot = resid.sum()
    resid = torch.where(tot > 0, resid / tot.clamp_min(1e-30), p[n_acc])
    nxt = torch.multinomial(resid, 1, generator=gen)[0]
    judged = (r <= n_acc).to(p.dtype)
    overlap = (torch.minimum(p[:g], q).sum(-1) * judged).sum()
    return n_acc, nxt, overlap


@dataclass
class SpecStats:
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    rejections: int = 0
    draft_calls: int = 0
    overlap_sum: float = 0.0
    accepted_per_round: list[int] = field(default_factory=list)

    def add(self, other: "SpecStats") -> None:
        for k in ("rounds", "proposed", "accepted", "rejections", "draft_calls", "overlap_sum"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.accepted_per_round += other.accepted_per_round

    @property
    def alpha(self) -> float:
        """accepted / (accepted + rejections): drafts after the first rejection are never judged."""
        return self.accepted / max(self.accepted + self.rejections, 1)

    @property
    def tokens_per_target_call(self) -> float:
        return (self.accepted + self.rounds) / max(self.rounds, 1)


class SpecEngine:
    """Holds caches + pre-captured graphs for a (target, draft, gamma) triple, reused across prompts."""

    def __init__(self, target: GPT2, draft: GPT2, gamma: int, tmax: int, use_graph: bool = True):
        self.t, self.d, self.gamma, self.use_graph = target, draft, gamma, use_graph
        self.tc = StaticKV(target.cfg, 1, tmax, target.dtype)
        self.dc = StaticKV(draft.cfg, 1, tmax, draft.dtype)
        if use_graph:
            self.t_step = {gamma + 1: GraphStep(target, self.tc, gamma + 1), 1: GraphStep(target, self.tc, 1)}
            self.d_step = {1: GraphStep(draft, self.dc, 1), 2: GraphStep(draft, self.dc, 2)}

    def _run(self, which: str, idx: torch.Tensor, pos: int) -> torch.Tensor:
        m, c = (self.t, self.tc) if which == "t" else (self.d, self.dc)
        if self.use_graph:
            return (self.t_step if which == "t" else self.d_step)[idx.shape[1]](idx, pos)[0]
        c.length = pos
        return m.forward(idx, c)[0]

    def prefill(self, prompt: list[int]) -> None:
        x = torch.tensor([prompt[:-1]], device=self.t.device)
        self.tc.length = self.dc.length = 0
        self.t.forward(x, self.tc)
        self.d.forward(x, self.dc)

    @torch.no_grad()
    def generate(self, prompt: list[int], n_new: int, temperature: float = 0.0,
                 gen: torch.Generator | None = None, prefill: bool = True) -> tuple[list[int], SpecStats]:
        """Cache invariant at round start: target holds seq[:-1]; draft holds a prefix of seq missing
        1 or 2 tokens. gamma is fixed for every round (output truncated to n_new)."""
        g, dev = self.gamma, self.t.device
        if len(prompt) + n_new + g + 2 > self.tc.tmax:
            raise ValueError("tmax too small")
        if prefill:
            self.prefill(prompt)
        seq, d_len = list(prompt), len(prompt) - 1
        st = SpecStats()
        while len(seq) - len(prompt) < n_new:
            missing = torch.tensor([seq[d_len:]], device=dev)
            logits = self._run("d", missing, d_len)[-1]
            pos = d_len + missing.shape[1]
            drafts, qs = [], []
            for i in range(g):
                if temperature == 0.0:
                    x = logits.argmax(-1, keepdim=True)
                else:
                    qi = torch.softmax(logits.float() / temperature, -1)
                    x = torch.multinomial(qi, 1, generator=gen)
                    qs.append(qi)
                drafts.append(x)
                if i < g - 1:
                    logits = self._run("d", x.view(1, 1), pos + i)[-1]
            st.draft_calls += g
            dr = torch.cat(drafts)
            tl = self._run("t", torch.cat([torch.tensor([seq[-1]], device=dev), dr]).view(1, -1), len(seq) - 1)
            if temperature == 0.0:
                tgt = tl.argmax(-1)
                n_acc = torch.cumprod((dr == tgt[:g]).long(), 0).sum()
                nxt, ov = tgt[n_acc], torch.zeros((), device=dev)
            else:
                p = torch.softmax(tl.float() / temperature, -1)
                n_acc, nxt, ov = accept_reject_torch(p, torch.stack(qs), dr, gen)
            vals = torch.stack([n_acc.float(), nxt.float(), ov.float()]).tolist()
            n_acc_i, nxt_i = int(vals[0]), int(vals[1])
            acc_tokens = dr[:n_acc_i].tolist() if n_acc_i else []
            seq += acc_tokens + [nxt_i]
            d_len = min(pos + g - 1, len(seq) - 1)
            st.rounds += 1
            st.proposed += g
            st.accepted += n_acc_i
            st.rejections += int(n_acc_i < g)
            st.overlap_sum += vals[2]
            st.accepted_per_round.append(n_acc_i)
        return seq[:len(prompt) + n_new], st

    @torch.no_grad()
    def target_only(self, prompt: list[int], n_new: int, temperature: float = 0.0,
                    gen: torch.Generator | None = None, prefill: bool = True,
                    record_margin: bool = False) -> tuple[list[int], list[float]]:
        """Baseline: target alone with the same sampler; uses the t=1 graph."""
        dev = self.t.device
        if prefill:
            x = torch.tensor([prompt[:-1]], device=dev)
            self.tc.length = 0
            self.t.forward(x, self.tc)
        P = len(prompt)
        cur = torch.tensor([[prompt[-1]]], device=dev)
        toks, margins = [], []
        for i in range(n_new):
            logits = self._run("t", cur, P - 1 + i)[-1]
            if record_margin:
                top = logits.float().topk(2).values
                margins.append(top[0] - top[1])
            cur = _pick(logits, temperature, gen).view(1, 1)
            toks.append(cur.view(1))
        out = torch.cat(toks).tolist()
        mg = torch.stack(margins).tolist() if record_margin else []
        return list(prompt) + out, mg
