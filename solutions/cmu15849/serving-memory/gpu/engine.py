"""A small real serving engine on one GPU: paged KV cache + iteration-level scheduling.

Policies (same knobs as ``servesim.scheduler``):
  static     request-level batching: pad prompts to the longest, decode until the longest output ends
  continuous Orca / vLLM-v0: re-batch every iteration, admitted prompts prefilled whole (<= max_batched_tokens)
  chunked    Sarathi-Serve: per-iteration token budget; decodes first, then prefill chunks
Every iteration is one fused forward over a flattened ragged batch: linear layers see all tokens,
decode attention uses the Triton paged kernel, prefill chunks use SDPA over gathered pages.
Time is a virtual clock advanced by the measured wall time of each iteration (idle gaps skipped,
so no sleeping); KV blocks are allocated on demand, recompute preemption when the pool runs out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

from paged_attn import paged_decode


@dataclass
class Req:
    rid: int
    arrival: float
    prompt: int
    output: int
    tokens: list = field(default_factory=list)
    target: int = 0  # tokens to prefill (prompt, padded prompt, or prompt+generated after preemption)
    kv_len: int = 0
    generated: int = 0
    blocks: list = field(default_factory=list)
    first: float = -1.0
    finish: float = -1.0
    emits: list = field(default_factory=list)
    preemptions: int = 0


class Engine:
    def __init__(self, model, kv_bytes: float, block_size: int = 16, policy: str = "continuous",
                 max_num_seqs: int = 128, max_batched_tokens: int = 8192, chunk_tokens: int = 512,
                 static_batch: int | None = None, max_len: int = 4096):
        self.m = model
        self.bs = block_size
        per_block = model.kv_bytes_per_token() * block_size
        self.num_blocks = int(kv_bytes // per_block)
        self.pool = torch.empty(model.L, 2, self.num_blocks, block_size, model.Hk, model.D,
                                device=model.device, dtype=model.dtype)
        self.free = list(range(self.num_blocks - 1, -1, -1))
        self.policy = policy
        self.max_num_seqs = max_num_seqs
        self.max_batched_tokens = max_batched_tokens
        self.chunk_tokens = chunk_tokens
        self.static_batch = static_batch or max(1, self.num_blocks * block_size // max_len)
        self.max_len = max_len

    # ------------------------------------------------------------------ blocks
    def _ensure(self, r: Req, upto: int) -> bool:
        need = -(-upto // self.bs) - len(r.blocks)
        if need > len(self.free):
            return False
        for _ in range(need):
            r.blocks.append(self.free.pop())
        return True

    def _release(self, r: Req) -> None:
        self.free.extend(r.blocks)
        r.blocks = []

    # ------------------------------------------------------------------ one iteration
    def _forward(self, decodes: list[Req], plan: list[tuple[Req, int]]) -> list[int]:
        m, bs = self.m, self.bs
        ids, pos, slots = [], [], []
        for r in decodes:
            ids.append(r.tokens[r.kv_len])
            pos.append(r.kv_len)
            slots.append(r.blocks[r.kv_len // bs] * bs + r.kv_len % bs)
        pre_meta = []
        off = len(decodes)
        for r, n in plan:
            p = np.arange(r.kv_len, r.kv_len + n)
            blk = np.asarray(r.blocks, dtype=np.int64)
            ids.extend(r.tokens[r.kv_len:r.kv_len + n])
            pos.extend(p.tolist())
            slots.extend((blk[p // bs] * bs + p % bs).tolist())
            nb = -(-(r.kv_len + n) // bs)
            pre_meta.append((off, n, r.kv_len, torch.tensor(r.blocks[:nb], device=m.device)))
            off += n
        dev = m.device
        ids_t = torch.tensor(ids, device=dev)
        pos_t = torch.tensor(pos, device=dev)
        slots_t = torch.tensor(slots, device=dev)
        nd = len(decodes)
        if nd:
            maxb = max(len(r.blocks) for r in decodes)
            bt = torch.tensor([r.blocks + [0] * (maxb - len(r.blocks)) for r in decodes],
                              dtype=torch.int32, device=dev)
            lens = torch.tensor([r.kv_len + 1 for r in decodes], dtype=torch.int32, device=dev)
        Hk, D = m.Hk, m.D
        rep = m.H // Hk

        def attn(i, q, k, v):
            kp, vp = self.pool[i, 0], self.pool[i, 1]
            kp.view(-1, Hk, D).index_copy_(0, slots_t, k)
            vp.view(-1, Hk, D).index_copy_(0, slots_t, v)
            out = torch.empty_like(q)
            if nd:
                out[:nd] = paged_decode(q[:nd], kp, vp, bt, lens)
            for (o, n, c0, blk) in pre_meta:
                qq = q[o:o + n].transpose(0, 1).unsqueeze(0)
                if c0 == 0:
                    kk = k[o:o + n].transpose(0, 1).unsqueeze(0)
                    vv = v[o:o + n].transpose(0, 1).unsqueeze(0)
                    oo = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True, enable_gqa=True)
                else:
                    # a boolean mask (or enable_gqa with a bias) drops SDPA to the unfused math path
                    L = c0 + n
                    kk = kp[blk].reshape(-1, Hk, D)[:L].transpose(0, 1).repeat_interleave(rep, 0).unsqueeze(0)
                    vv = vp[blk].reshape(-1, Hk, D)[:L].transpose(0, 1).repeat_interleave(rep, 0).unsqueeze(0)
                    oo = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=causal_lower_right(n, L))
                out[o:o + n] = oo[0].transpose(0, 1)
            return out

        rows = list(range(nd))
        samplers = []
        for (o, n, c0, _), (r, _) in zip(pre_meta, plan):
            if c0 + n == r.target:
                rows.append(o + n - 1)
                samplers.append(r)
        if not rows:
            m.forward(ids_t, pos_t, attn, logits_idx=torch.tensor([0], device=dev))
            if ids_t.is_cuda:
                torch.cuda.synchronize()  # keep the iteration's wall time honest
            return []
        logits = m.forward(ids_t, pos_t, attn, logits_idx=torch.tensor(rows, device=dev))
        return logits.argmax(-1).tolist()

    # ------------------------------------------------------------------ main loop
    def run(self, reqs: list[Req], vocab_hi: int = 32000, seed: int = 0, max_steps: int | None = None):
        rng = np.random.default_rng(seed)
        for r in reqs:
            r.tokens = rng.integers(0, vocab_hi, r.prompt).tolist()
            r.target = r.prompt
        reqs = sorted(reqs, key=lambda r: r.arrival)
        waiting: list[Req] = []
        running: list[Req] = []
        nxt = done = 0
        t = 0.0
        steps = []
        peak_blocks = 0
        static_active = False
        while done < len(reqs):
            while nxt < len(reqs) and reqs[nxt].arrival <= t:
                waiting.append(reqs[nxt])
                nxt += 1
            if not running and not waiting:
                t = reqs[nxt].arrival
                continue
            t0 = time.perf_counter()
            decodes: list[Req] = []
            plan: list[tuple[Req, int]] = []
            if self.policy == "static":
                if not static_active:
                    batch = waiting[: self.static_batch]
                    del waiting[: len(batch)]
                    maxp = max(r.prompt for r in batch)
                    for r in batch:  # left-pad to the longest prompt (pad tokens cost real compute)
                        r.tokens = [0] * (maxp - r.prompt) + r.tokens
                        r.target = maxp
                        self._ensure(r, maxp)
                    running = batch
                    plan = [(r, maxp) for r in running]
                    static_active = True
                else:
                    decodes = list(running)  # finished members keep decoding (wasted slots)
            else:
                decodes = [r for r in running if r.kv_len >= r.target]
                # one KV slot per decode; preempt the newest request (recompute) until it fits
                while decodes and any(not self._ensure(r, r.kv_len + 1) for r in decodes):
                    victim = running.pop()
                    self._release(victim)
                    victim.target = victim.prompt + victim.generated
                    victim.kv_len = 0
                    victim.preemptions += 1
                    waiting.insert(0, victim)
                    if victim in decodes:
                        decodes.remove(victim)
                budget = (self.chunk_tokens - len(decodes)) if self.policy == "chunked" else self.max_batched_tokens
                if self.policy == "chunked":
                    for r in running:
                        if r.kv_len < r.target and budget > 0:
                            n = min(budget, r.target - r.kv_len)
                            plan.append((r, n))
                            budget -= n
                while waiting and budget > 0 and len(running) < self.max_num_seqs:
                    r = waiting[0]
                    n = r.target if self.policy != "chunked" else min(budget, r.target)
                    if self.policy != "chunked" and n > budget and plan:
                        break
                    if len(self.free) * self.bs < r.target + 16 * len(running):  # watermark
                        break
                    waiting.pop(0)
                    running.append(r)
                    plan.append((r, n))
                    budget -= n
                for r, n in plan:
                    if not self._ensure(r, r.kv_len + n):
                        raise RuntimeError("KV pool exhausted during admission")
                if not decodes and not plan:
                    if nxt < len(reqs):
                        t = max(t, reqs[nxt].arrival)
                        continue
                    raise RuntimeError("stalled")
            if self.policy == "static" and decodes:
                for r in decodes:
                    self._ensure(r, r.kv_len + 1)
                    if len(r.tokens) <= r.kv_len:
                        r.tokens.append(0)
            peak_blocks = max(peak_blocks, self.num_blocks - len(self.free))
            ctx_sum = sum(r.kv_len + 1 for r in decodes)
            n_pre = sum(n for _, n in plan)
            pre_ctx = [(r.kv_len, n) for r, n in plan]
            out = self._forward(decodes, plan)
            dt = time.perf_counter() - t0
            t += dt
            steps.append({"dt": dt, "n_dec": len(decodes), "n_prefill": n_pre, "n_prefill_seqs": len(plan),
                          "ctx_sum": ctx_sum, "prefill_ctx": pre_ctx, "blocks_used": self.num_blocks - len(self.free),
                          "t": t})
            k = 0
            finished = []
            for r in decodes:
                tok = out[k]
                k += 1
                r.kv_len += 1
                if self.policy == "static" and r.generated >= r.output:
                    continue
                r.tokens.append(tok)
                r.generated += 1
                r.emits.append(t)
                if r.generated >= r.output:
                    r.finish = t
                    finished.append(r)
            for r, n in plan:
                r.kv_len += n
                if r.kv_len == r.target:
                    r.tokens.append(out[k])
                    k += 1
                    r.generated += 1
                    r.emits.append(t)
                    if r.first < 0:
                        r.first = t
                    if r.generated >= r.output:
                        r.finish = t
                        finished.append(r)
            if self.policy == "static":
                if all(r.generated >= r.output for r in running):
                    for r in running:
                        self._release(r)
                    done += len(running)
                    running = []
                    static_active = False
            else:
                for r in finished:
                    running.remove(r)
                    self._release(r)
                    done += 1
            if max_steps and len(steps) >= max_steps:
                break
        return reqs, steps, peak_blocks


def summarize(reqs: list[Req], steps: list[dict], ttft_slo: float, tpot_slo: float) -> dict:
    fin = [r for r in reqs if r.finish >= 0]
    t0 = min(r.arrival for r in fin)
    span = max(r.finish for r in fin) - t0
    ttft = np.array([r.first - r.arrival for r in fin])
    tpot = np.array([(r.finish - r.first) / max(r.output - 1, 1) for r in fin])
    e2e = np.array([r.finish - r.arrival for r in fin])
    itl = np.concatenate([np.diff(r.emits) for r in fin if len(r.emits) > 1])
    ok = (ttft <= ttft_slo) & (tpot <= tpot_slo)
    dts = np.array([s["dt"] for s in steps])
    return {"requests": len(fin), "makespan_s": span,
            "throughput_tok_s": sum(r.output for r in fin) / span, "throughput_req_s": len(fin) / span,
            "ttft_mean_s": float(ttft.mean()), "ttft_p50_s": float(np.median(ttft)),
            "ttft_p99_s": float(np.percentile(ttft, 99)),
            "tpot_mean_ms": float(tpot.mean() * 1e3), "tpot_p99_ms": float(np.percentile(tpot, 99) * 1e3),
            "itl_p50_ms": float(np.median(itl) * 1e3), "itl_p99_ms": float(np.percentile(itl, 99) * 1e3),
            "itl_max_ms": float(itl.max() * 1e3), "e2e_p50_s": float(np.median(e2e)),
            "e2e_p99_s": float(np.percentile(e2e, 99)),
            "slo_attain": float(ok.mean()), "goodput_req_s": float(ok.sum() / span),
            "steps": len(steps), "step_ms_mean": float(dts.mean() * 1e3), "step_ms_max": float(dts.max() * 1e3),
            "preemptions": int(sum(r.preemptions for r in fin))}
