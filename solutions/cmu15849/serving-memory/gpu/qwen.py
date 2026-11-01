"""Minimal Qwen2 (Qwen2.5-0.5B) forward over a flattened ragged token batch.

Weights come from the HF safetensors checkpoint; the attention step is a callback so the same
weights serve the contiguous-KV roofline benchmark and the paged-KV engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

AttnFn = Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def find_checkpoint(repo: str = "Qwen/Qwen2.5-0.5B", cache: str = "/root/autodl-tmp/hf") -> Path:
    base = Path(cache) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(base.iterdir())
    if not snaps:
        raise FileNotFoundError(base)
    return snaps[-1]


class Qwen2:
    def __init__(self, path: Path | str, device: str = "cuda", dtype=torch.bfloat16, max_pos: int = 16384):
        from safetensors.torch import load_file

        path = Path(path)
        c = json.loads((path / "config.json").read_text())
        self.cfg = c
        self.L = c["num_hidden_layers"]
        self.d = c["hidden_size"]
        self.H = c["num_attention_heads"]
        self.Hk = c["num_key_value_heads"]
        self.D = self.d // self.H
        self.ff = c["intermediate_size"]
        self.eps = c["rms_norm_eps"]
        self.vocab = c["vocab_size"]
        sd = {}
        for f in sorted(path.glob("*.safetensors")):
            sd.update(load_file(str(f), device=device))
        g = lambda k: sd[k].to(dtype)  # noqa: E731
        self.embed = g("model.embed_tokens.weight")
        self.lm_head = g("lm_head.weight") if "lm_head.weight" in sd else self.embed
        self.norm = g("model.norm.weight")
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            self.layers.append({
                "ln1": g(p + "input_layernorm.weight"),
                "ln2": g(p + "post_attention_layernorm.weight"),
                "wqkv": torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")]),
                "bqkv": torch.cat([g(a + "q_proj.bias"), g(a + "k_proj.bias"), g(a + "v_proj.bias")]),
                "wo": g(a + "o_proj.weight"),
                "wgu": torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]),
                "wd": g(p + "mlp.down_proj.weight"),
            })
        del sd
        theta = c.get("rope_theta", 10000.0)
        inv = 1.0 / (theta ** (torch.arange(0, self.D, 2, device=device, dtype=torch.float32) / self.D))
        fr = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), inv)
        emb = torch.cat([fr, fr], -1)
        self.cos, self.sin = emb.cos(), emb.sin()
        self.device, self.dtype = device, dtype

    # ------------------------------------------------------------------ pieces
    def num_params(self) -> int:
        n = self.embed.numel() + self.norm.numel()
        if self.lm_head is not self.embed:
            n += self.lm_head.numel()
        for lw in self.layers:
            n += sum(t.numel() for t in lw.values())
        return n

    def weight_bytes(self) -> int:
        return self.num_params() * self.embed.element_size()

    def kv_bytes_per_token(self) -> int:
        return 2 * self.L * self.Hk * self.D * self.embed.element_size()

    def rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * xf.to(x.dtype)

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos[pos].unsqueeze(1), self.sin[pos].unsqueeze(1)  # [T,1,D]
        xf = x.float()
        x1, x2 = xf[..., : self.D // 2], xf[..., self.D // 2:]
        rot = torch.cat([-x2, x1], -1)
        return (xf * cos + rot * sin).to(x.dtype)

    # ------------------------------------------------------------------ forward
    def forward(self, ids: torch.Tensor, pos: torch.Tensor, attn: AttnFn,
                logits_idx: torch.Tensor | None = None) -> torch.Tensor:
        """ids, pos: [T]. attn(layer, q[T,H,D], k[T,Hk,D], v[T,Hk,D]) -> o[T,H,D].
        Returns logits for rows ``logits_idx`` (default: all rows)."""
        x = self.embed[ids]
        T = ids.shape[0]
        H, Hk, D = self.H, self.Hk, self.D
        for i, lw in enumerate(self.layers):
            h = self.rms(x, lw["ln1"])
            qkv = F.linear(h, lw["wqkv"], lw["bqkv"])
            q, k, v = qkv.split([H * D, Hk * D, Hk * D], -1)
            q = self.rope(q.view(T, H, D), pos)
            k = self.rope(k.view(T, Hk, D), pos)
            o = attn(i, q, k, v.view(T, Hk, D))
            x = x + F.linear(o.reshape(T, H * D), lw["wo"])
            h = self.rms(x, lw["ln2"])
            gu = F.linear(h, lw["wgu"])
            gate, up = gu.split(self.ff, -1)
            x = x + F.linear(F.silu(gate) * up, lw["wd"])
        if logits_idx is not None:
            x = x[logits_idx]
        return F.linear(self.rms(x, self.norm), self.lm_head)


def sdpa_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """q [B,H,Tq,D], k/v [B,Hk,Tk,D] -> [B,H,Tq,D] with grouped-query attention."""
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)


def decode_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-token decode without KV expansion: fold the query-head group into the query length.
    q [B,H,D], k/v [B,Hk,L,D] -> [B,H,D]."""
    B, H, D = q.shape
    Hk = k.shape[1]
    o = F.scaled_dot_product_attention(q.view(B, Hk, H // Hk, D), k, v)
    return o.reshape(B, H, D)


class ContiguousCache:
    """[L, 2, B, Hk, S, D] cache for equal-length batches (roofline benchmark and HF check)."""

    def __init__(self, m: Qwen2, B: int, S: int):
        self.kv = torch.zeros(m.L, 2, B, m.Hk, S, m.D, device=m.device, dtype=m.dtype)
        self.B, self.len = B, 0

    def prefill_attn(self, s: int) -> AttnFn:
        B = self.B

        def fn(i, q, k, v):
            H, Hk, D = q.shape[1], k.shape[1], q.shape[2]
            k4 = k.view(B, s, Hk, D).transpose(1, 2)
            v4 = v.view(B, s, Hk, D).transpose(1, 2)
            self.kv[i, 0, :, :, self.len:self.len + s] = k4
            self.kv[i, 1, :, :, self.len:self.len + s] = v4
            o = sdpa_gqa(q.view(B, s, H, D).transpose(1, 2), k4, v4, causal=True)
            return o.transpose(1, 2).reshape(B * s, H, D)
        return fn

    def decode_attn(self, pos: int) -> AttnFn:
        """Write the new token at ``pos`` and attend over [0, pos]; static shapes (graph-safe)."""

        def fn(i, q, k, v):
            self.kv[i, 0, :, :, pos] = k
            self.kv[i, 1, :, :, pos] = v
            return decode_gqa(q, self.kv[i, 0, :, :, : pos + 1], self.kv[i, 1, :, :, : pos + 1])
        return fn
