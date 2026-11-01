"""Minimal GPT (pre-LN, learned positions, tied embeddings) with a switchable attention implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 512
    seq_len: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 4


def _attend(q, k, v, impl: str):
    if impl == "sdpa":
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if impl == "naive":
        from flash_triton import naive_attention
        return naive_attention(q, k, v, causal=True)
    if impl == "triton":
        from flash_triton import flash_attention
        if q.dtype == torch.float32:
            raise ValueError("triton attention needs fp16/bf16 activations (use autocast)")
        return flash_attention(q, k, v, causal=True)
    raise ValueError(impl)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc = nn.Linear(d, cfg.mlp_ratio * d)
        self.fc_out = nn.Linear(cfg.mlp_ratio * d, d)
        self.attn_impl = "sdpa"

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        y = _attend(q.contiguous(), k.contiguous(), v.contiguous(), self.attn_impl)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc_out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def set_attn_impl(self, impl: str) -> None:
        for blk in self.blocks:
            blk.attn_impl = impl

    def n_params(self, include_pos: bool = False) -> int:
        """Parameters that take part in matmuls (tied token embedding counted once, as the LM head)."""
        n = sum(p.numel() for p in self.parameters())
        return n if include_pos else n - self.pos.weight.numel()

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok.weight.t()
