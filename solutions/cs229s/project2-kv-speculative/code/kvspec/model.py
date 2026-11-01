"""Minimal nanoGPT-style decoder with an optional preallocated KV cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 28
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class KVCache:
    """Preallocated (layers, B, H, block_size, head_dim) K/V buffers; `length` = valid prefix.

    Rolling back (e.g. after rejected drafts) is just lowering `length`.
    """

    def __init__(self, cfg: GPTConfig, batch: int = 1, dtype: torch.dtype = torch.float32):
        hd = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch, cfg.n_head, cfg.block_size, hd)
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)
        self.length = 0
        self.max_len = cfg.block_size

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer: int) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q, k, v = (z.view(b, t, self.n_head, c // self.n_head).transpose(1, 2) for z in (q, k, v))
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            s = cache.length
            cache.k[layer, :, :, s:s + t] = k
            cache.v[layer, :, :, s:s + t] = v
            kk, vv = cache.k[layer, :, :, :s + t], cache.v[layer, :, :, :s + t]
            if t == 1:
                y = F.scaled_dot_product_attention(q, kk, vv)
            else:
                mask = torch.arange(s + t)[None, :] <= (s + torch.arange(t))[:, None]
                y = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd))

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer: int) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache, layer)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) - self.wpe.weight.numel()

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """idx: (B, T) new tokens. With a cache they occupy positions [cache.length, cache.length+T)."""
        t = idx.shape[1]
        start = cache.length if cache is not None else 0
        if start + t > self.cfg.block_size:
            raise ValueError("sequence exceeds block_size")
        x = self.wte(idx) + self.wpe(torch.arange(start, start + t))
        for i, blk in enumerate(self.blocks):
            x = blk(x, cache, i)
        if cache is not None:
            cache.length = start + t
        return self.lm_head(self.ln_f(x))


def _pick(logits: torch.Tensor, temperature: float, gen: torch.Generator | None) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, 1, generator=gen)


@torch.no_grad()
def generate_nocache(model: GPT, idx: torch.Tensor, n_new: int, temperature: float = 0.0,
                     gen: torch.Generator | None = None,
                     return_logits: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Re-run the full prefix every step: step t costs a forward over t tokens -> O(T^2) total."""
    logits_all = []
    for _ in range(n_new):
        logits = model(idx)[:, -1]
        if return_logits:
            logits_all.append(logits)
        idx = torch.cat([idx, _pick(logits, temperature, gen)], dim=1)
    return idx, logits_all


@torch.no_grad()
def generate_kv(model: GPT, idx: torch.Tensor, n_new: int, temperature: float = 0.0,
                gen: torch.Generator | None = None,
                return_logits: bool = False) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Prefill once, then one token per forward against the cached K/V -> O(T) linear-layer work."""
    cache = KVCache(model.cfg, idx.shape[0], dtype=next(model.parameters()).dtype)
    logits = model(idx, cache)[:, -1]
    logits_all = []
    for i in range(n_new):
        if return_logits:
            logits_all.append(logits)
        nxt = _pick(logits, temperature, gen)
        idx = torch.cat([idx, nxt], dim=1)
        if i < n_new - 1:
            logits = model(nxt, cache)[:, -1]
    return idx, logits_all
