"""Minimal GPT-2 (nanoGPT-style) with per-block recomputation modes and pipeline stage split.

Dropout is 0 and attention uses ``F.scaled_dot_product_attention`` (flash/mem-efficient on
CUDA), so the s^2 attention-score term of Korthikanti et al. is not stored.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class GPTConfig:
    name: str
    n_layer: int
    n_embd: int
    n_head: int
    vocab: int = 50257
    block_size: int = 1024

    def params(self) -> int:
        """Parameter count with tied embeddings (matches GPT.num_params())."""
        d, L = self.n_embd, self.n_layer
        per_block = 12 * d * d + 13 * d  # qkv(3d^2+3d) proj(d^2+d) fc(4d^2+4d) fc2(4d^2+d) 2xLN(4d)
        return L * per_block + self.vocab * d + self.block_size * d + 2 * d


CONFIGS = {
    "gpt2-124m": GPTConfig("gpt2-124m", 12, 768, 12),
    "gpt2-355m": GPTConfig("gpt2-355m", 24, 1024, 16),
    "gpt2-774m": GPTConfig("gpt2-774m", 36, 1280, 20),
    "tiny": GPTConfig("tiny", 4, 128, 4, vocab=512, block_size=128),
}


class Attention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.fc2 = nn.Linear(4 * c.n_embd, c.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc(x), approximate="tanh"))


class Block(nn.Module):
    """``ckpt``: 'none' | 'full' (whole block recomputed) | 'mlp' (only the MLP recomputed)
    | 'attn' (only the attention sub-layer recomputed)."""

    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = MLP(c)
        self.ckpt = "none"

    def _attn(self, x):
        return self.attn(self.ln1(x))

    def _mlp(self, x):
        return self.mlp(self.ln2(x))

    def _body(self, x):
        x = x + self._attn(x)
        return x + self._mlp(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ckpt == "full" and self.training:
            return checkpoint(self._body, x, use_reentrant=False)
        if self.ckpt == "attn" and self.training:
            x = x + checkpoint(self._attn, x, use_reentrant=False)
        else:
            x = x + self._attn(x)
        if self.ckpt == "mlp" and self.training:
            return x + checkpoint(self._mlp, x, use_reentrant=False)
        return x + self._mlp(x)


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.cfg = c
        self.wte = nn.Embedding(c.vocab, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def set_ckpt(self, mode: str, every: int = 1) -> None:
        """Apply ``mode`` to every ``every``-th block (others 'none')."""
        for i, b in enumerate(self.blocks):
            b.ckpt = mode if i % every == 0 else "none"

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.wte(idx) + self.wpe(pos)

    def head_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = F.linear(self.ln_f(x), self.wte.weight)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head_loss(x, targets)


class Stage(nn.Module):
    """A contiguous slice of a GPT for pipeline parallelism (shares modules with ``gpt``)."""

    def __init__(self, gpt: GPT, s: int, p: int):
        super().__init__()
        L = gpt.cfg.n_layer
        lo, hi = s * L // p, (s + 1) * L // p
        self.first, self.last = s == 0, s == p - 1
        self.gpt = [gpt]  # plain list: avoid registering the full model as a submodule
        self.blocks = nn.ModuleList(gpt.blocks[lo:hi])

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        g = self.gpt[0]
        if self.first:
            x = g.embed(x)
        for b in self.blocks:
            x = b(x)
        if self.last:
            return g.head_loss(x, targets)
        return x


def flops_per_token(c: GPTConfig, seq: int) -> float:
    """Training FLOPs/token: 6N (dense) + 12*L*d*seq (attention QK^T and PV, fwd+bwd, causal not halved)."""
    return 6.0 * c.params() + 12.0 * c.n_layer * c.n_embd * seq


# Saved bytes per (s*b*h) per block, counted tensor by tensor for this Block:
#   LN1 in, qkv in, q/k/v, sdpa out (= proj in: the attention kernel returns a [B,T,H,D] buffer,
#   so transpose+reshape is a view), LN2 in, fc in, gelu in (4h), fc2 in (4h)
#   = 1 + 1 + 3 + 1 + 1 + 1 + 4 + 4 = 16 elements.
# fp32: 16*4 = 64.  pure bf16: 16*2 = 32.  autocast bf16: the residual stream and LayerNorm
# inputs stay fp32 (+2 bytes for LN1 in and LN2 in) -> 36.
# Korthikanti et al. (2022) give 34 + 5as/h for fp16 *with* dropout masks and materialised scores.
ACT_BYTES_PER_SBH = {"fp32": 64, "amp_bf16": 36, "bf16": 32, "bf16_master": 32}


def activation_bytes_model(c: GPTConfig, b: int, s: int, mode: str = "amp_bf16") -> float:
    """Saved activations of all blocks in bytes (embedding output and logits excluded)."""
    return c.n_layer * ACT_BYTES_PER_SBH[mode] * s * b * c.n_embd


def logits_bytes(c: GPTConfig, b: int, s: int, act_bytes: int = 4) -> float:
    """Cross-entropy keeps the fp32 log-softmax output (b*s*V*4) in every precision mode;
    the bf16 logits themselves are not saved."""
    return b * s * c.vocab * 4


def amp_weight_copy_bytes(c: GPTConfig) -> float:
    """Autocast saves the bf16 cast of each Linear weight for backward (~2 bytes/param, embeddings excluded)."""
    return 2 * c.n_layer * (12 * c.n_embd * c.n_embd) + 2 * c.vocab * c.n_embd


def init_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
