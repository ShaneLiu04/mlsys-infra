"""Megatron-LM tensor parallelism on T simulated shards inside one process.

Shards are separate parameter tensors. Communication is explicit through two autograd
functions (the paper's ``f`` and ``g``), which count all-reduce calls and bytes:

* ``f`` (copy to shards): identity forward, all-reduce of input grads backward.
* ``g`` (reduce from shards): all-reduce forward, identity backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CommLog:
    calls: list[tuple[str, int]] = field(default_factory=list)  # (phase, bytes per rank)

    def count(self, phase: str) -> int:
        return sum(1 for p, _ in self.calls if p == phase)

    def nbytes(self, phase: str) -> int:
        return sum(b for p, b in self.calls if p == phase)


class _CopyToShards(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: int, log: CommLog):  # type: ignore[override]
        ctx.log = log
        return tuple(x.clone() for _ in range(t))

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):  # type: ignore[override]
        ctx.log.calls.append(("backward", grads[0].numel() * grads[0].element_size()))
        return torch.stack(grads).sum(0), None, None


class _ReduceFromShards(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log: CommLog, *parts: torch.Tensor):  # type: ignore[override]
        log.calls.append(("forward", parts[0].numel() * parts[0].element_size()))
        ctx.t = len(parts)
        return torch.stack(parts).sum(0)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return (None, *[grad.clone() for _ in range(ctx.t)])


class MLP(nn.Module):
    """Y = GeLU(X A + a) B + b."""

    def __init__(self, d: int, d_ff: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ParallelMLP(nn.Module):
    """A split by columns, B split by rows: one all-reduce forward, one backward."""

    def __init__(self, ref: MLP, t: int, log: CommLog) -> None:
        super().__init__()
        self.t, self.log = t, log
        w1 = ref.fc1.weight.detach().chunk(t, 0)
        b1 = ref.fc1.bias.detach().chunk(t, 0)
        w2 = ref.fc2.weight.detach().chunk(t, 1)
        self.w1 = nn.ParameterList([nn.Parameter(w.clone()) for w in w1])
        self.b1 = nn.ParameterList([nn.Parameter(b.clone()) for b in b1])
        self.w2 = nn.ParameterList([nn.Parameter(w.clone()) for w in w2])
        self.b2 = nn.Parameter(ref.fc2.bias.detach().clone())  # replicated, added after g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = _CopyToShards.apply(x, self.t, self.log)
        parts = [F.gelu(xs[i] @ self.w1[i].T + self.b1[i]) @ self.w2[i].T for i in range(self.t)]
        return _ReduceFromShards.apply(self.log, *parts) + self.b2


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).split(d, -1)
        return self.out(_heads_attend(q, k, v, self.h))


def _heads_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: int) -> torch.Tensor:
    b, s, dl = q.shape
    hd = dl // h
    q, k, v = (z.view(b, s, h, hd).transpose(1, 2) for z in (q, k, v))
    mask = torch.ones(s, s, dtype=torch.bool).tril()
    att = (q @ k.transpose(-1, -2) / hd ** 0.5).masked_fill(~mask, float("-inf")).softmax(-1)
    return (att @ v).transpose(1, 2).reshape(b, s, dl)


class ParallelAttention(nn.Module):
    """Heads split across shards: QKV column-parallel, output projection row-parallel."""

    def __init__(self, ref: Attention, t: int, log: CommLog) -> None:
        super().__init__()
        if ref.h % t:
            raise ValueError("heads must divide tensor-parallel size")
        self.t, self.log, self.h_local = t, log, ref.h // t
        d = ref.out.weight.shape[0]
        wq, wk, wv = ref.qkv.weight.detach().split(d, 0)
        bq, bk, bv = ref.qkv.bias.detach().split(d, 0)
        self.wqkv = nn.ParameterList()
        self.bqkv = nn.ParameterList()
        for i in range(t):
            self.wqkv.append(nn.Parameter(torch.cat([w.chunk(t, 0)[i] for w in (wq, wk, wv)]).clone()))
            self.bqkv.append(nn.Parameter(torch.cat([b.chunk(t, 0)[i] for b in (bq, bk, bv)]).clone()))
        self.wo = nn.ParameterList([nn.Parameter(w.clone()) for w in ref.out.weight.detach().chunk(t, 1)])
        self.bo = nn.Parameter(ref.out.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = _CopyToShards.apply(x, self.t, self.log)
        parts = []
        for i in range(self.t):
            q, k, v = (xs[i] @ self.wqkv[i].T + self.bqkv[i]).chunk(3, -1)
            parts.append(_heads_attend(q, k, v, self.h_local) @ self.wo[i].T)
        return _ReduceFromShards.apply(self.log, *parts) + self.bo


def unshard_grads(par: nn.Module, kind: str) -> dict[str, torch.Tensor]:
    """Reassemble shard grads into the layout of the unsharded module."""
    if kind == "mlp":
        return {"fc1.weight": torch.cat([w.grad for w in par.w1], 0),
                "fc1.bias": torch.cat([b.grad for b in par.b1], 0),
                "fc2.weight": torch.cat([w.grad for w in par.w2], 1),
                "fc2.bias": par.b2.grad}
    parts_w = [w.grad.chunk(3, 0) for w in par.wqkv]
    parts_b = [b.grad.chunk(3, 0) for b in par.bqkv]
    return {"qkv.weight": torch.cat([torch.cat([p[j] for p in parts_w]) for j in range(3)]),
            "qkv.bias": torch.cat([torch.cat([p[j] for p in parts_b]) for j in range(3)]),
            "out.weight": torch.cat([w.grad for w in par.wo], 1),
            "out.bias": par.bo.grad}
