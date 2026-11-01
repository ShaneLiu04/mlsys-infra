"""ZeRO stages 0-3 for data-parallel training of a small MLP on P simulated ranks.

Parameters live in one flat float64 vector (padded to a multiple of P) so that sharding is
a slice. Adam keeps fp32-style ``m`` and ``v`` (here float64 for exact comparison).
Stage 0 = plain DP with ring all-reduce. Stage 1 shards optimizer states, stage 2 also
gradients, stage 3 also parameters (all-gather before forward and again before backward).
Persistent per-rank memory is measured as the bytes of arrays each rank holds between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .collectives import SimNetwork, ring_all_gather, ring_allreduce, ring_reduce_scatter


@dataclass(frozen=True)
class MLPSpec:
    d_in: int = 16
    hidden: int = 64
    d_out: int = 4

    @property
    def shapes(self) -> list[tuple[int, ...]]:
        return [(self.hidden, self.d_in), (self.hidden,), (self.d_out, self.hidden), (self.d_out,)]

    @property
    def numel(self) -> int:
        return int(sum(np.prod(s) for s in self.shapes))


def init_params(spec: MLPSpec, world: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts = [rng.normal(0, 1 / np.sqrt(s[-1]), size=s).ravel() if len(s) == 2 else np.zeros(s)
             for s in spec.shapes]
    flat = np.concatenate(parts)
    pad = (-len(flat)) % world
    return np.concatenate([flat, np.zeros(pad)])


def make_data(spec: MLPSpec, n: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, spec.d_in))
    w = rng.normal(size=(spec.d_in, spec.d_out))
    return x, np.tanh(x @ w) + 0.1 * rng.normal(size=(n, spec.d_out))


def loss_and_grad(spec: MLPSpec, flat: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    p = torch.tensor(flat, dtype=torch.float64, requires_grad=True)
    views, off = [], 0
    for s in spec.shapes:
        k = int(np.prod(s))
        views.append(p[off:off + k].view(s))
        off += k
    w1, b1, w2, b2 = views
    out = torch.tanh(torch.from_numpy(x) @ w1.T + b1) @ w2.T + b2
    loss = ((out - torch.from_numpy(y)) ** 2).mean()
    loss.backward()
    return float(loss), p.grad.numpy().copy()


@dataclass
class Adam:
    lr: float = 1e-2
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def update(self, w: np.ndarray, g: np.ndarray, m: np.ndarray, v: np.ndarray, t: int) -> None:
        m *= self.b1
        m += (1 - self.b1) * g
        v *= self.b2
        v += (1 - self.b2) * g * g
        mh = m / (1 - self.b1 ** t)
        vh = v / (1 - self.b2 ** t)
        w -= self.lr * mh / (np.sqrt(vh) + self.eps)


def train_single(spec: MLPSpec, flat: np.ndarray, x: np.ndarray, y: np.ndarray,
                 steps: int, opt: Adam) -> np.ndarray:
    w = flat.copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, steps + 1):
        _, g = loss_and_grad(spec, w, x, y)
        opt.update(w, g, m, v, t)
    return w


@dataclass
class ZeroResult:
    params: np.ndarray
    mem_bytes: dict[str, int]  # persistent bytes on rank 0, by category
    comm_bytes_per_step: float  # bytes sent per rank per step
    losses: list[float] = field(default_factory=list)


def train_zero(stage: int, spec: MLPSpec, flat: np.ndarray, x: np.ndarray, y: np.ndarray,
               world: int, steps: int, opt: Adam) -> ZeroResult:
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0..3")
    n = len(flat)
    c = n // world
    xs, ys = np.split(x, world), np.split(y, world)
    own = [slice(r * c, (r + 1) * c) for r in range(world)]
    net = SimNetwork(world)

    ranks: list[dict[str, np.ndarray]] = []
    for r in range(world):
        st: dict[str, np.ndarray] = {}
        st["params"] = flat[own[r]].copy() if stage == 3 else flat.copy()
        osz = n if stage == 0 else c
        st["grads"] = np.zeros(n if stage < 2 else c)
        st["adam_m"], st["adam_v"] = np.zeros(osz), np.zeros(osz)
        ranks.append(st)

    losses = []
    for t in range(1, steps + 1):
        if stage == 3:
            full_fwd = ring_all_gather([st["params"] for st in ranks], net)
            losses.append(float(np.mean([loss_and_grad(spec, full_fwd[r], xs[r], ys[r])[0]
                                         for r in range(world)])))
            del full_fwd  # freed after forward; re-gathered for backward
            full = ring_all_gather([st["params"] for st in ranks], net)
        else:
            full = [st["params"] for st in ranks]
        local = [loss_and_grad(spec, full[r], xs[r], ys[r])[1] for r in range(world)]
        if stage != 3:
            losses.append(float(np.mean([loss_and_grad(spec, full[r], xs[r], ys[r])[0]
                                         for r in range(world)])))

        if stage == 0:
            reduced = ring_allreduce(local, net)
            for st, g in zip(ranks, reduced):
                st["grads"][:] = g / world
                opt.update(st["params"], st["grads"], st["adam_m"], st["adam_v"], t)
            continue

        shards = ring_reduce_scatter(local, net)
        for r, st in enumerate(ranks):
            g = shards[r] / world
            if stage == 1:
                st["grads"][:] = 0.0
                st["grads"][own[r]] = g
            else:
                st["grads"][:] = g
            if stage == 3:
                opt.update(st["params"], g, st["adam_m"], st["adam_v"], t)
            else:
                w_shard = st["params"][own[r]].copy()
                opt.update(w_shard, g, st["adam_m"], st["adam_v"], t)
                st["params"][own[r]] = w_shard
        if stage in (1, 2):
            gathered = ring_all_gather([st["params"][own[r]].copy() for r, st in enumerate(ranks)], net)
            for st, w in zip(ranks, gathered):
                st["params"][:] = w

    final = (np.concatenate([st["params"] for st in ranks]) if stage == 3 else ranks[0]["params"].copy())
    mem = {k: int(v.nbytes) for k, v in ranks[0].items()}
    mem["total"] = sum(mem.values())
    return ZeroResult(final, mem, float(net.stats.bytes_sent[0]) / steps, losses)
