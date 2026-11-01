"""Activation checkpointing on an n-layer chain (Chen et al. 2016; Checkmate-style optimum).

Cost model: layer i produces activation of size a_i and costs c_i to recompute. A policy
stores a set S of layer outputs (the last layer is always stored, the input is free).
Backward over the segment between consecutive stored outputs recomputes its interior and
holds it, so

    peak = sum_{i in S} a_i + max_segment sum_{interior} a_j,   recompute = sum_{i not in S} c_i.
"""

from __future__ import annotations

import itertools
import math
import weakref
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Plan:
    stored: tuple[int, ...]  # 1-based layer indices, always includes n
    peak: int
    recompute: int


def evaluate(sizes: list[int], costs: list[int], stored: list[int] | tuple[int, ...]) -> Plan:
    n = len(sizes)
    s = sorted(set(stored) | {n})
    peak_store = sum(sizes[i - 1] for i in s)
    worst = 0
    prev = 0
    for j in s:
        worst = max(worst, sum(sizes[prev:j - 1]))
        prev = j
    recompute = sum(costs[i - 1] for i in range(1, n + 1) if i not in s)
    return Plan(tuple(s), peak_store + worst, recompute)


def no_checkpoint(sizes: list[int], costs: list[int]) -> Plan:
    return evaluate(sizes, costs, list(range(1, len(sizes) + 1)))


def sqrt_checkpoint(sizes: list[int], costs: list[int]) -> Plan:
    n = len(sizes)
    k = max(1, round(math.sqrt(n)))
    return evaluate(sizes, costs, list(range(k, n + 1, k)))


def optimal_frontier(sizes: list[int], costs: list[int]) -> dict[int, Plan]:
    """Min recompute for every memory budget, by DP over (segment cap M, stored bytes).

    For a fixed cap M on segment interiors, dp[j][u] is the min recompute with output j
    stored and u bytes of stored outputs; transitions i -> j are allowed if the interior
    of (i, j) fits in M. Returns {peak: plan} for Pareto-optimal plans.
    """
    n = len(sizes)
    pre = np.concatenate([[0], np.cumsum(sizes)])
    cpre = np.concatenate([[0], np.cumsum(costs)])
    total = int(pre[-1])
    interior = lambda i, j: int(pre[j - 1] - pre[i])  # noqa: E731  layers i+1..j-1
    caps = sorted({interior(i, j) for i in range(n) for j in range(i + 1, n + 1)})
    inf = np.iinfo(np.int64).max // 4
    best: dict[int, tuple[int, tuple[int, ...]]] = {}
    for cap in caps:
        dp = np.full((n + 1, total + 1), inf, dtype=np.int64)
        arg = np.full((n + 1, total + 1), -1, dtype=np.int64)
        dp[0, 0] = 0
        for j in range(1, n + 1):
            a = sizes[j - 1]
            for i in range(j):
                if interior(i, j) > cap:
                    continue
                cand = dp[i, : total + 1 - a] + int(cpre[j - 1] - cpre[i])
                tgt = dp[j, a:]
                better = cand < tgt
                tgt[better] = cand[better]
                arg[j, a:][better] = i
        for u in np.nonzero(dp[n] < inf)[0]:
            peak = int(u) + cap
            rec = int(dp[n, u])
            if peak not in best or rec < best[peak][0]:
                stored, j, uu = [], n, int(u)
                while j > 0:
                    stored.append(j)
                    i = int(arg[j, uu])
                    uu -= sizes[j - 1]
                    j = i
                best[peak] = (rec, tuple(sorted(stored)))
    plans = sorted((evaluate(sizes, costs, stored) for _, stored in best.values()),
                   key=lambda p: (p.peak, p.recompute))
    frontier: dict[int, Plan] = {}
    running = inf
    for plan in plans:
        if plan.recompute < running:
            running = plan.recompute
            frontier[plan.peak] = plan
    return frontier


def best_under_budget(frontier: dict[int, Plan], budget: int) -> Plan | None:
    ok = [p for peak, p in frontier.items() if peak <= budget]
    return min(ok, key=lambda p: p.recompute) if ok else None


def brute_force(sizes: list[int], costs: list[int], budget: int) -> int | None:
    n = len(sizes)
    best = None
    for r in range(n):
        for extra in itertools.combinations(range(1, n), r):
            plan = evaluate(sizes, costs, list(extra))
            if plan.peak <= budget and (best is None or plan.recompute < best):
                best = plan.recompute
    return best


# --------------------------------------------------------------------------- torch check

class SavedBytesMeter:
    """Tracks live bytes of tensors saved for backward (deduplicated by storage, params excluded)."""

    class _Holder:
        __slots__ = ("t", "__weakref__")

        def __init__(self, t: torch.Tensor) -> None:
            self.t = t

    def __init__(self, exclude: list[torch.Tensor]) -> None:
        self.exclude = {p.untyped_storage().data_ptr() for p in exclude}
        self.refs: dict[int, int] = {}
        self.sizes: dict[int, int] = {}
        self.live = 0
        self.peak = 0

    def _release(self, key: int) -> None:
        self.refs[key] -= 1
        if self.refs[key] == 0:
            self.live -= self.sizes.pop(key)
            del self.refs[key]

    def pack(self, t: torch.Tensor):
        key = t.untyped_storage().data_ptr()
        h = self._Holder(t)
        if key in self.exclude:
            return h
        if key not in self.refs:
            self.refs[key] = 0
            self.sizes[key] = t.untyped_storage().nbytes()
            self.live += self.sizes[key]
            self.peak = max(self.peak, self.live)
        self.refs[key] += 1
        weakref.finalize(h, self._release, key)
        return h

    @staticmethod
    def unpack(h: "SavedBytesMeter._Holder") -> torch.Tensor:
        return h.t


def make_chain(n: int, d: int, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(*[nn.Sequential(nn.Linear(d, d), nn.Tanh()) for _ in range(n)])


def run_chain(model: nn.Sequential, x: torch.Tensor, segment: int | None) -> tuple[float, list[torch.Tensor], int]:
    """Forward+backward; ``segment`` = layers per checkpointed segment (None = no checkpoint).

    Returns loss, parameter grads, and peak live saved-activation bytes.
    """
    model.zero_grad()
    params = list(model.parameters())
    meter = SavedBytesMeter(params)
    x = x.clone().requires_grad_(True)
    layers = list(model)
    with torch.autograd.graph.saved_tensors_hooks(meter.pack, meter.unpack):
        h = x
        if segment is None:
            h = model(h)
        else:
            for s in range(0, len(layers), segment):
                seg = nn.Sequential(*layers[s:s + segment])
                h = checkpoint(seg, h, use_reentrant=True)
        loss = (h ** 2).mean()
        loss.backward()
    return float(loss), [p.grad.clone() for p in params], meter.peak
