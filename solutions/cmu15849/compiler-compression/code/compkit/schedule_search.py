"""Loop-tiling schedule space for C = A @ B and three search strategies.

Schedule = (loop order over tile indices, tm, tn, tk). Cost model for one H100 SM
(its 1/132 share of peak FLOP/s and HBM bandwidth, 228 KB shared memory as the cache):

* Operand traffic: a tile is re-fetched once per iteration of every loop from the outermost
  down to the innermost loop that indexes it; loops inside that keep it resident. C is read
  and written. If the three tiles exceed the cache, in-tile reuse is lost too: every FMA
  streams operands from DRAM with only register-block reuse (simd x reg_rows).
* Compute: FLOPs / (peak * min(1, tn / simd)) -- the j tile feeds the SIMD lanes.
* Overhead: a fixed cost per tile iteration.

time = max(compute, traffic / bw) + overhead.  This is a model, not a measured kernel.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

import mlsys_calc

ORDERS = tuple("".join(p) for p in itertools.permutations("ijk"))
TILES = tuple(2 ** k for k in range(2, 10))  # 4 .. 512


_H100 = mlsys_calc.GPUS["h100"]
_SMS = 132


@dataclass(frozen=True)
class Machine:
    peak_flops: float = _H100.bf16_flops / _SMS
    bw: float = _H100.hbm_bw / _SMS
    cache_bytes: int = 228 * 1024
    simd: int = 16
    reg_rows: int = 16
    tile_overhead_s: float = 20e-9
    bytes_per_el: int = 2


@dataclass(frozen=True)
class Schedule:
    order: str
    tm: int
    tn: int
    tk: int


def space() -> list[Schedule]:
    return [Schedule(o, a, b, c) for o in ORDERS for a in TILES for b in TILES for c in TILES]


def cost(s: Schedule, m: int, n: int, k: int, mach: Machine = Machine()) -> float:
    trips = {"i": math.ceil(m / s.tm), "j": math.ceil(n / s.tn), "k": math.ceil(k / s.tk)}
    tiles = {"A": (s.tm * s.tk, "ik"), "B": (s.tk * s.tn, "kj"), "C": (s.tm * s.tn, "ij")}
    ws = sum(t for t, _ in tiles.values()) * mach.bytes_per_el
    total_iters = trips["i"] * trips["j"] * trips["k"]
    if ws > mach.cache_bytes:
        traffic = m * n * k * (1 / mach.simd + 1 / mach.reg_rows)
    else:
        traffic = 0.0
        for name, (elems, deps) in tiles.items():
            inner = max(s.order.index(d) for d in deps)
            fetches = math.prod(trips[s.order[p]] for p in range(inner + 1))
            traffic += elems * fetches * (2 if name == "C" else 1)
    traffic *= mach.bytes_per_el
    flops = 2.0 * m * n * k
    t_compute = flops / (mach.peak_flops * min(1.0, s.tn / mach.simd) * min(1.0, s.tm / mach.simd))
    return max(t_compute, traffic / mach.bw) + total_iters * mach.tile_overhead_s


def tiled_matmul(a: np.ndarray, b: np.ndarray, s: Schedule) -> np.ndarray:
    """Execute the schedule literally (for correctness, not speed)."""
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n))
    ranges = {"i": range(0, m, s.tm), "j": range(0, n, s.tn), "k": range(0, k, s.tk)}
    for idx in itertools.product(*(ranges[d] for d in s.order)):
        pos = dict(zip(s.order, idx))
        i, j, kk = pos["i"], pos["j"], pos["k"]
        c[i:i + s.tm, j:j + s.tn] += a[i:i + s.tm, kk:kk + s.tk] @ b[kk:kk + s.tk, j:j + s.tn]
    return c


# --------------------------------------------------------------------------- search

def grid_search(costs: dict[Schedule, float], target: float) -> int:
    """Evaluations (in enumeration order) until a schedule within target is found."""
    for i, s in enumerate(space(), 1):
        if costs[s] <= target:
            return i
    return len(costs)


def random_search(costs: dict[Schedule, float], target: float, rng: np.random.Generator) -> int:
    sp = space()
    for i, j in enumerate(rng.permutation(len(sp)), 1):
        if costs[sp[j]] <= target:
            return i
    return len(sp)


def _mutate(s: Schedule, rng: np.random.Generator) -> Schedule:
    field = rng.integers(4)
    if field == 0:
        return Schedule(ORDERS[rng.integers(len(ORDERS))], s.tm, s.tn, s.tk)
    vals = [s.tm, s.tn, s.tk]
    i = TILES.index(vals[field - 1]) + int(rng.choice([-2, -1, 1, 2]))
    vals[field - 1] = TILES[min(max(i, 0), len(TILES) - 1)]
    return Schedule(s.order, *vals)


def _crossover(a: Schedule, b: Schedule, rng: np.random.Generator) -> Schedule:
    pick = rng.integers(2, size=4)
    fa, fb = (a.order, a.tm, a.tn, a.tk), (b.order, b.tm, b.tn, b.tk)
    return Schedule(*[fa[i] if pick[i] else fb[i] for i in range(4)])


def evolutionary_search(costs: dict[Schedule, float], target: float, rng: np.random.Generator,
                        pop: int = 16, budget: int | None = None) -> int:
    """Ansor-flavoured evolution (no learned cost model): tournament, crossover, mutation."""
    sp = space()
    budget = budget or len(sp)
    seen: dict[Schedule, float] = {}

    def ev(s: Schedule) -> bool:
        if s not in seen:
            seen[s] = costs[s]
        return seen[s] <= target

    population = [sp[j] for j in rng.choice(len(sp), pop, replace=False)]
    for s in population:
        if ev(s):
            return len(seen)

    def tournament() -> Schedule:
        cand = [population[j] for j in rng.choice(len(population), 3, replace=False)]
        return min(cand, key=lambda x: seen[x])

    for _ in range(50 * budget // pop):  # cap generations: converged populations re-propose seen points
        if len(seen) >= budget:
            break
        children = [sp[j] for j in rng.choice(len(sp), 2)]  # random immigrants keep diversity
        for _ in range(pop - 2):
            child = _crossover(tournament(), tournament(), rng) if rng.random() < 0.5 else tournament()
            child = _mutate(child, rng)
            if rng.random() < 0.5:
                child = _mutate(child, rng)
            children.append(child)
        for child in children:
            if ev(child):
                return len(seen)
        population = sorted(set(population + children), key=lambda x: seen[x])[:pop]
    return len(seen)
