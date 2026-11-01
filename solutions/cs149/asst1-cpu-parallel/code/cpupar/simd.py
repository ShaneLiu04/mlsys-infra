"""SIMD-divergence model (prog2/prog3/prog4 analogue).

A W-wide gang executes in lockstep: it keeps issuing instructions until its
slowest lane is done, with finished lanes masked off. Given per-element work
counts in program order, utilization is

    useful lane-iterations / sum over gangs of (W * max work in gang).
"""

from __future__ import annotations

import numpy as np

SQRT_THRESHOLD = np.float32(1e-5)
SQRT_FAST = np.float32(1.0)
SQRT_SLOW = np.float32(2.999)


def gang_utilization(work: np.ndarray, width: int) -> float:
    """Fraction of issued lane-slots doing useful work for gangs of ``width`` lanes.

    The tail is padded with zero-work lanes, as a masked ISPC tail gang would be.
    """
    w = np.asarray(work, dtype=np.int64).ravel()
    pad = (-w.size) % width
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.int64)])
    gangs = w.reshape(-1, width)
    issued = width * gangs.max(axis=1).sum()
    return float(w.sum() / issued) if issued else 1.0


def gang_issued_slots(work: np.ndarray, width: int) -> int:
    """Total lane-slots issued (W * max per gang), i.e. SIMD cost in lane-iterations."""
    w = np.asarray(work, dtype=np.int64).ravel()
    pad = (-w.size) % width
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.int64)])
    return int(width * w.reshape(-1, width).max(axis=1).sum())


def sqrt_scalar(x: float, threshold: float = 1e-5, max_iter: int = 10_000) -> tuple[float, int]:
    """prog4 Newton iteration for 1/sqrt(x) starting from guess 1; returns (sqrt(x), iterations)."""
    guess = 1.0
    err = abs(guess * guess * x - 1.0)
    it = 0
    while err > threshold and it < max_iter:
        guess = (3.0 * guess - x * guess * guess * guess) * 0.5
        err = abs(guess * guess * x - 1.0)
        it += 1
    return x * guess, it


def sqrt_iterations(x: np.ndarray, max_iter: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized prog4 Newton with compaction (float32); returns (sqrt(x), per-element iterations)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    guess_out = np.ones(n, dtype=np.float32)
    iters = np.zeros(n, dtype=np.int32)
    idx = np.arange(n)
    xv = x.copy()
    g = np.ones(n, dtype=np.float32)
    err = np.abs(g * g * xv - np.float32(1.0))
    for it in range(max_iter):
        done = err <= SQRT_THRESHOLD
        if done.any():
            guess_out[idx[done]] = g[done]
            iters[idx[done]] = it
            keep = ~done
            idx, xv, g = idx[keep], xv[keep], g[keep]
            if idx.size == 0:
                break
        g = (np.float32(3.0) * g - xv * g * g * g) * np.float32(0.5)
        err = np.abs(g * g * xv - np.float32(1.0))
    else:
        guess_out[idx] = g
        iters[idx] = max_iter
    return x * guess_out, iters


def sqrt_masked(x: np.ndarray, max_iter: int = 10_000) -> np.ndarray:
    """Vectorized prog4 Newton *without* compaction: every lane runs until the slowest lane is done."""
    x = np.asarray(x, dtype=np.float32)
    g = np.ones_like(x)
    for _ in range(max_iter):
        active = np.abs(g * g * x - np.float32(1.0)) > SQRT_THRESHOLD
        if not active.any():
            break
        g = np.where(active, (np.float32(3.0) * g - x * g * g * g) * np.float32(0.5), g)
    return x * g


def sqrt_inputs(pattern: str, n: int, width: int = 8, seed: int = 0) -> np.ndarray:
    """prog4 input patterns.

    - ``uniform``: the starter's ``0.001 + 2.998 * rand``.
    - ``all_fast`` / ``all_slow``: every element 1.0 / 2.999 (no divergence).
    - ``one_slow_per_gang``: element i is slow iff ``i % width == 0`` — one slow
      lane in every gang, maximal divergence for that W.
    """
    if pattern == "uniform":
        rng = np.random.default_rng(seed)
        return (0.001 + 2.998 * rng.random(n)).astype(np.float32)
    if pattern == "all_fast":
        return np.full(n, SQRT_FAST, dtype=np.float32)
    if pattern == "all_slow":
        return np.full(n, SQRT_SLOW, dtype=np.float32)
    if pattern == "one_slow_per_gang":
        x = np.full(n, SQRT_FAST, dtype=np.float32)
        x[::width] = SQRT_SLOW
        return x
    raise ValueError(f"unknown pattern {pattern!r}")
