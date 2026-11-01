"""SAXPY bandwidth (prog5 analogue)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import torch

BYTES_PER_ELEM = 4


def saxpy_numpy(a: float, x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    """Two numpy passes (multiply then add): numpy has no fused axpy ufunc."""
    np.multiply(x, np.float32(a), out=out)
    np.add(out, y, out=out)


def saxpy_torch(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Single fused pass ``out = y + a*x``; threads controlled by ``torch.set_num_threads``."""
    torch.add(y, x, alpha=a, out=out)


def saxpy_numpy_threads(pool: ThreadPoolExecutor, workers: int, a: float,
                        x: np.ndarray, y: np.ndarray, out: np.ndarray) -> None:
    """Split the arrays into ``workers`` slices; numpy ufuncs release the GIL."""
    bounds = np.linspace(0, x.size, workers + 1).astype(np.int64)
    futs = [pool.submit(saxpy_numpy, a, x[s:e], y[s:e], out[s:e]) for s, e in zip(bounds[:-1], bounds[1:])]
    for f in futs:
        f.result()


def effective_gbps(n: int, seconds: float) -> float:
    """Programmer-visible traffic: read x, read y, write out = 3 * 4 * N bytes."""
    return 3 * BYTES_PER_ELEM * n / seconds / 1e9


def time_best(fn: Callable[[], None], min_time: float = 0.05, repeats: int = 5) -> float:
    """Best-of-``repeats`` per-call time, each repeat running enough calls to last ``min_time``."""
    fn()
    t0 = time.perf_counter()
    fn()
    one = max(time.perf_counter() - t0, 1e-7)
    inner = max(1, int(min_time / one))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        best = min(best, (time.perf_counter() - t0) / inner)
    return best
