"""CPU GEMM variants for measuring what blocking and library kernels buy."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pure-Python i-j-k triple loop (interpreter-bound; only for small n)."""
    m, k = a.shape
    n = b.shape[1]
    al, bl = a.tolist(), b.tolist()
    c = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai, ci = al[i], c[i]
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += ai[kk] * bl[kk][j]
            ci[j] = s
    return np.array(c)


def matmul_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """i-k-j order with numpy row AXPYs: vectorized inner loop, no cache blocking."""
    m, k = a.shape
    c = np.zeros((m, b.shape[1]), dtype=np.result_type(a, b))
    for i in range(m):
        for kk in range(k):
            c[i] += a[i, kk] * b[kk]
    return c


def matmul_blocked(a: np.ndarray, b: np.ndarray, bs: int) -> np.ndarray:
    """bs x bs blocks, each block product by np.matmul; Python loop overhead ~ (n/bs)^3."""
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.result_type(a, b))
    for i in range(0, m, bs):
        for j in range(0, n, bs):
            acc = c[i:i + bs, j:j + bs]
            for kk in range(0, k, bs):
                acc += a[i:i + bs, kk:kk + bs] @ b[kk:kk + bs, j:j + bs]
    return c


def gflops(fn: Callable[[], object], flops: float, repeats: int = 3,
           min_time: float = 0.2) -> tuple[float, float]:
    """Best-of-`repeats` GFLOP/s; each repeat loops until min_time elapsed."""
    best = float("inf")
    for _ in range(repeats):
        n_iter, t0 = 0, time.perf_counter()
        while True:
            fn()
            n_iter += 1
            dt = time.perf_counter() - t0
            if dt >= min_time:
                break
        best = min(best, dt / n_iter)
    return flops / best / 1e9, best
