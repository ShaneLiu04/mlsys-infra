"""Exclusive scan three ways, each instrumented with work (adds) and span (dependent steps).

Each vectorized numpy statement stands for one parallel step (one CUDA kernel
launch / barrier); its element count is the number of adds performed in that step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanStats:
    adds: int = 0
    steps: int = 0
    padded_n: int = 0


def _acc_dtype(x: np.ndarray, dtype: np.dtype | None) -> np.dtype:
    if dtype is not None:
        return np.dtype(dtype)
    if np.issubdtype(x.dtype, np.bool_) or np.issubdtype(x.dtype, np.integer):
        return np.dtype(np.int64)
    return np.dtype(np.float64)


def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def exclusive_scan_sequential(x: np.ndarray, dtype: np.dtype | None = None) -> tuple[np.ndarray, ScanStats]:
    """out[i] = sum(x[:i]); one add per element, every add depends on the previous one."""
    x = np.asarray(x)
    out = np.empty(x.size, dtype=_acc_dtype(x, dtype))
    acc = out.dtype.type(0)
    for i in range(x.size):
        out[i] = acc
        acc = acc + out.dtype.type(x[i])
    n_adds = max(x.size - 1, 0)
    return out, ScanStats(adds=n_adds, steps=n_adds, padded_n=x.size)


def hillis_steele_exclusive(x: np.ndarray, dtype: np.dtype | None = None) -> tuple[np.ndarray, ScanStats]:
    """Hillis–Steele: ceil(log2 n) steps, step d does n - 2^d adds; O(n log n) work."""
    x = np.asarray(x)
    n = x.size
    y = x.astype(_acc_dtype(x, dtype), copy=True)
    st = ScanStats(padded_n=n)
    d = 1
    while d < n:
        # RHS is fully evaluated before the store: this is the double buffer.
        y[d:] = y[d:] + y[:-d]
        st.adds += n - d
        st.steps += 1
        d *= 2
    out = np.zeros_like(y)
    out[1:] = y[:-1]
    return out, st


def blelloch_exclusive(x: np.ndarray, dtype: np.dtype | None = None) -> tuple[np.ndarray, ScanStats]:
    """Work-efficient up-sweep / down-sweep on a power-of-two padded buffer.

    Mirrors the CUDA version: each level is one kernel launch over the
    ``m / two_dplus1`` active indices ``i = 0, two_dplus1, 2*two_dplus1, ...``,
    expressed here as strided views of the buffer.
    """
    x = np.asarray(x)
    n = x.size
    m = next_pow2(n)
    a = np.zeros(m, dtype=_acc_dtype(x, dtype))
    a[:n] = x
    st = ScanStats(padded_n=m)
    if n == 0:
        return a[:0], st
    two_d = 1
    while two_d < m:
        two_dp1 = 2 * two_d
        right = a[two_dp1 - 1::two_dp1]  # a[i + two_dp1 - 1] for i = 0, two_dp1, ...
        left = a[two_d - 1::two_dp1]     # a[i + two_d - 1]
        right += left
        st.adds += right.size
        st.steps += 1
        two_d = two_dp1
    a[m - 1] = 0
    two_d = m // 2
    while two_d >= 1:
        two_dp1 = 2 * two_d
        right = a[two_dp1 - 1::two_dp1]
        left = a[two_d - 1::two_dp1]
        t = left.copy()
        left[:] = right
        right += t
        st.adds += right.size
        st.steps += 1
        two_d //= 2
    return a[:n], st


def analytic_stats(algo: str, n: int) -> ScanStats:
    """Closed-form work/span for each algorithm (n = input length)."""
    if n <= 1:
        lg = 0
    else:
        lg = (n - 1).bit_length()
    if algo == "sequential":
        return ScanStats(adds=max(n - 1, 0), steps=max(n - 1, 0), padded_n=n)
    if algo == "hillis_steele":
        return ScanStats(adds=sum(n - (1 << k) for k in range(lg)), steps=lg, padded_n=n)
    if algo == "blelloch":
        m = next_pow2(n)
        return ScanStats(adds=2 * (m - 1) if n else 0, steps=2 * lg, padded_n=m)
    raise ValueError(algo)


def exclusive_scan(x: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Default scan used by the primitives: the work-efficient one."""
    return blelloch_exclusive(x, dtype)[0]
