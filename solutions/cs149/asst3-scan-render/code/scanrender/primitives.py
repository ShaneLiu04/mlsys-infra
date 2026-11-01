"""Scan-based data-parallel primitives: find_repeats, stream compaction, segmented scan."""

from __future__ import annotations

import numpy as np

from .scan import _acc_dtype, exclusive_scan


def _scatter_by_flags(values: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """flags -> exclusive scan -> scatter: the core compaction pattern."""
    flags = flags.astype(np.int64)
    if flags.size == 0:
        return values[:0]
    pos = exclusive_scan(flags)
    total = int(pos[-1] + flags[-1])
    out = np.empty(total, dtype=values.dtype)
    keep = flags.astype(bool)
    out[pos[keep]] = values[keep]
    return out


def find_repeats(a: np.ndarray) -> np.ndarray:
    """Indices i with a[i] == a[i+1], in increasing order (CS149 ``find_repeats``)."""
    a = np.asarray(a)
    if a.size < 2:
        return np.zeros(0, dtype=np.int64)
    flags = a[:-1] == a[1:]
    return _scatter_by_flags(np.arange(a.size - 1, dtype=np.int64), flags)


def stream_compact(values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Keep ``values[i]`` where ``keep[i]``, preserving order."""
    return _scatter_by_flags(np.asarray(values), np.asarray(keep, dtype=bool))


def segmented_exclusive_scan(values: np.ndarray, heads: np.ndarray, dtype: np.dtype | None = None) -> np.ndarray:
    """Exclusive scan restarting at every ``heads[i] == True`` (index 0 is always a head).

    Hillis–Steele over the associative operator
    (f1, v1) ⊕ (f2, v2) = (f1 | f2, v2 if f2 else v1 + v2), then shift within segments.
    """
    values = np.asarray(values)
    n = values.size
    v = values.astype(_acc_dtype(values, dtype), copy=True)
    f = np.asarray(heads, dtype=bool).copy()
    if n == 0:
        return v
    f[0] = True
    heads0 = f.copy()
    d = 1
    while d < n:
        new_v = np.where(f[d:], v[d:], v[:-d] + v[d:])
        new_f = f[d:] | f[:-d]
        v[d:] = new_v
        f[d:] = new_f
        d *= 2
    out = np.zeros_like(v)
    out[1:] = np.where(heads0[1:], 0, v[:-1])
    return out


def segmented_exclusive_scan_reference(values: np.ndarray, heads: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    out = np.zeros(values.size, dtype=_acc_dtype(values, None))
    acc = out.dtype.type(0)
    for i in range(values.size):
        if i == 0 or heads[i]:
            acc = out.dtype.type(0)
        out[i] = acc
        acc = acc + values[i]
    return out
