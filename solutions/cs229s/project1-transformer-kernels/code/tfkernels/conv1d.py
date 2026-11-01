"""Depthwise causal 1D convolution y[c, t] = sum_{j<K} h[c, j] * x[c, t-j] (x[c, <0] = 0)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv_loop(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Reference: explicit loops over time and taps (vectorized only over channels)."""
    c, n = x.shape
    k = h.shape[1]
    y = np.zeros((c, n), dtype=np.result_type(x, h))
    for t in range(n):
        for j in range(min(k, t + 1)):
            y[:, t] += h[:, j] * x[:, t - j]
    return y


def conv_direct(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """O(N*K) shift-and-add: one vector op over (C, N-j) per tap."""
    n = x.shape[1]
    y = np.zeros(x.shape, dtype=np.result_type(x, h))
    for j in range(min(h.shape[1], n)):
        y[:, j:] += h[:, j:j + 1] * x[:, :n - j]
    return y


def conv_windows(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """O(N*K) as a batched dot over sliding windows (what one CUDA thread per output would do)."""
    k = h.shape[1]
    xp = np.pad(x, ((0, 0), (k - 1, 0)))
    win = sliding_window_view(xp, k, axis=1)  # (C, N, K), win[c,t,i] = x[c, t-k+1+i]
    return np.einsum("cnk,ck->cn", win, h[:, ::-1])


def conv_fft(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """O(N log N): zero-pad to >= N+K-1 so circular convolution equals linear, keep first N."""
    n, k = x.shape[1], h.shape[1]
    size = 1 << (n + k - 2).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(h, size), size)
    return y[:, :n]
