"""Convolution as GEMM via im2col, against a direct shift-and-accumulate convolution."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def out_size(h: int, k: int, stride: int, pad: int) -> int:
    return (h + 2 * pad - k) // stride + 1


def conv2d_direct(x: np.ndarray, w: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """x: (N,C,H,W), w: (F,C,KH,KW). Sum over KH*KW shifted strided views; no patch matrix."""
    n, c, h, wd = x.shape
    f, c2, kh, kw = w.shape
    if c != c2:
        raise ValueError("channel mismatch")
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = out_size(h, kh, stride, pad), out_size(wd, kw, stride, pad)
    y = np.zeros((f, n, oh, ow), dtype=np.result_type(x, w))
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i:i + stride * oh:stride, j:j + stride * ow:stride]
            y += np.tensordot(w[:, :, i, j], patch, axes=([1], [1]))
    return y.transpose(1, 0, 2, 3)


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Return the (N*OH*OW, C*KH*KW) patch matrix (materialized copy)."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(xp, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
    n, c, oh, ow = win.shape[:4]
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(n * oh * ow, c * kh * kw)


def conv2d_im2col(x: np.ndarray, w: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    n, _, h, wd = x.shape
    f, c, kh, kw = w.shape
    oh, ow = out_size(h, kh, stride, pad), out_size(wd, kw, stride, pad)
    cols = im2col(x, kh, kw, stride, pad)
    y = cols @ w.reshape(f, c * kh * kw).T
    return y.reshape(n, oh, ow, f).transpose(0, 3, 1, 2)


def blowup_factor(c: int, h: int, w: int, kh: int, kw: int, stride: int = 1, pad: int = 0) -> float:
    """Patch-matrix elements / input elements, roughly KH*KW/stride^2."""
    oh, ow = out_size(h, kh, stride, pad), out_size(w, kw, stride, pad)
    return (oh * ow * c * kh * kw) / (c * h * w)
