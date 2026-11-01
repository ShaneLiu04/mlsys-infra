"""Post-training quantization primitives (simulated: quantize then dequantize in float64).

Convention: Y = X @ W with X [tokens, d_in] and W [d_in, d_out]. "per-channel" for weights
means one scale per output column; "per-group" splits each column into groups of ``g``
consecutive input rows; "per-token" for activations means one scale per row of X.
"""

from __future__ import annotations

import numpy as np


def _qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def quantize_absmax(x: np.ndarray, bits: int, axis: int | None) -> np.ndarray:
    """Symmetric round-to-nearest; ``axis`` is the axis reduced to get each scale (None = tensor)."""
    amax = np.abs(x).max(axis=axis, keepdims=axis is not None)
    scale = np.where(amax > 0, amax / _qmax(bits), 1.0)
    return np.clip(np.round(x / scale), -_qmax(bits), _qmax(bits)) * scale


def quantize_weight(w: np.ndarray, bits: int, granularity: str, group: int = 128) -> np.ndarray:
    if granularity == "tensor":
        return quantize_absmax(w, bits, None)
    if granularity == "channel":
        return quantize_absmax(w, bits, 0)
    if granularity == "group":
        d_in, d_out = w.shape
        if d_in % group:
            raise ValueError("d_in must be divisible by group size")
        g = w.reshape(d_in // group, group, d_out)
        return quantize_absmax(g, bits, 1).reshape(d_in, d_out)
    raise KeyError(granularity)


def quantize_act(x: np.ndarray, bits: int, granularity: str) -> np.ndarray:
    return quantize_absmax(x, bits, None if granularity == "tensor" else 1)


def rel_err(ref: np.ndarray, approx: np.ndarray) -> float:
    return float(np.linalg.norm(ref - approx) / np.linalg.norm(ref))


def w8a8(x: np.ndarray, w: np.ndarray, act: str = "tensor", weight: str = "tensor") -> np.ndarray:
    return quantize_act(x, 8, act) @ quantize_weight(w, 8, weight)


def llm_int8(x: np.ndarray, w: np.ndarray, threshold: float = 6.0) -> tuple[np.ndarray, int]:
    """Mixed decomposition: outlier input channels in full precision, the rest vector-wise int8."""
    outlier = np.abs(x).max(0) > threshold
    y = x[:, outlier] @ w[outlier]
    rest = ~outlier
    y += quantize_act(x[:, rest], 8, "token") @ quantize_weight(w[rest], 8, "channel")
    return y, int(outlier.sum())


def smooth(x: np.ndarray, w: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """SmoothQuant: X' = X diag(s)^-1, W' = diag(s) W with s_j = max|X_j|^a / max|W_j|^(1-a)."""
    s = np.abs(x).max(0) ** alpha / np.maximum(np.abs(w).max(1), 1e-8) ** (1 - alpha)
    s = np.maximum(s, 1e-5)
    return x / s, w * s[:, None]


def gptq(w: np.ndarray, x_cal: np.ndarray, bits: int = 4, damp: float = 0.01) -> np.ndarray:
    """GPTQ with per-output-channel scales: quantize input rows in order, push the error
    onto not-yet-quantized rows using the upper Cholesky factor of H^-1, H = X^T X."""
    w = w.copy()
    d_in = w.shape[0]
    h = x_cal.T @ x_cal
    h += damp * np.mean(np.diag(h)) * np.eye(d_in)
    u = np.linalg.cholesky(np.linalg.inv(h)).T
    scale = np.abs(w).max(0) / _qmax(bits)
    scale = np.where(scale > 0, scale, 1.0)
    q = np.zeros_like(w)
    for i in range(d_in):
        q[i] = np.clip(np.round(w[i] / scale), -_qmax(bits), _qmax(bits)) * scale
        err = (w[i] - q[i]) / u[i, i]
        w[i + 1:] -= np.outer(u[i, i + 1:], err)
    return q


def kmeans_quantize(w: np.ndarray, k: int = 16, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Deep Compression weight sharing: 1-D k-means with linear initialisation."""
    flat = w.ravel()
    c = np.linspace(flat.min(), flat.max(), k)
    for _ in range(iters):
        idx = np.abs(flat[:, None] - c[None, :]).argmin(1)
        for j in range(k):
            sel = flat[idx == j]
            if len(sel):
                c[j] = sel.mean()
    idx = np.abs(flat[:, None] - c[None, :]).argmin(1)
    return c[idx].reshape(w.shape), c


def synthetic_weight(d_in: int, d_out: int, rng: np.random.Generator, df: float | None = 4.0) -> np.ndarray:
    """Student-t weights (heavy tails, like trained LLM layers); ``df=None`` gives Gaussian."""
    if df is None:
        return 0.02 * rng.normal(size=(d_in, d_out))
    return 0.02 * rng.standard_t(df, size=(d_in, d_out))


def synthetic_activations(tokens: int, d: int, rng: np.random.Generator, n_outlier: int = 0,
                          outlier_scale: float = 40.0, rank_decay: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian activations, optionally correlated, with a few persistent outlier channels."""
    x = rng.normal(size=(tokens, d))
    if rank_decay:
        mix = rng.normal(size=(d, d)) * (np.arange(1, d + 1) ** -rank_decay)[:, None]
        x = x @ mix
        x /= x.std()
    idx = rng.choice(d, n_outlier, replace=False)
    x[:, idx] *= outlier_scale
    return x, idx
