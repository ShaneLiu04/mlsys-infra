"""Softmax attention in numpy: naive, online (streaming), and FlashAttention-2-style tiled.

All functions take q: (..., Tq, d), k, v: (..., Tk, d) and return o: (..., Tq, d).
Causal masking requires Tq == Tk (query i sees keys j <= i).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


def _scale(d: int, scale: float | None) -> float:
    return 1.0 / np.sqrt(d) if scale is None else scale


def _t(x: Array) -> Array:
    return np.swapaxes(x, -1, -2)


def _check(q: Array, k: Array, causal: bool) -> None:
    if causal and q.shape[-2] != k.shape[-2]:
        raise ValueError("causal attention needs Tq == Tk")


def naive_attention(q: Array, k: Array, v: Array, causal: bool = False,
                    scale: float | None = None) -> tuple[Array, Array]:
    """Materialize S and P (T x T). Returns (o, logsumexp of scaled scores)."""
    _check(q, k, causal)
    s = (q @ _t(k)) * _scale(q.shape[-1], scale)
    if causal:
        t = q.shape[-2]
        s = np.where(np.tril(np.ones((t, t), dtype=bool)), s, -np.inf)
    m = s.max(axis=-1, keepdims=True)
    p = np.exp(s - m)
    l = p.sum(axis=-1, keepdims=True)
    return (p / l) @ v, (m + np.log(l))[..., 0]


def naive_attention_backward(q: Array, k: Array, v: Array, do: Array, causal: bool = False,
                             scale: float | None = None) -> tuple[Array, Array, Array]:
    """dS = P * (dP - rowsum(dP * P)), with dP = dO V^T."""
    sc = _scale(q.shape[-1], scale)
    s = (q @ _t(k)) * sc
    if causal:
        t = q.shape[-2]
        s = np.where(np.tril(np.ones((t, t), dtype=bool)), s, -np.inf)
    p = np.exp(s - s.max(axis=-1, keepdims=True))
    p /= p.sum(axis=-1, keepdims=True)
    dv = _t(p) @ do
    dp = do @ _t(v)
    ds = p * (dp - (dp * p).sum(axis=-1, keepdims=True))
    return sc * ds @ k, sc * _t(ds) @ q, dv


def online_softmax(x: Array) -> Array:
    """Softmax over the last axis; max and normalizer from ONE streaming pass (Milakov & Gimelshein)."""
    m = np.full(x.shape[:-1], -np.inf)
    d = np.zeros(x.shape[:-1])
    for j in range(x.shape[-1]):
        xj = x[..., j]
        m_new = np.maximum(m, xj)
        d = d * np.exp(m - m_new) + np.exp(xj - m_new)
        m = m_new
    return np.exp(x - m[..., None]) / d[..., None]


def online_attention(q: Array, k: Array, v: Array, causal: bool = False,
                     scale: float | None = None) -> Array:
    """Single pass over keys, one key at a time; output rescaled as the running max moves."""
    _check(q, k, causal)
    sc = _scale(q.shape[-1], scale)
    tq = q.shape[-2]
    m = np.full(q.shape[:-1], -np.inf)
    l = np.zeros(q.shape[:-1])
    acc = np.zeros(q.shape[:-1] + (v.shape[-1],))
    rows = np.arange(tq)
    for j in range(k.shape[-2]):
        s = np.einsum("...td,...d->...t", q, k[..., j, :]) * sc
        if causal:
            s = np.where(rows >= j, s, -np.inf)
        m_new = np.maximum(m, s)
        alpha = np.exp(m - m_new)
        p = np.exp(s - m_new)
        l = alpha * l + p
        acc = alpha[..., None] * acc + p[..., None] * v[..., j, None, :]
        m = m_new
    return acc / l[..., None]


@dataclass
class IOCounter:
    """Elements moved between 'HBM' (full arrays) and 'SRAM' (blocks) by the tiled kernels."""

    reads: int = 0
    writes: int = 0
    blocks_visited: int = 0
    blocks_skipped: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.reads + self.writes


def _block_mask(r0: int, r1: int, c0: int, c1: int) -> Array:
    return np.arange(r0, r1)[:, None] >= np.arange(c0, c1)[None, :]


def flash_forward(q: Array, k: Array, v: Array, causal: bool = False, br: int = 64, bc: int = 64,
                  scale: float | None = None, io: IOCounter | None = None) -> tuple[Array, Array]:
    """FlashAttention-2 forward: outer loop over Q blocks, inner over K/V blocks.

    Keeps running max m, normalizer l and unnormalized output per Q block; never forms T x T.
    Causal: K/V blocks entirely above the diagonal are skipped, the diagonal block is masked.
    Returns (o, lse) where lse = m + log l is saved for the backward pass.
    """
    _check(q, k, causal)
    sc = _scale(q.shape[-1], scale)
    tq, tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    batch = int(np.prod(q.shape[:-2], dtype=np.int64))
    o = np.empty(q.shape[:-1] + (v.shape[-1],))
    lse = np.empty(q.shape[:-1])
    for r0 in range(0, tq, br):
        r1 = min(r0 + br, tq)
        qi = q[..., r0:r1, :]
        m = np.full(qi.shape[:-1], -np.inf)
        l = np.zeros(qi.shape[:-1])
        acc = np.zeros(qi.shape[:-1] + (v.shape[-1],))
        if io:
            io.reads += batch * (r1 - r0) * d
        for c0 in range(0, tk, bc):
            c1 = min(c0 + bc, tk)
            if causal and c0 > r1 - 1:
                if io:
                    io.blocks_skipped += -(-(tk - c0) // bc)
                break
            kj, vj = k[..., c0:c1, :], v[..., c0:c1, :]
            if io:
                io.reads += batch * (c1 - c0) * (d + v.shape[-1])
                io.blocks_visited += 1
            s = (qi @ _t(kj)) * sc
            if causal and c1 - 1 > r0:
                s = np.where(_block_mask(r0, r1, c0, c1), s, -np.inf)
            m_new = np.maximum(m, s.max(axis=-1))
            alpha = np.exp(m - m_new)
            p = np.exp(s - m_new[..., None])
            l = alpha * l + p.sum(axis=-1)
            acc = alpha[..., None] * acc + p @ vj
            m = m_new
        o[..., r0:r1, :] = acc / l[..., None]
        lse[..., r0:r1] = m + np.log(l)
        if io:
            io.writes += batch * (r1 - r0) * (v.shape[-1] + 1)
    return o, lse


def flash_backward(q: Array, k: Array, v: Array, o: Array, do: Array, lse: Array,
                   causal: bool = False, br: int = 64, bc: int = 64, scale: float | None = None,
                   io: IOCounter | None = None) -> tuple[Array, Array, Array]:
    """FlashAttention-2 backward: recompute P = exp(S - lse) blockwise instead of storing it.

    Outer loop over K/V blocks (dK, dV accumulate in SRAM), inner over Q blocks
    (dQ accumulated in HBM). D = rowsum(dO * O) replaces rowsum(dP * P).
    """
    _check(q, k, causal)
    sc = _scale(q.shape[-1], scale)
    tq, tk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    dv_dim = v.shape[-1]
    batch = int(np.prod(q.shape[:-2], dtype=np.int64))
    delta = (do * o).sum(axis=-1)
    if io:
        io.reads += batch * tq * 2 * dv_dim
        io.writes += batch * tq
    dq = np.zeros_like(q, dtype=np.result_type(q, np.float64))
    dk = np.zeros_like(k, dtype=dq.dtype)
    dv = np.zeros_like(v, dtype=dq.dtype)
    for c0 in range(0, tk, bc):
        c1 = min(c0 + bc, tk)
        kj, vj = k[..., c0:c1, :], v[..., c0:c1, :]
        dkj = np.zeros_like(kj, dtype=dq.dtype)
        dvj = np.zeros_like(vj, dtype=dq.dtype)
        if io:
            io.reads += batch * (c1 - c0) * (d + dv_dim)
        r_start = (c0 // br) * br if causal else 0
        for r0 in range(r_start, tq, br):
            r1 = min(r0 + br, tq)
            qi, doi = q[..., r0:r1, :], do[..., r0:r1, :]
            if io:
                io.reads += batch * (r1 - r0) * (2 * d + dv_dim + 2)
                io.writes += batch * (r1 - r0) * d
                io.blocks_visited += 1
            s = (qi @ _t(kj)) * sc
            if causal and c1 - 1 > r0:
                s = np.where(_block_mask(r0, r1, c0, c1), s, -np.inf)
            p = np.exp(s - lse[..., r0:r1, None])
            dvj += _t(p) @ doi
            dp = doi @ _t(vj)
            ds = p * (dp - delta[..., r0:r1, None])
            dq[..., r0:r1, :] += sc * ds @ kj
            dkj += sc * _t(ds) @ qi
        dk[..., c0:c1, :] = dkj
        dv[..., c0:c1, :] = dvj
        if io:
            io.writes += batch * (c1 - c0) * (d + dv_dim)
    return dq, dk, dv
