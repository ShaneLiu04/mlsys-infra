"""FlashAttention-2 forward + backward in Triton.

Layout: q, k, v are (B, H, T, D) contiguous fp16/bf16, D in {16, 32, 64, 128}.
Forward: one program per (Q block, batch*head); online softmax over K/V blocks in base 2
(exp2 with log2(e) folded into the scale); causal = unmasked off-diagonal blocks + masked
diagonal blocks, blocks above the diagonal are never loaded. Saves lse (natural log) for backward.
Backward: delta = rowsum(dO * O) (torch), then two kernels: dK/dV (outer loop over K/V blocks,
inner over Q blocks) and dQ (outer over Q blocks, inner over K/V blocks); P is recomputed as
exp(S - lse), never stored.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@triton.jit
def _fwd_inner(acc, l_i, m_i, q, k_base, v_base, offs_m, offs_n, offs_d, lo, hi, T, qk_scale,
               HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, MASK_CAUSAL: tl.constexpr,
               BOUNDARY: tl.constexpr):
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        k_ptrs = k_base + cols[None, :] * HEAD_DIM + offs_d[:, None]
        v_ptrs = v_base + cols[:, None] * HEAD_DIM + offs_d[None, :]
        if BOUNDARY:
            k = tl.load(k_ptrs, mask=cols[None, :] < T, other=0.0)
        else:
            k = tl.load(k_ptrs)
        qk = tl.dot(q, k) * qk_scale
        if MASK_CAUSAL:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, float("-inf"))
        if BOUNDARY:
            qk = tl.where(cols[None, :] < T, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        if BOUNDARY:
            v = tl.load(v_ptrs, mask=cols[:, None] < T, other=0.0)
        else:
            v = tl.load(v_ptrs)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(Q, K, V, O, LSE, sm_scale, T, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr, BOUNDARY: tl.constexpr):
    num_m = tl.cdiv(T, BLOCK_M)
    start_m = num_m - 1 - tl.program_id(0)  # heaviest causal rows first
    bh = tl.program_id(1).to(tl.int64)
    base = bh * T * HEAD_DIM
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    if BOUNDARY:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < T, other=0.0)
    else:
        q = tl.load(q_ptrs)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    if CAUSAL:
        lo_diag = (start_m * BLOCK_M // BLOCK_N) * BLOCK_N
        acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q, K + base, V + base, offs_m, offs_n, offs_d,
                                   0, lo_diag, T, qk_scale, HEAD_DIM, BLOCK_N, False, BOUNDARY)
        acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q, K + base, V + base, offs_m, offs_n, offs_d,
                                   lo_diag, tl.minimum((start_m + 1) * BLOCK_M, T), T, qk_scale,
                                   HEAD_DIM, BLOCK_N, True, BOUNDARY)
    else:
        acc, l_i, m_i = _fwd_inner(acc, l_i, m_i, q, K + base, V + base, offs_m, offs_n, offs_d,
                                   0, T, T, qk_scale, HEAD_DIM, BLOCK_N, False, BOUNDARY)
    o = acc / l_i[:, None]
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    o_ptrs = O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    if BOUNDARY:
        tl.store(o_ptrs, o.to(O.dtype.element_ty), mask=offs_m[:, None] < T)
        tl.store(LSE + bh * T + offs_m, lse, mask=offs_m < T)
    else:
        tl.store(o_ptrs, o.to(O.dtype.element_ty))
        tl.store(LSE + bh * T + offs_m, lse)


@triton.jit
def _attn_bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, sm_scale, T, HEAD_DIM: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_n = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * T * HEAD_DIM
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_m0 = tl.arange(0, BLOCK_M)
    nmask = offs_n < T
    k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=nmask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    lo = 0
    if CAUSAL:
        lo = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    for start_m in range(lo, T, BLOCK_M):
        offs_m = start_m + offs_m0
        mmask = offs_m < T
        q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mmask[:, None], other=0.0)
        do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mmask[:, None], other=0.0)
        lse = tl.load(LSE + bh * T + offs_m, mask=mmask, other=0.0) * 1.4426950408889634
        delta = tl.load(DELTA + bh * T + offs_m, mask=mmask, other=0.0)
        qk_t = tl.dot(k, tl.trans(q)) * qk_scale
        valid = mmask[None, :] & nmask[:, None]
        if CAUSAL:
            valid = valid & (offs_m[None, :] >= offs_n[:, None])
        p_t = tl.where(valid, tl.math.exp2(qk_t - lse[None, :]), 0.0)
        dv += tl.dot(p_t.to(do.dtype), do)
        dp_t = tl.dot(v, tl.trans(do))
        ds_t = p_t * (dp_t - delta[None, :])
        dk += tl.dot(ds_t.to(q.dtype), q)
    dk = dk * sm_scale
    tl.store(DK + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DV + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _attn_bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, sm_scale, T, HEAD_DIM: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * T * HEAD_DIM
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n0 = tl.arange(0, BLOCK_N)
    mmask = offs_m < T
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mmask[:, None], other=0.0)
    do = tl.load(DO + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=mmask[:, None], other=0.0)
    lse = tl.load(LSE + bh * T + offs_m, mask=mmask, other=0.0) * 1.4426950408889634
    delta = tl.load(DELTA + bh * T + offs_m, mask=mmask, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    hi = T
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, T)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n0
        nmask = offs_n < T
        k = tl.load(K + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        valid = mmask[:, None] & nmask[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        p = tl.where(valid, tl.math.exp2(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq = dq * sm_scale
    tl.store(DQ + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=mmask[:, None])


FWD_CONFIGS = {  # (BLOCK_M, BLOCK_N, num_warps, num_stages), tried in order until one fits SMEM
    64: [(128, 64, 4, 3), (128, 64, 4, 2), (64, 64, 4, 2), (64, 32, 4, 1)],
    128: [(128, 64, 8, 2), (64, 64, 4, 2), (64, 32, 4, 1), (32, 32, 4, 1)],
}
BWD_CONFIGS = {
    64: [(64, 64, 4, 2), (64, 64, 4, 1), (32, 32, 4, 1)],
    128: [(64, 64, 8, 2), (64, 64, 8, 1), (32, 64, 4, 1), (32, 32, 4, 1)],
}
_CHOSEN: dict = {}


def _is_resource_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__} {exc}".lower()
    return "outofresources" in msg or "shared memory" in msg or "out of resource" in msg


def _run_with_fallback(key, configs, launch):
    if key in _CHOSEN:
        return launch(*_CHOSEN[key])
    last = None
    for cfg in configs:
        try:
            out = launch(*cfg)
            _CHOSEN[key] = cfg
            return out
        except Exception as exc:  # noqa: BLE001
            if not _is_resource_error(exc):
                raise
            last = exc
    raise RuntimeError(f"no config fits for {key}: {last}")


def _check_inputs(q, k, v):
    assert q.shape == k.shape == v.shape and q.dim() == 4, "expects (B, H, T, D) self-attention"
    assert q.dtype in (torch.float16, torch.bfloat16) and q.is_cuda
    assert q.shape[-1] in (16, 32, 64, 128)
    return q.contiguous(), k.contiguous(), v.contiguous()


def flash_attn_fwd(q, k, v, causal: bool = False, sm_scale: float | None = None,
                   block_m: int | None = None, block_n: int | None = None,
                   num_warps: int | None = None, num_stages: int | None = None):
    """Returns (o, lse) with lse of shape (B, H, T) in float32 (natural log)."""
    q, k, v = _check_inputs(q, k, v)
    b, h, t, d = q.shape
    sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    o = torch.empty_like(q)
    lse = torch.empty((b, h, t), device=q.device, dtype=torch.float32)

    def launch(bm, bn, nw, ns):
        grid = (triton.cdiv(t, bm), b * h)
        _attn_fwd[grid](q, k, v, o, lse, sm_scale, t, HEAD_DIM=d, BLOCK_M=bm, BLOCK_N=bn,
                        CAUSAL=causal, BOUNDARY=(t % bm != 0) or (t % bn != 0),
                        num_warps=nw, num_stages=ns)
        return o, lse

    if block_m is not None:
        nw = num_warps or (8 if block_m * d >= 128 * 128 else 4)
        return launch(block_m, block_n, nw, num_stages or 2)
    configs = FWD_CONFIGS.get(d, FWD_CONFIGS[64])
    return _run_with_fallback(("fwd", d, q.dtype), configs, launch)


def flash_attn_bwd(q, k, v, o, lse, do, causal: bool = False, sm_scale: float | None = None):
    b, h, t, d = q.shape
    sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    do = do.contiguous()
    delta = (o.float() * do.float()).sum(-1)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

    def launch(bm, bn, nw, ns):
        _attn_bwd_dkdv[(triton.cdiv(t, bn), b * h)](q, k, v, do, lse, delta, dk, dv, sm_scale, t,
                                                     HEAD_DIM=d, BLOCK_M=bm, BLOCK_N=bn, CAUSAL=causal,
                                                     num_warps=nw, num_stages=ns)
        _attn_bwd_dq[(triton.cdiv(t, bm), b * h)](q, k, v, do, lse, delta, dq, sm_scale, t,
                                                   HEAD_DIM=d, BLOCK_M=bm, BLOCK_N=bn, CAUSAL=causal,
                                                   num_warps=nw, num_stages=ns)
        return dq, dk, dv

    return _run_with_fallback(("bwd", d, q.dtype), BWD_CONFIGS.get(d, BWD_CONFIGS[64]), launch)


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        o, lse = flash_attn_fwd(q, k, v, causal, sm_scale)
        ctx.save_for_backward(q.contiguous(), k.contiguous(), v.contiguous(), o, lse)
        ctx.causal, ctx.sm_scale = causal, sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_attn_bwd(q, k, v, o, lse, do, ctx.causal, ctx.sm_scale)
        return dq, dk, dv, None, None


def flash_attention(q, k, v, causal: bool = False, sm_scale: float | None = None):
    return FlashAttnFunc.apply(q, k, v, causal, sm_scale)


def naive_attention(q, k, v, causal: bool = False, sm_scale: float | None = None):
    """Eager PyTorch: materializes S = QK^T and P = softmax(S) as (B, H, T, T) tensors."""
    sm_scale = 1.0 / math.sqrt(q.shape[-1]) if sm_scale is None else sm_scale
    s = torch.matmul(q, k.transpose(-1, -2)) * sm_scale
    if causal:
        t = q.shape[-2]
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def reference_fp32(q, k, v, causal: bool = False, chunk: int = 2048):
    """fp32 attention computed in query chunks (memory O(chunk * T)); returns (o, lse)."""
    qf, kf, vf = q.float(), k.float(), v.float()
    t = q.shape[-2]
    scale = 1.0 / math.sqrt(q.shape[-1])
    outs, lses = [], []
    for r0 in range(0, t, chunk):
        r1 = min(r0 + chunk, t)
        s = torch.matmul(qf[..., r0:r1, :], kf.transpose(-1, -2)) * scale
        if causal:
            rows = torch.arange(r0, r1, device=q.device)[:, None]
            cols = torch.arange(t, device=q.device)[None, :]
            s = s.masked_fill(cols > rows, float("-inf"))
        lses.append(torch.logsumexp(s, -1))
        outs.append(torch.matmul(torch.softmax(s, -1), vf))
        del s
    return torch.cat(outs, -2), torch.cat(lses, -1)
