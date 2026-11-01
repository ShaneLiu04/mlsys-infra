"""Handwritten Triton kernels: fused RMSNorm, fused bias+GELU, tiled matmul, weight-only GEMV."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------- fused RMSNorm

@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    tl.store(y_ptr + row * n_cols + cols, (x * rstd * w).to(y_ptr.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    y = torch.empty_like(x2)
    block = triton.next_power_of_2(x2.shape[1])
    _rmsnorm_kernel[(x2.shape[0],)](x2, w, y, x2.shape[1], eps, BLOCK=block,
                                    num_warps=min(16, max(4, block // 256)))
    return y.view_as(x)


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


# --------------------------------------------------------------------------- fused bias + GELU

@triton.jit
def _bias_gelu_kernel(x_ptr, b_ptr, y_ptr, n_elem, n_cols, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elem
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs % n_cols, mask=mask, other=0.0).to(tl.float32)
    v = x + b
    y = 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476))
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def bias_gelu(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    block = 4096
    _bias_gelu_kernel[(triton.cdiv(n, block),)](x, b, y, n, x.shape[-1], BLOCK=block, num_warps=8)
    return y


# --------------------------------------------------------------------------- tiled matmul

@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, sam, sak, sbk, sbn, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % group) % gsize)
    pid_n = (pid % group) // gsize
    rm = (pid_m * BM + tl.arange(0, BM)) % M
    rn = (pid_n * BN + tl.arange(0, BN)) % N
    rk = tl.arange(0, BK)
    a = a_ptr + rm[:, None] * sam + rk[None, :] * sak
    b = b_ptr + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        av = tl.load(a, mask=rk[None, :] < K - k * BK, other=0.0)
        bv = tl.load(b, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(av, bv, acc)
        a += BK * sak
        b += BK * sbk
    cm = pid_m * BM + tl.arange(0, BM)
    cn = pid_n * BN + tl.arange(0, BN)
    c = c_ptr + cm[:, None] * scm + cn[None, :] * scn
    tl.store(c, acc.to(c_ptr.dtype.element_ty), mask=(cm[:, None] < M) & (cn[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor, bm: int, bn: int, bk: int, num_warps: int,
           num_stages: int, group_m: int = 8) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _matmul_kernel[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1), BM=bm, BN=bn, BK=bk, GROUP_M=group_m,
                         num_warps=num_warps, num_stages=num_stages)
    return c


# Triton tutorial style candidate list for @triton.autotune.
AUTOTUNE_CONFIGS = [
    triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 64, "BN": 256, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 128, "BN": 32, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BM": 64, "BN": 32, "BK": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BM": 32, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
    triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
    triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
]

_autotuned = triton.autotune(configs=AUTOTUNE_CONFIGS, key=["M", "N", "K"])(_matmul_kernel)


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),)

    _autotuned[grid](a, b, c, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1))
    return c


def autotune_pick() -> dict:
    cfg = next(iter(_autotuned.cache.values()))
    return {**cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}


# --------------------------------------------------------------------------- weight-only GEMV

@triton.jit
def _gemv_w8_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    """y[n] = s[n] * sum_k x[k] * W[n, k], W int8 row-major [N, K]."""
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rk).to(tl.float32)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    s = tl.load(s_ptr + rn).to(tl.float32)
    tl.store(y_ptr + rn, (acc * s).to(y_ptr.dtype.element_ty))


@triton.jit
def _gemv_w4_kernel(x_ptr, w_ptr, s_ptr, y_ptr, N, K, G: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr):
    """W packed two int4 per byte along K: byte j of row n holds k=2j (low) and 2j+1 (high).
    Group scales s[n, k // G]; values are signed int4 stored with +8 offset."""
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    half: tl.constexpr = BK // 2
    for k0 in range(0, K, BK):
        rj = k0 // 2 + tl.arange(0, half)
        packed = tl.load(w_ptr + rn[:, None] * (K // 2) + rj[None, :]).to(tl.int32)
        lo = ((packed & 0xF) - 8).to(tl.float32)
        hi = (((packed >> 4) & 0xF) - 8).to(tl.float32)
        xe = tl.load(x_ptr + k0 + 2 * tl.arange(0, half)).to(tl.float32)
        xo = tl.load(x_ptr + k0 + 2 * tl.arange(0, half) + 1).to(tl.float32)
        s = tl.load(s_ptr + rn * (K // G) + k0 // G).to(tl.float32)
        acc += s * tl.sum(lo * xe[None, :] + hi * xo[None, :], axis=1)
    tl.store(y_ptr + rn, acc.to(y_ptr.dtype.element_ty))


def quant_w8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = w.float().abs().amax(1).clamp_min(1e-8) / 127
    q = torch.round(w.float() / s[:, None]).clamp(-127, 127).to(torch.int8)
    return q, s.to(torch.float16)


def quant_w4(w: torch.Tensor, g: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    n, k = w.shape
    wg = w.float().view(n, k // g, g)
    s = wg.abs().amax(-1).clamp_min(1e-8) / 7
    q = (torch.round(wg / s[..., None]).clamp(-8, 7) + 8).to(torch.uint8).view(n, k)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, s.to(torch.float16).contiguous()


def dequant_w4(packed: torch.Tensor, s: torch.Tensor, g: int = 128) -> torch.Tensor:
    n = packed.shape[0]
    lo = (packed & 0xF).to(torch.int16) - 8
    hi = (packed >> 4).to(torch.int16) - 8
    q = torch.stack([lo, hi], -1).view(n, -1).float()
    return (q.view(n, -1, g) * s.float()[..., None]).view(n, -1)


def gemv_w8(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor, bn: int = 16, bk: int = 256):
    n, k = q.shape
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    _gemv_w8_kernel[(n // bn,)](x, q, s, y, n, k, BN=bn, BK=bk, num_warps=4)
    return y


def gemv_w4(x: torch.Tensor, packed: torch.Tensor, s: torch.Tensor, g: int = 128, bn: int = 16,
            bk: int = 128):
    n = packed.shape[0]
    k = packed.shape[1] * 2
    assert g % bk == 0 and k % bk == 0
    y = torch.empty(n, device=x.device, dtype=x.dtype)
    _gemv_w4_kernel[(n // bn,)](x, packed, s, y, n, k, G=g, BN=bn, BK=bk, num_warps=4)
    return y
