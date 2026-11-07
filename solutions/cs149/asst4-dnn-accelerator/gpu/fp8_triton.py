"""Hand-scheduled FP8 GEMM on Ada tensor cores, via Triton.

cuBLAS `_scaled_mm` is the ceiling. The fp32 ladder already sits near 80% of cuBLAS and
stops moving; FP8 is the unused 2x. This sweeps tile, pipeline stages, warp count and
the L2-swizzled launch order, and records each config against `_scaled_mm`.

    python gpu/fp8_triton.py
Writes artifacts/raw/gpu/fp8_triton.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

_LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB.parents[2]))
from mlsys_harness.paths import raw_dir  # noqa: E402

RAW = raw_dir(_LAB, "FP8_OUT")
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, group_m)
CONFIGS = [
    (64, 64, 64, 4, 3, 8),
    (128, 64, 64, 4, 3, 8),
    (64, 128, 64, 4, 3, 8),
    (128, 128, 64, 4, 3, 8),
    (128, 128, 64, 8, 3, 8),
    (128, 128, 128, 8, 3, 8),
    (128, 256, 64, 8, 3, 8),
    (256, 128, 64, 8, 3, 8),
    (128, 128, 64, 8, 4, 8),
    (128, 128, 64, 8, 2, 8),
    (128, 128, 64, 8, 3, 1),  # same tile, row-major grid: the L2-swizzle ablation
]
SIZES = (1024, 2048, 4096, 8192)


@triton.jit
def _fp8_gemm(a_ptr, b_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.bfloat16)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, c, mask=mask)


def _ms(fn, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _cublas(a, b):
    # b is the same (K, N) column-major view the Triton kernel reads.
    one = torch.ones(1, device=a.device, dtype=torch.float32)
    return torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)


def _triton(a, b, cfg):
    bm, bn, bk, warps, stages, group = cfg
    m, k = a.shape
    n = b.shape[0]
    c = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _fp8_gemm[grid](
        a, b, c, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group,
        num_warps=warps, num_stages=stages,
    )
    return c


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "fp8_triton.jsonl"
    rows: list[dict] = []
    device = torch.cuda.get_device_properties(0)
    print(f"fp8 sweep on {device.name} cc={device.major}.{device.minor}", flush=True)
    # One correctness check, then the sweep. fp8 quantization noise dominates, so the
    # reference is cuBLAS on the same fp8 inputs, not the fp32 product.
    torch.manual_seed(0)
    # (N, K) row-major, transposed to the (K, N) column-major operand cuBLAS requires.
    # Triton consumes that view through its strides, so both sides multiply the same B.
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
    try:
        got = _triton(a, b, CONFIGS[4])
        ref = _cublas(a, b)
        err = (got.float() - ref.float()).abs().max().item()
    except Exception as exc:
        print("fp8 kernel failed:", exc, flush=True)
        rows.append({"part": "error", "error": str(exc)[:400]})
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return 1
    print(f"correctness max abs vs cuBLAS {err:.4f}", flush=True)
    rows.append({"part": "correctness", "N": 1024, "max_abs_err": err, "config": list(CONFIGS[4])})
    if err > 1.0:
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        print("correctness failed, not sweeping", flush=True)
        return 1

    for n in SIZES:
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t()
        ms_ref = _ms(lambda: _cublas(a, b))
        tflops_ref = 2 * n ** 3 / ms_ref / 1e9
        rows.append({"part": "cublas", "N": n, "ms": ms_ref, "tflops": tflops_ref})
        print(f"cublas N={n} {tflops_ref:7.1f} TFLOP/s", flush=True)
        best = 0.0
        for cfg in CONFIGS:
            try:
                ms = _ms(lambda cfg=cfg: _triton(a, b, cfg))
                tf = 2 * n ** 3 / ms / 1e9
                err_s = ""
            except Exception as exc:
                ms, tf, err_s = float("nan"), 0.0, str(exc)[:160]
            best = max(best, tf)
            row = {"part": "triton", "N": n, "BM": cfg[0], "BN": cfg[1], "BK": cfg[2],
                   "warps": cfg[3], "stages": cfg[4], "group_m": cfg[5],
                   "ms": ms, "tflops": tf, "frac_of_cublas": tf / tflops_ref if tflops_ref else 0.0,
                   "error": err_s}
            rows.append(row)
            print(f"  {cfg} {tf:7.1f} TFLOP/s  {row['frac_of_cublas']:.2f} {err_s}", flush=True)
        rows.append({"part": "best", "N": n, "tflops": best, "cublas_tflops": tflops_ref,
                     "frac_of_cublas": best / tflops_ref if tflops_ref else 0.0})
        print(f"best N={n} {best:.1f} / cublas {tflops_ref:.1f}", flush=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
