"""Attention benchmarks on GPU: Triton FlashAttention vs naive PyTorch vs SDPA backends.

python gpu/bench_attention.py --part fwd|bwd|bh|blocks
fwd    : T = 256..32768, D = 64/128, causal/non-causal, fp16/bf16, B*H = 2*16; time, TFLOP/s, peak memory,
         max |err| vs fp32 reference (first 2 heads, chunked), OOM point of materializing methods.
bwd    : forward+backward time and gradient error (fp16).
bh     : batch*heads sweep at T=4096 (occupancy).
blocks : Triton BLOCK_M x BLOCK_N sweep with the IO-model HBM bytes for each block size.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import JsonlWriter, cuda_time_ms, env_info, is_oom, peak_extra_bytes
from flash_triton import flash_attention, flash_attn_fwd, naive_attention, reference_fp32

DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
SDPA = {"sdpa_flash": SDPBackend.FLASH_ATTENTION, "sdpa_efficient": SDPBackend.EFFICIENT_ATTENTION,
        "sdpa_math": SDPBackend.MATH}
METHODS = ["triton", "sdpa_flash", "sdpa_efficient", "sdpa_math", "naive"]


def fwd_fn(method: str, q, k, v, causal: bool):
    if method == "triton":
        return lambda: flash_attn_fwd(q, k, v, causal)[0]
    if method == "naive":
        return lambda: naive_attention(q, k, v, causal)

    def run():
        with sdpa_kernel(SDPA[method]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return run


def attn_flops(bh: int, t: int, d: int, causal: bool) -> float:
    """QK^T and PV: 4*T^2*d per head; causal counts only the useful half."""
    return 4.0 * bh * t * t * d * (0.5 if causal else 1.0)


def free():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def part_fwd(w: JsonlWriter, quick: bool) -> None:
    b, h = 2, 16
    ts = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    if quick:
        ts = ts[:3]
    for dtype in ("fp16", "bf16"):
        for d in (64, 128):
            for causal in (False, True):
                dead: set[str] = set()
                for t in ts:
                    torch.manual_seed(t)
                    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=DT[dtype]) for _ in range(3))
                    o_ref, _ = reference_fp32(q[:, :1], k[:, :1], v[:, :1], causal)
                    o_ref = o_ref[0, 0]  # batch 0, head 0
                    for m in METHODS:
                        row = {"part": "fwd", "method": m, "dtype": dtype, "B": b, "H": h, "T": t, "D": d,
                               "causal": causal, "flops": attn_flops(b * h, t, d, causal),
                               "input_bytes": 3 * q.numel() * q.element_size()}
                        if m in dead:
                            row["status"] = "skipped_after_oom"
                            w.write(row)
                            continue
                        fn = fwd_fn(m, q, k, v, causal)
                        try:
                            out = fn()
                            row["max_err"] = (out[0, 0].float() - o_ref).abs().max().item()
                            row["mean_err"] = (out[0, 0].float() - o_ref).abs().mean().item()
                            del out
                            row["peak_extra_bytes"] = peak_extra_bytes(fn)
                            row["ms"] = cuda_time_ms(fn, warmup=2, reps=10)
                            row["tflops"] = row["flops"] / row["ms"] / 1e9
                            row["status"] = "ok"
                        except Exception as exc:  # noqa: BLE001
                            row["status"] = "oom" if is_oom(exc) else "error"
                            row["error"] = str(exc).splitlines()[0][:200]
                            if is_oom(exc):
                                dead.add(m)
                        free()
                        w.write(row)
                        print(f"fwd {dtype} d={d} c={int(causal)} T={t:<6} {m:<15} {row['status']:<6} "
                              f"{row.get('ms', float('nan')):9.3f} ms {row.get('tflops', 0):7.1f} TF "
                              f"err={row.get('max_err', float('nan')):.2e} "
                              f"mem={row.get('peak_extra_bytes', 0) / 2**20:.0f}MiB", flush=True)
                    del q, k, v, o_ref
                    free()


def part_bwd(w: JsonlWriter, quick: bool) -> None:
    b, h = 2, 16
    ts = [256, 512, 1024, 2048, 4096, 8192, 16384]
    if quick:
        ts = ts[:2]
    methods = ["triton", "sdpa_flash", "sdpa_efficient", "naive"]
    for d in (64, 128):
        for causal in (False, True):
            dead: set[str] = set()
            for t in ts:
                torch.manual_seed(t)
                q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.float16, requires_grad=True)
                           for _ in range(3))
                do = torch.randn(b, h, t, d, device="cuda", dtype=torch.float16)
                ref = None
                if t <= 4096:
                    qf, kf, vf = (x[:, :2].detach().float().requires_grad_(True) for x in (q, k, v))
                    with sdpa_kernel(SDPBackend.MATH):
                        F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal).backward(do[:, :2].float())
                    ref = (qf.grad, kf.grad, vf.grad)
                for m in methods:
                    row = {"part": "bwd", "method": m, "dtype": "fp16", "B": b, "H": h, "T": t, "D": d,
                           "causal": causal, "flops": 3.5 * attn_flops(b * h, t, d, causal)}
                    if m in dead:
                        row["status"] = "skipped_after_oom"
                        w.write(row)
                        continue
                    if m == "triton":
                        f = lambda: flash_attention(q, k, v, causal)  # noqa: E731
                    else:
                        f = fwd_fn(m, q, k, v, causal)

                    def step():
                        for x in (q, k, v):
                            x.grad = None
                        f().backward(do)
                    try:
                        step()
                        if ref is not None:
                            row["max_err_dq"], row["max_err_dk"], row["max_err_dv"] = (
                                (x.grad[:, :2].float() - r).abs().max().item() for x, r in zip((q, k, v), ref))
                        row["peak_extra_bytes"] = peak_extra_bytes(step)
                        row["ms"] = cuda_time_ms(step, warmup=2, reps=10)
                        row["tflops"] = row["flops"] / row["ms"] / 1e9
                        row["status"] = "ok"
                    except Exception as exc:  # noqa: BLE001
                        row["status"] = "oom" if is_oom(exc) else "error"
                        row["error"] = str(exc).splitlines()[0][:200]
                        if is_oom(exc):
                            dead.add(m)
                    for x in (q, k, v):
                        x.grad = None
                    free()
                    w.write(row)
                    print(f"bwd d={d} c={int(causal)} T={t:<6} {m:<15} {row['status']:<6} "
                          f"{row.get('ms', float('nan')):9.3f} ms {row.get('tflops', 0):7.1f} TF "
                          f"err_dq={row.get('max_err_dq', float('nan')):.2e}", flush=True)
                del q, k, v, do, ref
                free()


def part_bh(w: JsonlWriter, quick: bool) -> None:
    t, d = 4096, 64
    for causal in (False, True):
        for b, h in [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (2, 16), (4, 16), (8, 16), (16, 16)]:
            q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.float16) for _ in range(3))
            for m in ("triton", "sdpa_flash", "naive"):
                fn = fwd_fn(m, q, k, v, causal)
                row = {"part": "bh", "method": m, "dtype": "fp16", "B": b, "H": h, "T": t, "D": d,
                       "causal": causal, "flops": attn_flops(b * h, t, d, causal)}
                try:
                    row["ms"] = cuda_time_ms(fn, warmup=2, reps=20)
                    row["tflops"] = row["flops"] / row["ms"] / 1e9
                    row["status"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "oom" if is_oom(exc) else "error"
                w.write(row)
                print(f"bh c={int(causal)} BH={b * h:<4} {m:<12} {row.get('ms', float('nan')):8.3f} ms "
                      f"{row.get('tflops', 0):6.1f} TF", flush=True)
                free()
            del q, k, v


def model_bytes(bh: int, t: int, d: int, bm: int, bn: int, causal: bool, el: int = 2) -> int:
    """IO model of the kernel: Q and O once, K/V once per (Q block, visited K/V block), lse once."""
    kv = 0
    for r0 in range(0, t, bm):
        r1 = min(r0 + bm, t)
        keys = min(t, -(-r1 // bn) * bn) if causal else t
        kv += keys * 2 * d
    return bh * (el * (2 * t * d + kv) + 4 * t)


def part_blocks(w: JsonlWriter, quick: bool) -> None:
    b, h = 2, 16
    for t in (4096, 16384):
        for d in (64, 128):
            q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=torch.float16) for _ in range(3))
            o_ref, _ = reference_fp32(q[:, :1], k[:, :1], v[:, :1], False)
            for bm in (16, 32, 64, 128, 256):
                for bn in (16, 32, 64, 128):
                    row = {"part": "blocks", "method": "triton", "dtype": "fp16", "B": b, "H": h, "T": t, "D": d,
                           "causal": False, "BLOCK_M": bm, "BLOCK_N": bn, "flops": attn_flops(b * h, t, d, False),
                           "model_bytes": model_bytes(b * h, t, d, bm, bn, False)}
                    nw = 8 if bm * d >= 128 * 128 or bm >= 256 else 4
                    for ns in (2, 1):
                        try:
                            fn = lambda: flash_attn_fwd(q, k, v, False, block_m=bm, block_n=bn,  # noqa: E731
                                                        num_warps=nw, num_stages=ns)[0]
                            out = fn()
                            row["max_err"] = (out[0, 0].float() - o_ref[0, 0]).abs().max().item()
                            row["ms"] = cuda_time_ms(fn, warmup=2, reps=10)
                            row.update({"num_warps": nw, "num_stages": ns, "status": "ok",
                                        "tflops": row["flops"] / row["ms"] / 1e9,
                                        "model_gbps": row["model_bytes"] / row["ms"] / 1e6})
                            break
                        except Exception as exc:  # noqa: BLE001
                            row["status"] = "error"
                            row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                    w.write(row)
                    print(f"blocks T={t} d={d} BM={bm:<3} BN={bn:<3} {row['status']:<5} "
                          f"{row.get('ms', float('nan')):8.3f} ms {row.get('tflops', 0):6.1f} TF "
                          f"model {row['model_bytes'] / 1e9:.2f} GB", flush=True)
                    free()
            del q, k, v
            free()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["fwd", "bwd", "bh", "blocks"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    w = JsonlWriter(f"attention_{a.part}.jsonl")
    w.write({"part": "meta", **env_info()})
    t0 = time.time()
    {"fwd": part_fwd, "bwd": part_bwd, "bh": part_bh, "blocks": part_blocks}[a.part](w, a.quick)
    w.write({"part": "meta_end", "elapsed_s": round(time.time() - t0, 1)})
    w.close()


if __name__ == "__main__":
    main()
