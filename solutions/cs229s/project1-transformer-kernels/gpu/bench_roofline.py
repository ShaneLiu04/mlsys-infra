"""5090 roofline: measured copy bandwidth and BF16 matmul peak, then the pieces of one Llama-2-7B layer
(QKV projection, attention, output projection, gated MLP) for prefill and decode shapes.

Analytic FLOPs/bytes per op come from tools/mlsys_calc.py (gemm_intensity, kv_cache_bytes) so measured
points can be compared with the closed-form intensities.
"""

from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import RAW_GPU, JsonlWriter, cuda_time_ms, env_info

import mlsys_calc as calc  # noqa: E402  (path set up by common)

MODEL = calc.MODELS["llama2-7b"]
BF16 = torch.bfloat16


def bench_copy(w: JsonlWriter) -> float:
    best = 0.0
    for log2 in range(16, 32):  # 64 KiB .. 2 GiB per buffer
        n = (1 << log2) // 2
        x = torch.empty(n, device="cuda", dtype=BF16).normal_()
        y = torch.empty_like(x)
        ms = cuda_time_ms(lambda: y.copy_(x), warmup=5, reps=50 if log2 < 26 else 20)
        moved = 2 * n * 2
        gbps = moved / ms / 1e6
        w.write({"part": "copy", "buffer_bytes": n * 2, "bytes_moved": moved, "ms": ms, "gbps": gbps})
        print(f"copy {n * 2 / 2**20:9.2f} MiB  {gbps:8.1f} GB/s", flush=True)
        if n * 2 >= 256 * 2**20:  # well beyond the 96 MB L2: DRAM bandwidth
            best = max(best, gbps)
        del x, y
    # triad-like read-only reduction as a second estimate of read bandwidth
    x = torch.empty(1 << 29, device="cuda", dtype=BF16).normal_()
    ms = cuda_time_ms(lambda: x.sum(dtype=torch.float32), warmup=3, reps=20)
    w.write({"part": "read", "buffer_bytes": x.numel() * 2, "bytes_moved": x.numel() * 2, "ms": ms,
             "gbps": x.numel() * 2 / ms / 1e6})
    print(f"read-only sum 1 GiB {x.numel() * 2 / ms / 1e6:.1f} GB/s")
    return best * 1e9


def bench_matmul(w: JsonlWriter) -> float:
    best = 0.0
    for dtype_name, dtype in (("bf16", BF16), ("fp16", torch.float16)):
        for n in (256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384):
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            b = torch.randn(n, n, device="cuda", dtype=dtype)
            ms = cuda_time_ms(lambda: a @ b, warmup=5, reps=30)
            tf = 2 * n ** 3 / ms / 1e9
            w.write({"part": "matmul", "dtype": dtype_name, "M": n, "N": n, "K": n, "ms": ms, "tflops": tf,
                     "intensity": calc.gemm_intensity(n, n, n)})
            print(f"matmul {dtype_name} {n:6d}  {tf:7.1f} TFLOP/s", flush=True)
            if dtype_name == "bf16":
                best = max(best, tf)
            del a, b
    a = torch.randn(8192, 8192, device="cuda")
    ms = cuda_time_ms(lambda: a @ a, warmup=3, reps=10)
    w.write({"part": "matmul", "dtype": "fp32", "M": 8192, "N": 8192, "K": 8192, "ms": ms,
             "tflops": 2 * 8192 ** 3 / ms / 1e9})
    torch.backends.cuda.matmul.allow_tf32 = True
    ms = cuda_time_ms(lambda: a @ a, warmup=3, reps=10)
    torch.backends.cuda.matmul.allow_tf32 = False
    w.write({"part": "matmul", "dtype": "tf32", "M": 8192, "N": 8192, "K": 8192, "ms": ms,
             "tflops": 2 * 8192 ** 3 / ms / 1e9})
    return best * 1e12


def gemm_cost(m: int, n: int, k: int) -> tuple[float, float]:
    flops = 2.0 * m * n * k
    return flops, flops / calc.gemm_intensity(m, n, k, 2)


def layer_ops(b: int, t_new: int, ctx: int):
    """Yield (op, fn, flops, bytes) for one layer; ctx = number of keys attended (= t_new for prefill)."""
    d, hd, h, ff = MODEL.d_model, MODEL.head_dim, MODEL.n_heads, MODEL.d_ff
    m = b * t_new
    x = torch.randn(m, d, device="cuda", dtype=BF16)
    w_qkv = torch.randn(d, 3 * d, device="cuda", dtype=BF16) * 0.02
    w_o = torch.randn(d, d, device="cuda", dtype=BF16) * 0.02
    w_up = torch.randn(d, 2 * ff, device="cuda", dtype=BF16) * 0.02
    w_down = torch.randn(ff, d, device="cuda", dtype=BF16) * 0.02
    yield ("qkv_proj", lambda: x @ w_qkv, *gemm_cost(m, 3 * d, d))
    q = torch.randn(b, h, t_new, hd, device="cuda", dtype=BF16)
    k = torch.randn(b, h, ctx, hd, device="cuda", dtype=BF16)
    v = torch.randn(b, h, ctx, hd, device="cuda", dtype=BF16)
    causal = t_new == ctx and t_new > 1
    attn_flops = 4.0 * b * h * t_new * ctx * hd * (0.5 if causal else 1.0)
    kv_bytes = calc.kv_cache_bytes(MODEL, ctx, b, 2) / MODEL.layers
    attn_bytes = kv_bytes + 2 * 2 * m * d  # K,V once + Q read + O write (flash-ideal)

    def attn():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    yield ("attention", attn, attn_flops, attn_bytes)
    a = torch.randn(m, d, device="cuda", dtype=BF16)
    yield ("out_proj", lambda: a @ w_o, *gemm_cost(m, d, d))

    def mlp():
        gu = x @ w_up
        g, u = gu.split(ff, dim=-1)
        return (F.silu(g) * u) @ w_down
    f1, b1 = gemm_cost(m, 2 * ff, d)
    f2, b2 = gemm_cost(m, d, ff)
    yield ("mlp", mlp, f1 + f2, b1 + b2 + 2 * 3 * m * ff)  # + silu*mul: read 2, write 1 (m x ff)


def bench_layer(w: JsonlWriter, gpu: calc.GPU) -> None:
    shapes = [("prefill", 1, t, t) for t in (128, 512, 2048, 8192)] + [("prefill", 8, 2048, 2048)]
    shapes += [("decode", b, 1, 4096) for b in (1, 4, 16, 64, 256)]
    for phase, b, t_new, ctx in shapes:
        for op, fn, flops, nbytes in layer_ops(b, t_new, ctx):
            ms = cuda_time_ms(fn, warmup=5, reps=30)
            t_pred, bound = calc.roofline_time(flops, nbytes, gpu)
            row = {"part": "layer", "phase": phase, "B": b, "T_new": t_new, "ctx": ctx, "op": op,
                   "flops": flops, "bytes": nbytes, "intensity": flops / nbytes, "ms": ms,
                   "tflops": flops / ms / 1e9, "gbps": nbytes / ms / 1e6, "roofline_ms": t_pred * 1e3,
                   "bound": bound, "roofline_efficiency": t_pred * 1e3 / ms}
            w.write(row)
            print(f"{phase:<7} B={b:<4} T={t_new:<5} ctx={ctx:<5} {op:<10} I={row['intensity']:8.1f} "
                  f"{ms:8.3f} ms {row['tflops']:7.1f} TF {row['gbps']:7.0f} GB/s eff={row['roofline_efficiency']:.2f}",
                  flush=True)
        torch.cuda.empty_cache()


def main() -> None:
    torch.set_num_threads(2)
    w = JsonlWriter("roofline.jsonl")
    w.write({"part": "meta", **env_info()})
    t0 = time.time()
    bw = bench_copy(w)
    peak = bench_matmul(w)
    gpu = calc.GPU("RTX 5090 (measured)", peak, bw, 32 * calc.GB, 64 * calc.GB)
    peaks = {"peak_bf16_flops": peak, "peak_copy_bw": bw, "ridge": gpu.ridge, "spec_bf16_flops": 209.5e12,
             "spec_bw": 1.792e12}
    w.write({"part": "peaks", **peaks})
    (RAW_GPU / "peaks.json").write_text(json.dumps(peaks, indent=2))
    print("peaks", peaks, flush=True)
    bench_layer(w, gpu)
    w.write({"part": "meta_end", "elapsed_s": round(time.time() - t0, 1)})
    w.close()


if __name__ == "__main__":
    main()
