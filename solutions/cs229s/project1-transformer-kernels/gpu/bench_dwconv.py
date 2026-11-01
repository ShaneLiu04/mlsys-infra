"""Depthwise causal conv1d on GPU: naive / smem / regblock CUDA kernels vs cuDNN conv1d (groups=C) vs FFT.

Sweeps: kernel length K at fixed N (short conv K=3/4 as in Mamba/H3 up to long conv K=1024),
sequence length N at K=4 and at K=N (Hyena long conv). B=4, C=768, fp32. Correctness vs fp64 cuDNN.
"""

from __future__ import annotations

import time

import torch

from common import JsonlWriter, cuda_time_ms, env_info, is_oom
from dwconv_cuda import conv_fft, conv_torch, load_ext, run_kernel

B, C = 4, 768


def case(w: JsonlWriter, ext, sweep: str, n: int, k: int) -> None:
    torch.manual_seed(n * 7 + k)
    x = torch.randn(B, C, n, device="cuda")
    h = torch.randn(C, k, device="cuda") / k ** 0.5
    ref = None
    if n * k <= 2 ** 24:  # fp64 cuDNN reference (slow for very long kernels)
        ref = conv_torch(x.double(), h.double())
    methods = {"naive": lambda: run_kernel(ext, "naive", x, h), "smem": lambda: run_kernel(ext, "smem", x, h),
               "regblock": lambda: run_kernel(ext, "regblock", x, h), "cudnn": lambda: conv_torch(x, h),
               "fft": lambda: conv_fft(x, h)}
    flops = 2.0 * B * C * n * k
    io_bytes = 4.0 * (2 * B * C * n + C * k)
    for name, fn in methods.items():
        row = {"sweep": sweep, "method": name, "B": B, "C": C, "N": n, "K": k, "flops_direct": flops,
               "io_bytes": io_bytes}
        if name in ("naive", "smem") and flops > 2e12:
            row["status"] = "skipped_slow"
            w.write(row)
            continue
        try:
            out = fn()
            if ref is not None:
                row["max_err"] = (out.double() - ref).abs().max().item()
            del out
            row["ms"] = cuda_time_ms(fn, warmup=2, reps=10)
            row["gflops_direct_equiv"] = flops / row["ms"] / 1e6
            row["gbps"] = io_bytes / row["ms"] / 1e6
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "oom" if is_oom(exc) else "error"
            row["error"] = str(exc).splitlines()[0][:200]
        torch.cuda.empty_cache()
        w.write(row)
        print(f"{sweep:<6} N={n:<6} K={k:<5} {name:<8} {row['status']:<7} {row.get('ms', float('nan')):9.3f} ms "
              f"{row.get('gflops_direct_equiv', 0):9.0f} GFLOP/s {row.get('gbps', 0):7.0f} GB/s "
              f"err={row.get('max_err', float('nan')):.1e}", flush=True)


def main() -> None:
    torch.set_num_threads(2)
    ext = load_ext()
    w = JsonlWriter("dwconv.jsonl")
    w.write({"part": "meta", **env_info()})
    t0 = time.time()
    for k in (3, 4, 7, 16, 32, 64, 128, 256, 512, 1024):
        case(w, ext, "K", 16384, k)
    for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        case(w, ext, "N_K4", n, 4)
    for n in (256, 512, 1024, 2048, 4096, 8192):
        case(w, ext, "K_eq_N", n, n)
    w.write({"part": "meta_end", "elapsed_s": round(time.time() - t0, 1)})
    w.close()


if __name__ == "__main__":
    main()
