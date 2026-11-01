"""Q2: tile/schedule search for a Triton matmul -- measured landscape vs autotune vs cuBLAS.

Grid: BM, BN in {32..256}, BK in {32, 64}, num_warps in {4, 8}, num_stages in {2, 3, 4}.
Each config is compiled, checked against cuBLAS, and timed. Out-of-resource configs are recorded.
"""

from __future__ import annotations

import argparse
import itertools

import torch

from common import Writer, bench, env_info, set_threads
import triton_kernels as tk

SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (256, 4096, 4096)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    set_threads(2)
    w = Writer("matmul_search")
    w(part="env", **env_info())
    bms = [32, 64, 128, 256]
    grid = list(itertools.product(bms, bms, [32, 64], [4, 8], [2, 3, 4]))
    shapes = SHAPES
    if a.quick:
        grid = grid[:6]
        shapes = [(1024, 1024, 1024)]
    for (m, n, k) in shapes:
        torch.manual_seed(0)
        A = torch.randn(m, k, device="cuda", dtype=torch.float16)
        B = torch.randn(k, n, device="cuda", dtype=torch.float16)
        ref = A @ B
        flops = 2 * m * n * k
        t = bench(lambda: A @ B, 0.05)
        w(part="cublas", M=m, N=n, K=k, time_s=t, tflops=flops / t / 1e12)
        for bm, bn, bk, nw, ns in grid:
            if (m, n, k) == (8192, 8192, 8192) and (bm < 64 or bn < 64):
                continue  # tiny tiles at 8192^3 take seconds each and are never competitive
            row = dict(part="grid", M=m, N=n, K=k, BM=bm, BN=bn, BK=bk, num_warps=nw, num_stages=ns)
            try:
                f = lambda: tk.matmul(A, B, bm, bn, bk, nw, ns)  # noqa: E731
                c = f()
                torch.cuda.synchronize()
                err = float((c.float() - ref.float()).abs().max() / ref.float().abs().max())
                t = bench(f, 0.03)
                w(**row, ok=True, time_s=t, tflops=flops / t / 1e12, rel_err=err)
            except Exception as e:  # noqa: BLE001  (OutOfResources: shared memory / registers)
                w(**row, ok=False, error=type(e).__name__ + ": " + str(e)[:120])
                torch.cuda.synchronize()
        c = tk.matmul_autotuned(A, B)
        torch.cuda.synchronize()
        err = float((c.float() - ref.float()).abs().max() / ref.float().abs().max())
        t = bench(lambda: tk.matmul_autotuned(A, B), 0.05)
        w(part="autotune", M=m, N=n, K=k, time_s=t, tflops=flops / t / 1e12, rel_err=err,
          pick=tk.autotune_pick())
        tk._autotuned.cache.clear()


if __name__ == "__main__":
    main()
