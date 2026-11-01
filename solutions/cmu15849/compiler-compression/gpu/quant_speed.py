"""Q3b: low-precision matmul throughput and weight-only GEMV bandwidth on RTX 5090.

* bf16 (cuBLAS) vs int8 (torch._int_mm, int32 accumulate) vs fp8 e4m3 (torch._scaled_mm)
  for decode-like (small M) and prefill-like (large M) shapes.
* batch-1 GEMV: bf16 cuBLAS vs Triton W8A16 / W4A16 (in-kernel dequant) vs dequant-then-matmul.
"""

from __future__ import annotations

import argparse

import torch

from common import RTX5090, Writer, bench, env_info, set_threads
import triton_kernels as tk


def gemm_part(w: Writer, ms, nks):
    for n, k in nks:
        torch.manual_seed(0)
        wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        wi8 = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8)
        fp8_ok = hasattr(torch, "float8_e4m3fn")
        wf8 = wb.to(torch.float8_e4m3fn) if fp8_ok else None
        one = torch.tensor(1.0, device="cuda")
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            flops = 2 * m * n * k
            t = bench(lambda: x @ wb.t())
            w(part="gemm", M=m, N=n, K=k, dtype="bf16", time_s=t, tflops=flops / t / 1e12,
              bytes=(m * k + n * k + m * n) * 2, ok=True)
            xi8 = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8)
            try:
                y = torch._int_mm(xi8, wi8.t())
                ref = xi8.double() @ wi8.t().double()
                exact = bool(torch.equal(y.double(), ref))
                t = bench(lambda: torch._int_mm(xi8, wi8.t()))
                w(part="gemm", M=m, N=n, K=k, dtype="int8", time_s=t, tflops=flops / t / 1e12,
                  bytes=m * k + n * k + m * n * 4, ok=True, exact=exact)
            except Exception as e:  # noqa: BLE001
                w(part="gemm", M=m, N=n, K=k, dtype="int8", ok=False, error=str(e)[:160])
            if fp8_ok:
                xf8 = x.to(torch.float8_e4m3fn)
                try:
                    f = lambda: torch._scaled_mm(xf8, wf8.t(), scale_a=one, scale_b=one,  # noqa: E731
                                                 out_dtype=torch.bfloat16)
                    y = f()
                    ref = xf8.float() @ wf8.float().t()
                    rel = float((y.float() - ref).norm() / ref.norm())
                    t = bench(f)
                    w(part="gemm", M=m, N=n, K=k, dtype="fp8", time_s=t, tflops=flops / t / 1e12,
                      bytes=m * k + n * k + m * n * 2, ok=True, rel_err=rel)
                except Exception as e:  # noqa: BLE001
                    w(part="gemm", M=m, N=n, K=k, dtype="fp8", ok=False, error=str(e)[:160])


class Rotating:
    """Cycle through `copies` replicas so the working set exceeds the 96 MB L2 ('cold' weights)."""

    def __init__(self, make, copies: int):
        self.items = [make() for _ in range(copies)]
        self.i = 0

    def next(self):
        self.i = (self.i + 1) % len(self.items)
        return self.items[self.i]


def gemv_part(w: Writer, nks):
    for n, k in nks:
        torch.manual_seed(0)
        wb = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** 0.5
        wh = wb.half()
        x = torch.randn(k, device="cuda", dtype=torch.float16)
        q8, s8 = tk.quant_w8(wh)
        q4, s4 = tk.quant_w4(wh)
        ref8 = (q8.float() * s8.float()[:, None]) @ x.float()
        ref4 = tk.dequant_w4(q4, s4) @ x.float()
        b16, b8, b4 = n * k * 2, n * k + n * 2, n * k // 2 + s4.numel() * 2
        for cache in ["hot", "cold"]:
            def copies(nb):
                return 1 if cache == "hot" else max(2, -(-384 * 2 ** 20 // nb))
            rb = Rotating(lambda: wb.clone(), copies(b16))
            rh = Rotating(lambda: wh.clone(), copies(b16))
            r8 = Rotating(lambda: (q8.clone(), s8.clone()), copies(b8))
            r4 = Rotating(lambda: (q4.clone(), s4.clone()), copies(b4))
            impls = {
                "bf16_cublas": (lambda: rb.next() @ x.bfloat16(), b16, None),
                "fp16_cublas": (lambda: rh.next() @ x, b16, None),
                "w8a16_triton": (lambda: tk.gemv_w8(x, *r8.next()), b8, ref8),
                "w4a16_triton": (lambda: tk.gemv_w4(x, *r4.next()), b4, ref4),
                "w4_dequant_then_matmul": (lambda: tk.dequant_w4(*r4.next()).half() @ x, b4, ref4),
            }
            for name, (f, nbytes, ref) in impls.items():
                y = f()
                torch.cuda.synchronize()
                err = float((y.float() - ref).norm() / ref.norm()) if ref is not None else 0.0
                t = bench(f, 0.03)
                w(part="gemv", N=n, K=k, impl=name, cache=cache, time_s=t, weight_bytes=nbytes,
                  eff_bw=nbytes / t, bw_frac=nbytes / t / RTX5090["hbm_bw"], rel_err=err,
                  fits_l2=bool(nbytes < RTX5090["l2_bytes"]))
            del rb, rh, r8, r4
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    set_threads(2)
    w = Writer("quant_speed")
    w(part="env", **env_info())
    if a.quick:
        gemm_part(w, [32, 1024], [(1024, 1024)])
        gemv_part(w, [(1024, 1024)])
        return
    gemm_part(w, [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], [(4096, 4096), (16384, 4096)])
    gemv_part(w, [(4096, 4096), (11008, 4096), (4096, 11008), (16384, 8192), (32768, 8192)])


if __name__ == "__main__":
    main()
