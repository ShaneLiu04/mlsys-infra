"""GPU analogue of CS149 Asst4 on an RTX 5090. Writes artifacts/raw/gpu/*.jsonl.

    python gpu/run_gpu.py build | micro | ladder | tiles | skinny | conv | fusion | meta
Each subcommand is one short GPU job (run it under flock on the shared box).
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(LAB.parents[2] / "tools"))

import kernels as K  # noqa: E402
from dnnaccel.tiling import Accel, tile_stats  # noqa: E402

try:
    from mlsys_calc import gemm_intensity  # noqa: E402
except Exception:  # pragma: no cover - tools/ missing in a partial checkout
    def gemm_intensity(m, n, k, bytes_per_el=2):
        return 2 * m * n * k / (bytes_per_el * (m * k + k * n + m * n))

RAW = LAB / "artifacts" / "raw" / "gpu"
DEV = "cuda"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def emit(name: str, rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {name} ({len(rows)} rows)", flush=True)


_SCRUB: torch.Tensor | None = None


def scrub_l2() -> None:
    """Overwrite a 256 MiB buffer (> 96 MB L2) so the next kernel starts with a cold L2."""
    global _SCRUB
    if _SCRUB is None:
        _SCRUB = torch.empty(64 * 2**20, device=DEV)
    _SCRUB.fill_(1.0)


def bench(fn, min_time: float = 0.08, max_reps: int = 100, min_reps: int = 3, cold: bool = False) -> float:
    """Median seconds per call, one cudaEvent pair per call, after one warm-up call.
    cold=True scrubs L2 before every timed call (outside the event window)."""
    fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize()
    est = s.elapsed_time(e) / 1e3
    reps = max(min_reps, min(max_reps, int(min_time / max(est, 1e-6))))
    if cold:
        reps = min(reps, 30)
    ts = []
    for _ in range(reps):
        if cold:
            scrub_l2()
        s.record(); fn(); e.record(); e.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def rel_err(c: torch.Tensor, ref: torch.Tensor) -> float:
    return float((c.double() - ref).abs().max() / ref.abs().max())


def model_bytes(m: int, n: int, k: int, bm: int, bn: int, b_in: int, b_out: int = 4) -> int:
    """DRAM bytes if every block re-reads its A/B panels from DRAM (no L2 reuse); = tiling.py with b=4."""
    gm, gn = math.ceil(m / bm), math.ceil(n / bn)
    return b_in * (m * k * gn + k * n * gm) + b_out * m * n


def set_tf32(on: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


# ====================================================================== roofline microbenchmarks
def exp_micro() -> None:
    rows = []
    for mib in (4, 16, 32, 64, 128, 256, 512, 1024):
        n = mib * 2**20 // 4
        a = torch.rand(n, device=DEV); b = torch.empty_like(a); c = torch.rand(n, device=DEV)
        for blocks in (170 * 4, 170 * 16):
            t = bench(lambda: K.bw_copy(a, b, blocks), min_time=0.05)
            rows.append({"kernel": "copy", "mib_per_array": mib, "blocks": blocks, "s": t,
                         "bytes": 2 * n * 4, "gbps": 2 * n * 4 / t / 1e9})
            t = bench(lambda: K.bw_triad(b, a, c, blocks), min_time=0.05)
            rows.append({"kernel": "triad", "mib_per_array": mib, "blocks": blocks, "s": t,
                         "bytes": 3 * n * 4, "gbps": 3 * n * 4 / t / 1e9})
        t = bench(lambda: b.copy_(a), min_time=0.05)
        rows.append({"kernel": "torch_copy", "mib_per_array": mib, "blocks": 0, "s": t,
                     "bytes": 2 * n * 4, "gbps": 2 * n * 4 / t / 1e9})
        del a, b, c
        print(rows[-1], flush=True)
    emit("micro_bw.jsonl", rows)

    frows = []
    out = torch.zeros(1, device=DEV)
    iters = 4096
    for chains in (1, 2, 4, 8, 16):
        for threads in (128, 256, 512, 1024):
            for bps in (1, 2, 4, 8):
                blocks = 170 * bps
                if threads * bps > 2048:
                    continue
                t = bench(lambda: K.fma_chain(out, blocks, threads, iters, chains), min_time=0.05)
                fl = 2.0 * blocks * threads * iters * chains
                frows.append({"chains": chains, "threads": threads, "blocks": blocks, "iters": iters,
                              "s": t, "flops": fl, "tflops": fl / t / 1e12,
                              "warps_per_sm": threads * bps // 32})
    emit("micro_fma.jsonl", frows)
    print(max(frows, key=lambda r: r["tflops"]), flush=True)

    # tensor-core peak via cuBLAS: large square GEMMs
    trows = []
    for dt_name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16), ("tf32", torch.float32)):
        for n in (4096, 8192, 16384):
            set_tf32(dt_name == "tf32")
            a = torch.randn(n, n, device=DEV).to(dt); b = torch.randn(n, n, device=DEV).to(dt)
            t = bench(lambda: a @ b, min_time=0.2)
            trows.append({"dtype": dt_name, "n": n, "s": t, "tflops": 2 * n**3 / t / 1e12})
            print(trows[-1], flush=True)
            del a, b
    set_tf32(False)
    emit("micro_tensor.jsonl", trows)


# ====================================================================== SGEMM ladder
LADDER_SIZES = (256, 512, 1024, 2048, 4096, 8192)


def _gemm_row(name: str, dtype: str, m: int, n: int, k: int, t: float, err: float | None, b_in: int) -> dict:
    bm, bn = K.SGEMM_TILES.get(name, (128, 128))
    mb = model_bytes(m, n, k, bm, bn, b_in)
    fl = 2.0 * m * n * k
    return {"variant": name, "dtype": dtype, "M": m, "N": n, "K": k, "s": t, "gflops": fl / t / 1e9,
            "max_rel_err": err, "tile_bm": bm, "tile_bn": bn, "model_bytes": mb, "model_intensity": fl / mb,
            "compulsory_intensity": gemm_intensity(m, n, k, b_in)}


def run_gemm_set(m: int, n: int, k: int, variants: list[int], rows: list[dict], with_lib: bool = True) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, k, device=DEV); b = torch.randn(k, n, device=DEV)
    ref = a.double() @ b.double()
    c = torch.empty(m, n, device=DEV)
    for v in variants:
        name = K.SGEMM_VARIANTS[v]
        if v in (7, 8, 9) and (m % 128 or n % 128):
            continue
        c.zero_()
        K.sgemm(v, a, b, c)
        err = rel_err(c, ref)
        t = bench(lambda: K.sgemm(v, a, b, c), min_time=0.08 if m * n * k < 2**36 else 0.3)
        rows.append(_gemm_row(name, "fp32", m, n, k, t, err, 4))
        print(f"{name:<10} {m}x{n}x{k} {rows[-1]['gflops']:9.1f} GFLOP/s err={err:.2e}", flush=True)
    # cuBLAS references
    for label, tf32 in (("cublas_fp32", False), ("cublas_tf32", True)):
        set_tf32(tf32)
        out = a @ b
        rows.append(_gemm_row(label, "fp32" if not tf32 else "tf32", m, n, k, bench(lambda: a @ b),
                              rel_err(out, ref), 4))
    set_tf32(False)
    for dt_name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        ah, bh = a.to(dt), b.to(dt)
        refh = ah.double() @ bh.double()
        out = ah @ bh
        rows.append(_gemm_row(f"cublas_{dt_name}", dt_name, m, n, k, bench(lambda: ah @ bh), rel_err(out, refh), 2))
        if with_lib and m % 128 == 0 and n % 128 == 0 and k % 32 == 0:
            cw = torch.empty(m, n, device=DEV)
            K.hgemm_wmma(ah, bh, cw)
            rows.append(_gemm_row(f"wmma_{dt_name}", dt_name, m, n, k, bench(lambda: K.hgemm_wmma(ah, bh, cw)),
                                  rel_err(cw, refh), 2))
        print(f"cublas/wmma {dt_name} {m}: {rows[-1]['gflops']:.0f} GFLOP/s", flush=True)
    # FP8 e4m3 via torch._scaled_mm (B must be column-major)
    try:
        sa = (a.abs().max() / 448.0).float(); sb = (b.abs().max() / 448.0).float()
        a8 = (a / sa).to(torch.float8_e4m3fn)
        b8 = (b.t().contiguous() / sb).to(torch.float8_e4m3fn).t()
        ref8 = (a8.double() * sa.double()) @ (b8.double() * sb.double())

        def f8():
            return torch._scaled_mm(a8, b8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        out = f8()
        rows.append(_gemm_row("cublas_fp8", "fp8_e4m3", m, n, k, bench(f8), rel_err(out.float(), ref8), 1))
    except Exception as exc:  # noqa: BLE001
        rows.append({"variant": "cublas_fp8", "M": m, "N": n, "K": k, "error": repr(exc)[:300]})
    del a, b, ref, c
    torch.cuda.empty_cache()


def exp_ladder() -> None:
    rows: list[dict] = []
    for s in LADDER_SIZES:
        variants = list(range(11)) if s <= 4096 else list(range(1, 11))
        run_gemm_set(s, s, s, variants, rows)
    emit("gemm_ladder.jsonl", rows)


def exp_tiles() -> None:
    rows = []
    for s in (2048, 4096):
        a = torch.randn(s, s, device=DEV); b = torch.randn(s, s, device=DEV)
        ref = a.double() @ b.double()
        c = torch.empty(s, s, device=DEV)
        for bm in (32, 64, 128, 256):
            for bn in (32, 64, 128, 256):
                c.zero_()
                K.sgemm_tile(bm, bn, a, b, c)
                err = rel_err(c, ref)
                t = bench(lambda: K.sgemm_tile(bm, bn, a, b, c))
                mb = model_bytes(s, s, s, bm, bn, 4)
                ts = tile_stats(s, s, s, bm, bn, 8, Accel(bytes_per_el=4), with_cycles=False)
                rows.append({"M": s, "BM": bm, "BN": bn, "threads": (bm // 8) * (bn // 8), "s": t,
                             "gflops": 2 * s**3 / t / 1e9, "max_rel_err": err, "model_bytes": mb,
                             "model_intensity": 2 * s**3 / mb, "tiling_py_bytes": ts.hbm_bytes,
                             "smem_bytes": 4 * 8 * (bm + bn)})
                print(rows[-1], flush=True)
    emit("gemm_tiles.jsonl", rows)


def exp_skinny() -> None:
    """Decode-like GEMM, N=K=4096 (B = 64 MiB fp32, which fits the 96 MB L2): warm-L2 and cold-L2 timings."""
    rows = []
    n = k = 4096
    b = torch.randn(k, n, device=DEV)
    bh = b.to(torch.bfloat16)
    for m in (1, 2, 4, 8, 16, 32, 64, 128):
        a = torch.randn(m, k, device=DEV)
        ref = a.double() @ b.double()
        c = torch.zeros(m, n, device=DEV)
        byt = 4 * (m * k + k * n + m * n)
        fl = 2.0 * m * n * k
        ah = a.to(torch.bfloat16)
        for l2 in ("warm", "cold"):
            cold = l2 == "cold"

            def rec(name, t, err, bytes_=byt):
                rows.append({"variant": name, "l2": l2, "M": m, "N": n, "K": k, "s": t, "gflops": fl / t / 1e9,
                             "gbps": bytes_ / t / 1e9, "bytes": bytes_, "max_rel_err": err,
                             "compulsory_intensity": fl / bytes_})
            for v in (1, 4, 6):
                c.zero_(); K.sgemm(v, a, b, c)
                rec(K.SGEMM_VARIANTS[v], bench(lambda: K.sgemm(v, a, b, c), cold=cold), rel_err(c, ref))
            best = None
            for ks in (8, 16, 32, 64):
                c.zero_(); K.sgemm_skinny(a, b, c, ks)
                err = rel_err(c, ref)

                def fsk():
                    c.zero_()
                    K.sgemm_skinny(a, b, c, ks)
                t = bench(fsk, cold=cold)
                rec(f"skinny_k{ks}", t, err)
                if best is None or t < best[0]:
                    best = (t, err)
            rec("skinny_best", best[0], best[1])
            rec("cublas_fp32", bench(lambda: a @ b, cold=cold), rel_err(a @ b, ref))
            rec("cublas_bf16", bench(lambda: ah @ bh, cold=cold),
                rel_err((ah @ bh).float(), ah.double() @ bh.double()), bytes_=2 * (m * k + k * n) + 2 * m * n)
            print(f"M={m} {l2}: " + ", ".join(f"{r['variant']}={r['gbps']:.0f}GB/s" for r in rows
                                             if r["M"] == m and r["l2"] == l2), flush=True)
    emit("gemm_skinny.jsonl", rows)


# ====================================================================== conv
CONV_LAYERS = [  # name, C, H, F, K, stride, pad   (batch CONV_N)
    ("conv1 7x7/2 @224", 3, 224, 64, 7, 2, 3),
    ("3x3 @56 C64", 64, 56, 64, 3, 1, 1),
    ("3x3 @28 C128", 128, 28, 128, 3, 1, 1),
    ("3x3 @14 C256", 256, 14, 256, 3, 1, 1),
    ("3x3 @7 C512", 512, 7, 512, 3, 1, 1),
    ("3x3/2 @56 C128", 128, 56, 128, 3, 2, 1),
    ("1x1 @56 C64->256", 64, 56, 256, 1, 1, 0),
    ("1x1 @14 C1024->256", 1024, 14, 256, 1, 1, 0),
]
CONV_N = 8


def exp_conv() -> None:
    import torch.nn.functional as F
    rows = []
    for name, c, h, f, kk, s, p in CONV_LAYERS:
        torch.manual_seed(0)
        x = torch.randn(CONV_N, c, h, h, device=DEV); w = torch.randn(f, c, kk, kk, device=DEV) / math.sqrt(c * kk * kk)
        ref = F.conv2d(x.double(), w.double(), stride=s, padding=p)
        oh = K.out_hw(h, kk, s, p)
        fl = 2.0 * CONV_N * f * oh * oh * c * kk * kk
        base = {"layer": name, "N": CONV_N, "C": c, "H": h, "F": f, "K": kk, "stride": s, "pad": p, "OH": oh,
                "flops": fl, "x_bytes": x.numel() * 4, "w_bytes": w.numel() * 4, "y_bytes": CONV_N * f * oh * oh * 4}
        y = torch.empty(CONV_N, f, oh, oh, device=DEV)

        def rec(method, t, err, **kw):
            rows.append({**base, "method": method, "s": t, "gflops": fl / t / 1e9, "max_rel_err": err, **kw})
            print(f"{name:<22} {method:<16} {t*1e3:8.3f} ms {fl/t/1e9:9.1f} GFLOP/s err={err:.1e}", flush=True)

        K.conv_direct(x, w, y, s, p)
        rec("direct", bench(lambda: K.conv_direct(x, w, y, s, p)), rel_err(y, ref))
        # explicit im2col + cuBLAS fp32
        rr = c * kk * kk
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        cols = torch.empty(CONV_N, rr, oh * oh, device=DEV)
        wm = w.reshape(f, rr)

        def im2col_gemm():
            K.conv_im2col(x, cols, kk, kk, s, p)
            return torch.matmul(wm, cols)
        out = im2col_gemm()
        peak = torch.cuda.max_memory_allocated() - m0
        t_tot = bench(im2col_gemm)
        t_cols = bench(lambda: K.conv_im2col(x, cols, kk, kk, s, p))
        t_mm = bench(lambda: torch.matmul(wm, cols))
        rec("im2col+cublas", t_tot, rel_err(out.view_as(y), ref), cols_bytes=cols.numel() * 4,
            blowup=cols.numel() / x.numel(), peak_extra_bytes=peak, t_im2col=t_cols, t_gemm=t_mm)
        del cols, out
        K.conv_igemm(x, w, y, s, p)
        rec("implicit_gemm", bench(lambda: K.conv_igemm(x, w, y, s, p)), rel_err(y, ref))
        for lab, tf in (("cudnn_fp32", False), ("cudnn_tf32", True)):
            set_tf32(tf)
            out = F.conv2d(x, w, stride=s, padding=p)
            rec(lab, bench(lambda: F.conv2d(x, w, stride=s, padding=p)), rel_err(out, ref))
        set_tf32(False)
        xh, wh = x.half().to(memory_format=torch.channels_last), w.half().to(memory_format=torch.channels_last)
        out = F.conv2d(xh, wh, stride=s, padding=p)
        rec("cudnn_fp16_nhwc", bench(lambda: F.conv2d(xh, wh, stride=s, padding=p)),
            rel_err(out.float(), F.conv2d(xh.double(), wh.double(), stride=s, padding=p)))
        del x, w, y, ref
        torch.cuda.empty_cache()
    emit("conv.jsonl", rows)


FUSION_LAYERS = [("3x3 @112 C64", 64, 112, 64), ("3x3 @56 C128", 128, 56, 128),
                 ("3x3 @28 C256", 256, 28, 256), ("3x3 @14 C512", 512, 14, 512)]


def exp_fusion() -> None:
    """Batch 8 keeps the conv output (<= 26 MB) L2-resident; batch 64 pushes it to 103-206 MB, beyond L2."""
    import torch.nn.functional as F
    set_tf32(False)
    rows = []
    for (name, c, h, f), n in ((L, nb) for nb in (8, 64) for L in FUSION_LAYERS):
        torch.manual_seed(0)
        x = torch.randn(n, c, h, h, device=DEV); w = torch.randn(f, c, 3, 3, device=DEV) / math.sqrt(9 * c)
        ref = F.max_pool2d(F.relu(F.conv2d(x.double(), w.double(), padding=1)), 2)
        Y = n * f * h * h * 4  # conv output bytes (stride 1, pad 1 -> OH = H)
        io = x.numel() * 4 + w.numel() * 4
        fl = 2.0 * n * f * h * h * c * 9
        y = torch.empty(n, f, h, h, device=DEV); y2 = torch.empty_like(y)
        out = torch.empty(n, f, h // 2, h // 2, device=DEV)
        base = {"layer": name, "N": n, "C": c, "H": h, "F": f, "flops": fl, "conv_out_bytes": Y, "io_bytes": io}

        def rec(method, stages: dict, err, inter_bytes):
            tot = sum(stages.values())
            rows.append({**base, "method": method, "s": tot, "stages": stages, "max_rel_err": err,
                         "model_hbm_bytes": io + inter_bytes, "intermediate_bytes": inter_bytes,
                         "gflops": fl / tot / 1e9})
            print(f"{name:<14} N={n:<3} {method:<22} {tot*1e3:8.3f} ms err={err:.1e} {stages}", flush=True)

        # A. direct 2x2 conv, then separate ReLU and maxpool kernels
        K.conv2x2(False, x, w, y, 1, 1); K.relu(y, y2); K.maxpool2x2(y2, out)
        e = rel_err(out, ref)
        rec("direct_unfused", {"conv": bench(lambda: K.conv2x2(False, x, w, y, 1, 1)),
                               "relu": bench(lambda: K.relu(y, y2)), "pool": bench(lambda: K.maxpool2x2(y2, out))},
            e, Y + 2 * Y + Y + Y // 4)
        out.zero_(); K.conv2x2(True, x, w, out, 1, 1)
        rec("direct_fused", {"conv+relu+pool": bench(lambda: K.conv2x2(True, x, w, out, 1, 1))}, rel_err(out, ref),
            Y // 4)
        # B. cuDNN conv + torch relu + torch maxpool (3 kernels) vs cuDNN + my fused relu+pool epilogue
        yc = F.conv2d(x, w, padding=1)
        o1 = F.max_pool2d(F.relu(yc), 2)
        rec("cudnn_unfused", {"conv": bench(lambda: F.conv2d(x, w, padding=1)), "relu": bench(lambda: F.relu(yc)),
                              "pool": bench(lambda: F.max_pool2d(yc, 2))}, rel_err(o1, ref), Y + 2 * Y + Y + Y // 4)
        K.maxpool2x2(yc, out, with_relu=True)
        rec("cudnn+fused_relu_pool", {"conv": bench(lambda: F.conv2d(x, w, padding=1)),
                                      "relu+pool": bench(lambda: K.maxpool2x2(yc, out, with_relu=True))},
            rel_err(out, ref), Y + Y + Y // 4)
        # C. my implicit GEMM with ReLU in the epilogue, then pool
        K.conv_igemm(x, w, y, 1, 1, relu=True); K.maxpool2x2(y, out)
        rec("igemm_relu+pool", {"conv+relu": bench(lambda: K.conv_igemm(x, w, y, 1, 1, relu=True)),
                                "pool": bench(lambda: K.maxpool2x2(y, out))}, rel_err(out, ref), Y + Y + Y // 4)
        # end-to-end timing of the whole unfused / fused sequences (one event pair around all kernels)
        t_u = bench(lambda: (K.conv2x2(False, x, w, y, 1, 1), K.relu(y, y2), K.maxpool2x2(y2, out)))
        t_f = bench(lambda: K.conv2x2(True, x, w, out, 1, 1))
        t_cu = bench(lambda: F.max_pool2d(F.relu(F.conv2d(x, w, padding=1)), 2))
        rows.append({**base, "method": "e2e", "direct_unfused_s": t_u, "direct_fused_s": t_f, "cudnn_unfused_s": t_cu})
        del x, w, y, y2, out, yc
        torch.cuda.empty_cache()
    emit("conv_fusion.jsonl", rows)


def exp_info() -> None:
    """Registers / smem / theoretical occupancy per kernel (ncu is not permitted in the container)."""
    p = torch.cuda.get_device_properties(0)
    max_warps = p.max_threads_per_multi_processor // 32
    rows = []
    for i in range(len(K.KERNEL_INFO_NAMES)):
        r = K.kernel_info(i)
        r["theoretical_occupancy"] = r["warps_per_sm"] / max_warps
        r["max_warps_per_sm"] = max_warps
        rows.append(r)
        print(r, flush=True)
    emit("kernel_info.jsonl", rows)


def exp_meta() -> None:
    p = torch.cuda.get_device_properties(0)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    nvcc = subprocess.run([K.NVCC, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    meta = {"gpu": p.name, "sm_count": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
            "total_mem_gb": p.total_memory / 1e9, "l2_mb": getattr(p, "L2_cache_size", 0) / 2**20,
            "nvidia_smi": smi, "nvcc": nvcc, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
            "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RAW / "meta_gpu.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(meta)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "build":
        K.build(force=True)
        sys.exit(0)
    K.build()
    RAW.mkdir(parents=True, exist_ok=True)
    fns = {"micro": exp_micro, "ladder": exp_ladder, "tiles": exp_tiles, "skinny": exp_skinny, "conv": exp_conv,
           "fusion": exp_fusion, "info": exp_info, "meta": exp_meta}
    for name in (fns if cmd == "all" else [cmd]):
        t0 = time.time()
        fns[name]()
        print(f"[{name}] {time.time() - t0:.1f} s", flush=True)
