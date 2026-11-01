"""Produce every CSV/figure cited in the README. CPU only, ~1-2 minutes."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from dnnaccel.gemm_cpu import gflops, matmul_blocked, matmul_naive, matmul_rows  # noqa: E402
from dnnaccel.im2col import blowup_factor, conv2d_direct, conv2d_im2col, im2col  # noqa: E402
from dnnaccel.systolic import systolic_matmul  # noqa: E402
from dnnaccel.tiling import Accel, best_square_tile, fits, search_tiles, tile_stats  # noqa: E402

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
MNK = 4096


def write_csv(name: str, rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)")


def exp_tiling() -> None:
    acc = Accel()
    rows = []
    for t in (128, 256, 512, 1024, 2048, 4096):
        s = tile_stats(MNK, MNK, MNK, t, t, 128, acc)
        rows.append({"tile": t, "tk": 128, "intensity_flop_per_byte": round(s.intensity, 2),
                     "hbm_gb": round(s.hbm_bytes / 1e9, 4), "dma_count": s.dma_count,
                     "sbuf_kib": s.sbuf_footprint // 1024, "psum_kib": s.psum_footprint // 1024,
                     "fits_default": fits(t, t, 128, acc), "compute_ms": round(s.compute_s * 1e3, 3),
                     "dma_ms": round(s.dma_s * 1e3, 3), "bound": s.bound})
    write_csv("tiling_square.csv", rows)

    tk_rows = []
    for tk in (128, 256, 512, 1024, 2048, 4096):
        s = tile_stats(MNK, MNK, MNK, 512, 512, tk, acc, with_cycles=False)
        tk_rows.append({"tm": 512, "tn": 512, "tk": tk, "intensity_flop_per_byte": round(s.intensity, 2),
                        "dma_count": s.dma_count, "sbuf_kib": s.sbuf_footprint // 1024,
                        "fits_default": fits(512, 512, tk, acc), "dma_ms": round(s.dma_s * 1e3, 3)})
    write_csv("tiling_tk.csv", tk_rows)

    cap_rows = []
    for kib in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24576):
        a = replace(acc, sbuf_bytes=kib * 1024)
        s = search_tiles(MNK, MNK, MNK, a)
        assert s is not None
        sbuf_util = s.sbuf_footprint / a.sbuf_bytes
        psum_util = s.psum_footprint / a.psum_bytes
        cap_rows.append({"sbuf_kib": kib, "psum_kib": a.psum_bytes // 1024, "tm": s.tm, "tn": s.tn, "tk": s.tk,
                         "intensity_flop_per_byte": round(s.intensity, 2), "hbm_gb": round(s.hbm_bytes / 1e9, 4),
                         "dma_count": s.dma_count, "sbuf_used": round(sbuf_util, 3),
                         "psum_used": round(psum_util, 3),
                         "binding": "psum" if psum_util >= sbuf_util else "sbuf",
                         "compute_ms": round(s.compute_s * 1e3, 3), "dma_ms": round(s.dma_s * 1e3, 3),
                         "bound": s.bound, "ridge": round(a.ridge, 1)})
    write_csv("tiling_capacity.csv", cap_rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = [r["tile"] for r in rows]
    ax[0].loglog(t, [r["intensity_flop_per_byte"] for r in rows], "o-", base=2,
                 label="model (M=N=K=4096, bf16)")
    ax[0].loglog(t, [x / 2 for x in t], "k--", lw=0.8, base=2, label="t/2 = 2/(b(1/t+1/t)), b=2")
    tmax = best_square_tile(acc)
    ax[0].axvline(tmax, color="r", ls=":", label=f"PSUM 2 MiB limit (t={tmax})")
    ax[0].axhline(acc.ridge, color="g", ls=":", label=f"ridge {acc.ridge:.0f} F/B")
    ax[0].set(xlabel="square tile tm=tn (tk=128)", ylabel="FLOP / HBM byte", title="Intensity vs tile")
    ax[0].legend(fontsize=7)
    kibs = [r["sbuf_kib"] for r in cap_rows]
    ax[1].semilogx(kibs, [r["intensity_flop_per_byte"] for r in cap_rows], "o-", base=2)
    for i, r in enumerate(cap_rows):
        if i and (r["tm"], r["tn"]) == (cap_rows[i - 1]["tm"], cap_rows[i - 1]["tn"]):
            continue
        ax[1].annotate(f"{r['tm']}x{r['tn']}\n{r['binding']}", (r["sbuf_kib"], r["intensity_flop_per_byte"]),
                       fontsize=6, textcoords="offset points", xytext=(0, 6), ha="center")
    ax[1].axhline(acc.ridge, color="g", ls=":", label="ridge")
    ax[1].set(xlabel="SBUF capacity (KiB)", ylabel="best FLOP / byte", title="Best tile under capacity")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "tiling_intensity.png", dpi=130)
    plt.close(fig)


def exp_systolic(rng: np.random.Generator) -> None:
    def run(n: int, p: int) -> dict:
        a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
        c, st = systolic_matmul(a, b, p)
        return {"n": n, "P": p, "blocks": st.blocks, "serial_cycles": st.serial_cycles,
                "pipelined_cycles": st.pipelined_cycles, "util_serial": round(st.utilization(), 4),
                "util_pipelined": round(st.utilization(pipelined=True), 4),
                "max_abs_err": float(np.abs(c - a @ b).max())}

    size_rows = [run(n, 16) for n in (8, 16, 32, 64, 128, 256)]
    write_csv("systolic_vs_size.csv", size_rows)
    p_rows = [run(128, p) for p in (4, 8, 16, 32, 64, 128)]
    write_csv("systolic_vs_p.csv", p_rows)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, lab in (("util_serial", "serial weight load"), ("util_pipelined", "double-buffered")):
        ax[0].semilogx([r["n"] for r in size_rows], [r[key] for r in size_rows], "o-", base=2, label=lab)
        ax[1].semilogx([r["P"] for r in p_rows], [r[key] for r in p_rows], "o-", base=2, label=lab)
    ax[0].set(xlabel="n (square C = A@B)", ylabel="PE utilization", title="P=16: utilization vs size", ylim=(0, 1))
    ax[1].set(xlabel="array size P", ylabel="PE utilization", title="n=128: utilization vs P", ylim=(0, 1))
    for a in ax:
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "systolic_utilization.png", dpi=130)
    plt.close(fig)


def exp_cpu_gemm(rng: np.random.Generator) -> None:
    rows = []

    def rec(method: str, n: int, bs: int | str, fn, repeats: int = 5) -> None:
        g, sec = gflops(fn, 2.0 * n ** 3, repeats=repeats)
        rows.append({"method": method, "n": n, "block": bs, "gflops": round(g, 3), "seconds": f"{sec:.4g}"})
        print(f"  {method:<8} n={n:<5} bs={bs!s:<5} {g:9.3f} GFLOP/s")

    mats = {n: (rng.standard_normal((n, n)), rng.standard_normal((n, n))) for n in (32, 64, 128, 256, 512, 1024, 2048)}
    for n in (32, 64, 128):
        a, b = mats[n]
        rec("naive", n, "-", lambda: matmul_naive(a, b), repeats=1)
    for n in (64, 128, 256):
        a, b = mats[n]
        rec("rows", n, "-", lambda: matmul_rows(a, b), repeats=1)
    a, b = mats[1024]
    for bs in (16, 32, 64, 128, 256, 512, 1024):
        rec("blocked", 1024, bs, lambda: matmul_blocked(a, b, bs))
    for n in (64, 128, 256, 512, 1024, 2048):
        a, b = mats[n]
        rec("np.matmul", n, "-", lambda: a @ b)
    write_csv("cpu_gemm.csv", rows)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m, st in (("naive", "o-"), ("rows", "s-"), ("np.matmul", "^-")):
        rr = [r for r in rows if r["method"] == m]
        ax[0].loglog([r["n"] for r in rr], [r["gflops"] for r in rr], st, base=2, label=m)
    ax[0].set(xlabel="n", ylabel="GFLOP/s (float64)", title="GEMM variants on CPU")
    ax[0].legend(fontsize=8)
    rb = [r for r in rows if r["method"] == "blocked"]
    ax[1].semilogx([r["block"] for r in rb], [r["gflops"] for r in rb], "o-", base=2, label="blocked, n=1024")
    ref = next(r["gflops"] for r in rows if r["method"] == "np.matmul" and r["n"] == 1024)
    ax[1].axhline(ref, color="k", ls="--", label="np.matmul n=1024")
    ax[1].set(xlabel="block size", ylabel="GFLOP/s", title="Python-level blocking")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "cpu_gemm.png", dpi=130)
    plt.close(fig)


def best_of(fn, repeats: int = 5):
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def exp_im2col(rng: np.random.Generator) -> None:
    layers = [("resnet conv1", 3, 224, 64, 7, 2, 3), ("3x3 @56", 64, 56, 64, 3, 1, 1),
              ("3x3 @28", 128, 28, 128, 3, 1, 1), ("3x3 s2 @28", 128, 28, 256, 3, 2, 1),
              ("1x1 @14", 256, 14, 1024, 1, 1, 0), ("5x5 @32", 32, 32, 64, 5, 1, 2)]
    rows = []
    for name, c, h, f, k, s, p in layers:
        x = rng.standard_normal((1, c, h, h)).astype(np.float32)
        w = rng.standard_normal((f, c, k, k)).astype(np.float32)
        yd, td = best_of(lambda: conv2d_direct(x, w, s, p))
        yi, ti = best_of(lambda: conv2d_im2col(x, w, s, p))
        cols, tc = best_of(lambda: im2col(x, k, k, s, p))
        rows.append({"layer": name, "C": c, "H": h, "F": f, "K": k, "stride": s, "pad": p,
                     "blowup": round(blowup_factor(c, h, h, k, k, s, p), 3),
                     "input_mib": round(x.nbytes / 2**20, 3), "im2col_mib": round(cols.nbytes / 2**20, 3),
                     "direct_ms": round(td * 1e3, 2), "im2col_total_ms": round(ti * 1e3, 2),
                     "im2col_copy_ms": round(tc * 1e3, 2),
                     "max_rel_err": float(np.abs(yd - yi).max() / np.abs(yd).max())})
    write_csv("im2col.csv", rows)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r["input_mib"] for r in rows], 0.4, label="input")
    ax.bar(xs + 0.2, [r["im2col_mib"] for r in rows], 0.4, label="im2col matrix")
    for i, r in enumerate(rows):
        ax.text(i + 0.2, r["im2col_mib"], f"x{r['blowup']:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs, [r["layer"] for r in rows], rotation=20, fontsize=7)
    ax.set(ylabel="MiB (fp32, batch 1)", title="im2col memory blow-up")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "im2col_blowup.png", dpi=130)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    exp_tiling()
    exp_systolic(rng)
    exp_cpu_gemm(rng)
    exp_im2col(rng)
    meta = {"machine": platform.processor(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "accel_model": Accel().__dict__,
            "accel_peak_tflops": Accel().peak_flops / 1e12, "elapsed_s": round(time.perf_counter() - t0, 1)}
    (RAW / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done in {meta['elapsed_s']} s")


if __name__ == "__main__":
    main()
