"""Figures for the C++/CUDA version of asst1 (README PNGs + report PDFs).

Reads artifacts/raw/cpp/<platform>/*.jsonl and artifacts/raw/gpu/*.jsonl and writes
artifacts/figures/cpp_*.png|pdf. Server numbers are the median over the repeated campaigns
(server, server_r2, server_r3) with min-max error bars.

Usage: python scripts/plot_cpp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SERVER_REPS = ["server", "server_r2", "server_r3"]
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
C = {"block": "#d62728", "interleaved": "#1f77b4", "dynamic": "#2ca02c", "scalar": "#7f7f7f",
     "avx2": "#ff7f0e", "avx512": "#1f77b4", "avx512x2": "#9467bd", "simd": "#1f77b4",
     "stream": "#2ca02c", "read": "#8c564b", "gpu": "#76b900"}


def jl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def load(name: str, platforms=None) -> pd.DataFrame:
    frames = []
    for p in platforms or SERVER_REPS:
        d = jl(RAW / "cpp" / p / f"{name}.jsonl")
        if not d.empty:
            frames.append(d.assign(rep=p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gpu(name: str) -> pd.DataFrame:
    df = jl(RAW / "gpu" / f"{name}.jsonl")
    if name == "mandel_iters" and not df.empty and "iters" not in df:
        # Older a1_gpu builds did not print max_iter; wave2.sh ran iters (64,256,1024,4096) x view (1,2), 12 shapes each.
        df["iters"] = [(64, 256, 1024, 4096)[i // 24] for i in range(len(df))]
    return df


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def agg(d: pd.DataFrame, keys: list[str], val: str) -> pd.DataFrame:
    g = d.groupby(keys)[val]
    return pd.DataFrame({"med": g.median(), "lo": g.min(), "hi": g.max()}).reset_index()


def errbar(ax, x, a, **kw):
    ax.errorbar(x, a["med"], yerr=[a["med"] - a["lo"], a["hi"] - a["med"]], capsize=2, **kw)


# ------------------------------------------------------------------ Mandelbrot images
def mandel_np(view: int, w=1600, h=1200, it=256) -> np.ndarray:
    x0, y0, x1, y1 = -2.0, -1.0, 1.0, 1.0
    if view == 2:
        s, sx, sy = 0.015, -0.986, 0.30
        x0, y0, x1, y1 = x0 * s + sx, y0 * s + sy, x1 * s + sx, y1 * s + sy
    xs = np.float32(x0) + np.arange(w, dtype=np.float32) * np.float32((x1 - x0) / w)
    ys = np.float32(y0) + np.arange(h, dtype=np.float32) * np.float32((y1 - y0) / h)
    cr, ci = np.meshgrid(xs, ys)
    zr, zi = cr.copy(), ci.copy()
    out = np.full(cr.shape, it, dtype=np.int32)
    alive = np.ones(cr.shape, bool)
    for k in range(it):
        zr2, zi2 = zr * zr, zi * zi
        esc = alive & (zr2 + zi2 > 4)
        out[esc] = k
        alive &= ~esc
        if not alive.any():
            break
        zi = np.where(alive, 2 * zr * zi + ci, zi)
        zr = np.where(alive, zr2 - zi2 + cr, zr)
    return out


def fig_views():
    rows = load("prog1_rows", ["server"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for r, view in enumerate((1, 2)):
        img = mandel_np(view)
        ax = axes[r, 0]
        ax.imshow(np.log1p(img), cmap="magma", aspect="auto")
        ax.set_title(f"view {view}: log(1+iterations), 1600x1200, max 256")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        ax = axes[r, 1]
        d = rows[rows.view == view].sort_values("row")
        ax.plot(d["work"] / 1e3, d["row"], color="k", lw=0.8)
        ax.invert_yaxis()
        for P, col in ((4, C["block"]), (12, "#9467bd")):
            for b in range(1, P):
                ax.axhline(1200 * b / P, color=col, lw=0.6, ls="--" if P == 12 else "-", alpha=0.8)
        tot = d["work"].sum()
        for P, col in ((4, C["block"]), (12, "#9467bd")):
            parts = [d["work"].values[int(1200 * k / P):int(1200 * (k + 1) / P)].sum() for k in range(P)]
            ax.text(0.98, 0.02 + 0.08 * (P == 12), f"P={P} block: max/mean = {max(parts) / (tot / P):.2f}",
                    transform=ax.transAxes, ha="right", color=col, fontsize=8.5)
        ax.set_xlabel("work per row (k iterations)"), ax.set_ylabel("row")
        ax.set_title(f"view {view}: per-row cost and contiguous-block cuts")
    save(fig, "cpp_mandel_views")


# ------------------------------------------------------------------ prog1
def fig_prog1_speedup():
    d = load("prog1_threads")
    loc = load("prog1_threads", ["local4c"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, view in enumerate((1, 2)):
        ax = axes[i]
        dv = d[(d.view == view) & (d.threads <= 12)]
        for s in ("block", "interleaved", "dynamic"):
            a = agg(dv[dv.sched == s], ["threads"], "speedup")
            errbar(ax, a.threads, a, marker="o", ms=3.5, color=C[s], label=s)
        ax.plot([1, 12], [1, 12], "k:", lw=1, label="ideal")
        ax.set_title(f"server (12 pinned cores), view {view}")
        ax.set_xlabel("threads"), ax.set_ylabel("speedup vs serial")
        ax.legend(loc="upper left")
    ax = axes[2]
    for view, ls in ((1, "-"), (2, "--")):
        dv = loc[loc.view == view]
        for s in ("block", "interleaved", "dynamic"):
            a = dv[dv.sched == s].sort_values("threads")
            ax.plot(a.threads, a.speedup, ls=ls, marker="o", ms=3.5, color=C[s], label=f"{s} v{view}")
    ax.plot([1, 4], [1, 4], "k:", lw=1)
    ax.axvline(4, color="grey", lw=0.6)
    ax.set_title("local 4C/8T Ice Lake (threads > 4 use SMT)")
    ax.set_xlabel("threads"), ax.set_ylabel("speedup vs serial"), ax.legend(fontsize=7.5, ncol=2)
    save(fig, "cpp_prog1_speedup")


def fig_prog1_thread_time():
    d = load("prog1_threads", ["server"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, s in zip(axes, ("block", "interleaved", "dynamic")):
        r = d[(d.view == 1) & (d.sched == s) & (d.threads == 8)].iloc[0]
        t = np.array(r.thread_s) * 1e3
        ax.bar(range(8), t, color=C[s])
        ax.axhline(r.wall_s * 1e3, color="k", ls="--", lw=0.8, label=f"wall {r.wall_s * 1e3:.1f} ms")
        ax.set_title(f"{s}: speedup {r.speedup:.2f}x"), ax.set_xlabel("thread id")
        ax.legend()
    axes[0].set_ylabel("compute time per thread (ms)")
    fig.suptitle("view 1, 8 threads: who waits for whom", y=1.02)
    save(fig, "cpp_prog1_thread_time")


def fig_prog1_chunk():
    d = load("prog1_chunk")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for view, col in ((1, "#1f77b4"), (2, "#ff7f0e")):
        a = agg(d[d.view == view], ["chunk"], "speedup")
        errbar(ax, a.chunk, a, marker="o", color=col, label=f"view {view}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rows per dynamic chunk (100 = one chunk per thread)"), ax.set_ylabel("speedup (12 threads)")
    ax.set_title("dynamic scheduling: chunk-size trade-off")
    ax.legend()
    save(fig, "cpp_prog1_chunk")


def fig_smt():
    d = load("prog1_smt")
    if d.empty:
        return
    d["label"] = d.apply(lambda r: "scalar (prog1)" if r["kind"] == "run" else str(r.get("isa", "avx512")), axis=1)
    layouts = list(dict.fromkeys(d.layout))
    labels = ["scalar (prog1)", "avx512", "avx512x2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, view in zip(axes, (1, 2)):
        w = 0.26
        for k, lab in enumerate(labels):
            vals, los, his = [], [], []
            for lay in layouts:
                x = d[(d.view == view) & (d.layout == lay) & (d.label == lab)]["wall_s"] * 1e3
                vals.append(x.median() if len(x) else np.nan)
                los.append(x.min() if len(x) else np.nan), his.append(x.max() if len(x) else np.nan)
            vals, los, his = map(np.array, (vals, los, his))
            ax.bar(np.arange(len(layouts)) + (k - 1) * w, vals, w, yerr=[vals - los, his - vals], capsize=2,
                   color=[C["scalar"], C["avx512"], C["avx512x2"]][k], label=lab)
        ax.set_yscale("log")
        ax.set_xticks(range(len(layouts)), [l.replace("_", " ") for l in layouts], fontsize=8)
        ax.set_ylabel("wall time (ms, log)"), ax.set_title(f"view {view}: same threads, fewer physical cores")
        ax.legend(loc="upper right", fontsize=8, ncol=3, bbox_to_anchor=(1.0, 1.0))
        ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    save(fig, "cpp_smt")


# ------------------------------------------------------------------ prog2
def fig_prog2():
    d = load("prog2_vecintrin", ["server"])
    ce = d[d.kind == "clamped_exp"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for dist, mk in zip(["constant", "sorted", "uniform", "skewed", "one_slow_per_vector"], "osD^v"):
        a = ce[ce.dist == dist].sort_values("width")
        ax.plot(a.width, a.utilization, marker=mk, label=dist)
    ax.set_xscale("log", base=2), ax.set_xticks([2, 4, 8, 16], [2, 4, 8, 16])
    ax.set_xlabel("VECTOR_WIDTH"), ax.set_ylabel("lane utilization"), ax.set_ylim(0.4, 1.0)
    ax.set_title("clampedExp: utilization vs width (N=10000)"), ax.legend(fontsize=8)
    ax = axes[1]
    for dist in ["constant", "sorted", "uniform", "skewed", "one_slow_per_vector"]:
        a = ce[ce.dist == dist].sort_values("width")
        ax.plot(a.width, a.instructions, marker="o", label=dist)
    ax.set_xscale("log", base=2), ax.set_yscale("log"), ax.set_xticks([2, 4, 8, 16], [2, 4, 8, 16])
    ax.set_xlabel("VECTOR_WIDTH"), ax.set_ylabel("vector instructions issued")
    ax.set_title("wider vectors issue fewer instructions, but waste more lanes")
    save(fig, "cpp_prog2_utilization")


# ------------------------------------------------------------------ prog3
def fig_prog3_simd():
    srv, loc = load("prog3_simd"), load("prog3_simd", ["local4c"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, d) in zip(axes, (("server Xeon 6459C", srv), ("local Ice Lake", loc))):
        if d.empty:
            continue
        cols = [("avx2_speedup", "AVX2 (8)", C["avx2"]), ("avx512_speedup", "AVX-512 (16)", C["avx512"]),
                ("avx512x2_speedup", "AVX-512 x2 ILP", C["avx512x2"])]
        for k, (col, lab, c) in enumerate(cols):
            if col not in d:
                continue
            a = agg(d, ["view"], col)
            ax.bar(a.view + (k - 1) * 0.27, a.med, 0.27, yerr=[a.med - a.lo, a.hi - a.med], capsize=2, color=c, label=lab)
        u = d.groupby("view")[["util_w8", "util_w16"]].first()
        for v in (1, 2):
            ax.plot([v - 0.4, v - 0.14], [8 * u.loc[v, "util_w8"]] * 2, "k-", lw=1.5)
            ax.plot([v - 0.13, v + 0.13], [16 * u.loc[v, "util_w16"]] * 2, "k-", lw=1.5)
        ax.plot([], [], "k-", label="model W x utilization")
        ax.set_xticks([1, 2], ["view 1", "view 2"]), ax.set_ylabel("speedup vs scalar (1 thread)")
        ax.set_title(name), ax.legend(fontsize=8)
    save(fig, "cpp_prog3_simd")


def fig_prog3_tasks():
    d = load("prog3_tasks")
    d = d[d.get("scaling").isna()] if "scaling" in d else d
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, view in zip(axes, (1, 2)):
        for isa in ("avx2", "avx512", "avx512x2"):
            a = agg(d[(d.view == view) & (d.isa == isa)], ["tasks"], "speedup_vs_simd1")
            if len(a):
                errbar(ax, a.tasks, a, marker="o", ms=3.5, color=C[isa], label=isa)
        ax.axhline(12, color="k", ls=":", lw=1, label="12 cores")
        ax.set_xscale("log"), ax.set_xlabel("ISPC-style tasks (12 worker threads)")
        ax.set_ylabel("speedup vs same ISA, 1 thread"), ax.set_title(f"view {view}: how many tasks?")
        ax.legend(fontsize=8)
    save(fig, "cpp_prog3_tasks")


def fig_pin():
    d = load("prog3_pin")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    pins = ["cpu0", "cpu_first_core", "unpinned"]
    for k, view in enumerate((1, 2)):
        vals = [agg(d[(d.view == view) & (d.pin == p)], ["view"], "speedup_vs_simd1") for p in pins]
        med = np.array([v.med.iloc[0] if len(v) else np.nan for v in vals])
        lo = np.array([v.lo.iloc[0] if len(v) else np.nan for v in vals])
        hi = np.array([v.hi.iloc[0] if len(v) else np.nan for v in vals])
        ax.bar(np.arange(3) + (k - 0.5) * 0.38, med, 0.38, yerr=[med - lo, hi - med], capsize=2, label=f"view {view}")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(range(3), ["pinned CPU 0", "pinned CPU 1", "unpinned"])
    ax.set_ylabel("1 worker thread / main thread speed"), ax.set_title("one task on one worker: where it runs matters")
    ax.legend()
    save(fig, "cpp_pin")


# ------------------------------------------------------------------ prog4
def fig_prog4():
    it = load("prog4_iters", ["server"])
    d = load("prog4_sqrt")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    ax.plot(it.x, it.iterations, color="k")
    ax.set_xlabel("input x"), ax.set_ylabel("Newton iterations to 1e-5"), ax.set_title("prog4: work depends on the input")
    ax.axvspan(0.95, 1.05, color="green", alpha=0.15), ax.text(1.0, 2, "x~1: 0 iters", ha="center", fontsize=8)
    ax = axes[1]
    pats = ["random", "all_slow", "all_one", "one_slow_8", "one_slow_16"]
    simd = d[d.kind == "simd"]
    tasks = d[(d.kind == "tasks")]
    P = int(tasks.threads.max())
    series = [("avx2_speedup", simd, "AVX2", C["avx2"]), ("avx512_speedup", simd, "AVX-512", C["avx512"])]
    w = 0.27
    for k, (col, src, lab, c) in enumerate(series):
        a = agg(src, ["pattern"], col).set_index("pattern").loc[pats]
        ax.bar(np.arange(5) + (k - 1) * w, a.med, w, yerr=[a.med - a.lo, a.hi - a.med], capsize=2, color=c, label=lab)
    a = agg(tasks[tasks.threads == P], ["pattern"], "speedup_vs_scalar").set_index("pattern").loc[pats]
    ax.bar(np.arange(5) + w, a.med, w, yerr=[a.med - a.lo, a.hi - a.med], capsize=2, color="#2ca02c", label=f"AVX-512 + {P} threads")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_yscale("log"), ax.set_xticks(range(5), pats, fontsize=8.5)
    ax.set_ylabel("speedup vs scalar (log)"), ax.set_title("best / worst inputs for SIMD"), ax.legend(fontsize=8)
    for i, p in enumerate(pats):
        mi = simd[simd.pattern == p].mean_iters.iloc[0]
        ax.text(i, 0.35, f"{mi:.1f} it", ha="center", fontsize=7.5, color="grey")
    save(fig, "cpp_prog4_sqrt")


# ------------------------------------------------------------------ prog5
def fig_prog5_sizes():
    d = load("prog5_sizes")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, T in zip(axes, (1, int(d.threads.max()))):
        for k in ("scalar", "simd", "stream", "read"):
            a = agg(d[(d.threads == T) & (d.kernel == k)], ["bytes_ws"], "visible_gbs")
            errbar(ax, a.bytes_ws, a, marker=".", color=C[k], label=k)
        per = {"L1 48K": 48e3, "L2 2M": 2e6, "L3 60M": 60e6}
        for lab, b in per.items():
            agg_b = b * (T if lab != "L3 60M" else 1)
            ax.axvline(agg_b, color="grey", ls="--", lw=0.7)
            ax.text(agg_b, ax.get_ylim()[1] if False else 2.2, f"{lab}{' x' + str(T) if T > 1 and lab != 'L3 60M' else ''}",
                    rotation=90, fontsize=7, va="bottom", ha="right", color="grey")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlabel("working set (bytes, 3 arrays)"), ax.set_ylabel("programmer-visible GB/s")
        ax.set_title(f"saxpy bandwidth, {T} thread(s), server"), ax.legend(fontsize=8)
    save(fig, "cpp_prog5_sizes")


def fig_prog5_threads():
    d, numa = load("prog5_threads"), load("prog5_numa")
    loc = load("prog5_threads", ["local4c"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 0.9]})
    ax = axes[0]
    for k in ("scalar", "simd", "stream", "read"):
        a = agg(d[d.kernel == k], ["threads"], "real_gbs")
        errbar(ax, a.threads, a, marker="o", ms=3.5, color=C[k], label=k)
    ax.set_xlabel("threads"), ax.set_ylabel("real DRAM GB/s (incl. write-allocate)")
    ax.set_title("server: 768 MB saxpy, one socket"), ax.legend(fontsize=8)
    ax = axes[1]
    for k in ("simd", "stream"):
        a = loc[loc.kernel == k].sort_values("threads")
        ax.plot(a.threads, a.real_gbs, marker="o", color=C[k], label=k)
    ax.set_xlabel("threads"), ax.set_ylabel("real GB/s"), ax.set_title("local Ice Lake VM"), ax.legend(fontsize=8)
    ax = axes[2]
    if not numa.empty:
        pl = ["local", "split", "remote"]
        for j, k in enumerate(("simd", "stream")):
            a = agg(numa[numa.kernel == k], ["placement"], "real_gbs").set_index("placement").loc[pl]
            ax.bar(np.arange(3) + (j - 0.5) * 0.38, a.med, 0.38, yerr=[a.med - a.lo, a.hi - a.med], capsize=2,
                   color=C[k], label=k)
        ax.set_xticks(range(3), ["threads on\nmemory's socket", "half/half", "other socket"], fontsize=8)
        ax.set_ylabel("real GB/s (12 threads)"), ax.set_title("NUMA: first-touch on socket 0"), ax.legend(fontsize=8)
    save(fig, "cpp_prog5_threads")


# ------------------------------------------------------------------ prog6
def fig_prog6():
    d = load("prog6_kmeans")
    m = d.groupby(["ver", "threads"])[["assign_s", "centroids_s", "cost_s", "total_s", "speedup"]].median().reset_index()
    order = [("baseline", 1)] + [(v, t) for v in ("v1", "v2", "v3") for t in sorted(m[m.ver == v].threads)]
    m = m.set_index(["ver", "threads"]).loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(m))
    bot = np.zeros(len(m))
    for col, lab, c in (("assign_s", "computeAssignments", "#1f77b4"), ("centroids_s", "computeCentroids", "#ff7f0e"),
                        ("cost_s", "computeCost", "#2ca02c")):
        ax.bar(x, m[col], bottom=bot, color=c, label=lab)
        bot += m[col].values
    for i, r in m.iterrows():
        ax.text(i, r.total_s + 0.02, f"{r.speedup:.1f}x", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"{v}\n{t}T" for v, t in zip(m.ver, m.threads)], fontsize=8)
    ax.set_ylabel("seconds (M=1e6, N=100, K=3)")
    ax.set_title("K-means: v1 parallel assignments only (Amdahl), v2 + parallel centroids/cost, v3 + interchange/no-sqrt/AVX-512")
    ax.legend()
    save(fig, "cpp_prog6_kmeans")


# ------------------------------------------------------------------ GPU
def fig_gpu_mandel():
    d = gpu("mandel")
    if d.empty:
        return
    d = d[d.scale == 4]
    d["shape"] = d.bx.astype(str) + "x" + d.by.astype(str)
    shapes = ["32x1", "64x1", "128x1", "256x1", "1x32", "16x2", "8x4", "4x8", "8x8", "16x16", "32x4", "32x8"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, view in zip(axes, (1, 2)):
        a = d[d.view == view].set_index("shape").loc[shapes]
        cols = ["#d62728" if bx * by == 32 else "#1f77b4" for bx, by in zip(a.bx, a.by)]
        ax.bar(range(len(shapes)), a.ms, color=cols)
        ax2 = ax.twinx()
        ax2.plot(range(len(shapes)), a.warp_eff, "ko-", ms=3)
        ax2.set_ylabel("warp execution efficiency (o)"), ax2.grid(False)
        ax.set_xticks(range(len(shapes)), shapes, rotation=45, fontsize=8)
        ax.set_ylabel("kernel ms (6400x4800)"), ax.set_title(f"RTX 5090 view {view}: block shape (red = 32-thread blocks)")
    save(fig, "cpp_gpu_mandel_shapes")


def fig_gpu_sqrt():
    d = gpu("sqrt_repeat")
    if d.empty:
        d = gpu("sqrt").assign(repeat=1)
    cpu = load("prog4_sqrt")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    pats = ["random", "all_slow", "all_one", "one_slow_8", "one_slow_16", "one_slow_32", "one_slow_64", "sorted_random"]
    for r in sorted(d["repeat"].unique()):
        a = d[d["repeat"] == r].set_index("pattern")
        base = a.loc["all_slow", "ms"]
        ax.plot(range(len(pats)), [a.loc[p, "ms"] / base if p in a.index else np.nan for p in pats], marker="o", label=f"repeat={r}")
    ax.set_xticks(range(len(pats)), pats, rotation=35, fontsize=8)
    ax.set_ylabel("time / time(all_slow)"), ax.set_title("GPU sqrt: divergence only shows once compute-bound")
    ax.legend(fontsize=8)
    ax = axes[1]
    a = d[d["repeat"] == 1].set_index("pattern")
    c = cpu[(cpu.kind == "simd")].groupby("pattern").avx512_s.median()
    P = int(cpu[cpu.kind == "tasks"].threads.max())
    ct = cpu[(cpu.kind == "tasks") & (cpu.threads == P)].groupby("pattern").wall_s.median()
    common = [p for p in ["random", "all_slow", "all_one", "one_slow_8", "one_slow_16"] if p in a.index]
    w = 0.27
    ax.bar(np.arange(len(common)) - w, [c[p] * 1e3 for p in common], w, color=C["avx512"], label="CPU AVX-512 1T")
    ax.bar(np.arange(len(common)), [ct[p] * 1e3 for p in common], w, color="#2ca02c", label=f"CPU AVX-512 {P}T")
    ax.bar(np.arange(len(common)) + w, [a.loc[p, "ms"] for p in common], w, color=C["gpu"], label="RTX 5090")
    ax.set_yscale("log"), ax.set_xticks(range(len(common)), common, fontsize=8)
    ax.set_ylabel("ms for 20M elements (log)"), ax.set_title("same Newton sqrt: CPU vs GPU")
    ax.set_ylim(top=ax.get_ylim()[1] * 6), ax.legend(fontsize=8, ncol=3, loc="upper center")
    save(fig, "cpp_gpu_sqrt")


def fig_bw_compare():
    g = gpu("saxpy")
    c = load("prog5_sizes")
    if g.empty or c.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(g.bytes, g.vec4_gbs, "o-", color=C["gpu"], ms=3, label="RTX 5090 kernel (float4)")
    ax.plot(g.bytes, g.e2e_gbs, "s--", color=C["gpu"], ms=3, label="RTX 5090 incl. PCIe copies")
    T = int(c.threads.max())
    for th, ls in ((1, ":"), (T, "-")):
        a = agg(c[(c.threads == th) & (c.kernel == "simd")], ["bytes_ws"], "visible_gbs")
        ax.plot(a.bytes_ws, a.med, ls=ls, color=C["simd"], label=f"Xeon AVX-512, {th} thread(s)")
    ax.axhline(1792, color=C["gpu"], lw=0.6, ls=":"), ax.text(2e4, 1900, "GDDR7 spec 1792 GB/s", fontsize=7.5, color=C["gpu"])
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("working set (bytes)"), ax.set_ylabel("GB/s (12 B/element)")
    ax.set_title("saxpy: GPU wins on-device, loses if data crosses PCIe"), ax.legend(fontsize=8)
    save(fig, "cpp_bw_cpu_vs_gpu")


def fig_ladder():
    simd, tasks = load("prog3_simd"), load("prog3_tasks")
    g = gpu("mandel")
    gi = gpu("mandel_iters")
    if simd.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, view in enumerate((1, 2)):
        s = simd[simd.view == view]
        sc = s.scalar_s.median()
        steps = [("scalar\n1 thread", 1.0), ("AVX2", s.avx2_speedup.median()), ("AVX-512", s.avx512_speedup.median())]
        if "avx512x2_speedup" in s and s.avx512x2_speedup.notna().any():
            steps.append(("AVX-512\nx2 ILP", s.avx512x2_speedup.median()))
        t = tasks[(tasks.view == view) & (tasks.get("scaling").isna() if "scaling" in tasks else True)]
        best = t.groupby(["isa", "tasks"]).speedup_vs_scalar.median()
        steps.append((f"best SIMD +\n12 threads", best.max()))
        gg = g[(g.view == view) & (g.scale == 1)]
        if len(gg):
            steps.append(("RTX 5090\n1600x1200", sc * 1e3 / gg.ms.min()))
        if not gi.empty:
            g4 = gi[(gi.view == view) & (gi.iters == 256)]
            if len(g4):
                px = 1600 * 1200
                steps.append(("RTX 5090\nper-pixel @16x", sc * 1e3 / (g4.ms.min() * px / (6400 * 4800))))
        x = np.arange(len(steps))
        ax.plot(x, [v for _, v in steps], "o-", color=["#1f77b4", "#ff7f0e"][k], label=f"view {view}")
        for i, (_, v) in enumerate(steps):
            ax.text(i, v * (1.25 if k == 0 else 0.7), f"{v:.0f}x" if v >= 10 else f"{v:.1f}x", ha="center", fontsize=7.5,
                    color=["#1f77b4", "#ff7f0e"][k])
        ax.set_xticks(x, [n for n, _ in steps], fontsize=8)
    ax.set_yscale("log"), ax.set_ylabel("speedup vs scalar 1 thread (log)")
    ax.set_title("Mandelbrot speedup ladder: SIMD x ILP x cores x GPU"), ax.legend()
    save(fig, "cpp_ladder")


def fig_gpu_iters():
    gi = gpu("mandel_iters")
    tasks = load("prog3_tasks")
    if gi.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for view, c in ((1, "#1f77b4"), (2, "#ff7f0e")):
        a = gi[gi.view == view].groupby("iters").giters_per_s.max()
        ax.plot(a.index, a.values, "o-", color=c, label=f"RTX 5090 view {view}")
    ax.set_xscale("log", base=2), ax.set_xlabel("max iterations"), ax.set_ylabel("G lane-iterations / s (useful)")
    ax.set_title("GPU throughput rises until launch/latency is amortised"), ax.legend(fontsize=8)
    save(fig, "cpp_gpu_iters")


def main():
    for f in (fig_views, fig_prog1_speedup, fig_prog1_thread_time, fig_prog1_chunk, fig_smt, fig_prog2, fig_prog3_simd,
              fig_prog3_tasks, fig_pin, fig_prog4, fig_prog5_sizes, fig_prog5_threads, fig_prog6, fig_gpu_mandel,
              fig_gpu_sqrt, fig_bw_compare, fig_ladder, fig_gpu_iters):
        try:
            f()
        except Exception as e:  # keep going so one missing file does not block the rest
            print(f"FAILED {f.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
