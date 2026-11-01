"""Plot the CUDA results in artifacts/raw/gpu/ into artifacts/figures/gpu_*.png|pdf."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
GDDR7_GBS = 1792.0  # RTX 5090 spec: 28 Gbps x 512 bit / 8
PCIE5_GBS = 64.0    # PCIe 5.0 x16, one direction, nominal

SCAN_ALGOS = ["naive", "smem_nopad", "smem", "raking", "thrust", "cub"]
SCAN_LABEL = {"naive": "naive global (1 launch/level)", "smem_nopad": "smem Blelloch, no padding",
              "smem": "smem Blelloch + padding", "raking": "raking + warp shuffle",
              "thrust": "thrust::exclusive_scan", "cub": "cub::DeviceScan"}
STAGES = ["naive_per_circle", "allcircles", "binned", "binned_smem", "blockscan_smem"]
STAGE_LABEL = {"naive_per_circle": "naive per-circle (WRONG)", "allcircles": "per-pixel, all circles",
               "binned": "global binning (scan+sort)", "binned_smem": "binning + smem batches",
               "blockscan_smem": "in-block scan binning + smem"}
SCENES = ["rgb", "rand10k", "rand100k", "biglittle", "littlebig", "pattern", "snowsingle", "rand1M", "micro2M"]


def load(name: str) -> list[dict]:
    return [json.loads(x) for x in (RAW / name).read_text().splitlines() if x.strip()]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def fig_saxpy() -> None:
    rows = [r for r in load("saxpy.jsonl") if r["kind"] == "saxpy"]
    n = np.array([r["n"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, lab, st in [("kernel_gbs", "kernel only (device memory)", "o-"),
                         ("streams_e2e_gbs", "end-to-end, pinned, 4 streams x 16 chunks", "d-"),
                         ("pinned_e2e_gbs", "end-to-end, pinned (cudaMallocHost)", "s-"),
                         ("pageable_e2e_gbs", "end-to-end, pageable (std::vector)", "^-")]:
        ax.plot(n, [r[key] for r in rows], st, label=lab)
    ax.axhline(GDDR7_GBS, color="k", ls="--", lw=0.8)
    ax.text(n[0], GDDR7_GBS * 1.08, "GDDR7 peak 1.79 TB/s", fontsize=8)
    ax.axhline(PCIE5_GBS, color="r", ls="--", lw=0.8)
    ax.text(n[0], PCIE5_GBS * 1.08, "PCIe 5.0 x16 nominal 64 GB/s", fontsize=8, color="r")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N (floats per vector)")
    ax.set_ylabel("effective GB/s  (12N bytes / time)")
    ax.set_title("SAXPY: GPU memory vs PCIe bound")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "gpu_saxpy_bandwidth")

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    sel = [r for r in rows if r["n"] in (1 << 20, 1 << 24, 1 << 28)]
    labels, h2d, ker, d2h = [], [], [], []
    for r in sel:
        for mode in ("pinned", "pageable"):
            labels.append(f"N=2^{int(np.log2(r['n']))}\n{mode}")
            h2d.append(r[f"{mode}_h2d_ms"])
            d2h.append(r[f"{mode}_d2h_ms"])
            ker.append(r["kernel_ms"])
    tot = np.array(h2d) + np.array(ker) + np.array(d2h)
    x = np.arange(len(labels))
    ax.bar(x, np.array(h2d) / tot * 100, label="H2D copy (x, y)")
    ax.bar(x, np.array(ker) / tot * 100, bottom=np.array(h2d) / tot * 100, label="kernel")
    ax.bar(x, np.array(d2h) / tot * 100, bottom=(np.array(h2d) + np.array(ker)) / tot * 100, label="D2H copy (y)")
    for i, t in enumerate(tot):
        ax.text(i, 101, f"{t:.2f} ms", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("% of end-to-end time")
    ax.set_ylim(0, 110)
    ax.set_title("SAXPY end-to-end breakdown: the kernel is a rounding error")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "gpu_saxpy_breakdown")


def scan_rows() -> dict:
    by = defaultdict(list)
    for r in load("scan.jsonl"):
        by[r["algo"]].append(r)
    for a in by:
        by[a].sort(key=lambda r: r["n"])
    return by


def fig_scan() -> None:
    by = scan_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for a in SCAN_ALGOS:
        rs = by[a]
        axes[0].plot([r["n"] for r in rs], [r["ms"] * 1e3 for r in rs], "o-", ms=3, label=SCAN_LABEL[a])
        axes[1].plot([r["n"] for r in rs], [r["eff_gbs"] for r in rs], "o-", ms=3, label=SCAN_LABEL[a])
    n = np.array([r["n"] for r in by["cub"]], dtype=float)
    axes[0].plot(n, 8 * n / (GDDR7_GBS * 1e9) * 1e6, "k--", lw=0.8, label="8N bytes at 1.79 TB/s")
    axes[1].axhline(GDDR7_GBS, color="k", ls="--", lw=0.8)
    axes[0].set_ylabel("time (us)")
    axes[0].set_title("exclusive scan time vs N (int32)")
    axes[1].set_ylabel("effective GB/s = 8N / time")
    axes[1].set_title("effective bandwidth (read N + write N)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.legend(fontsize=7)
    save(fig, "gpu_scan_time_bw")

    # small-N latency floor: time vs launches
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a in SCAN_ALGOS:
        rs = [r for r in by[a] if r["n"] <= 1 << 20]
        ax.plot([r["n"] for r in rs], [r["ms"] * 1e3 for r in rs], "o-", ms=3,
                label=f"{SCAN_LABEL[a]} ({rs[-1]['kernel_launches']} launches @2^20)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("time (us)")
    ax.set_title("small N: latency floor set by number of kernel launches")
    ax.legend(fontsize=7)
    save(fig, "gpu_scan_latency_floor")

    # non power of two
    pairs = [((1 << 20), (1 << 20) + 1), ((1 << 24) - 1, 1 << 24), (1 << 26, 100000000), (100000000, 1 << 27)]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.13
    for i, a in enumerate(SCAN_ALGOS):
        t = {r["n"]: r["ms"] for r in by[a]}
        vals = [t[b] / t[aa] * (aa / b) for aa, b in pairs]  # time per element ratio b vs a
        ax.bar(np.arange(len(pairs)) + (i - 2.5) * w, vals, w, label=SCAN_LABEL[a])
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(pairs)), [f"{b:,} vs {a:,}" for a, b in pairs], fontsize=8)
    ax.set_ylabel("time per element, ratio (right / left size)")
    ax.set_title("non-power-of-2 sizes: only the padded naive scan pays (2^20+1 pads to 2^21)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "gpu_scan_nonpow2")


def fig_find_repeats() -> None:
    rows = load("find_repeats.jsonl")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rng in zip(axes, (100, 2)):
        for a, st in (("scan_smem", "o-"), ("scan_raking", "s-"), ("thrust_copy_if", "^-")):
            rs = sorted([r for r in rows if r["algo"] == a and r["value_range"] == rng], key=lambda r: r["n"])
            ax.plot([r["n"] for r in rs], [r["gelem_s"] for r in rs], st, label=a)
        frac = np.mean([r["repeats"] / r["n"] for r in rows if r["value_range"] == rng])
        ax.set_title(f"find_repeats, values in [0,{rng}) -> {frac * 100:.0f}% repeats")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("G elements / s")
        ax.legend(fontsize=8)
    save(fig, "gpu_find_repeats")


def render_rows() -> list[dict]:
    return [r for r in load("render.jsonl") if r["kind"] == "render"]


def best(rows, scene, stage):
    rs = [r for r in rows if r["scene"] == scene and r["stage"] == stage]
    return min(rs, key=lambda r: r["ms"]) if rs else None


def fig_render() -> None:
    rows = render_rows()
    scenes = [s for s in SCENES if any(r["scene"] == s for r in rows)]
    cpu = {r["scene"]: r["cpu_ref_ms"] for r in load("render.jsonl") if r["kind"] == "scene"}
    # stage comparison
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.14
    for i, st in enumerate(["cpu"] + STAGES):
        vals = []
        for s in scenes:
            if st == "cpu":
                vals.append(cpu[s])
            else:
                b = best(rows, s, st)
                vals.append(b["ms"] if b else np.nan)
        ax.bar(np.arange(len(scenes)) + (i - 2.5) * w, vals, w,
               label="CPU sequential reference (1 core)" if st == "cpu" else STAGE_LABEL[st],
               hatch="//" if st == "naive_per_circle" else None)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(scenes)), scenes)
    ax.set_ylabel("ms per frame (best tile size, incl. binning)")
    ax.set_title("Circle renderer 1024x1024: time per optimization stage")
    ax.legend(fontsize=7, ncol=3)
    save(fig, "gpu_render_stages")

    # tile sweep heatmap for the three tiled stages
    stg = ["binned", "binned_smem", "blockscan_smem"]
    cols = [(s, t) for s in stg for t in (8, 16, 32)]
    M = np.full((len(scenes), len(cols)), np.nan)
    for i, sc in enumerate(scenes):
        for j, (s, t) in enumerate(cols):
            r = [x for x in rows if x["scene"] == sc and x["stage"] == s and x["tile"] == t]
            if r:
                M[i, j] = r[0]["ms"]
    rel = M / np.nanmin(M, axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(np.log2(rel), cmap="viridis_r", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center", fontsize=7,
                    color="w" if rel[i, j] > 2 else "k")
    ax.set_xticks(range(len(cols)), [f"{s}\nT={t}" for s, t in cols], fontsize=7)
    ax.set_yticks(range(len(scenes)), scenes)
    fig.colorbar(im, ax=ax, label="log2(time / best in row)")
    ax.set_title("Tile-size sweep (ms per frame; colour = slowdown vs best for the scene)")
    save(fig, "gpu_render_tile_sweep")

    # binning vs render breakdown at T=16
    fig, ax = plt.subplots(figsize=(10, 4))
    bins, rend = [], []
    for s in scenes:
        r = [x for x in rows if x["scene"] == s and x["stage"] == "binned_smem" and x["tile"] == 16][0]
        bins.append(r["bin_ms"])
        rend.append(r["render_ms"])
    x = np.arange(len(scenes))
    ax.bar(x, bins, label="binning: count + 2 scans + emit + radix sort")
    ax.bar(x, rend, bottom=bins, label="render kernel (smem batches)")
    for i in range(len(scenes)):
        ax.text(i, bins[i] + rend[i], f"{bins[i] / (bins[i] + rend[i]) * 100:.0f}%", ha="center",
                va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x, scenes)
    ax.set_ylabel("ms (log)")
    ax.set_title("binned_smem, T=16: where the time goes (label = binning share)")
    ax.legend(fontsize=8)
    save(fig, "gpu_render_breakdown")

    # list-length statistics (load balance) vs tile size
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for s in scenes:
        rs = sorted([x for x in rows if x["scene"] == s and x["stage"] == "binned"], key=lambda r: r["tile"])
        axes[0].plot([r["tile"] for r in rs], [r["pairs"] for r in rs], "o-", label=s)
        axes[1].plot([r["tile"] for r in rs], [r["max_list"] / max(r["mean_list"], 1e-9) for r in rs], "o-", label=s)
    axes[0].set_ylabel("(tile, circle) pairs")
    axes[0].set_title("binning work grows as tiles shrink")
    axes[1].set_ylabel("max / mean circles per tile")
    axes[1].set_title("load imbalance across tiles")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("tile size T (pixels)")
        ax.legend(fontsize=7)
    save(fig, "gpu_render_lists")

    # correctness: naive error per scene
    fig, ax = plt.subplots(figsize=(10, 3.8))
    nv = [best(rows, s, "naive_per_circle") for s in scenes]
    frac = [r["pixels_diff_gt_1e-3"] / (1024 * 1024) * 100 for r in nv]
    nd = [r.get("pixels_differ_between_runs", 0) / (1024 * 1024) * 100 for r in nv]
    ax.bar(x - 0.2, frac, 0.4, label="pixels wrong vs reference (>1e-3)")
    ax.bar(x + 0.2, nd, 0.4, label="pixels differing between two naive runs")
    for i, r in enumerate(nv):
        ax.text(i - 0.2, frac[i] + 1, f"max {r['max_abs_diff']:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x, scenes)
    ax.set_ylabel("% of 1024x1024 pixels")
    ax.set_title("naive one-thread-per-circle kernel: wrong and nondeterministic (correct stages: max diff 0)")
    ax.legend(fontsize=8)
    save(fig, "gpu_render_naive_error")


def fig_sweep() -> None:
    rows = load("render_sweep.jsonl")
    cpu = sorted([r for r in rows if r["kind"] == "scene"], key=lambda r: r["n"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot([r["n"] for r in cpu], [r["cpu_ref_ms"] for r in cpu], "k*-", label="CPU sequential reference")
    for st, mk in zip(STAGES, "xosd^"):
        rs = sorted([r for r in rows if r["kind"] == "render" and r["stage"] == st], key=lambda r: r["n"])
        ax.plot([r["n"] for r in rs], [r["ms"] for r in rs], mk + "-", label=STAGE_LABEL[st])
    n = np.array([r["n"] for r in cpu], dtype=float)
    a0 = min((r for r in rows if r["kind"] == "render" and r["stage"] == "allcircles"), key=lambda r: r["n"])
    ax.plot(n, a0["ms"] * n / a0["n"], "C1:", lw=0.8, label="slope 1 (N x pixels)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N circles (radius scaled so ~45 circles cover each pixel)")
    ax.set_ylabel("ms per 1024x1024 frame (T=16)")
    ax.set_title("Scaling with N at constant overdraw")
    ax.legend(fontsize=7)
    save(fig, "gpu_render_sweep")


def fig_images() -> None:
    img = RAW / "images"
    from matplotlib.image import imread
    show = [s for s in ["rgb", "rand10k", "pattern", "biglittle", "littlebig", "snowsingle", "rand100k", "rand1M"]
            if (img / f"{s}_ref.png").exists()]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, s in zip(axes.flat, show):
        ax.imshow(imread(img / f"{s}_ref.png"))
        ax.set_title(s, fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Sequential-reference images (identical to every correct CUDA stage)")
    save(fig, "gpu_render_gallery")

    sc = [s for s in ["rgb", "rand10k", "pattern", "biglittle"] if (img / f"{s}_naive.png").exists()]
    fig, axes = plt.subplots(3, len(sc), figsize=(3.2 * len(sc), 9.4))
    for j, s in enumerate(sc):
        ref = imread(img / f"{s}_ref.png")[..., :3]
        bad = imread(img / f"{s}_naive.png")[..., :3]
        axes[0, j].imshow(ref)
        axes[0, j].set_title(f"{s}: ordered reference", fontsize=9)
        axes[1, j].imshow(bad)
        axes[1, j].set_title("naive per-circle CUDA", fontsize=9)
        diff = imread(img / f"{s}_naive_absdiff.png")  # full-res max-channel |diff|, 8-bit / 255
        im = axes[2, j].imshow(diff, cmap="inferno", vmin=0, vmax=0.8)
        axes[2, j].set_title("|naive - ref| (1024x1024, 8-bit)", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.03)
    fig.savefig(FIG / "gpu_render_naive_images.png", dpi=110)
    fig.savefig(FIG / "gpu_render_naive_images.pdf")
    plt.close(fig)
    print("  wrote gpu_render_naive_images")


def fig_roofline() -> None:
    """Achieved bandwidth of the memory-bound kernels against the GDDR7 roof."""
    sax = [r for r in load("saxpy.jsonl") if r["kind"] == "saxpy"]
    by = scan_rows()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = [("saxpy kernel 2^28", [r for r in sax if r["n"] == 1 << 28][0]["kernel_gbs"])]
    for a in SCAN_ALGOS:
        bars.append((f"scan {a} 2^28", [r for r in by[a] if r["n"] == 1 << 28][0]["eff_gbs"]))
    fr = load("find_repeats.jsonl")
    for a in ("scan_raking", "thrust_copy_if"):
        r = max((x for x in fr if x["algo"] == a and x["value_range"] == 100), key=lambda x: x["n"])
        bars.append((f"find_repeats {a} 1e8", r["gelem_s"] * 4))
    y = np.arange(len(bars))
    ax.barh(y, [b[1] for b in bars], color=["C0"] + ["C1"] * len(SCAN_ALGOS) + ["C2"] * 2)
    for i, b in enumerate(bars):
        ax.text(b[1], i, f" {b[1]:.0f} ({b[1] / GDDR7_GBS * 100:.0f}%)", va="center", fontsize=7)
    ax.axvline(GDDR7_GBS, color="k", ls="--", lw=0.8)
    ax.set_yticks(y, [b[0] for b in bars], fontsize=8)
    ax.set_xlabel("effective GB/s (compulsory bytes / time); find_repeats counts 4N")
    ax.set_title("Memory-bound kernels vs the 1.79 TB/s GDDR7 roof")
    ax.set_xlim(0, GDDR7_GBS * 1.25)
    save(fig, "gpu_bandwidth_roof")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_saxpy()
    fig_scan()
    fig_find_repeats()
    fig_render()
    fig_sweep()
    fig_images()
    fig_roofline()


if __name__ == "__main__":
    main()
