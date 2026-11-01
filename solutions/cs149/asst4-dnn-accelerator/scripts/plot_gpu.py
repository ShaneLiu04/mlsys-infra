"""Figures for the GPU part (reads artifacts/raw/gpu/*, writes artifacts/figures/gpu_*.png|pdf). Local, CPU only."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
SPEC_FP32 = 104.8e12  # RTX 5090 datasheet fp32 (non-tensor) at boost clock
SPEC_BW = 1.792e12  # GDDR7 datasheet

LADDER_ORDER = ["naive", "coalesced", "smem8", "smem16", "smem32", "reg1d", "reg2d", "reg2d_nc", "vec4", "vec4_db",
                "vec4_db_lb2",
                "cublas_fp32", "cublas_tf32", "wmma_fp16", "wmma_bf16", "cublas_fp16", "cublas_bf16", "cublas_fp8"]


def load(name: str) -> list[dict]:
    p = RAW / name
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def peaks() -> dict:
    bw = load("micro_bw.jsonl")
    fma = load("micro_fma.jsonl")
    tc = load("micro_tensor.jsonl")
    dram = max(r["gbps"] for r in bw if r["kernel"] in ("copy", "triad") and r["mib_per_array"] >= 256) * 1e9
    l2 = max(r["gbps"] for r in bw if r["kernel"] in ("copy", "triad")) * 1e9
    fp32 = max(r["tflops"] for r in fma) * 1e12
    out = {"dram": dram, "l2": l2, "fp32": fp32}
    for dt in ("bf16", "fp16", "tf32"):
        v = [r["tflops"] for r in tc if r["dtype"] == dt]
        if v:
            out[dt] = max(v) * 1e12
    return out


def ncu_rows() -> list[dict]:
    p = RAW / "ncu_raw.csv"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="replace")
    i = txt.find('"ID"')
    if i < 0:
        return []
    rows = list(csv.DictReader(io.StringIO(txt[i:])))
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["ID"], r["Kernel Name"])
        d = out.setdefault(key, {"id": int(r["ID"]), "kernel": r["Kernel Name"]})
        try:
            d[r["Metric Name"]] = float(r["Metric Value"].replace(",", ""))
        except ValueError:
            d[r["Metric Name"]] = r["Metric Value"]
        d[r["Metric Name"] + "@unit"] = r["Metric Unit"]
    return sorted(out.values(), key=lambda d: d["id"])


NCU_LABELS = ["naive", "coalesced", "smem8", "smem32", "reg1d", "reg2d", "vec4", "vec4_db", "wmma_fp16",
              "cublas_fp32", "cublas_fp16", "triad", "conv2x2_unfused", "relu", "pool", "conv2x2_fused",
              "igemm_relu", "cudnn_conv"]


def label_ncu(rows: list[dict]) -> list[dict]:
    """ncu_targets.py launches kernels in a fixed order; torch ops may add helper kernels, so match by name."""
    seq = []
    for r in rows:
        k = r["kernel"]
        if k.startswith("k_naive"): lab = "naive"
        elif k.startswith("k_coalesced"): lab = "coalesced"
        elif k.startswith("k_smem<8"): lab = "smem8"
        elif k.startswith("k_smem<32"): lab = "smem32"
        elif k.startswith("k_reg1d"): lab = "reg1d"
        elif k.startswith("k_reg2d"): lab = "reg2d"
        elif k.startswith("k_vec<false"): lab = "vec4"
        elif k.startswith("k_vec<true"): lab = "vec4_db"
        elif k.startswith("k_wmma"): lab = "wmma_fp16"
        elif k.startswith("k_triad"): lab = "triad"
        elif k.startswith("k_conv2x2<false"): lab = "conv2x2_unfused"
        elif k.startswith("k_conv2x2<true"): lab = "conv2x2_fused"
        elif k.startswith("k_relu"): lab = "relu"
        elif k.startswith("k_pool"): lab = "pool"
        elif k.startswith("k_conv_igemm"): lab = "igemm_relu"
        elif "gemm" in k.lower() or "cutlass" in k.lower() or "xmma" in k.lower() or "sm80" in k or "sm90" in k:
            lab = "cublas_fp32" if not any(s["label"] == "cublas_fp32" for s in seq) else "cublas_fp16"
            if any(s["label"] == "cublas_fp16" for s in seq):
                lab = "cudnn_conv"
        elif "conv" in k.lower() or "implicit" in k.lower() or "cudnn" in k.lower():
            lab = "cudnn_conv"
        else:
            continue
        seq.append({**r, "label": lab})
    return seq


# ---------------------------------------------------------------------------------------------- figures
def fig_roofline(pk: dict) -> None:
    lad = [r for r in load("gemm_ladder.jsonl") if r.get("M") == 4096 and "gflops" in r]
    conv = [r for r in load("conv.jsonl") if r["layer"] == "3x3 @56 C64"]
    nc = {r["label"]: r for r in label_ncu(ncu_rows())}
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.logspace(-2, 4, 300)
    roofs = [("fp32 FMA", pk["fp32"], "C0"), ("bf16 tensor (cuBLAS)", pk.get("bf16", 0), "C3")]
    for lab, peak, col in roofs:
        if peak:
            ax.loglog(x, np.minimum(peak, pk["dram"] * x) / 1e12, color=col, lw=1.6,
                      label=f"{lab} {peak/1e12:.1f} TFLOP/s, DRAM {pk['dram']/1e12:.2f} TB/s")
    ax.loglog(x, np.minimum(pk["fp32"], pk["l2"] * x) / 1e12, color="C0", ls=":", lw=1,
              label=f"fp32 with L2-resident roof {pk['l2']/1e12:.2f} TB/s")
    for i, r in enumerate(lad):
        if r["variant"] == "cublas_fp8":
            continue
        ai = r["model_intensity"]
        mk = "o" if r["dtype"] == "fp32" else ("s" if r["dtype"] in ("fp16", "bf16") else "D")
        ax.scatter(ai, r["gflops"] / 1e3, marker=mk, s=36, zorder=3, color=plt.cm.tab20(i % 20))
        ax.annotate(r["variant"], (ai, r["gflops"] / 1e3), fontsize=6.5, xytext=(4, -3), textcoords="offset points")
        if r["variant"] in nc and "dram__bytes_read.sum" in nc[r["variant"]]:
            d = nc[r["variant"]]
            byt = d["dram__bytes_read.sum"] * _unit(d, "dram__bytes_read.sum") + \
                d["dram__bytes_write.sum"] * _unit(d, "dram__bytes_write.sum")
            ai2 = 2 * 4096**3 / byt
            ax.scatter(ai2, r["gflops"] / 1e3, marker=mk, s=36, facecolors="none", edgecolors="k", zorder=3)
            ax.plot([ai, ai2], [r["gflops"] / 1e3] * 2, color="grey", lw=0.5, ls="--")
    for r in conv:
        ai = r["flops"] / (r["x_bytes"] + r["w_bytes"] + r["y_bytes"] + r.get("cols_bytes", 0) * 2)
        ax.scatter(ai, r["gflops"] / 1e3, marker="^", color="k", s=30, zorder=3)
        ax.annotate("conv:" + r["method"], (ai, r["gflops"] / 1e3), fontsize=6, xytext=(4, 3), textcoords="offset points")
    ax.set(xlabel="arithmetic intensity (FLOP / DRAM byte, tile model without L2 reuse)", ylabel="TFLOP/s",
           title="RTX 5090 measured roofline: GEMM variants at 4096^3 (o fp32, s fp16/bf16), conv 3x3@56 (^)")
    ax.set_xlim(0.05, 5000); ax.set_ylim(0.05, 600)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "gpu_roofline")


def _unit(d: dict, m: str) -> float:
    u = str(d.get(m + "@unit", "byte")).lower()
    return {"byte": 1, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12}.get(u, 1)


def fig_ladder(pk: dict) -> None:
    lad = {r["variant"]: r for r in load("gemm_ladder.jsonl") if r.get("M") == 4096 and "gflops" in r}
    names = [n for n in LADDER_ORDER if n in lad]
    vals = [lad[n]["gflops"] / 1e3 for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = ["C0" if lad[n]["dtype"] == "fp32" and not n.startswith("cublas") else
            ("C2" if n.startswith("cublas") else "C3") for n in names]
    bars = ax.bar(range(len(names)), vals, color=cols)
    ax.set_yscale("log")
    for b, n, v in zip(bars, names, vals):
        ref = pk["fp32"] if lad[n]["dtype"] == "fp32" else pk.get(lad[n]["dtype"], pk.get("bf16", pk["fp32"]))
        if lad[n]["dtype"] == "fp8_e4m3":
            ref = 2 * pk.get("bf16", pk["fp32"])
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.1f}\n{100*v*1e12/ref:.0f}%", ha="center", fontsize=6.5)
    ax.axhline(pk["fp32"] / 1e12, color="C0", ls="--", lw=0.8, label=f"measured fp32 FMA peak {pk['fp32']/1e12:.1f}")
    if "bf16" in pk:
        ax.axhline(pk["bf16"] / 1e12, color="C3", ls="--", lw=0.8, label=f"cuBLAS bf16 best {pk['bf16']/1e12:.1f}")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="TFLOP/s (log)", title="GEMM ladder, M=N=K=4096. blue: my fp32, red: my WMMA, green: cuBLAS\n"
           "label: TFLOP/s and % of measured peak (fp32: FMA chain; fp16/bf16: best cuBLAS; fp8: 2x bf16)")
    ax.legend(fontsize=7)
    save(fig, "gpu_sgemm_ladder")


def fig_sweep() -> None:
    lad = [r for r in load("gemm_ladder.jsonl") if "gflops" in r]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, n in enumerate(LADDER_ORDER):
        rr = sorted([r for r in lad if r["variant"] == n], key=lambda r: r["M"])
        if not rr:
            continue
        a = ax[0] if (rr[0]["dtype"] == "fp32" or n == "cublas_tf32") else ax[1]
        a.loglog([r["M"] for r in rr], [r["gflops"] / 1e3 for r in rr], "o-", base=2, ms=3,
                 label=n, color=plt.cm.tab20(i % 20))
    ax[0].set(xlabel="M=N=K", ylabel="TFLOP/s", title="fp32 / tf32 kernels vs size")
    ax[1].set(xlabel="M=N=K", ylabel="TFLOP/s", title="tensor-core (fp16/bf16/fp8) vs size")
    for a in ax:
        a.legend(fontsize=6.5, ncol=2)
        a.grid(True, which="both", alpha=0.2)
    save(fig, "gpu_sgemm_sweep")


def fig_tiles() -> None:
    t = load("gemm_tiles.jsonl")
    if not t:
        return
    sizes = sorted({r["M"] for r in t})
    fig, ax = plt.subplots(1, len(sizes) + 1, figsize=(4.2 * (len(sizes) + 1), 3.9))
    bs = [32, 64, 128, 256]
    for a, s in zip(ax, sizes):
        g = np.full((4, 4), np.nan)
        for r in t:
            if r["M"] == s:
                g[bs.index(r["BM"]), bs.index(r["BN"])] = r["gflops"] / 1e3
        im = a.imshow(g, cmap="viridis", origin="lower")
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{g[i, j]:.1f}", ha="center", va="center", fontsize=8,
                       color="w" if g[i, j] < np.nanmax(g) * 0.6 else "k")
        a.set_xticks(range(4), bs); a.set_yticks(range(4), bs)
        a.set(xlabel="BN", ylabel="BM", title=f"reg2d TFLOP/s, {s}^3 (TM=TN=8, BK=8)")
        fig.colorbar(im, ax=a, fraction=0.046)
    g = np.full((4, 4), np.nan)
    for r in t:
        if r["M"] == sizes[-1]:
            g[bs.index(r["BM"]), bs.index(r["BN"])] = r["model_intensity"]
    im = ax[-1].imshow(g, cmap="magma", origin="lower")
    for i in range(4):
        for j in range(4):
            ax[-1].text(j, i, f"{g[i, j]:.0f}", ha="center", va="center", fontsize=8, color="w" if g[i, j] < 40 else "k")
    ax[-1].set_xticks(range(4), bs); ax[-1].set_yticks(range(4), bs)
    ax[-1].set(xlabel="BN", ylabel="BM", title="model FLOP/DRAM byte (no L2 reuse)")
    fig.colorbar(im, ax=ax[-1], fraction=0.046)
    save(fig, "gpu_tile_heatmap")


def fig_skinny(pk: dict) -> None:
    s = load("gemm_skinny.jsonl")
    if not s:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for i, n in enumerate(["coalesced", "smem32", "reg2d", "skinny_best", "cublas_fp32", "cublas_bf16"]):
        for l2, ls in (("warm", "--"), ("cold", "-")):
            rr = sorted([r for r in s if r["variant"] == n and r.get("l2", "warm") == l2], key=lambda r: r["M"])
            a = ax[0] if l2 == "warm" else ax[1]
            a.semilogx([r["M"] for r in rr], [r["gbps"] for r in rr], "o" + ls, base=2, ms=3, label=n, color=f"C{i}")
            if l2 == "cold":
                ax[2].loglog([r["M"] for r in rr], [r["gflops"] / 1e3 for r in rr], "o-", base=2, ms=3, label=n,
                             color=f"C{i}")
    for a in ax[:2]:
        a.axhline(pk["dram"] / 1e9, color="k", ls="--", lw=0.8, label="measured DRAM copy peak")
        a.axhline(pk["l2"] / 1e9, color="grey", ls=":", lw=0.8, label="L2-resident copy peak")
    ms = sorted({r["M"] for r in s})
    ax[2].loglog(ms, [min(pk["fp32"], pk["dram"] * 2 * m * 4096 * 4096 / (4 * (m * 4096 + 4096**2 + m * 4096))) / 1e12
                      for m in ms], "k--", lw=0.8, label="fp32 roofline (compulsory bytes, DRAM)")
    ax[0].set(xlabel="M (N=K=4096)", ylabel="compulsory bytes / time (GB/s)", title="Warm L2 (B=64 MiB stays in 96 MB L2)")
    ax[1].set(xlabel="M (N=K=4096)", ylabel="compulsory bytes / time (GB/s)", title="Cold L2 (256 MiB scrub before each call)")
    ax[2].set(xlabel="M", ylabel="TFLOP/s", title="Cold-L2 throughput vs DRAM roofline")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(True, which="both", alpha=0.2)
    save(fig, "gpu_skinny")


def fig_micro(pk: dict) -> None:
    bw = load("micro_bw.jsonl")
    fma = load("micro_fma.jsonl")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, st in (("copy", "o-"), ("triad", "s-"), ("torch_copy", "^-")):
        rr = {}
        for r in bw:
            if r["kernel"] == k:
                rr[r["mib_per_array"]] = max(rr.get(r["mib_per_array"], 0), r["gbps"])
        xs = sorted(rr)
        ax[0].semilogx(xs, [rr[x] for x in xs], st, base=2, label=k)
    ax[0].axhline(SPEC_BW / 1e9, color="k", ls="--", lw=0.8, label="datasheet 1792 GB/s")
    ax[0].axvline(96, color="r", ls=":", lw=0.8, label="L2 = 96 MB")
    ax[0].set(xlabel="MiB per array", ylabel="GB/s", title="Streaming bandwidth vs working set")
    ax[0].legend(fontsize=7)
    chains = sorted({r["chains"] for r in fma})
    warps = sorted({r["warps_per_sm"] for r in fma})
    g = np.full((len(chains), len(warps)), np.nan)
    for r in fma:
        i, j = chains.index(r["chains"]), warps.index(r["warps_per_sm"])
        g[i, j] = np.nanmax([g[i, j], r["tflops"]])
    im = ax[1].imshow(g, origin="lower", cmap="viridis", aspect="auto")
    for i in range(len(chains)):
        for j in range(len(warps)):
            if not np.isnan(g[i, j]):
                ax[1].text(j, i, f"{g[i, j]:.0f}", ha="center", va="center", fontsize=7,
                           color="w" if g[i, j] < np.nanmax(g) * 0.6 else "k")
    ax[1].set_xticks(range(len(warps)), warps); ax[1].set_yticks(range(len(chains)), chains)
    ax[1].set(xlabel="launched warps per SM (threads x blocks/SM / 32; at most 48 resident)",
              ylabel="independent FMA chains / thread",
              title=f"FMA-chain TFLOP/s (best {np.nanmax(g):.1f}; datasheet {SPEC_FP32/1e12:.1f})")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    save(fig, "gpu_micro")


def fig_conv() -> None:
    c = load("conv.jsonl")
    if not c:
        return
    layers = list(dict.fromkeys(r["layer"] for r in c))
    methods = ["direct", "im2col+cublas", "implicit_gemm", "cudnn_fp32", "cudnn_tf32", "cudnn_fp16_nhwc"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        v = [next((r["gflops"] / 1e3 for r in c if r["layer"] == L and r["method"] == m), np.nan) for L in layers]
        ax.bar(np.arange(len(layers)) + (j - len(methods) / 2 + 0.5) * w, v, w, label=m)
    ax.set_yscale("log")
    ax.set_xticks(range(len(layers)), layers, rotation=20, fontsize=7.5)
    ax.set(ylabel="TFLOP/s (log)", title="Conv2d on RTX 5090 (batch 8, fp32 unless noted)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis="y", which="both", alpha=0.2)
    save(fig, "gpu_conv")


def fig_im2col() -> None:
    c = [r for r in load("conv.jsonl") if r["method"] == "im2col+cublas"]
    if not c:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(len(c))
    ax[0].bar(xs - 0.2, [r["x_bytes"] / 2**20 for r in c], 0.4, label="input x")
    ax[0].bar(xs + 0.2, [r["cols_bytes"] / 2**20 for r in c], 0.4, label="im2col buffer")
    for i, r in enumerate(c):
        ax[0].text(i + 0.2, r["cols_bytes"] / 2**20, f"x{r['blowup']:.2f}", ha="center", va="bottom", fontsize=7)
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs, [r["layer"] for r in c], rotation=25, fontsize=7)
    ax[0].set(ylabel="MiB (fp32, batch 8, log)", title="im2col memory blow-up on GPU")
    ax[0].legend(fontsize=7)
    ax[1].bar(xs, [r["t_im2col"] * 1e3 for r in c], label="im2col kernel")
    ax[1].bar(xs, [r["t_gemm"] * 1e3 for r in c], bottom=[r["t_im2col"] * 1e3 for r in c], label="cuBLAS GEMM")
    ig = {r["layer"]: r for r in load("conv.jsonl") if r["method"] == "implicit_gemm"}
    cd = {r["layer"]: r for r in load("conv.jsonl") if r["method"] == "cudnn_fp32"}
    ax[1].plot(xs, [ig[r["layer"]]["s"] * 1e3 for r in c], "k^", label="my implicit GEMM")
    ax[1].plot(xs, [cd[r["layer"]]["s"] * 1e3 for r in c], "rv", label="cuDNN fp32")
    ax[1].set_xticks(xs, [r["layer"] for r in c], rotation=25, fontsize=7)
    ax[1].set(ylabel="ms", title="Explicit im2col: copy vs GEMM time")
    ax[1].legend(fontsize=7)
    save(fig, "gpu_im2col")


def fig_fusion() -> None:
    f = [r for r in load("conv_fusion.jsonl") if r["method"] != "e2e"]
    if not f:
        return
    layers = list(dict.fromkeys(r["layer"] for r in f))
    batches = sorted({r["N"] for r in f})
    methods = ["direct_unfused", "direct_fused", "cudnn_unfused", "cudnn+fused_relu_pool", "igemm_relu+pool"]
    fig, ax = plt.subplots(len(batches), 2, figsize=(13, 4.2 * len(batches)), squeeze=False)
    w = 0.16
    for bi, nb in enumerate(batches):
        for li, L in enumerate(layers):
            for j, m in enumerate(methods):
                r = next(x for x in f if x["layer"] == L and x["method"] == m and x["N"] == nb)
                xpos = li + (j - 2) * w
                ax[bi, 0].bar(xpos, r["io_bytes"] / 2**20, w, color="C7",
                              label="x+w read" if li == 0 and j == 0 else None)
                ax[bi, 0].bar(xpos, r["intermediate_bytes"] / 2**20, w, bottom=r["io_bytes"] / 2**20, color=f"C{j}",
                              label=m + " (conv out / relu / pool traffic)" if li == 0 else None)
                bottom = 0
                for k, (sn, sv) in enumerate(r["stages"].items()):
                    ax[bi, 1].bar(xpos, sv * 1e3, w, bottom=bottom, color=f"C{j}", alpha=1 - 0.3 * k,
                                  edgecolor="k", lw=0.3, label=m if (li == 0 and k == 0) else None)
                    bottom += sv * 1e3
        ax[bi, 0].set_xticks(range(len(layers)), layers, fontsize=8)
        ax[bi, 0].set(ylabel="model DRAM MiB", title=f"Conv+ReLU+maxpool DRAM traffic, batch {nb} (stacked)")
        ax[bi, 0].legend(fontsize=6)
        ax[bi, 1].set_xticks(range(len(layers)), layers, fontsize=8)
        ax[bi, 1].set(ylabel="ms (stacked per kernel)", title=f"Time per stage, batch {nb} (lighter = relu / pool)")
        ax[bi, 1].legend(fontsize=6)
    save(fig, "gpu_fusion")


def fig_fusion_saving(pk: dict) -> None:
    f = [r for r in load("conv_fusion.jsonl") if r["method"] != "e2e"]
    if not f:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    pairs = [("direct_unfused", "direct_fused", "o"), ("cudnn_unfused", "cudnn+fused_relu_pool", "s")]
    for un, fu, mk in pairs:
        for nb, col in ((8, "C0"), (64, "C3")):
            xs, ys = [], []
            for L in dict.fromkeys(r["layer"] for r in f):
                a = next(r for r in f if r["layer"] == L and r["method"] == un and r["N"] == nb)
                b = next(r for r in f if r["layer"] == L and r["method"] == fu and r["N"] == nb)
                xs.append((a["model_hbm_bytes"] - b["model_hbm_bytes"]) / 2**20)
                ys.append((a["s"] - b["s"]) * 1e6)
            ax.scatter(xs, ys, marker=mk, color=col, label=f"{un} -> {fu}, batch {nb}")
    x = np.linspace(0, ax.get_xlim()[1], 50)
    ax.plot(x, x * 2**20 / pk["dram"] * 1e6, "k--", lw=0.8, label="saved bytes / DRAM bandwidth")
    ax.plot(x, x * 2**20 / pk["l2"] * 1e6, "k:", lw=0.8, label="saved bytes / L2 bandwidth")
    ax.set(xlabel="model DRAM traffic saved by fusion (MiB)", ylabel="time saved (us)",
           title="Fusion saving tracks bytes / bandwidth once intermediates exceed L2")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.2)
    save(fig, "gpu_fusion_saving")


def fig_intensity_model() -> None:
    import sys
    sys.path.insert(0, str(LAB / "code"))
    from dnnaccel.tiling import Accel, tile_stats
    ts = [1, 8, 16, 32, 64, 128, 256, 512]
    n = 4096
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for b, lab in ((4, "fp32 (GPU kernels)"), (2, "bf16 (CPU-lab accelerator model)")):
        v = [tile_stats(n, n, n, t, t, 128, Accel(bytes_per_el=b), with_cycles=False).intensity for t in ts]
        ax.loglog(ts, v, "o-", base=2, label=lab)
    pk = peaks()
    ax.axhline(pk["fp32"] / pk["dram"], color="C0", ls=":", label=f"5090 fp32 ridge {pk['fp32']/pk['dram']:.0f}")
    if "bf16" in pk:
        ax.axhline(pk["bf16"] / pk["dram"], color="C3", ls=":", label=f"5090 bf16 ridge {pk['bf16']/pk['dram']:.0f}")
    ax.axhline(Accel().ridge, color="g", ls=":", label=f"Trn-like model ridge {Accel().ridge:.0f}")
    lad = {r["variant"]: r for r in load("gemm_ladder.jsonl") if r.get("M") == 4096 and "gflops" in r}
    for nme in ("smem8", "smem16", "smem32", "reg1d", "reg2d"):
        if nme in lad:
            ax.annotate(nme, (lad[nme]["tile_bm"], lad[nme]["model_intensity"]), fontsize=7, xytext=(3, -10),
                        textcoords="offset points")
    ax.set(xlabel="square block tile BM=BN", ylabel="FLOP / DRAM byte (4096^3)",
           title="tiling.py traffic formula: I ~ t/b for a square t x t tile (b = bytes/element)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "gpu_intensity_model")


def fig_kernel_info() -> None:
    """ncu is blocked in the container: occupancy from the CUDA occupancy API, conflicts from gpu/bankmodel.py."""
    import sys
    sys.path.insert(0, str(LAB / "gpu"))
    from bankmodel import rows as bank_rows
    info = load("kernel_info.jsonl")
    br = bank_rows()
    with open(RAW / "bank_model.jsonl", "w", encoding="utf-8") as fh:
        for r in br:
            fh.write(json.dumps(r) + "\n")
    lad = {r["variant"]: r for r in load("gemm_ladder.jsonl") if r.get("M") == 4096 and "gflops" in r}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    names = [r["kernel"] for r in info]
    xs = np.arange(len(names))
    ax[0].bar(xs - 0.2, [r["theoretical_occupancy"] * 100 for r in info], 0.4, label="theoretical occupancy %")
    ax[0].bar(xs + 0.2, [r["regs"] for r in info], 0.4, label="registers / thread")
    for i, r in enumerate(info):
        g = lad.get(r["kernel"])
        if g:
            ax[0].text(i, max(r["regs"], r["theoretical_occupancy"] * 100) + 3, f"{g['gflops']/1e3:.1f}T",
                       ha="center", fontsize=6.5)
    ax[0].set_xticks(xs, names, rotation=40, ha="right", fontsize=7.5)
    ax[0].set(title="Occupancy API: regs and warps/SM (label: TFLOP/s at 4096^3)")
    ax[0].legend(fontsize=7)
    kern = list(dict.fromkeys(r["kernel"] for r in br))
    worst = [max(r["conflict_degree"] for r in br if r["kernel"] == k) for k in kern]
    ax[1].bar(range(len(kern)), worst, color="C3")
    for i, k in enumerate(kern):
        acc = [r for r in br if r["kernel"] == k and r["conflict_degree"] > 1]
        ax[1].text(i, worst[i] + 0.15, "\n".join(a["access"].split(" ")[0] + " " + a["access"].split(" ")[1]
                                                for a in acc), ha="center", fontsize=6)
    ax[1].set_xticks(range(len(kern)), kern, fontsize=8)
    ax[1].set(ylabel="worst bank-conflict degree (wavefronts / ideal)", ylim=(0, 10),
              title="Analytic shared-memory bank conflicts (warp 0, inner loop)")
    save(fig, "gpu_kernel_info")


def fig_ncu() -> None:
    rows = label_ncu(ncu_rows())
    if not rows:
        return
    labs = [r["label"] for r in rows]
    mets = [("sm__warps_active.avg.pct_of_peak_sustained_active", "achieved occupancy %"),
            ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM throughput % of peak"),
            ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM throughput %")]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    w = 0.27
    for j, (m, lab) in enumerate(mets):
        ax[0].bar(np.arange(len(rows)) + (j - 1) * w, [r.get(m, np.nan) if isinstance(r.get(m), float) else np.nan
                                                       for r in rows], w, label=lab)
    ax[0].set_xticks(range(len(rows)), labs, rotation=45, ha="right", fontsize=7)
    ax[0].set(ylabel="%", title="ncu: occupancy / DRAM / SM utilisation")
    ax[0].legend(fontsize=7)
    ld = [r.get("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", np.nan) for r in rows]
    wf = [r.get("l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", np.nan) for r in rows]
    ratio = [(a / b * 100) if isinstance(a, float) and isinstance(b, float) and b > 0 else np.nan for a, b in zip(ld, wf)]
    ax[1].bar(range(len(rows)), ratio, color="C3")
    ax[1].set_xticks(range(len(rows)), labs, rotation=45, ha="right", fontsize=7)
    ax[1].set(ylabel="% of shared-load wavefronts that are bank-conflict replays",
              title="ncu: shared-memory load bank conflicts")
    save(fig, "gpu_ncu")


def fig_precision() -> None:
    lad = [r for r in load("gemm_ladder.jsonl") if r.get("M") == 4096 and r.get("max_rel_err") is not None]
    if not lad:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for r in lad:
        e = max(r["max_rel_err"], 1e-9)
        ax.scatter(r["gflops"] / 1e3, e, s=30, label=None)
        ax.annotate(f"{r['variant']} ({r['dtype']})", (r["gflops"] / 1e3, e), fontsize=6.5, xytext=(3, 2),
                    textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel="TFLOP/s at 4096^3", ylabel="max |C - C_fp64| / max |C_fp64|",
           title="Accuracy vs speed (reference: fp64 on the same rounded inputs)")
    ax.grid(True, which="both", alpha=0.2)
    save(fig, "gpu_precision")


def main() -> None:
    pk = peaks()
    print({k: f"{v/1e12:.3f}T" for k, v in pk.items()})
    fig_micro(pk)
    fig_roofline(pk)
    fig_ladder(pk)
    fig_sweep()
    fig_tiles()
    fig_skinny(pk)
    fig_conv()
    fig_im2col()
    fig_fusion()
    fig_fusion_saving(pk)
    fig_intensity_model()
    fig_kernel_info()
    fig_ncu()
    fig_precision()


if __name__ == "__main__":
    main()
