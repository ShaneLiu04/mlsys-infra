"""Figures for the GPU part, from artifacts/raw/gpu/*.jsonl (runs locally, no GPU needed).

python scripts/plot_gpu.py
Writes artifacts/figures/gpu_*.png (README) and .pdf; the LaTeX report reads this directory directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(LAB.parents[2] / "tools"))

COL = {"triton": "C3", "sdpa_flash": "C0", "sdpa_efficient": "C2", "sdpa_math": "C4", "naive": "C1",
       "regblock": "C3", "smem": "C2", "cudnn": "C0", "fft": "C4"}
MK = {"triton": "o", "sdpa_flash": "s", "sdpa_efficient": "^", "sdpa_math": "v", "naive": "x",
      "regblock": "o", "smem": "^", "cudnn": "s", "fft": "D"}


def load(name: str) -> pd.DataFrame:
    rows = [json.loads(x) for x in (RAW / name).read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.DataFrame([r for r in rows if not str(r.get("part", "")).startswith("meta")])


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=140)
    plt.close(fig)
    print("saved", name)


def peaks() -> dict:
    return json.loads((RAW / "peaks.json").read_text())


# ----------------------------------------------------------------------------------------- attention

def fig_attn_fwd(df: pd.DataFrame) -> None:
    d16 = df[df.dtype == "fp16"]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    for j, (d, causal) in enumerate([(64, False), (64, True), (128, False), (128, True)]):
        sub = d16[(d16.D == d) & (d16.causal == causal)]
        for m, g in sub.groupby("method"):
            ok = g[g.status == "ok"].sort_values("T")
            ax[0, j].loglog(ok["T"], ok["ms"], marker=MK[m], color=COL[m], base=2, label=m)
            ax[1, j].semilogx(ok["T"], ok["tflops"], marker=MK[m], color=COL[m], base=2, label=m)
            oom = g[g.status == "oom"]
            if len(oom):
                t_oom = oom["T"].min()
                ax[0, j].axvline(t_oom, color=COL[m], ls=":", lw=1)
                ax[0, j].annotate(f"{m} OOM", (t_oom, ok["ms"].max() if len(ok) else 1), color=COL[m],
                                  fontsize=7, rotation=90, ha="right", va="top")
        ax[0, j].set(title=f"fp16 fwd, d={d}, {'causal' if causal else 'non-causal'}, B*H=32",
                     xlabel="sequence length T", ylabel="time (ms)")
        ax[1, j].set(xlabel="sequence length T", ylabel="TFLOP/s (useful FLOPs)")
        ax[0, j].grid(alpha=.3, which="both")
        ax[1, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    save(fig, "gpu_attn_fwd_time_tflops")


def fig_attn_memory(df: pd.DataFrame) -> None:
    sub = df[(df.dtype == "fp16") & (df.D == 64) & (~df.causal)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for m, g in sub.groupby("method"):
        ok = g[g.status == "ok"].sort_values("T")
        ax[0].loglog(ok["T"], ok["peak_extra_bytes"] / 2**20, marker=MK[m], color=COL[m], base=2, label=m)
        oom = g[g.status == "oom"]
        if len(oom):
            ax[0].scatter([oom["T"].min()], [32 * 1024], marker="X", s=90, color=COL[m], zorder=5)
    ts = np.array(sorted(sub["T"].unique()), dtype=float)
    ax[0].loglog(ts, 32 * ts ** 2 * 2 / 2**20, "k--", lw=1, base=2, label="one fp16 T×T score tensor (B*H=32)")
    ax[0].axhline(32 * 1024, color="gray", ls=":", lw=1)
    ax[0].text(ts[0], 32 * 1024 * 1.1, "32 GB VRAM (X = OOM)", fontsize=8)
    ax[0].set(xlabel="T", ylabel="peak extra memory (MiB)", title="Peak memory above inputs, fp16 fwd, d=64")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3, which="both")
    # ratio peak / (one score tensor)
    for m, g in sub[sub.status == "ok"].groupby("method"):
        g = g.sort_values("T")
        ax[1].semilogx(g["T"], g["peak_extra_bytes"] / (32 * g["T"] ** 2 * 2), marker=MK[m], color=COL[m],
                       base=2, label=m)
    ax[1].set(xlabel="T", ylabel="peak extra / (B·H·T²·2 bytes)", title="Memory in units of one fp16 score matrix",
              yscale="log")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")
    save(fig, "gpu_attn_peak_memory")


def fig_attn_error(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for i, dt in enumerate(("fp16", "bf16")):
        sub = df[(df.dtype == dt) & (df.D == 128) & (df.causal) & (df.status == "ok")]
        for m, g in sub.groupby("method"):
            g = g.sort_values("T")
            ax[i].loglog(g["T"], g["max_err"], marker=MK[m], color=COL[m], base=2, label=m)
        ax[i].set(xlabel="T", ylabel="max |O - O_fp32| (head 0)", title=f"{dt} forward error vs fp32 reference, d=128, causal")
        ax[i].grid(alpha=.3, which="both")
        ax[i].legend(fontsize=7)
    save(fig, "gpu_attn_error")


def fig_attn_bwd(dfb: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for j, (d, causal) in enumerate([(64, False), (64, True), (128, False), (128, True)]):
        sub = dfb[(dfb.D == d) & (dfb.causal == causal)]
        for m, g in sub.groupby("method"):
            ok = g[g.status == "ok"].sort_values("T")
            ax[j].semilogx(ok["T"], ok["tflops"], marker=MK[m], color=COL[m], base=2, label=m)
            oom = g[g.status == "oom"]
            if len(oom):
                ax[j].axvline(oom["T"].min(), color=COL[m], ls=":", lw=1)
        ax[j].set(title=f"fwd+bwd fp16, d={d}, {'causal' if causal else 'non-causal'}", xlabel="T",
                  ylabel="TFLOP/s (3.5 x fwd FLOPs)")
        ax[j].grid(alpha=.3)
    ax[0].legend(fontsize=8)
    save(fig, "gpu_attn_fwd_bwd")


def fig_attn_bh(dfh: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, causal in enumerate((False, True)):
        sub = dfh[(dfh.causal == causal) & (dfh.status == "ok")].copy()
        sub["BH"] = sub.B * sub.H
        for m, g in sub.groupby("method"):
            g = g.sort_values("BH")
            ax[i].semilogx(g["BH"], g["tflops"], marker=MK[m], color=COL[m], base=2, label=m)
        for bh in (1, 2, 4, 8):
            ctas = bh * 4096 // 128
            ax[i].annotate(f"{ctas} CTAs", (bh, 2), fontsize=7, rotation=90)
        ax[i].axvline(170 * 128 / 4096, color="gray", ls=":", lw=1)
        ax[i].text(170 * 128 / 4096 * 1.05, 5, "B·H·T/128 = 170 SMs", fontsize=7)
        ax[i].set(xlabel="batch × heads", ylabel="TFLOP/s", title=f"T=4096, d=64, fp16, {'causal' if causal else 'non-causal'}")
        ax[i].grid(alpha=.3)
        ax[i].legend(fontsize=8)
    save(fig, "gpu_attn_batch_heads")


def fig_attn_blocks(dfk: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    bms, bns = [16, 32, 64, 128, 256], [16, 32, 64, 128]
    for i, d in enumerate((64, 128)):
        sub = dfk[(dfk["T"] == 16384) & (dfk.D == d)]
        grid = np.full((len(bms), len(bns)), np.nan)
        for _, r in sub.iterrows():
            if r.status == "ok":
                grid[bms.index(r.BLOCK_M), bns.index(r.BLOCK_N)] = r.tflops
        im = ax[i].imshow(grid, cmap="viridis", origin="lower")
        for a in range(len(bms)):
            for b in range(len(bns)):
                ax[i].text(b, a, "fail" if np.isnan(grid[a, b]) else f"{grid[a, b]:.0f}", ha="center", va="center",
                           color="w" if np.isnan(grid[a, b]) or grid[a, b] < np.nanmax(grid) * .6 else "k", fontsize=8)
        ax[i].set_xticks(range(len(bns)), bns)
        ax[i].set_yticks(range(len(bms)), bms)
        ax[i].set(xlabel="BLOCK_N (K/V tile)", ylabel="BLOCK_M (Q tile)", title=f"Triton fwd TFLOP/s, T=16384, d={d}, fp16")
        fig.colorbar(im, ax=ax[i], fraction=.046)
    ok = dfk[dfk.status == "ok"]
    for (t, d), g in ok.groupby(["T", "D"]):
        ax[2].scatter(g["model_bytes"] / 1e9, g["ms"], label=f"T={t}, d={d}", s=18)
    ax[2].set(xscale="log", yscale="log", xlabel="IO-model HBM bytes (GB): Q+O once, K/V once per Q block",
              ylabel="measured time (ms)", title="Time vs modelled traffic over all block shapes")
    bw = peaks()["peak_copy_bw"]
    xs = np.logspace(np.log10(ok["model_bytes"].min() / 1e9), np.log10(ok["model_bytes"].max() / 1e9), 20)
    ax[2].plot(xs, xs * 1e9 / bw * 1e3, "k--", lw=1, label="bytes / copy bandwidth")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3, which="both")
    save(fig, "gpu_attn_block_sweep")


def eager_naive_bytes(t: int, d: int, bh: int = 32, causal: bool = False) -> float:
    """Eager PyTorch naive attention, fp16: QK^T writes S; *scale reads+writes; [masked_fill reads+writes +
    bool mask]; softmax reads+writes; PV reads P  => 6 (8 causal) T^2 fp16 accesses + 4Td."""
    t2 = (8 if causal else 6) * t * t * 2 + (2 * t * t if causal else 0) / bh
    return bh * (t2 + 4 * t * d * 2)


def fig_attn_io(df: pd.DataFrame) -> None:
    """Measured time vs IO-model bytes: naive (4Td+4T^2 elements) vs flash (one K/V pass per Q block)."""
    from tfkernels.io_model import naive_elements
    bw = peaks()["peak_copy_bw"]
    sub = df[(df.dtype == "fp16") & (~df.causal) & (df.status == "ok") & df.method.isin(["naive", "triton", "sdpa_flash"])]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bh, bm = 32, 128
    for d, ls in ((64, "-"), (128, "--")):
        for m, lbl in (("naive", "naive, textbook 4Td+4T²"), ("eager", "naive, eager 4Td+6T²"), ("triton", "flash (Br=128)")):
            g = sub[(sub.D == d) & (sub.method == ("naive" if m == "eager" else m))].sort_values("T")
            if m == "naive":
                model = np.array([naive_elements(int(t), d) * 2 * bh for t in g["T"]], float)
            elif m == "eager":
                model = np.array([eager_naive_bytes(int(t), d, bh) for t in g["T"]], float)
            else:
                model = np.array([bh * 2 * (2 * t * d + -(-t // bm) * 2 * t * d) for t in g["T"]], float)
            color = {"naive": "C1", "eager": "C5", "triton": "C3"}[m]
            ax[0].loglog(g["T"], model / 1e9, ls=ls, marker=MK.get(m, "+"), color=color, base=2, label=f"{lbl}, d={d}")
            ax[1].semilogx(g["T"], model / (g["ms"].to_numpy() * 1e-3) / bw, ls=ls, marker=MK.get(m, "+"),
                           color=color, base=2, label=f"{lbl}, d={d}")
    ax[0].set(xlabel="T", ylabel="modelled HBM bytes (GB), B·H=32", title="IO models (fp16, non-causal)")
    ax[1].axhline(1, color="k", lw=1)
    ax[1].set(xlabel="T", ylabel="(model bytes / time) / copy bandwidth",
              title="Implied bandwidth: >1 means the model overcounts DRAM traffic (L2 reuse)", yscale="log")
    for a in ax:
        a.legend(fontsize=7)
        a.grid(alpha=.3, which="both")
    save(fig, "gpu_attn_io_model")


# ----------------------------------------------------------------------------------------- roofline

def fig_roofline(dfr: pd.DataFrame) -> None:
    pk = peaks()
    peak, bw = pk["peak_bf16_flops"], pk["peak_copy_bw"]
    lay = dfr[dfr.part == "layer"]
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.logspace(-1, 4, 200)
    ax.loglog(xs, np.minimum(peak, xs * bw) / 1e12, "k-", lw=2, label=f"measured roof: {peak / 1e12:.0f} TFLOP/s, {bw / 1e12:.2f} TB/s")
    ax.loglog(xs, np.minimum(pk["spec_bf16_flops"], xs * pk["spec_bw"]) / 1e12, "k:", lw=1,
              label=f"spec roof: {pk['spec_bf16_flops'] / 1e12:.1f} TFLOP/s, {pk['spec_bw'] / 1e12:.3f} TB/s")
    mm = dfr[(dfr.part == "matmul") & (dfr.dtype == "bf16")]
    ax.scatter(mm["intensity"], mm["tflops"], marker=".", color="gray", label="square bf16 matmuls 256..16384")
    ops = {"qkv_proj": "o", "attention": "s", "out_proj": "^", "mlp": "D"}
    for (phase, op), g in lay.groupby(["phase", "op"]):
        ax.scatter(g["intensity"], g["tflops"], marker=ops[op], s=50, facecolors="none" if phase == "decode" else None,
                   edgecolors="C0" if phase == "prefill" else "C3", color="C0" if phase == "prefill" else "C3",
                   label=f"{phase} {op}")
        if op in ("qkv_proj", "attention"):
            for _, r in g.iterrows():
                lbl = f"B{int(r.B)}" if phase == "decode" else f"T{int(r.T_new)}" + (f"×B{int(r.B)}" if r.B > 1 else "")
                ax.annotate(lbl, (r.intensity, r.tflops), fontsize=6, xytext=(4, -6), textcoords="offset points")
    ax.axvline(peak / bw, color="gray", ls="--", lw=1)
    ax.text(peak / bw * 1.05, 0.2, f"ridge {peak / bw:.0f} FLOP/B", fontsize=8)
    ax.set(xlabel="arithmetic intensity (FLOP/byte, analytic)", ylabel="attained TFLOP/s",
           title="RTX 5090 roofline: one Llama-2-7B layer (bf16), prefill (filled) vs decode ctx=4096 (hollow)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=.3, which="both")
    save(fig, "gpu_roofline_layer")


def fig_peaks(dfr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    cp = dfr[dfr.part == "copy"]
    ax[0].semilogx(cp["buffer_bytes"] / 2**20, cp["gbps"], "o-", base=2)
    ax[0].axvline(96, color="gray", ls=":")
    ax[0].text(100, cp["gbps"].min(), "96 MB L2", fontsize=8)
    ax[0].axhline(1792, color="k", ls="--", lw=1)
    ax[0].text(cp["buffer_bytes"].min() / 2**20, 1830, "spec 1792 GB/s", fontsize=8)
    ax[0].set(xlabel="buffer size (MiB), copy y.copy_(x)", ylabel="GB/s (read + write)", title="Copy bandwidth vs buffer size")
    ax[0].grid(alpha=.3)
    mm = dfr[dfr.part == "matmul"]
    for dt, g in mm[mm.dtype.isin(["bf16", "fp16"])].groupby("dtype"):
        ax[1].semilogx(g["M"], g["tflops"], "o-", base=2, label=dt)
    for dt in ("fp32", "tf32"):
        r = mm[mm.dtype == dt]
        if len(r):
            ax[1].scatter(r["M"], r["tflops"], marker="*", s=80, label=f"{dt} 8192³")
    ax[1].axhline(209.5, color="k", ls="--", lw=1)
    ax[1].text(256, 214, "spec dense bf16 209.5", fontsize=8)
    ax[1].set(xlabel="square matmul size n", ylabel="TFLOP/s", title="Matmul throughput (torch.matmul / cuBLAS)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)
    save(fig, "gpu_peak_bw_flops")


def fig_layer_breakdown(dfr: pd.DataFrame) -> None:
    lay = dfr[dfr.part == "layer"].copy()
    lay["shape"] = lay.apply(lambda r: f"prefill\nB{int(r.B)} T{int(r.T_new)}" if r.phase == "prefill"
                             else f"decode\nB{int(r.B)}", axis=1)
    order = list(dict.fromkeys(lay["shape"]))
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    piv = lay.pivot_table(index="shape", columns="op", values="ms").loc[order][["qkv_proj", "attention", "out_proj", "mlp"]]
    frac = piv.div(piv.sum(axis=1), axis=0)
    bottom = np.zeros(len(order))
    for i, op in enumerate(frac.columns):
        ax[0].bar(range(len(order)), frac[op], bottom=bottom, label=op, color=f"C{i}")
        bottom += frac[op].to_numpy()
    ax[0].set_xticks(range(len(order)), order, fontsize=7)
    ax[0].set(ylabel="fraction of layer time", title="Where one layer's time goes")
    ax[0].legend(fontsize=8)
    eff = lay.pivot_table(index="shape", columns="op", values="roofline_efficiency").loc[order][frac.columns]
    for i, op in enumerate(eff.columns):
        ax[1].plot(range(len(order)), eff[op], "o-", label=op, color=f"C{i}")
    ax[1].set_xticks(range(len(order)), order, fontsize=7)
    ax[1].axhline(1, color="k", lw=.8, ls=":")
    ax[1].set(ylabel="roofline time / measured time", title="Fraction of the (measured) roofline achieved (>1: L2-resident weights)",
              ylim=(0, 1.6))
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)
    save(fig, "gpu_layer_breakdown")


# ----------------------------------------------------------------------------------------- dwconv

def fig_dwconv(dfc: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    for i, (sw, x, title) in enumerate([("K", "K", "N=16384, sweep kernel length K"),
                                        ("N_K4", "N", "short conv K=4 (Mamba/H3), sweep N"),
                                        ("K_eq_N", "N", "long conv K=N (Hyena), sweep N")]):
        sub = dfc[(dfc.sweep == sw) & (dfc.status == "ok")]
        for m, g in sub.groupby("method"):
            g = g.sort_values(x)
            ax[i].loglog(g[x], g["ms"], marker=MK[m], color=COL[m], base=2, label=m)
        ax[i].set(xlabel=x, ylabel="time (ms), B=4, C=768, fp32", title=title)
        ax[i].grid(alpha=.3, which="both")
        ax[i].legend(fontsize=8)
    save(fig, "gpu_dwconv_time")


def fig_dwconv_throughput(dfc: pd.DataFrame) -> None:
    sub = dfc[(dfc.sweep == "K") & (dfc.status == "ok")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for m, g in sub.groupby("method"):
        g = g.sort_values("K")
        ax[0].loglog(g["K"], g["gflops_direct_equiv"] / 1e3, marker=MK[m], color=COL[m], base=2, label=m)
        ax[1].semilogx(g["K"], g["gbps"], marker=MK[m], color=COL[m], base=2, label=m)
    ax[0].axhline(104.8, color="k", ls="--", lw=1)
    ax[0].text(3, 110, "fp32 CUDA-core peak ≈104.8 TFLOP/s (spec)", fontsize=7)
    ax[0].set(xlabel="K", ylabel="direct-equivalent TFLOP/s (2·B·C·N·K / t)", title="Compute throughput, N=16384")
    pk = peaks()
    ax[1].axhline(pk["peak_copy_bw"] / 1e9, color="k", ls="--", lw=1)
    ax[1].text(3, pk["peak_copy_bw"] / 1e9 * 1.03, "measured copy bandwidth", fontsize=7)
    ax[1].set(xlabel="K", ylabel="GB/s of compulsory traffic (read x, write y)", title="Memory throughput, N=16384")
    for a in ax:
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
    save(fig, "gpu_dwconv_throughput")


# ----------------------------------------------------------------------------------------- training

def fig_mqar(dfm: pd.DataFrame) -> None:
    cur = dfm[dfm.part == "mqar_curve"]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for i, (run, g) in enumerate(cur.groupby("run", sort=False)):
        ax[0].plot(g["step"], g["loss"], label=run, color=f"C{i}")
        ax[1].plot(g["step"], g["acc"], label=run, color=f"C{i}")
        ax[2].plot(g["elapsed_s"], g["acc"], label=run, color=f"C{i}")
    ax[0].set(xlabel="step", ylabel="train loss (answer tokens)", title="MQAR learning curves (batch 256)", yscale="log")
    ax[1].set(xlabel="step", ylabel="train accuracy on answers", title="Accuracy vs step")
    ax[2].set(xlabel="wall-clock seconds", ylabel="train accuracy", title="Accuracy vs wall clock")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=7)
    save(fig, "gpu_mqar_curves")


def fig_mqar_sweep(dfs: pd.DataFrame) -> None:
    """Phase transition of MQAR vs task difficulty / lr (sweep) plus the first, failed setting."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    cur = dfs[dfs.part == "sweep_curve"]
    for i, (run, g) in enumerate(cur.groupby("run", sort=False)):
        ax[0].plot(g["step"], g["acc"], label=f"{run} (6k)", color=f"C{i}")
        ax[1].plot(g["step"], g["loss"], label=f"{run} (6k)", color=f"C{i}")
    s2k = RAW / "train_mqar_sweep_2k.jsonl"
    if s2k.exists():
        old = load(s2k.name)
        old = old[old.part == "sweep_curve"]
        for i, (run, g) in enumerate(old.groupby("run", sort=False)):
            ax[0].plot(g["step"], g["acc"], ":", lw=1, label=f"{run} (2k)", color=f"C{i + 5}")
            ax[1].plot(g["step"], g["loss"], ":", lw=1, label=f"{run} (2k)", color=f"C{i + 5}")
    v1 = RAW / "train_mqar_v1_k256_P32_lr1e-3.jsonl"
    if v1.exists():
        old = load(v1.name)
        old = old[(old.part == "mqar_curve") & (old.run == "L2_bf16_sdpa")]
        ax[0].plot(old["step"], old["acc"], "k--", label="k256_P32_lr0.001 + wd0.1 (first try)")
        ax[1].plot(old["step"], old["loss"], "k--", label="k256_P32_lr0.001 + wd0.1 (first try)")
    ax[0].set(xlabel="step", ylabel="train accuracy on answers", title="MQAR sweep, 2-layer GPT d=128 (bf16, SDPA)")
    ax[1].set(xlabel="step", ylabel="train loss", title="Loss plateau then drop = induction head forms", yscale="log")
    for a in ax:
        a.grid(alpha=.3)
        a.legend(fontsize=6, ncol=2)
    save(fig, "gpu_mqar_sweep")


def fig_throughput(dft: pd.DataFrame) -> None:
    main = dft[(dft.sweep == "main")]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    labels = [f"{r.precision}\n{r.impl}" for r in main.itertuples()]
    ok = main.status == "ok"
    ax[0].bar(range(len(main)), np.where(ok, main["tokens_per_s"].fillna(0) / 1e3, 0),
              color=["C0" if p == "bf16" else ("C2" if p == "tf32" else "C1") for p in main.precision])
    for i, r in enumerate(main.itertuples()):
        if r.status == "ok":
            ax[0].text(i, r.tokens_per_s / 1e3, f"{r.mfu_6N * 100:.0f}%", ha="center", va="bottom", fontsize=8)
        else:
            ax[0].text(i, 1, r.status, ha="center", fontsize=8)
    ax[0].set_xticks(range(len(main)), labels, fontsize=8)
    ax[0].set(ylabel="k tokens/s (label: MFU, 6N)", title="GPT-2-small shape (12L, d=768, T=1024, B=8) training")
    ax[1].bar(range(len(main)), np.where(ok, main["peak_mem_gb"].fillna(0), 0), color="C7")
    ax[1].set_xticks(range(len(main)), labels, fontsize=8)
    ax[1].set(ylabel="peak allocated GB", title="Training peak memory")
    sl = dft[dft.sweep == "seqlen"]
    for m, g in sl.groupby("impl"):
        okg = g[g.status == "ok"].sort_values("seq_len")
        ax[2].semilogx(okg["seq_len"], okg["tokens_per_s"] / 1e3, marker=MK.get(m, "s"), color=COL.get(m, "C0"),
                       base=2, label=m)
        bad = g[g.status != "ok"]
        for r in bad.itertuples():
            ax[2].annotate(f"{m} {r.status}", (r.seq_len, 5), color=COL.get(m, "k"), fontsize=7, rotation=90)
    ax[2].set(xlabel="sequence length T (tokens/step fixed at 16384)", ylabel="k tokens/s", title="bf16 training throughput vs T")
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8)
    save(fig, "gpu_train_throughput")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9})
    jobs = {"attention_fwd.jsonl": (fig_attn_fwd, fig_attn_memory, fig_attn_error, fig_attn_io),
            "attention_bwd.jsonl": (fig_attn_bwd,), "attention_bh.jsonl": (fig_attn_bh,),
            "attention_blocks.jsonl": (fig_attn_blocks,),
            "roofline.jsonl": (fig_peaks, fig_roofline, fig_layer_breakdown),
            "dwconv.jsonl": (fig_dwconv, fig_dwconv_throughput),
            "train_mqar.jsonl": (fig_mqar,), "train_mqar_sweep.jsonl": (fig_mqar_sweep,),
            "train_throughput.jsonl": (fig_throughput,)}
    for name, fns in jobs.items():
        if not (RAW / name).exists():
            print("missing", name)
            continue
        df = load(name)
        for fn in fns:
            fn(df)


if __name__ == "__main__":
    main()
