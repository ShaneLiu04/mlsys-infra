"""Generate every figure of reports/survey/survey.tex.

All analytic numbers come from tools/mlsys_calc.py; all lab numbers are read from
solutions/*/*/artifacts/raw/*.csv. Figures are vector PDFs with English labels
(Chinese explanations live in the LaTeX captions).

    python reports/survey/make_figures.py            # all figures, parallel
    python reports/survey/make_figures.py roofline   # a single figure
"""

from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIG = HERE / "figures"
RAW = REPO / "solutions"
PREVIEW = os.environ.get("SURVEY_PREVIEW") == "1"  # also write PNG previews to figures/_preview
sys.path.insert(0, str(REPO / "tools"))
import mlsys_calc as mc  # noqa: E402

COL = 3.45      # IEEE single column width (in)
DCOL = 7.16     # IEEE double column width (in)
C = ["#1f4e79", "#c0392b", "#2e8b57", "#d68910", "#7d3c98", "#5d6d7e", "#17a2b8", "#8b4513"]

H100, A100 = mc.GPUS["h100"], mc.GPUS["a100"]
L8, L70, L2_7 = mc.MODELS["llama3-8b"], mc.MODELS["llama3-70b"], mc.MODELS["llama2-7b"]


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    if PREVIEW:
        (FIG / "_preview").mkdir(exist_ok=True)
        fig.savefig(FIG / "_preview" / f"{name}.png", dpi=150)
    plt.close(fig)


def raw(rel: str) -> pd.DataFrame:
    return pd.read_csv(RAW / rel)


# --------------------------------------------------------------------------- analytic figures

def fig_roofline() -> None:
    fig, ax = plt.subplots(figsize=(COL, 2.45))
    x = np.logspace(-1.5, 4, 400)
    for g, c in ((H100, C[0]), (A100, C[1])):
        y = np.minimum(g.bf16_flops, x * g.hbm_bw) / 1e12
        ax.loglog(x, y, color=c, label=f"{g.name.split()[0]} ({g.bf16_flops/1e12:.0f} TF/s, "
                                       f"{g.hbm_bw/1e12:.2f} TB/s)")
        ax.axvline(g.ridge, color=c, ls=":", lw=0.8)
        ax.text(g.ridge * 1.08, 3 if g is H100 else 1.2, f"ridge {g.ridge:.0f}", color=c, fontsize=6,
                rotation=90)
    d = L8.d_model
    flash = raw("cs229s/project1-transformer-kernels/artifacts/raw/attention_io.csv")
    row = flash[(flash["T"] == 4096) & (flash["d"] == 128) & (~flash["causal"])].iloc[0]
    attn_flops = 4 * 4096**2 * 128
    pts = [
        ("elementwise add", 1 / 6),
        ("LayerNorm/softmax", 1.25),
        ("decode GEMV (B=1)", mc.gemm_intensity(1, d, d)),
        ("decode attn GQA-8", 4.0),
        ("decode GEMM B=64", mc.gemm_intensity(64, d, d)),
        ("naive attn (S,P 1x)", attn_flops / (row["naive_mb"] * 1e6)),
        ("FlashAttn fwd", attn_flops / (row["flash_mb"] * 1e6)),
        ("prefill GEMM 4096^3", mc.gemm_intensity(4096, 4096, 4096)),
    ]
    lines = []
    for i, (lab, inten) in enumerate(pts, start=1):
        perf = min(H100.bf16_flops, inten * H100.hbm_bw) / 1e12
        ax.plot(inten, perf, "o", ms=6.5, color="white", mec=C[0], mew=0.8, zorder=5)
        ax.text(inten, perf, str(i), fontsize=5, ha="center", va="center", zorder=6)
        lines.append(f"{i}  {lab}: I={inten:.3g}, {perf:.3g} TF/s")
    ax.text(0.04, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=5.0, va="top",
            family="DejaVu Sans Mono")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("attainable TFLOP/s (bf16 dense)")
    ax.set_ylim(0.1, 4e5)
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    save(fig, "fig_roofline")


def fig_amdahl() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(COL, 1.85))
    n = np.unique(np.logspace(0, 4, 60).astype(int))
    for p, c in zip((0.9, 0.95, 0.99, 0.999), C):
        axs[0].loglog(n, [mc.amdahl_speedup(p, k) for k in n], color=c, label=f"p={p}")
        axs[1].loglog(n, [mc.gustafson_speedup(p, k) for k in n], color=c, label=f"p={p}")
    axs[0].loglog(n, n, "k:", lw=0.8)
    axs[1].loglog(n, n, "k:", lw=0.8)
    axs[0].set_title("Amdahl (fixed size)")
    axs[1].set_title("Gustafson (scaled size)")
    for a in axs:
        a.set_xlabel("workers n")
    axs[0].set_ylabel("speedup")
    axs[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_amdahl")


def fig_intensity_batch() -> None:
    fig, ax = plt.subplots(figsize=(COL, 2.25))
    b = np.unique(np.logspace(0, 10, 80, base=2).astype(int))
    for seq, c in zip((0, 1024, 4096, 32768), C):
        inten = [mc.decode_step_time(L8, H100, seq, int(k))["intensity"] for k in b]
        ax.loglog(b, inten, color=c, label=f"decode, ctx={seq}")
    pre = [mc.gemm_intensity(512 * int(k), L8.d_model, L8.d_model) for k in b]
    ax.loglog(b, pre, color=C[4], ls="--", label="prefill 512 tok/seq (linear layers)")
    for g, c in ((H100, "k"), (A100, "gray")):
        ax.axhline(g.ridge, color=c, ls=":", lw=0.9)
        ax.text(1.1, g.ridge * 1.1, f"{g.name.split()[0]} ridge {g.ridge:.0f}", fontsize=6, color=c)
    ax.set_xlabel("batch size B (sequences)")
    ax.set_ylabel("FLOP / HBM byte")
    ax.set_title("Llama-3-8B, bf16: prefill vs decode intensity")
    ax.legend(frameon=False, fontsize=5.8, loc="center right")
    save(fig, "fig_intensity_batch")


def fig_decode_throughput() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.25))
    b = np.unique(np.logspace(0, 11, 90, base=2).astype(int))
    node = mc.GPU("8xH100 (ideal TP=8)", 8 * H100.bf16_flops, 8 * H100.hbm_bw, 8 * H100.hbm_bytes, H100.link_bw)
    for model, gpu, ax in ((L8, H100, axs[0]), (L70, node, axs[1])):
        for seq, c in zip((0, 1024, 4096, 16384), C):
            tps, fit = [], []
            for k in b:
                r = mc.decode_step_time(model, gpu, seq, int(k))
                tps.append(r["tokens_per_s"])
                fit.append((r["weight_gb"] + r["kv_gb"]) * 1e9 <= gpu.hbm_bytes)
            tps, fit = np.array(tps), np.array(fit)
            ax.loglog(b[fit], tps[fit], color=c, label=f"ctx={seq}")
            ax.loglog(b[~fit], tps[~fit], color=c, ls=":", lw=0.9)
        ridge = mc.critical_batch(model, gpu)
        ax.axvline(ridge, color="k", ls="--", lw=0.8)
        ax.text(ridge * 1.08, tps[0] * 1.3, f"B* = ridge = {ridge:.0f}", fontsize=6)
        ax.set_title(f"{model.name} on {gpu.name}")
        ax.set_xlabel("batch size B")
    axs[0].set_ylabel("roofline upper bound (tokens/s)")
    axs[0].legend(frameon=False, loc="upper left")
    fig.text(0.5, -0.02, "solid: weights+KV fit in HBM; dotted: would not fit", ha="center", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_decode_throughput")


def act_bytes(layers: int, hidden: int, seq: int = 4096, mb: int = 1, full_recompute: bool = False) -> float:
    """Korthikanti et al. 2023: 34*s*b*h per layer w/o attention scores (FlashAttention);
    full recomputation keeps only the 2*s*b*h layer input."""
    per = (2 if full_recompute else 34) * seq * mb * hidden
    return per * layers


def fig_train_memory() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.3))
    cases = [("7B (L=32, h=4096), DP=8", 7e9, 8, 32, 4096, axs[0]),
             ("70B (L=80, h=8192), DP=64", L70.params(), 64, 80, 8192, axs[1])]
    for title, n, dp, layers, h, ax in cases:
        xs, labels = [], []
        for z in range(4):
            mem = mc.training_memory_bytes(n, dp, z)
            for j, rc in enumerate((False, True)):
                x = z * 2.6 + j
                xs.append(x)
                labels.append(f"Z{z}\n{'recomp' if rc else 'no rc'}")
                bottom = 0.0
                for key, c in (("weights", C[0]), ("grads", C[3]), ("optimizer", C[1])):
                    v = mem[key] / 1e9
                    ax.bar(x, v, bottom=bottom, color=c, width=0.85,
                           label=key if (z == 0 and j == 0) else None)
                    bottom += v
                a = act_bytes(layers, h, full_recompute=rc) / 1e9
                ax.bar(x, a, bottom=bottom, color=C[2], width=0.85, hatch="//" if rc else None,
                       label=("activations (1x4096 seq)" if (z == 0 and j == 0) else None))
                ax.text(x, (bottom + a) * 1.02, f"{bottom + a:.0f}", ha="center", va="bottom", fontsize=5.6)
        ax.axhline(80, color="k", ls="--", lw=0.8)
        ax.text(xs[-1] + 0.5, 82, "80 GB", fontsize=6, ha="right", va="bottom")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=5.4)
        ax.set_ylabel("GB per GPU (linear)")
        ax.set_title(title)
    axs[0].legend(frameon=False, loc="upper right", fontsize=6)
    axs[0].set_ylim(0, 150)
    fig.tight_layout()
    save(fig, "fig_train_memory")


def fig_kv_cache() -> None:
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    T = 2 ** np.arange(10, 18)
    mha70 = mc.Model("70B-MHA", 80, 8192, 64, 64, 28672, 128256)
    series = [(L8, "Llama-3-8B GQA-8 (128 KiB/tok)", C[0], "-"),
              (L2_7, "8B-class MHA (Llama-2-7B, 512 KiB/tok)", C[0], "--"),
              (L70, "Llama-3-70B GQA-8 (320 KiB/tok)", C[1], "-"),
              (mha70, "70B MHA counterfactual (2.5 MiB/tok)", C[1], "--")]
    for m, lab, c, ls in series:
        ax.loglog(T, [mc.kv_cache_bytes(m, int(t)) / 1e9 for t in T], color=c, ls=ls, marker="o",
                  ms=2.5, label=lab)
    ax.axhline(80, color="k", ls=":", lw=0.9)
    ax.text(1100, 90, "one H100 (80 GB)", fontsize=6)
    ax.axhline(L8.params() * 2 / 1e9, color=C[0], ls=":", lw=0.7)
    ax.text(1100, L8.params() * 2 / 1e9 * 1.12, "8B bf16 weights 16 GB", fontsize=6, color=C[0])
    ax.set_xticks(T)
    ax.set_xticklabels([f"{t // 1024}K" for t in T])
    ax.set_xlabel("context length T (batch = 1, bf16)")
    ax.set_ylabel("KV cache (GB)")
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    save(fig, "fig_kv_cache")


def fig_allreduce() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.2))
    df = raw("cmu15849/distributed-training/artifacts/raw/allreduce_sweep.csv")
    df = df[df["P"] == 8]
    for (alg, c) in zip(("naive", "ring", "halving_doubling", "tree"), C):
        s = df[df["alg"] == alg].sort_values("bytes")
        axs[0].loglog(s["bytes"], s["time_full_s"] * 1e6, color=c, label=alg)
        axs[0].loglog(s["bytes"], s["time_ring_topo_s"] * 1e6, color=c, ls=":", lw=0.9)
    axs[0].set_title("lab: simulated all-reduce, P=8 (solid switch, dotted ring topo)")
    axs[0].set_xlabel("message size S (bytes)")
    axs[0].set_ylabel("time (us)")
    axs[0].legend(frameon=False)
    S = np.logspace(3, 10.5, 200)
    alpha, bw = 5e-6, 450e9
    for p, c in zip((8, 64, 512), C):
        ring = [mc.ring_allreduce_time(s, p, bw, alpha) for s in S]
        lg = math.ceil(math.log2(p))
        tree = [2 * lg * (alpha + s / bw) for s in S]
        axs[1].loglog(S, np.array(ring) * 1e3, color=c, label=f"ring P={p}")
        axs[1].loglog(S, np.array(tree) * 1e3, color=c, ls="--", label=f"tree P={p}")
    s7 = 14e9
    t7 = mc.ring_allreduce_time(s7, 8, bw)
    axs[1].plot(s7, t7 * 1e3, "k*", ms=7)
    axs[1].annotate(f"7B bf16 grads, 8xH100:\n{t7 * 1e3:.1f} ms", (s7, t7 * 1e3), xytext=(3e8, 0.04),
                    fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.6))
    axs[1].set_title("alpha-beta model: alpha=5 us, beta^-1=450 GB/s")
    axs[1].set_xlabel("message size S (bytes)")
    axs[1].set_ylabel("time (ms)")
    axs[1].legend(frameon=False, ncol=2, fontsize=5.6)
    fig.tight_layout()
    save(fig, "fig_allreduce")


def fig_pipeline() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.1))
    m = np.arange(1, 65)
    for v, c in zip((1, 2, 4), C):
        axs[0].plot(m, [mc.pipeline_bubble(8, int(k), v) for k in m], color=c,
                    label=("GPipe = 1F1B" if v == 1 else f"interleaved v={v}") + " (p=8)")
    axs[0].plot(m, [mc.pipeline_bubble(4, int(k)) for k in m], color=C[5], ls="--", label="GPipe/1F1B p=4")
    df = raw("cmu15849/distributed-training/artifacts/raw/pipeline_bubble.csv")
    for sch, mk in (("gpipe", "o"), ("1f1b", "x"), ("interleaved", "s")):
        s = df[(df["schedule"] == sch) & (df["p"] == 8)]
        axs[0].scatter(s["m"], s["bubble_sim"], marker=mk, s=14, color="k", zorder=5,
                       label=f"lab sim: {sch}")
    b = mc.pipeline_bubble(8, 32)
    axs[0].annotate(f"p=8, m=32: {b:.1%}", (32, b), xytext=(24, 0.03), fontsize=6.3,
                    arrowprops=dict(arrowstyle="->", lw=0.6))
    axs[0].set_xlabel("micro-batches m")
    axs[0].set_ylabel("bubble fraction")
    axs[0].set_ylim(0, 1)
    axs[0].legend(frameon=False, fontsize=5.6)
    for sch, c in zip(("gpipe", "1f1b", "interleaved"), C):
        s = df[(df["schedule"] == sch) & (df["p"] == 8)].sort_values("m")
        axs[1].plot(s["m"], s["peak_act_stage0"], marker="o", ms=3, color=c, label=sch)
    axs[1].set_xlabel("micro-batches m (p=8)")
    axs[1].set_ylabel("in-flight activations on stage 0\n(units of one micro-batch)")
    axs[1].set_title("lab: event simulation, t_b = 2 t_f")
    axs[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_pipeline")


def fig_speculative() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(DCOL, 2.1))
    g = np.arange(0, 13)
    for a, c in zip((0.5, 0.6, 0.7, 0.8, 0.9, 0.95), C):
        axs[0].plot(g, [mc.speculative_expected_tokens(a, int(k)) for k in g], color=c, label=f"alpha={a}")
    e = mc.speculative_expected_tokens(0.8, 4)
    axs[0].plot(4, e, "k*", ms=7)
    axs[0].annotate(f"alpha=0.8, gamma=4: {e:.2f}", (4, e), xytext=(5, 1.3), fontsize=6,
                    arrowprops=dict(arrowstyle="->", lw=0.6))
    axs[0].set_xlabel("draft length gamma")
    axs[0].set_ylabel("E[tokens / target forward]")
    axs[0].legend(frameon=False, fontsize=5.6)
    g1 = np.arange(1, 17)
    for cr, c in zip((0.02, 0.05, 0.1, 0.25), C):
        sp = [mc.speculative_speedup(0.8, int(k), cr) for k in g1]
        axs[1].plot(g1, sp, color=c, label=f"c={cr}")
        k = int(np.argmax(sp))
        axs[1].plot(g1[k], sp[k], "o", color=c, ms=3)
    axs[1].axhline(1, color="k", lw=0.6)
    axs[1].set_xlabel("draft length gamma (alpha=0.8)")
    axs[1].set_ylabel("ideal speedup E/(gamma c+1)")
    axs[1].legend(frameon=False, fontsize=5.6)
    df = raw("cs229s/project2-kv-speculative/artifacts/raw/speculative.csv")
    for mode, mk, c in (("greedy", "o", C[0]), ("sample", "s", C[1])):
        s = df[df["mode"] == mode]
        axs[2].plot(s["gamma"], s["tokens_per_target_call"], mk + "-", color=c, ms=3.5,
                    label=f"{mode}: measured tok/call")
        axs[2].plot(s["gamma"], s["formula_tokens_per_call"], mk, color=c, mfc="none", ms=6,
                    label=f"{mode}: formula")
        axs[2].plot(s["gamma"], s["measured_speedup"], mk + "--", color=c, ms=3, alpha=0.7,
                    label=f"{mode}: CPU wall speedup")
    axs[2].set_xlabel("gamma")
    axs[2].set_title("lab (CPU, batch 1)")
    axs[2].legend(frameon=False, fontsize=5.0)
    fig.tight_layout()
    save(fig, "fig_speculative")


def fig_chinchilla() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.2))
    Cs = np.logspace(20, 26, 100)
    n_opt = np.sqrt(Cs / 120)
    axs[0].loglog(Cs, n_opt, color=C[0], label="Chinchilla rule N* = sqrt(C/120)")
    axs[0].loglog(Cs, 20 * n_opt / 1e3, color=C[1], ls="--", label="D* = 20 N* (shown /1000)")
    models = [("GPT-3 175B, 0.3T", 175e9, 300e9), ("Chinchilla 70B, 1.4T", 70e9, 1.4e12),
              ("Llama-2-7B, 2T", 6.74e9, 2e12), ("Llama-3-8B, 15T", L8.params(), 15e12),
              ("Llama-3-70B, 15T", L70.params(), 15e12), ("Llama-3.1-405B, 15.6T", 405e9, 15.6e12)]
    marks = "osD^vP"
    for i, (lab, n, d) in enumerate(models):
        cc = mc.train_flops(n, d)
        axs[0].plot(cc, n, marks[i], color=C[2 + i % 6], ms=4.5, ls="none",
                    label=f"{lab} ({d / n:.0f} tok/param)")
    axs[0].set_xlabel("training compute C = 6ND (FLOP)")
    axs[0].set_ylabel("parameters N")
    axs[0].legend(frameon=False, loc="upper left", fontsize=5.0)
    axs[0].set_ylim(1e7, 1e14)
    gpus = np.logspace(2, 4.5, 50)
    for (lab, n), c in zip((("8B", L8.params()), ("70B", L70.params())), C):
        for mfu, ls in ((0.4, "-"), (0.55, "--")):
            days = mc.train_flops(n, 15e12) / (gpus * H100.bf16_flops * mfu) / 86400
            axs[1].loglog(gpus, days, color=c, ls=ls, label=f"{lab}, 15T tok, MFU {mfu:.0%}")
    axs[1].axhline(30, color="k", ls=":", lw=0.7)
    axs[1].text(120, 33, "30 days", fontsize=6)
    axs[1].set_xlabel("number of H100 GPUs")
    axs[1].set_ylabel("wall-clock days")
    axs[1].legend(frameon=False, fontsize=5.8)
    fig.tight_layout()
    save(fig, "fig_chinchilla")


def fig_capability_map() -> None:
    fig, ax = plt.subplots(figsize=(DCOL, 3.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    color = {"C": "#d6eaf8", "M": "#fdebd0", "N": "#d5f5e3"}
    edge = {"C": C[0], "M": C[3], "N": C[2]}
    layers = [
        ("Hardware &\nparallel prog.\n(CS149)", 52,
         [("M01", "perf models\nroofline", "C"), ("M02", "multicore\nSIMD/SMT", "C"),
          ("M03", "scheduling\nwork-span", "C"), ("M04", "coherence\nsync", "N"),
          ("M05", "GPU/CUDA", "C"), ("M06", "scan/reduce\nSpark", "N")]),
        ("Kernels &\naccelerators", 41,
         [("M07", "GEMM tiling\nMFU", "C"), ("M08", "attention\nFlashAttn", "M"),
          ("M09", "systolic\nTPU/Trainium", "C")]),
        ("Compilers &\nframeworks", 30,
         [("M10", "ML compilers\nfusion/TVM", "C"), ("M11", "autodiff\nruntime", "M")]),
        ("Training\nsystems\n(15-849)", 19,
         [("M12", "DP/all-reduce\nZeRO", "N"), ("M13", "TP/PP\nauto-parallel", "N"),
          ("M14", "recompute\nmixed prec.", "M")]),
        ("Inference &\nefficiency\n(CS229S)", 8,
         [("M15", "KV cache\nspeculative", "M"), ("M16", "LLM serving\nvLLM/SGLang", "M"),
          ("M17", "quant/prune\nLoRA", "M"), ("M18", "MoE, SSM\nclusters", "N")]),
    ]
    centers = {}
    for name, y, mods in layers:
        ax.text(0, y + 3.1, name, fontsize=6.0, va="center", fontweight="bold")
        w = 9.2
        for i, (mid, lab, q) in enumerate(mods):
            x = 19.5 + i * 10.4
            ax.add_patch(FancyBboxPatch((x, y), w, 6.2, boxstyle="round,pad=0.25,rounding_size=0.8",
                                        fc=color[q], ec=edge[q], lw=0.9))
            ax.text(x + w / 2, y + 4.6, mid, ha="center", fontsize=6.6, fontweight="bold")
            ax.text(x + w / 2, y + 2.0, lab, ha="center", va="center", fontsize=5.3)
            centers[mid] = (x + w / 2, y)
    deps = [("M01", "M07"), ("M05", "M07"), ("M05", "M08"), ("M07", "M08"), ("M07", "M09"),
            ("M07", "M10"), ("M09", "M10"), ("M10", "M11"), ("M11", "M12"), ("M06", "M12"),
            ("M12", "M13"), ("M13", "M14"), ("M08", "M15"), ("M15", "M16"), ("M14", "M17"),
            ("M13", "M18"), ("M03", "M06"), ("M04", "M12")]
    for a, b in deps:
        (xa, ya), (xb, yb) = centers[a], centers[b]
        ax.annotate("", xy=(xb, yb + 6.4), xytext=(xa, ya - 0.2),
                    arrowprops=dict(arrowstyle="->", lw=0.5, color="#555555", alpha=0.7,
                                    connectionstyle="arc3,rad=0.08"))
    roles = [("Training infra", "M07 M11-M14 M18", 50), ("Inference / serving", "M05 M07 M08 M15-M17", 38),
             ("Kernel / compiler / chip", "M02 M04-M10", 26), ("General ML eng.", "M01 M08 M12 M15 M16", 14)]
    ax.text(91, 59, "job roles", ha="center", fontsize=7, fontweight="bold")
    for r, mods, y in roles:
        ax.add_patch(FancyBboxPatch((84.5, y), 15, 7, boxstyle="round,pad=0.3", fc="#f2f3f4", ec="#566573", lw=0.8))
        ax.text(92, y + 4.6, r, ha="center", fontsize=5.8, fontweight="bold")
        ax.text(92, y + 1.8, mods, ha="center", fontsize=5.0)
    for q, lab, x in (("C", "compute: fast enough?", 17),
                      ("M", "memory: does it fit?", 40), ("N", "communication: can we move it?", 62)):
        ax.add_patch(FancyBboxPatch((x, 0.2), 2.5, 2.2, boxstyle="round,pad=0.1", fc=color[q], ec=edge[q]))
        ax.text(x + 3.3, 1.3, lab, fontsize=6, va="center")
    save(fig, "fig_capability_map")


# --------------------------------------------------------------------------- lab re-plots

def fig_lab_flash() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.1))
    df = raw("cs229s/project1-transformer-kernels/artifacts/raw/attention_io.csv")
    for d, c in ((64, C[0]), (128, C[1])):
        s = df[(df["d"] == d) & (~df["causal"])]
        sc = df[(df["d"] == d) & (df["causal"])]
        axs[0].loglog(s["T"], s["naive_mb"], color=c, ls="--", label=f"naive d={d}")
        axs[0].loglog(s["T"], s["flash_mb"], color=c, marker="o", ms=2.5, label=f"flash d={d}")
        axs[0].loglog(sc["T"], sc["flash_mb"], color=c, ls=":", marker="x", ms=3, label=f"flash causal d={d}")
    t = np.array([512, 65536])
    axs[0].loglog(t, 2.3593 * (t / 512) ** 2, color="gray", lw=0.6)
    axs[0].text(3000, 200, "slope 2 (T^2)", fontsize=6, color="gray", rotation=28)
    axs[0].set_xlabel("sequence length T")
    axs[0].set_ylabel("HBM traffic per head (MB, bf16)")
    axs[0].set_title("lab IO model, SRAM 100 KiB")
    axs[0].legend(frameon=False, fontsize=5.6)
    sv = raw("cs229s/project1-transformer-kernels/artifacts/raw/attention_io_vs_sram.csv")
    axs[1].semilogx(sv["sram_kib"], sv["ratio"], "o-", color=C[0], ms=3, label="naive / flash, T=8192, d=64")
    ax2 = axs[1].twinx()
    ax2.semilogx(sv["sram_kib"], sv["Br"], "s--", color=C[3], ms=3, label="Br (rows of Q block)")
    ax2.semilogx(sv["sram_kib"], sv["Bc"], "^:", color=C[2], ms=3, label="Bc (rows of K/V block)")
    ax2.set_ylabel("block size")
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    axs[1].set_xticks(sv["sram_kib"])
    axs[1].set_xticklabels([str(v) for v in sv["sram_kib"]])
    axs[1].minorticks_off()
    axs[1].set_xlabel("SRAM per SM (KiB)")
    axs[1].set_ylabel("traffic ratio")
    axs[1].set_ylim(0.8, 2.3)
    h1, l1 = axs[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.6, loc="center right")
    axs[1].set_title("Br = min(ceil(M/4d), d) caps the gain")
    fig.tight_layout()
    save(fig, "fig_lab_flash")


def fig_lab_serving() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(DCOL, 2.1))
    w = raw("cmu15849/serving-memory/artifacts/raw/kv_waste.csv")
    labels = ["max4096", "exact len"] + [
        ("ideal" if s == "ideal" or b == 1 else f"paged {b}") for s, b in zip(w["scheme"][2:], w["block"][2:])]
    axs[0].bar(range(len(w)), w["waste_frac"] * 100, color=C[1])
    axs[0].set_xticks(range(len(w)))
    axs[0].set_xticklabels(labels, rotation=40, fontsize=5.5, ha="right")
    axs[0].set_ylabel("lifetime KV waste (%)", color=C[1])
    ax2 = axs[0].twinx()
    ax2.plot(range(len(w)), w["concurrent_seqs"], "o-", color=C[0], ms=3)
    ax2.set_ylabel("concurrent sequences", color=C[0])
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)
    axs[0].set_title("KV allocation (Llama-3-8B, 55.9 GB)")
    sw = raw("cmu15849/serving-memory/artifacts/raw/scheduler_sweep.csv")
    for pol, c in zip(("static", "continuous", "chunked"), C):
        s = sw[sw["policy"] == pol]
        axs[1].plot(s["rate_req_s"], s["goodput_req_s"], "o-", ms=2.5, color=c, label=pol)
        axs[2].semilogy(s["rate_req_s"], s["ttft_p99_s"], "o-", ms=2.5, color=c, label=pol)
    axs[1].plot([0, 160], [0, 160], "k:", lw=0.6)
    axs[1].set_xlabel("arrival rate (req/s)")
    axs[1].set_ylabel("goodput (req/s)")
    axs[1].set_title("SLO: TTFT<=1 s, TPOT<=50 ms")
    axs[1].legend(frameon=False)
    axs[2].axhline(1.0, color="k", ls="--", lw=0.7)
    axs[2].set_xlabel("arrival rate (req/s)")
    axs[2].set_ylabel("P99 TTFT (s)")
    axs[2].set_title("roofline-timed simulation")
    fig.tight_layout()
    save(fig, "fig_lab_serving")


def fig_lab_compress() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.0))
    q = raw("cmu15849/compiler-compression/artifacts/raw/quant_outliers.csv")
    cols = [("W8A8_actTensor_wChannel", "per-tensor act"), ("W8A8_actToken_wChannel", "per-token act"),
            ("SmoothQuant0.5_actTensor_wChannel", "SmoothQuant a=0.5"), ("LLM.int8_threshold6", "LLM.int8 (thr 6)"),
            ("normal_channel_part_err_actTensor", "per-tensor: normal-channel part")]
    for (col, lab), c in zip(cols, C):
        axs[0].plot(q["outlier_scale"], q[col] * 100, marker="o", ms=2.5, color=c, label=lab,
                    ls="--" if "normal" in col else "-")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("outlier magnitude (8 of 1024 channels)")
    axs[0].set_ylabel("relative output error (%)")
    axs[0].legend(frameon=False, fontsize=5.6)
    f = raw("cmu15849/compiler-compression/artifacts/raw/fusion_totals.csv")
    x = np.arange(len(f))
    axs[1].bar(x - 0.2, f["hbm_vs_unfused"], 0.4, color=C[0], label="HBM traffic (rel.)")
    axs[1].bar(x + 0.2, f["time_vs_unfused"], 0.4, color=C[3], label="roofline time (rel.)")
    for i, r in f.iterrows():
        axs[1].text(i - 0.2, r["hbm_vs_unfused"] + 0.02, f"{r['hbm_GB']:.3f} GB", ha="center", fontsize=5.5)
        axs[1].text(i + 0.2, r["time_vs_unfused"] + 0.02, f"{r['time_us_roofline']:.0f} us", ha="center", fontsize=5.5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{v}\n({k} kernels)" for v, k in zip(f["variant"], f["kernels"])], fontsize=6)
    axs[1].set_ylim(0, 1.2)
    axs[1].legend(frameon=False, loc="upper right")
    axs[1].set_title("MLP block fusion, T=4096, d=4096, H100 model")
    fig.tight_layout()
    save(fig, "fig_lab_compress")


def fig_lab_kv_systolic() -> None:
    fig, axs = plt.subplots(1, 3, figsize=(DCOL, 2.0))
    k = raw("cs229s/project2-kv-speculative/artifacts/raw/kv_timing.csv")
    axs[0].loglog(k["total_len"], k["nocache_tok_s"], "o-", ms=3, color=C[1], label="no cache")
    axs[0].loglog(k["total_len"], k["kv_tok_s"], "s-", ms=3, color=C[0], label="KV cache")
    for _, r in k.iterrows():
        axs[0].text(r["total_len"], r["kv_tok_s"] * 1.15, f"{r['speedup']:.1f}x", fontsize=5.5, ha="center")
    axs[0].set_xlabel("total length T")
    axs[0].set_ylabel("tokens/s (CPU, batch 1)")
    axs[0].legend(frameon=False)
    axs[0].set_title("lab: KV cache")
    s = raw("cs149/asst4-dnn-accelerator/artifacts/raw/systolic_vs_size.csv")
    axs[1].semilogx(s["n"], s["util_serial"], "o-", ms=3, color=C[1], label="serial weight load")
    axs[1].semilogx(s["n"], s["util_pipelined"], "s-", ms=3, color=C[0], label="double-buffered")
    axs[1].set_xlabel("matrix size n (P=16 array)")
    axs[1].set_ylabel("PE utilization")
    axs[1].legend(frameon=False)
    axs[1].set_title("lab: weight-stationary systolic")
    ck = raw("cmu15849/distributed-training/artifacts/raw/ckpt_frontier.csv")
    for chain, c in (("uniform64", C[0]), ("hetero32", C[1])):
        dp = ck[(ck["chain"] == chain) & (ck["policy"] == "dp_optimal")]
        axs[2].plot(dp["peak"], dp["recompute"], "-", color=c, label=f"{chain}: DP frontier")
        sq = ck[(ck["chain"] == chain) & (ck["policy"] == "sqrt_n")]
        axs[2].plot(sq["peak"], sq["recompute"], "*", color=c, ms=8, label=f"{chain}: sqrt(n)")
    hs = ck[ck["policy"] == "dp_at_sqrt_budget"]
    axs[2].plot(hs["peak"], hs["recompute"], "D", color=C[1], mfc="none", ms=5,
                label="hetero32: DP at sqrt(n) budget")
    axs[2].set_xlabel("peak stored activations")
    axs[2].set_ylabel("recomputed cost")
    axs[2].legend(frameon=False, fontsize=5.6)
    axs[2].set_title("lab: checkpointing frontier")
    fig.tight_layout()
    save(fig, "fig_lab_kv_systolic")


FIGS = {f.__name__[4:]: f for f in (
    fig_roofline, fig_amdahl, fig_intensity_batch, fig_decode_throughput, fig_train_memory,
    fig_kv_cache, fig_allreduce, fig_pipeline, fig_speculative, fig_chinchilla, fig_capability_map,
    fig_lab_flash, fig_lab_serving, fig_lab_compress, fig_lab_kv_systolic)}


def run(name: str) -> str:
    style()
    FIGS[name]()
    return name


def main(argv: list[str]) -> int:
    names = argv or list(FIGS)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for n in ex.map(run, names):
            print("wrote", FIG / f"fig_{n}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
