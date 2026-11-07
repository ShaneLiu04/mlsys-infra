"""Plot every GPU figure from artifacts/raw/gpu/*.jsonl (runs locally, no GPU needed).

Also writes artifacts/raw/gpu/summary_gpu.json with the derived numbers quoted in the README/report
(speedups, Spearman correlations, lab cost-model predictions for the RTX 5090).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[2]))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parents[2] / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import mlsys_calc  # noqa: E402
from compkit.graph_ir import fuse_elementwise, kernel_cost, transformer_mlp  # noqa: E402
from compkit.schedule_search import Machine, Schedule, cost  # noqa: E402
from mlsys_harness.devices import get  # noqa: E402

RAW = ROOT / "artifacts" / "raw" / "gpu"
FIG = ROOT / "artifacts" / "figures"
_D5090 = get("rtx5090")
HBM = _D5090.spec_hbm_bw
SAXPY = _D5090.reference_copy_bw
PEAK_BF16 = _D5090.spec_bf16_flops
SMS = _D5090.sms
GPU5090 = mlsys_calc.GPU("RTX 5090 32GB", PEAK_BF16, HBM, 32e9, 64e9)
SUMMARY: dict = {}
MODE_C = {"eager": "tab:gray", "default": "tab:blue", "max-autotune": "tab:orange",
          "max-autotune-no-cudagraphs": "tab:green", "inductor": "tab:blue", "triton": "tab:red"}


def load(name: str) -> list[dict]:
    p = RAW / f"{name}.jsonl"
    if not p.exists():
        print("missing", p)
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- Q1 fusion

def fig_fusion():
    rows = [r for r in load("fusion_chains") if r.get("part") == "chains"]
    if not rows:
        return
    chains = list(dict.fromkeys(r["chain"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    by = {(r["chain"], r["M"], r["mode"]): r for r in rows}
    sizes = sorted({r["M"] for r in rows})

    fig, axes = plt.subplots(1, len(chains), figsize=(4 * len(chains), 3.6), sharey=True)
    for ax, c in zip(axes, chains):
        for mode in modes:
            ys = [by[(c, m, "eager")]["time_s"] / by[(c, m, mode)]["time_s"] for m in sizes if (c, m, mode) in by]
            ax.plot(sizes[:len(ys)], ys, "o-", label=mode, color=MODE_C.get(mode))
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_title(c)
        ax.set_xlabel("rows M (N=4096, bf16)")
    axes[0].set_ylabel("speedup over eager")
    axes[0].legend(fontsize=7)
    save(fig, "gpu_fusion_speedup")

    fig, axes = plt.subplots(1, len(chains), figsize=(4 * len(chains), 3.6), sharey=True)
    for ax, c in zip(axes, chains):
        for mode in modes:
            pts = [(m, by[(c, m, mode)]["achieved_bw"] / 1e12) for m in sizes if (c, m, mode) in by]
            ax.plot(*zip(*pts), "o-", label=mode, color=MODE_C.get(mode))
        ax.axhline(HBM / 1e12, ls="--", color="k", lw=0.8, label="GDDR7 peak 1.79")
        ax.axhline(SAXPY / 1e12, ls=":", color="k", lw=0.8, label="saxpy 1.55")
        l2_rows = [m for m in sizes if by[(c, m, "eager")]["ideal_bytes"] < 96 * 2 ** 20]
        if l2_rows:
            ax.axvspan(sizes[0] / 1.3, max(l2_rows) * 1.3, color="yellow", alpha=0.15, label="ideal bytes < L2")
        ax.set_xscale("log", base=2)
        ax.set_title(c)
        ax.set_xlabel("rows M")
    axes[0].set_ylabel("ideal bytes / time (TB/s)")
    axes[0].legend(fontsize=6)
    save(fig, "gpu_fusion_bandwidth")

    big = max(sizes)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(chains))
    wdt = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        a1.bar(x + i * wdt, [by[(c, big, mode)]["kernels"] for c in chains], wdt, label=mode, color=MODE_C.get(mode))
    a1.set_xticks(x + 0.4 - wdt / 2, chains, rotation=20, fontsize=8)
    a1.set_ylabel(f"CUDA kernels per call (M={big})")
    a1.legend(fontsize=7)
    eb = [by[(c, big, "eager")]["eager_bytes"] / 1e6 for c in chains]
    ib = [by[(c, big, "eager")]["ideal_bytes"] / 1e6 for c in chains]
    a2.bar(x - 0.2, eb, 0.4, label="eager (sum over aten ops)", color="tab:gray")
    a2.bar(x + 0.2, ib, 0.4, label="fused ideal (inputs + output once)", color="tab:blue")
    for i in range(len(chains)):
        a2.text(x[i], eb[i], f"-{100 * (1 - ib[i] / eb[i]):.0f}%", ha="center", va="bottom", fontsize=8)
    a2.set_xticks(x, chains, rotation=20, fontsize=8)
    a2.set_ylabel("HBM bytes per call (MB)")
    a2.legend(fontsize=7)
    save(fig, "gpu_fusion_kernels_bytes")
    SUMMARY["fusion_big"] = {c: {m: {"time_us": by[(c, big, m)]["time_s"] * 1e6, "kernels": by[(c, big, m)]["kernels"],
                                     "bw_TBs": by[(c, big, m)]["achieved_bw"] / 1e12} for m in modes}
                             for c in chains}
    SUMMARY["fusion_bytes_saved_frac"] = {c: 1 - ib[i] / eb[i] for i, c in enumerate(chains)}


def lab_mlp_model(tokens: int, d: int = 1024) -> dict:
    g = transformer_mlp(tokens, d, 4 * d)
    out = {}
    for name, gg in [("unfused", g), ("elementwise", fuse_elementwise(g)), ("epilogue", fuse_elementwise(g, epilogue=True))]:
        rows = kernel_cost(gg, 2, GPU5090)
        out[name] = {"kernels": len(rows), "time_us": sum(r["time_us"] for r in rows),
                     "bytes": sum(r["bytes"] for r in rows)}
    return out


def fig_mlp():
    rows = [r for r in load("fusion_mlp") if r.get("part") == "mlp"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, kind in zip(axes, ["mlp", "block"]):
        rr = [r for r in rows if r["chain"] == kind]
        toks = sorted({r["M"] for r in rr})
        modes = list(dict.fromkeys(r["mode"] for r in rr))
        x = np.arange(len(toks))
        wdt = 0.8 / (len(modes) + (1 if kind == "mlp" else 0))
        for i, mode in enumerate(modes):
            vals = [next((r["tflops"] for r in rr if r["M"] == t and r["mode"] == mode), np.nan) for t in toks]
            ax.bar(x + i * wdt, vals, wdt, label=mode, color=MODE_C.get(mode))
        if kind == "mlp":
            pred = []
            for t in toks:
                lm = lab_mlp_model(t)
                fl = next(r["flops"] for r in rr if r["M"] == t)
                pred.append(fl / (lm["epilogue"]["time_us"] * 1e-6) / 1e12)
            ax.bar(x + len(modes) * wdt, pred, wdt, label="lab roofline model (epilogue-fused)", color="white",
                   edgecolor="k", hatch="//")
        ax.axhline(PEAK_BF16 / 1e12, ls="--", color="k", lw=0.8, label="bf16 dense peak 209.5")
        ax.set_xticks(x + 0.4 - wdt / 2, [str(t) for t in toks])
        ax.set_xlabel("tokens")
        ax.set_ylabel("achieved TFLOP/s")
        ax.set_title("Transformer MLP (d=1024, x + W2 GELU(W1 LN(x)))" if kind == "mlp" else "pre-LN block (d=1024, 16 heads, seq 1024)")
        ax.legend(fontsize=7)
    save(fig, "gpu_mlp_block")
    mm = {}
    for r in rows:
        mm.setdefault(r["chain"], {}).setdefault(str(r["M"]), {})[r["mode"]] = {
            "time_us": r["time_s"] * 1e6, "kernels": r["kernels"], "tflops": r["tflops"], "compile_s": r["compile_s"],
            "max_err": r["max_err"], "eager_bytes": r["eager_bytes"], "ideal_bytes": r["ideal_bytes"]}
    SUMMARY["mlp_block"] = mm
    SUMMARY["lab_mlp_model_5090"] = {str(t): lab_mlp_model(t) for t in sorted({r["M"] for r in rows if r["chain"] == "mlp"})}


def fig_triton():
    rows = [r for r in load("fusion_triton") if r.get("part") == "triton"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, name in zip(axes, ["rmsnorm", "bias_gelu"]):
        rr = [r for r in rows if r["chain"] == name]
        keys = sorted({(r["M"], r["N"]) for r in rr})
        x = np.arange(len(keys))
        for i, impl in enumerate(["eager", "inductor", "triton"]):
            vals = [next(r["achieved_bw"] for r in rr if (r["M"], r["N"]) == k and r["impl"] == impl) / 1e12 for k in keys]
            ax.bar(x + i * 0.27, vals, 0.27, label=impl, color=MODE_C[impl])
        ax.axhline(HBM / 1e12, ls="--", color="k", lw=0.8)
        ax.set_xticks(x + 0.27, [f"{m}x{n}" for m, n in keys], rotation=45, fontsize=7)
        ax.set_title(f"{name}: handwritten Triton vs eager vs inductor")
        ax.set_ylabel("ideal bytes / time (TB/s)")
        ax.legend(fontsize=7)
    save(fig, "gpu_triton_fused")
    SUMMARY["triton_handwritten"] = [{k: r[k] for k in ("chain", "M", "N", "impl", "time_s", "achieved_bw", "max_err", "kernels")} for r in rows]


def fig_compile_dynamic():
    ch = [r for r in load("fusion_chains") if r.get("part") == "chains" and r["mode"] != "eager"]
    ml = [r for r in load("fusion_mlp") if r.get("part") == "mlp" and r["mode"] != "eager"]
    dy = load("fusion_dynamic")
    if not (ch or dy):
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    groups = defaultdict(list)
    for r in ch + ml:
        groups[(r["chain"], r["mode"])].append(r["compile_s"])
    names = list(dict.fromkeys(k[0] for k in groups))
    modes = list(dict.fromkeys(k[1] for k in groups))
    x = np.arange(len(names))
    for i, mode in enumerate(modes):
        a1.bar(x + i * 0.27, [np.median(groups.get((n, mode), [np.nan])) for n in names], 0.27, label=mode, color=MODE_C.get(mode))
    a1.set_yscale("log")
    a1.set_xticks(x + 0.27, names, rotation=25, fontsize=8)
    a1.set_ylabel("first-call compile time (s, median over sizes)")
    a1.legend(fontsize=7)
    steps = [r for r in dy if r.get("part") == "dynamic"]
    for strat, c in [("static", "tab:red"), ("automatic", "tab:blue"), ("mark_dynamic", "tab:green")]:
        rr = [r for r in steps if r["strategy"] == strat]
        if rr:
            a2.plot([r["step"] for r in rr], [r["first_call_s"] for r in rr], "o-", color=c,
                    label=f"{strat}: {rr[-1]['graphs']} graphs, total {rr[-1]['cum_s']:.1f}s")
    if steps:
        seqs = [r["seq"] for r in steps if r["strategy"] == "static"]
        a2.set_xticks(range(len(seqs)), [str(s) for s in seqs], fontsize=7)
    a2.set_yscale("log")
    a2.set_xlabel("call sequence (tokens of each new input)")
    a2.set_ylabel("first-call latency (s)")
    a2.set_title("recompilation under changing sequence length (MLP)")
    a2.legend(fontsize=7)
    save(fig, "gpu_compile_dynamic")
    SUMMARY["dynamic"] = [r for r in dy if r.get("part") in ("dynamic_total", "dynamic_steady")]
    SUMMARY["compile_s_median"] = {f"{n}|{m}": float(np.median(v)) for (n, m), v in groups.items()}


# --------------------------------------------------------------------------- Q2 matmul search

def model2(r: dict) -> float:
    """Wave-quantised roofline: each C tile re-reads A and B strips; tiles run in waves of 170."""
    m, n, k, bm, bn = r["M"], r["N"], r["K"], r["BM"], r["BN"]
    tiles = -(-m // bm) * -(-n // bn)
    waves = -(-tiles // SMS)
    t_tile = max(2 * bm * bn * k / (PEAK_BF16 / SMS), 2 * (bm + bn) * k / (HBM / SMS))
    return waves * t_tile


def fig_matmul():
    rows = load("matmul_search")
    grid = [r for r in rows if r.get("part") == "grid"]
    if not grid:
        return
    shapes = list(dict.fromkeys((r["M"], r["N"], r["K"]) for r in grid))
    mach = Machine(peak_flops=PEAK_BF16 / SMS, bw=HBM / SMS, cache_bytes=99 * 1024)
    SUMMARY["matmul"] = {}
    fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4.2))
    fig2, axes2 = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4.2))
    for ax, ax2, shp in zip(np.atleast_1d(axes), np.atleast_1d(axes2), shapes):
        rr = [r for r in grid if (r["M"], r["N"], r["K"]) == shp]
        ok = [r for r in rr if r["ok"]]
        cub = next(r for r in rows if r.get("part") == "cublas" and (r["M"], r["N"], r["K"]) == shp)
        at = next((r for r in rows if r.get("part") == "autotune" and (r["M"], r["N"], r["K"]) == shp), None)
        bms = sorted({r["BM"] for r in rr})
        bns = sorted({r["BN"] for r in rr})
        hm = np.full((len(bms), len(bns)), np.nan)
        for r in ok:
            i, j = bms.index(r["BM"]), bns.index(r["BN"])
            hm[i, j] = np.nanmax([hm[i, j], r["tflops"]])
        im = ax.imshow(hm, cmap="viridis", origin="lower")
        for i in range(len(bms)):
            for j in range(len(bns)):
                ax.text(j, i, "OOR" if np.isnan(hm[i, j]) else f"{hm[i, j]:.0f}", ha="center", va="center",
                        color="w" if np.isnan(hm[i, j]) or hm[i, j] < np.nanmax(hm) * 0.7 else "k", fontsize=8)
        ax.set_xticks(range(len(bns)), bns)
        ax.set_yticks(range(len(bms)), bms)
        ax.set_xlabel("BLOCK_N")
        ax.set_ylabel("BLOCK_M")
        best = max(ok, key=lambda r: r["tflops"])
        ax.set_title(f"{shp[0]}x{shp[1]}x{shp[2]} fp16 TFLOP/s (best over BK/warps/stages)\n"
                     f"grid best {best['tflops']:.0f}, autotune {at['tflops'] if at else float('nan'):.0f}, cuBLAS {cub['tflops']:.0f}",
                     fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8)
        # cost-model ranking over tile triples (best measured over warps/stages)
        trip = {}
        for r in ok:
            key = (r["BM"], r["BN"], r["BK"])
            if key not in trip or r["time_s"] < trip[key]["time_s"]:
                trip[key] = r
        meas = np.array([v["time_s"] for v in trip.values()])
        lab = np.array([cost(Schedule("ijk", *key), *shp, mach) for key in trip])
        m2 = np.array([model2(v) for v in trip.values()])
        rho_lab = spearmanr(lab, meas).statistic
        rho_m2 = spearmanr(m2, meas).statistic
        allv = np.array([r["time_s"] for r in ok])
        lab_all = np.array([cost(Schedule("ijk", r["BM"], r["BN"], r["BK"]), *shp, mach) for r in ok])
        rho_all = spearmanr(lab_all, allv).statistic
        ax2.scatter(lab / lab.min(), meas / meas.min(), s=18, label=f"lab cost model, rho={rho_lab:.2f}")
        ax2.scatter(m2 / m2.min(), meas / meas.min(), s=18, marker="x", label=f"wave-quantised roofline, rho={rho_m2:.2f}")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("predicted time / best predicted")
        ax2.set_ylabel("measured time / best measured")
        ax2.set_title(f"{shp[0]}x{shp[1]}x{shp[2]}: does the model rank tiles?", fontsize=9)
        ax2.legend(fontsize=7)
        lab_pick = list(trip)[int(np.argmin(lab))]
        m2_pick = list(trip)[int(np.argmin(m2))]
        top = sorted(ok, key=lambda r: r["time_s"])[:5]
        SUMMARY["matmul"][f"{shp[0]}x{shp[1]}x{shp[2]}"] = {
            "cublas_tflops": cub["tflops"], "grid_best": {k: best[k] for k in ("BM", "BN", "BK", "num_warps", "num_stages", "tflops")},
            "autotune": at and {"tflops": at["tflops"], "pick": at["pick"]},
            "n_configs": len(rr), "n_ok": len(ok), "n_oor": len(rr) - len(ok),
            "spearman_lab_tiles": rho_lab, "spearman_wave_tiles": rho_m2, "spearman_lab_all": rho_all,
            "lab_pick": lab_pick, "lab_pick_tflops": trip[lab_pick]["tflops"],
            "wave_pick": m2_pick, "wave_pick_tflops": trip[m2_pick]["tflops"],
            "within5pct": int(sum(r["time_s"] <= best["time_s"] * 1.05 for r in ok)),
            "top5": [{k: r[k] for k in ("BM", "BN", "BK", "num_warps", "num_stages", "tflops")} for r in top],
            "median_tflops": float(np.median([r["tflops"] for r in ok]))}
    save(fig, "gpu_matmul_heatmap")
    save(fig2, "gpu_costmodel_rank")

    shp = shapes[0]
    ok = [r for r in grid if (r["M"], r["N"], r["K"]) == shp and r["ok"]]
    best = max(ok, key=lambda r: r["tflops"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ws, ss = [4, 8], [2, 3, 4]
    hm = np.full((len(ws), len(ss)), np.nan)
    for r in ok:
        if (r["BM"], r["BN"], r["BK"]) == (best["BM"], best["BN"], best["BK"]):
            hm[ws.index(r["num_warps"]), ss.index(r["num_stages"])] = r["tflops"]
    a1.imshow(hm, cmap="magma", origin="lower")
    for i in range(len(ws)):
        for j in range(len(ss)):
            a1.text(j, i, "OOR" if np.isnan(hm[i, j]) else f"{hm[i, j]:.0f}", ha="center", va="center", color="c")
    a1.set_xticks(range(len(ss)), ss)
    a1.set_yticks(range(len(ws)), ws)
    a1.set_xlabel("num_stages")
    a1.set_ylabel("num_warps")
    a1.set_title(f"best tile {best['BM']}x{best['BN']}x{best['BK']}: warps x stages (TFLOP/s)", fontsize=9)
    vals = np.sort([r["tflops"] for r in ok])[::-1]
    a2.plot(np.arange(1, len(vals) + 1), vals, ".-")
    cub = next(r for r in rows if r.get("part") == "cublas" and (r["M"], r["N"], r["K"]) == shp)
    a2.axhline(cub["tflops"], color="k", ls="--", label="cuBLAS")
    a2.set_xscale("log")
    a2.set_xlabel("config rank")
    a2.set_ylabel("TFLOP/s")
    a2.set_title(f"{shp[0]}^3 landscape: sorted configs", fontsize=9)
    a2.legend()
    save(fig, "gpu_matmul_landscape")


# --------------------------------------------------------------------------- Q3 quantization

ORDER = ["fp16", "w8_tensor", "w8_channel", "w4_channel", "gptq_w4_channel", "w4_g128", "awq_w4_g128", "gptq_w4_g128",
         "w3_g128", "awq_w3_g128", "gptq_w3_g128", "w8a8_tensor", "w8a8_token", "smoothquant_tensor",
         "smoothquant_tensor_a0.8", "smoothquant_token", "smoothquant_token_a0.8", "llm_int8"]


def fig_quant():
    models = [m for m in ["gpt2", "gpt2-medium"] if load(f"quant_ppl_{m}")]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(12, 4.5))
    wdt = 0.8 / len(models)
    SUMMARY["quant_ppl"] = {}
    for i, m in enumerate(models):
        rr = {r["method"]: r for r in load(f"quant_ppl_{m}") if r.get("part") == "ppl"}
        meths = [k for k in ORDER if k in rr]
        x = np.arange(len(meths))
        bars = ax.bar(x + i * wdt, [rr[k]["ppl"] for k in meths], wdt, label=m)
        for b, k in zip(bars, meths):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{rr[k]['ppl']:.1f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)
        ax.axhline(rr["fp16"]["ppl"], color=bars[0].get_facecolor(), lw=0.6, ls="--")
        SUMMARY["quant_ppl"][m] = {k: rr[k]["ppl"] for k in meths}
    ax.set_xticks(np.arange(len(meths)) + 0.4 - wdt / 2, meths, rotation=40, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("wikitext-2 test perplexity (log)")
    ax.set_title("GPT-2 post-training quantization (weight-only: RTN/AWQ/GPTQ; W8A8 simulated)")
    ax.legend()
    save(fig, "gpu_quant_ppl")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m, ls in zip(models, ["-", "--"]):
        o = [r for r in load(f"quant_ppl_{m}") if r.get("part") == "outliers"]
        for kind, c in [("attn.c_attn", "tab:blue"), ("attn.c_proj", "tab:orange"), ("mlp.c_fc", "tab:green"), ("mlp.c_proj", "tab:red")]:
            rr = [r for r in o if r["layer"].endswith(kind)]
            layers = [int(r["layer"].split(".h.")[1].split(".")[0]) for r in rr]
            axes[0].plot(layers, [r["absmax"] for r in rr], ls, color=c, marker=".", label=f"{m} {kind}")
            axes[1].plot(layers, [r["n_gt6"] for r in rr], ls, color=c, marker=".", label=f"{m} {kind}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("max |x| over input channels")
    axes[0].set_title("activation outliers at each linear's input")
    axes[1].set_xlabel("block")
    axes[1].set_ylabel("# input channels with max|x| > 6")
    axes[1].set_title("LLM.int8 threshold-6 outlier channels")
    axes[0].legend(fontsize=6, ncol=2)
    save(fig, "gpu_act_outliers")
    SUMMARY["outliers"] = {m: sorted(({k: r[k] for k in ("layer", "absmax", "median_absmax", "ratio", "n_gt6", "top_channels")}
                                      for r in load(f"quant_ppl_{m}") if r.get("part") == "outliers"),
                                     key=lambda r: -r["absmax"])[:6] for m in models}


def fig_qspeed():
    rows = load("quant_speed")
    g = [r for r in rows if r.get("part") == "gemm" and r.get("ok")]
    if not g:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    shapes = list(dict.fromkeys((r["N"], r["K"]) for r in g))
    for ax, (n, k) in zip(axes, shapes):
        for dt, c in [("bf16", "tab:blue"), ("int8", "tab:orange"), ("fp8", "tab:green")]:
            rr = sorted([r for r in g if (r["N"], r["K"]) == (n, k) and r["dtype"] == dt], key=lambda r: r["M"])
            if rr:
                ax.plot([r["M"] for r in rr], [r["tflops"] for r in rr], "o-", color=c, label=dt)
        ax.axhline(PEAK_BF16 / 1e12, ls="--", color="tab:blue", lw=0.7, label="bf16 dense peak")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("M (tokens)")
        ax.set_ylabel("TFLOP/s or TOP/s")
        ax.set_title(f"GEMM [M x {k}] x [{k} x {n}]")
        ax.legend(fontsize=7)
    save(fig, "gpu_lowprec_gemm")
    SUMMARY["gemm"] = [{k: r[k] for k in ("M", "N", "K", "dtype", "tflops", "time_s")} for r in g]

    v = [r for r in rows if r.get("part") == "gemv"]
    if not v:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    shapes = list(dict.fromkeys((r["N"], r["K"]) for r in v))
    impls = ["bf16_cublas", "fp16_cublas", "w8a16_triton", "w4a16_triton", "w4_dequant_then_matmul"]
    x = np.arange(len(shapes))
    for i, im in enumerate(impls):
        vals = [next((r["time_s"] * 1e6 for r in v if (r["N"], r["K"]) == s and r["impl"] == im and r["cache"] == "cold"), np.nan) for s in shapes]
        a1.bar(x + i * 0.16, vals, 0.16, label=im)
        vals = [next((r["eff_bw"] / 1e12 for r in v if (r["N"], r["K"]) == s and r["impl"] == im and r["cache"] == "cold"), np.nan) for s in shapes]
        a2.bar(x + i * 0.16, vals, 0.16, label=im)
    for a in (a1, a2):
        a.set_xticks(x + 0.32, [f"{n}x{k}" for n, k in shapes], fontsize=8)
        a.set_xlabel("weight N x K (batch 1, cold = rotating copies > L2)")
    a1.set_yscale("log")
    a1.set_ylabel("time per GEMV (us)")
    a1.legend(fontsize=7)
    a2.axhline(HBM / 1e12, ls="--", color="k", lw=0.8, label="GDDR7 peak")
    a2.set_ylabel("weight bytes / time (TB/s)")
    a2.legend(fontsize=7)
    a1.set_title("weight-only quant at batch 1: bytes, not FLOPs")
    save(fig, "gpu_weight_only_gemv")
    SUMMARY["gemv"] = [{k: r[k] for k in ("N", "K", "impl", "cache", "time_s", "eff_bw", "rel_err")} for r in v]


# --------------------------------------------------------------------------- Q4 pruning

def fig_prune():
    pp = [r for r in load("prune_ppl_gpt2") if r.get("part") == "prune_ppl"]
    sp = load("prune_speed")
    if not (pp or sp):
        return
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
    if pp:
        dense = next(r["ppl"] for r in pp if r["method"] == "dense")
        mag = sorted([r for r in pp if r["method"].startswith("magnitude_0")], key=lambda r: r["sparsity"])
        a1.plot([0] + [r["sparsity"] for r in mag], [dense] + [r["ppl"] for r in mag], "o-", label="magnitude (per layer)")
        for r in pp:
            if r["method"] not in ("dense",) and not r["method"].startswith("magnitude_0"):
                a1.scatter(r["sparsity"], r["ppl"], marker="s", zorder=3)
                a1.annotate(r["method"], (r["sparsity"], r["ppl"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        a1.set_yscale("log")
        a1.set_xlabel("sparsity of block linears")
        a1.set_ylabel("wikitext-2 ppl (GPT-2 124M)")
        a1.set_title("one-shot pruning, no fine-tuning")
        a1.legend(fontsize=7)
        SUMMARY["prune_ppl"] = {r["method"]: {"sparsity": r["sparsity"], "ppl": r["ppl"]} for r in pp}
    s24 = [r for r in sp if r.get("part") == "sparse24"]
    shapes = list(dict.fromkeys((r["N"], r["K"]) for r in s24))
    for s in shapes:
        rr = sorted([r for r in s24 if (r["N"], r["K"]) == s and r["impl"] == "cusparselt" and r["ok"]], key=lambda r: r["M"])
        if rr:
            a2.plot([r["M"] for r in rr], [r["speedup"] for r in rr], "o-", label=f"W {s[0]}x{s[1]}")
    a2.axhline(1, color="k", lw=0.6)
    a2.axhline(2, color="k", lw=0.6, ls=":", label="ideal 2x")
    a2.set_xscale("log", base=2)
    a2.set_xlabel("M (tokens)")
    a2.set_ylabel("2:4 cuSPARSELt speedup over dense")
    a2.set_title("semi-structured 2:4 (fp16) on sm_120")
    a2.legend(fontsize=7)
    csr = [r for r in sp if r.get("part") == "csr" and r["ok"]]
    for m in sorted({r["M"] for r in csr}):
        rr = sorted([r for r in csr if r["M"] == m], key=lambda r: r["sparsity"])
        a3.plot([r["sparsity"] for r in rr], [r["speedup"] for r in rr], "o-", label=f"M={m}")
    a3.axhline(1, color="k", lw=0.6)
    a3.set_yscale("log")
    a3.set_xlabel("unstructured sparsity")
    a3.set_ylabel("CSR (cuSPARSE) speedup over dense")
    a3.set_title("unstructured sparsity needs >95% to pay")
    a3.legend(fontsize=7)
    save(fig, "gpu_pruning")
    SUMMARY["sparse24"] = [{k: r.get(k) for k in ("N", "K", "M", "impl", "time_s", "speedup", "rel_err", "ok", "error")} for r in s24]
    SUMMARY["csr"] = [{k: r[k] for k in ("M", "sparsity", "dense_s", "csr_s", "speedup")} for r in csr]


# --------------------------------------------------------------------------- Q5 LoRA

def fig_lora():
    rows = load("lora_ft")
    curves = [r for r in rows if r.get("part") == "curve"]
    summ = [r for r in rows if r.get("part") == "summary"]
    if not summ:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for cfg in dict.fromkeys(r["config"] for r in curves):
        rr = [r for r in curves if r["config"] == cfg]
        loss = np.array([r["loss"] for r in rr])
        k = 10
        sm = np.convolve(loss, np.ones(k) / k, mode="valid")
        a1.plot(np.arange(k - 1, len(loss)), sm, label=cfg, lw=1.2)
        vv = [(r["step"], r["val_loss"]) for r in rr if "val_loss" in r]
        a2.plot(*zip(*vv), "o-", label=cfg, ms=3)
    a1.set_xlabel("step")
    a1.set_ylabel("train loss (10-step moving avg)")
    a1.set_title("GPT-2 medium on wikitext-2")
    a2.set_xlabel("step")
    a2.set_ylabel("validation loss")
    a1.legend(fontsize=7)
    a2.legend(fontsize=7)
    save(fig, "gpu_lora_curves")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    names = [r["config"] for r in summ]
    x = np.arange(len(names))
    axes[0].bar(x, [r["trainable"] / 1e6 for r in summ], color="tab:purple")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("trainable params (M, log)")
    axes[1].bar(x, [r["peak_mem_gb"] for r in summ], color="tab:red", label="peak")
    axes[1].bar(x, [r["static_mem_gb"] for r in summ], color="tab:gray", label="weights before step")
    axes[1].set_ylabel("GPU memory (GiB)")
    axes[1].legend(fontsize=7)
    axes[2].bar(x, [r["step_s"] * 1e3 for r in summ], color="tab:blue")
    axes[2].set_ylabel("step time (ms, median)")
    for a in axes:
        a.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
    save(fig, "gpu_lora_cost")
    SUMMARY["lora"] = summ


def main():
    for f in [fig_fusion, fig_mlp, fig_triton, fig_compile_dynamic, fig_matmul, fig_quant, fig_qspeed, fig_prune, fig_lora]:
        try:
            f()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("FAILED", f.__name__, e)
    (RAW / "summary_gpu.json").write_text(json.dumps(SUMMARY, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
