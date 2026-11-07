"""Figures + summary tables for the GPU part (reads artifacts/raw/gpu/*.jsonl; runs locally).

    python scripts/plot_gpu.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
sys.path.insert(0, str(LAB.parents[2]))
sys.path.insert(0, str(LAB.parents[2] / "tools"))
import mlsys_calc as mc  # noqa: E402

MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ARCH = {"gpt2": (12, 768, 12), "gpt2-medium": (24, 1024, 16), "gpt2-large": (36, 1280, 20),
        "gpt2-xl": (48, 1600, 25)}
COL = dict(zip(MODELS, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))
TCOL = {0.0: "tab:blue", 0.7: "tab:orange", 1.0: "tab:red"}
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def load(name: str) -> list[dict]:
    p = RAW / f"{name}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def load_merged(name: str) -> list[dict]:
    """gpt2/gpt2-medium were re-run in v2/ after adding the bmm attention path; v2 rows replace them."""
    v2 = load(f"v2/{name}")
    redo = {r["model"] for r in v2}
    return [r for r in load(name) if r["model"] not in redo] + v2


def save(fig, name: str) -> None:
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", name)


HW = {r["kind"]: r for r in load("hw") if r["kind"] != "gemm_fp16"}
GEMM = max((r["TFLOPs"] for r in load("hw") if r["kind"] == "gemm_fp16"), default=209.5)
BW = HW.get("copy_bw", {}).get("GBps", 1550.0) * 1e9
_GPU_NAME = HW.get("device", {}).get("name", "GPU")
GPU = mc.GPU(f"{_GPU_NAME} (measured)", GEMM * 1e12, BW, 32e9, 64e9)


def mc_model(name: str) -> mc.Model:
    L, d, h = ARCH[name]
    return mc.Model(name, L, d, h, h, 4 * d, 50257, gated_mlp=False, tied_embeddings=True)


def fig_parity():
    rows = load("verify")
    lg = [r for r in rows if r["kind"] == "logits"]
    gr = [r for r in rows if r["kind"] == "greedy"]
    if not lg:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(lg))
    for k, (key, lab) in enumerate([("nocache_vs_hf_maxabs", "no-cache vs HF (max |Δlogit|)"),
                                    ("kv_vs_hf_maxabs", "KV cache vs HF (max |Δlogit|)"),
                                    ("rel_err", "max |Δ| / max |logit|"),
                                    ("prob_maxabs", "max |Δprob|")]):
        ax[0].bar(x + (k - 1.5) * 0.2, [r[key] for r in lg], 0.2, label=lab)
    ax[0].set_yscale("log")
    ax[0].axhline(1e-4, color="k", ls=":", lw=1)
    ax[0].set_xticks(x, [r["model"] for r in lg])
    ax[0].set_title("fp32 logits vs HF transformers (2x128 tokens)")
    ax[0].legend(fontsize=7)
    variants = [("kv_eq_nocache", "KV eager"), ("graph_eq_nocache", "KV+CUDA graph"), ("hf_eq_nocache", "HF fp32 generate")]
    mat = np.zeros((len(MODELS) * 2, len(variants)))
    labels = []
    for i, (m, dt) in enumerate([(m, dt) for m in MODELS for dt in ("float32", "float16")]):
        sel = [r for r in gr if r["model"] == m and r["dtype"] == dt]
        labels.append(f"{m} {dt[-2:]}")
        for j, (k, _) in enumerate(variants):
            mat[i, j] = sum(r[k] for r in sel) / max(len(sel), 1)
    im = ax[1].imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax[1].text(j, i, f"{mat[i, j] * 5:.0f}/5", ha="center", va="center", fontsize=7)
    ax[1].set_xticks(range(len(variants)), [v for _, v in variants])
    ax[1].set_yticks(range(len(labels)), labels, fontsize=7)
    ax[1].set_title("greedy 64-token outputs identical to no-cache")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    save(fig, "gpu_parity")


def fig_latency():
    st = [r for r in load_merged("latency") if r["kind"] == "step"]
    if not st:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODELS:
        s = sorted([r for r in st if r["model"] == m], key=lambda r: r["ctx"])
        c = [r["ctx"] for r in s]
        ax[0].plot(c, [r["nocache_ms"] for r in s], "o-", color=COL[m], label=f"{m} no-cache")
        ax[0].plot(c, [r["kv_eager_ms"] for r in s], "s--", color=COL[m], alpha=0.7, label=f"{m} KV eager")
        ax[0].plot(c, [r["kv_graph_ms"] for r in s], "^:", color=COL[m], label=f"{m} KV graph")
        ax[1].plot(c, [r["kv_graph_ms"] for r in s], "^-", color=COL[m], label=f"{m} graph, SDPA+mask")
        if all("kv_graph_bmm_ms" in r for r in s):
            ax[1].plot(c, [r["kv_graph_bmm_ms"] for r in s], "o-.", color=COL[m], ms=3, label=f"{m} graph, bmm attn")
        pred = [mc.decode_step_time(mc_model(m), GPU, r["ctx"], 1)["time_s"] * 1e3 for r in s]
        ax[1].plot(c, pred, "--", color=COL[m], alpha=0.6, label=f"{m} roofline")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("context length (tokens)")
    ax[0].set_ylabel("ms per generated token")
    ax[0].set_title("per-token latency, batch 1, fp16")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("context length (tokens)")
    ax[1].set_ylabel("ms per decode step")
    ax[1].set_title("CUDA-graph decode vs mlsys_calc.decode_step_time")
    ax[1].legend(fontsize=6, ncol=2)
    save(fig, "gpu_latency_vs_ctx")


def fig_model_size():
    st = [r for r in load_merged("latency") if r["kind"] == "step" and r["ctx"] == 128]
    if not st:
        return
    st.sort(key=lambda r: r["weight_bytes"])
    wb = np.array([r["weight_bytes"] for r in st]) / 1e9
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(wb, [r["kv_graph_ms"] for r in st], "o-", label="KV + CUDA graph")
    ax.plot(wb, [r["kv_eager_ms"] for r in st], "s-", label="KV eager (launch-bound)")
    ax.plot(wb, [r["nocache_ms"] for r in st], "^-", label="no cache (128-token forward)")
    xs = np.linspace(0, wb.max() * 1.05, 50)
    ax.plot(xs, xs * 1e9 / BW * 1e3, "k--", label=f"weights / {BW / 1e12:.2f} TB/s")
    for r, w in zip(st, wb):
        ax.annotate(r["model"], (w, r["kv_graph_ms"]), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax.set_xlabel("fp16 weight bytes (GB)")
    ax.set_ylabel("ms per token (ctx=128)")
    ax.set_title("decode latency vs model size")
    ax.legend(fontsize=7)
    save(fig, "gpu_latency_vs_model")


def fig_prefill_decode():
    g = [r for r in load_merged("latency") if r["kind"] == "generate"]
    if not g:
        return
    fig, ax = plt.subplots(figsize=(10, 3.8))
    labels, pre, dec = [], [], []
    for m in MODELS:
        for P in (128, 512, 896):
            for v in ("nocache", "kv_eager", "kv_graph"):
                r = [x for x in g if x["model"] == m and x["prompt"] == P and x["variant"] == v]
                if r:
                    labels.append(f"{m.replace('gpt2-', '').replace('gpt2', 'small')}\nP={P}\n{v.replace('kv_', '')}")
                    pre.append(r[0]["prefill_ms"])
                    dec.append(r[0]["decode_ms"])
    x = np.arange(len(labels))
    ax.bar(x, pre, label="prefill (P tokens, 1 forward)")
    ax.bar(x, dec, bottom=pre, label="decode (127 tokens)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=5)
    ax.set_ylabel("ms (log)")
    ax.set_title("prefill vs decode time, generate 128 new tokens, fp16 batch 1")
    ax.legend(fontsize=7)
    save(fig, "gpu_prefill_decode")


def batch_rows():
    rows = []
    for r in load_merged("batch") + load("ctx32/batch"):
        if "skipped" in r:
            continue
        p = mc.decode_step_time(mc_model(r["model"]), GPU, r["ctx"], r["B"])
        rows.append({"model": r["model"], "B": r["B"], "ctx": r["ctx"], "measured_ms": round(r["graph_ms"], 4),
                     "eager_ms": round(r["eager_ms"], 4), "bmm_ms": round(r.get("graph_bmm_ms", float("nan")), 4),
                     "predicted_ms": round(p["time_s"] * 1e3, 4),
                     "measured_over_pred": round(r["graph_ms"] / (p["time_s"] * 1e3), 3),
                     "tok_s": round(r["tok_s_graph"], 1), "pred_tok_s": round(p["tokens_per_s"], 1),
                     "intensity": round(p["intensity"], 2), "bound": p["bound"],
                     "achieved_TFLOPs": round(2 * mc_model(r["model"]).params() * r["B"] / r["graph_ms"] / 1e9, 2),
                     "achieved_GBps": round((p["weight_gb"] + p["kv_gb"]) * 1e9 / r["graph_ms"] / 1e6, 1)})
    return rows


def fig_batch():
    rows = batch_rows()
    if not rows:
        return
    write_csv("model_vs_measured_batch.csv", rows)
    ctxs = sorted({r["ctx"] for r in rows})
    fig, axs = plt.subplots(1, len(ctxs), figsize=(5 * len(ctxs), 3.8), squeeze=False)
    for ax, ctx in zip(axs[0], ctxs):
        for m in MODELS:
            s = [r for r in rows if r["model"] == m and r["ctx"] == ctx]
            B = [r["B"] for r in s]
            ax.plot(B, [r["tok_s"] for r in s], "o-", color=COL[m], label=f"{m} measured (graph)")
            ax.plot(B, [r["pred_tok_s"] for r in s], "--", color=COL[m], alpha=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("decode tokens/s")
        ax.set_title(f"ctx={ctx}: measured (solid) vs mlsys_calc roofline (dashed)")
        ax.legend(fontsize=7)
    save(fig, "gpu_batch_throughput")

    fig, axs = plt.subplots(1, len(ctxs), figsize=(5 * len(ctxs), 3.8), squeeze=False)
    for ax, ctx in zip(axs[0], ctxs):
        for m in MODELS:
            s = [r for r in rows if r["model"] == m and r["ctx"] == ctx]
            ax.plot([r["B"] for r in s], [r["measured_ms"] for r in s], "o-", color=COL[m], label=f"{m} graph")
            ax.plot([r["B"] for r in s], [r["eager_ms"] for r in s], "x:", color=COL[m], alpha=0.6,
                    label=f"{m} eager")
            ax.plot([r["B"] for r in s], [r["predicted_ms"] for r in s], "--", color=COL[m], alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("ms per decode step")
        ax.set_title(f"ctx={ctx}: step time vs batch (dashed: roofline)")
        ax.legend(fontsize=6, ncol=2)
    save(fig, "gpu_batch_steptime")
    rows = [r for r in rows if r["ctx"] == ctxs[0]] + [r for r in rows if r["ctx"] != ctxs[0]]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    I = np.logspace(-0.5, 3.2, 100)
    ax[0].plot(I, np.minimum(GPU.bf16_flops, I * BW) / 1e12, "k-", lw=1.5, label="roofline (measured peaks)")
    for m in MODELS:
        for ctx, mk in zip(ctxs, "os^"):
            s = [r for r in rows if r["model"] == m and r["ctx"] == ctx]
            ax[0].plot([r["intensity"] for r in s], [r["achieved_TFLOPs"] for r in s], mk + "-", color=COL[m],
                       label=f"{m} ctx={ctx}", alpha=1 if ctx == ctxs[0] else 0.6)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("arithmetic intensity (FLOP/byte, weights+KV)")
    ax[0].set_ylabel("achieved TFLOP/s (2NB / t)")
    ax[0].set_title("decode roofline on RTX 5090")
    ax[0].legend(fontsize=6, ncol=2)
    Bs = sorted({r["B"] for r in rows})
    keys = [(m, c) for c in ctxs for m in MODELS]
    mat = np.full((len(keys), len(Bs)), np.nan)
    for r in rows:
        mat[keys.index((r["model"], r["ctx"])), Bs.index(r["B"])] = r["measured_over_pred"]
    im = ax[1].imshow(mat, cmap="viridis", aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax[1].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax[1].set_xticks(range(len(Bs)), Bs)
    ax[1].set_yticks(range(len(keys)), [f"{m} ctx={c}" for m, c in keys], fontsize=7)
    ax[1].set_xlabel("batch size")
    ax[1].set_title("measured / mlsys_calc prediction")
    ax[1].grid(False)
    fig.colorbar(im, ax=ax[1], fraction=0.04)
    save(fig, "gpu_roofline_ratio")


def fig_overhead():
    """Batch-1 gap between measured graph decode and the roofline bound, per layer."""
    st = [r for r in load_merged("latency") if r["kind"] == "step"]
    if not st:
        return
    out = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for ctx, mk in ((16, "o"), (128, "s"), (1024, "^")):
        xs, ys = [], []
        for m in MODELS:
            r = [x for x in st if x["model"] == m and x["ctx"] == ctx]
            if not r:
                continue
            r = r[0]
            pred = mc.decode_step_time(mc_model(m), GPU, ctx, 1)["time_s"] * 1e3
            impl = "kv_graph_bmm_ms" if "kv_graph_bmm_ms" in r else "kv_graph_ms"
            gap = r[impl] - pred
            xs.append(ARCH[m][0])
            ys.append(gap)
            out.append({"model": m, "ctx": ctx, "layers": ARCH[m][0], "measured_ms": round(r[impl], 4),
                        "roofline_ms": round(pred, 4), "gap_ms": round(gap, 4),
                        "gap_us_per_layer": round(gap / ARCH[m][0] * 1e3, 1),
                        "measured_over_pred": round(r[impl] / pred, 3)})
        k = np.polyfit(xs, ys, 1)
        ax[0].plot(xs, ys, mk, label=f"ctx={ctx}: slope {k[0] * 1e3:.1f} µs/layer")
        ax[0].plot(xs, np.polyval(k, xs), "--", lw=1)
    ax[0].set_xlabel("number of layers")
    ax[0].set_ylabel("measured − roofline (ms)")
    ax[0].set_title("batch-1 decode overhead grows ~linearly with layers")
    ax[0].legend(fontsize=7)
    s128 = [o for o in out if o["ctx"] == 128]
    x = np.arange(len(s128))
    ax[1].bar(x, [o["roofline_ms"] for o in s128], label="roofline (bytes / BW)")
    ax[1].bar(x, [o["gap_ms"] for o in s128], bottom=[o["roofline_ms"] for o in s128], label="per-kernel overhead")
    for i, o in enumerate(s128):
        ax[1].text(i, o["measured_ms"], f"{o['measured_over_pred']:.2f}x", ha="center", va="bottom", fontsize=7)
    ax[1].set_xticks(x, [o["model"] for o in s128])
    ax[1].set_ylabel("ms per token (ctx=128, CUDA graph)")
    ax[1].set_title("where batch-1 decode time goes")
    ax[1].legend(fontsize=7)
    write_csv("decode_overhead.csv", out)
    save(fig, "gpu_decode_overhead")


def fig_memory():
    rows = load("memory")
    if not rows:
        return
    out = []
    for r in rows:
        L, d, h = ARCH[r["model"]]
        calc = mc.kv_cache_bytes(mc_model(r["model"]), r["T"], r["B"], 2)
        out.append({"model": r["model"], "T": r["T"], "B": r["B"], "formula_bytes": r["formula_bytes"],
                    "mlsys_calc_bytes": calc, "allocated_bytes": r["allocated_bytes"],
                    "alloc_over_formula": round(r["allocated_bytes"] / r["formula_bytes"], 4),
                    "decode_peak_bytes": r["decode_peak_bytes"],
                    "workspace_bytes": r["decode_peak_bytes"] - r["allocated_bytes"]})
    write_csv("kv_memory_summary.csv", out)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in MODELS:
        s = [r for r in out if r["model"] == m]
        ax[0].loglog([r["formula_bytes"] / 2**20 for r in s], [r["allocated_bytes"] / 2**20 for r in s], "o",
                     color=COL[m], label=m)
        for T in (1024,):
            ss = sorted([r for r in s if r["T"] == T], key=lambda r: r["B"])
            ax[1].plot([r["B"] for r in ss], [r["allocated_bytes"] / 2**30 for r in ss], "o-", color=COL[m],
                       label=f"{m} T={T}")
        ss = sorted([r for r in s if r["B"] == 8], key=lambda r: r["T"])
        ax[1].plot([r["B"] * 0 + 70 for r in ss][:0], [], color=COL[m])
    lim = [min(r["formula_bytes"] for r in out) / 2**20, max(r["formula_bytes"] for r in out) / 2**20]
    ax[0].plot(lim, lim, "k--", lw=1, label="y = 2·L·B·H·hd·T·2 bytes")
    ax[0].set_xlabel("formula (MiB)")
    ax[0].set_ylabel("torch.cuda.memory_allocated delta (MiB)")
    ax[0].set_title("KV cache: measured vs formula")
    ax[0].legend(fontsize=7)
    ax[1].axhline(32 * 1e9 / 2**30, color="k", ls=":", lw=1, label="32 GB VRAM")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("batch size")
    ax[1].set_ylabel("KV cache (GiB), fp16")
    ax[1].set_title("KV cache grows linearly in B (T=1024)")
    ax[1].legend(fontsize=7)
    save(fig, "gpu_kv_memory")


def spec_rows():
    rows = load("spec")
    base = {(r["target"], r["draft"], r["temperature"]): r for r in rows if r["kind"] == "baseline"}
    out = []
    for r in rows:
        if r["kind"] != "spec":
            continue
        g, a = r["gamma"], r["alpha"]
        c = r["t_draft1_ms"] / r["t_target1_ms"]
        v = r["t_target_gamma1_ms"] / r["t_target1_ms"]
        E = mc.speculative_expected_tokens(a, g)
        out.append({"target": r["target"], "draft": r["draft"], "gamma": g, "T": r["temperature"],
                    "alpha": round(a, 4), "accept_fraction": round(r["accept_fraction"], 4),
                    "alpha_overlap": round(r["alpha_overlap"], 4) if r["temperature"] > 0 else "",
                    "tokens_per_call": round(r["tokens_per_target_call"], 3), "E_formula": round(E, 3),
                    "c": round(c, 3), "v": round(v, 3),
                    "pred_speedup_free_verify": round(mc.speculative_speedup(a, g, c), 3),
                    "pred_speedup_measured_v": round(E / (g * c + v), 3),
                    "speedup": round(r["speedup"], 3), "tok_s": round(r["tok_s"], 1),
                    "base_tok_s": round(base[(r["target"], r["draft"], r["temperature"])]["tok_s"], 1),
                    "greedy_identical": r["greedy_identical"] if r["greedy_identical"] is not None else "",
                    "n_prompts": r["n_prompts"], "hist": r["accepted_hist"]})
    return out


def fig_spec():
    rows = spec_rows()
    if not rows:
        return
    write_csv("spec_summary.csv", [{k: v for k, v in r.items() if k != "hist"} for r in rows])
    pairs = sorted({(r["draft"], r["target"]) for r in rows}, key=lambda p: (p[1], p[0]))
    fig, axs = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), squeeze=False)
    for ax, (d, t) in zip(axs[0], pairs):
        for T in (0.0, 0.7, 1.0):
            s = sorted([r for r in rows if r["draft"] == d and r["target"] == t and r["T"] == T], key=lambda r: r["gamma"])
            if not s:
                continue
            ax.plot([r["gamma"] for r in s], [r["alpha"] for r in s], "o-", color=TCOL[T], label=f"α T={T}")
            ax.plot([r["gamma"] for r in s], [r["accept_fraction"] for r in s], "x--", color=TCOL[T], alpha=0.6,
                    label=f"accepted/proposed T={T}")
            if T > 0:
                ax.plot([r["gamma"] for r in s], [r["alpha_overlap"] for r in s], ":", color=TCOL[T], alpha=0.8,
                        label=f"E[Σmin(p,q)] T={T}")
        ax.set_title(f"{d} → {t}")
        ax.set_xlabel("γ")
        ax.set_ylabel("acceptance")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6)
    save(fig, "gpu_spec_alpha")

    fig, axs = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), squeeze=False)
    for ax, (d, t) in zip(axs[0], pairs):
        for T in (0.0, 0.7, 1.0):
            s = sorted([r for r in rows if r["draft"] == d and r["target"] == t and r["T"] == T], key=lambda r: r["gamma"])
            if not s:
                continue
            ax.plot([r["gamma"] for r in s], [r["tokens_per_call"] for r in s], "o-", color=TCOL[T], label=f"measured T={T}")
            ax.plot([r["gamma"] for r in s], [r["E_formula"] for r in s], "--", color=TCOL[T], alpha=0.7,
                    label=f"(1−α^(γ+1))/(1−α) T={T}")
        ax.set_title(f"{d} → {t}: tokens per target forward")
        ax.set_xlabel("γ")
        ax.legend(fontsize=6)
    save(fig, "gpu_spec_tokens")

    fig, axs = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.6), squeeze=False)
    for ax, (d, t) in zip(axs[0], pairs):
        for T in (0.0, 0.7, 1.0):
            s = sorted([r for r in rows if r["draft"] == d and r["target"] == t and r["T"] == T], key=lambda r: r["gamma"])
            if not s:
                continue
            g = [r["gamma"] for r in s]
            ax.plot(g, [r["speedup"] for r in s], "o-", color=TCOL[T], label=f"measured T={T}")
            ax.plot(g, [r["pred_speedup_measured_v"] for r in s], "--", color=TCOL[T], alpha=0.7,
                    label=f"E/(γc+v) T={T}")
        ax.axhline(1, color="k", lw=1)
        ax.set_title(f"{d} → {t}: wall-clock speedup (c={s[0]['c']:.2f})")
        ax.set_xlabel("γ")
        ax.legend(fontsize=6)
    save(fig, "gpu_spec_speedup")

    fig, axs = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 2.8), squeeze=False)
    for ax, (d, t) in zip(axs[0], pairs):
        Ts = sorted({r["T"] for r in rows if r["draft"] == d and r["target"] == t})
        gs = sorted({r["gamma"] for r in rows if r["draft"] == d and r["target"] == t})
        mat = np.full((len(Ts), len(gs)), np.nan)
        for r in rows:
            if r["draft"] == d and r["target"] == t:
                mat[Ts.index(r["T"]), gs.index(r["gamma"])] = r["speedup"]
        im = ax.imshow(mat, cmap="RdBu", vmin=0.4, vmax=1.6, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(gs)), gs)
        ax.set_yticks(range(len(Ts)), [f"T={x}" for x in Ts])
        ax.set_xlabel("γ")
        ax.set_title(f"{d} → {t} speedup")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, "gpu_spec_heatmap")

    sel = [r for r in rows if r["gamma"] in (2, 4, 8) and r["draft"] == "gpt2"]
    if sel:
        fig, axs = plt.subplots(1, len(sel[:6]), figsize=(2.4 * len(sel[:6]), 2.8), squeeze=False)
        sel = sorted(sel, key=lambda r: (r["target"], r["T"], r["gamma"]))
        sel = [r for r in sel if r["T"] in (0.0, 1.0)][:6]
        for ax, r in zip(axs[0], sel):
            h = np.array(r["hist"], float)
            h /= h.sum()
            g, a = r["gamma"], r["alpha"]
            pred = np.array([(1 - a) * a ** k for k in range(g)] + [a ** g])
            ax.bar(range(g + 1), h, label="measured")
            ax.plot(range(g + 1), pred, "ko-", ms=3, label="trunc. geometric")
            ax.set_title(f"{r['target'].replace('gpt2-', '')} γ={g} T={r['T']}", fontsize=8)
            ax.set_xlabel("accepted per round")
        axs[0][0].legend(fontsize=6)
        save(fig, "gpu_spec_accept_hist")


def fig_spec_batch():
    cost = load("spec_cost")
    srows = spec_rows()
    if not cost or not srows:
        return
    out = []
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, t in enumerate(sorted({r["target"] for r in cost})):
        s = sorted([r for r in cost if r["target"] == t], key=lambda r: r["B"])
        for T, ls in ((0.0, "-"), (1.0, "--")):
            for g, mk in ((1, "o"), (2, "s"), (4, "^"), (8, "v")):
                a = [r for r in srows if r["target"] == t and r["draft"] == "gpt2" and r["T"] == T and r["gamma"] == g]
                if not a:
                    continue
                al = a[0]["alpha"]
                E = mc.speculative_expected_tokens(al, g)
                sp = []
                for r in s:
                    c = r["t_draft1_ms"] / r["t_target1_ms"]
                    v = r[f"t_target{g + 1}_ms"] / r["t_target1_ms"]
                    sp.append(E / (g * c + v))
                    out.append({"target": t, "T": T, "gamma": g, "B": r["B"], "alpha": al, "c": round(c, 3),
                                "v": round(v, 3), "pred_speedup": round(sp[-1], 3)})
                axs[k].plot([r["B"] for r in s], sp, ls, marker=mk, label=f"γ={g} T={T} (α={al:.2f})")
        axs[k].axhline(1, color="k", lw=1)
        axs[k].set_xscale("log", base=2)
        axs[k].set_xlabel("batch size")
        axs[k].set_ylabel("predicted speedup E/(γc(B)+v(B))")
        axs[k].set_title(f"gpt2 → {t}: measured c(B), v(B), α")
        axs[k].legend(fontsize=6, ncol=2)
    write_csv("spec_batch_model.csv", out)
    save(fig, "gpu_spec_batch")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    for t in sorted({r["target"] for r in cost}):
        s = sorted([r for r in cost if r["target"] == t], key=lambda r: r["B"])
        for n, mk in ((2, "o"), (5, "^"), (9, "v")):
            ax.plot([r["B"] for r in s], [r[f"t_target{n}_ms"] / r["t_target1_ms"] for r in s], marker=mk,
                    label=f"{t} verify {n} tok")
        ax.plot([r["B"] for r in s], [r["t_draft1_ms"] / r["t_target1_ms"] for r in s], "k:", marker="x",
                label=f"c = t_draft/t_target ({t})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("cost relative to 1-token target step")
    ax.set_title("verification is ~free only at small batch")
    ax.legend(fontsize=6)
    save(fig, "gpu_spec_cost")


def fig_dist():
    rows = load("dist")
    if not rows:
        return
    fig, axs = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 3.6), squeeze=False)
    for ax, r in zip(axs[0], rows):
        k = 15
        x = np.arange(k)
        ax.bar(x - 0.3, r["top_exact_p"][:k], 0.2, label="exact target p")
        ax.bar(x - 0.1, r["top_spec_freq"][:k], 0.2, label=f"speculative ({r['trials']} runs)")
        ax.bar(x + 0.1, r["top_base_freq"][:k], 0.2, label="target-only sampling")
        ax.bar(x + 0.3, r["top_exact_q"][:k], 0.2, label="draft q")
        ax.set_xticks(x, [str(t) for t in r["top_tokens"][:k]], rotation=60, fontsize=6)
        ax.set_title(f"T={r['temperature']}: χ² p(spec vs p)={r['tok1_spec_vs_p_pvalue']:.2f}, "
                     f"p(spec vs q)={r['tok1_spec_vs_q_pvalue']:.1e}", fontsize=8)
        ax.set_xlabel("token id (top-15 of target)")
        ax.set_ylabel("probability")
        ax.legend(fontsize=6)
    save(fig, "gpu_spec_distribution")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_parity()
    fig_latency()
    fig_model_size()
    fig_prefill_decode()
    fig_batch()
    fig_overhead()
    fig_memory()
    fig_spec()
    fig_spec_batch()
    fig_dist()


if __name__ == "__main__":
    main()
