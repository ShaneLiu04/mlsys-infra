"""Figures (PNG + PDF) and a summary JSON from artifacts/raw/gpu/*.jsonl.

python scripts/plot_gpu.py      # runs locally, no GPU needed
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LAB / "gpu"), str(LAB / "code"), str(LAB.parents[2] / "tools")]
import mlsys_calc  # noqa: E402
from gpt import CONFIGS, activation_bytes_model, amp_weight_copy_bytes, logits_bytes  # noqa: E402

RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
GB = 1e9
GiB = 2**30
NOMINAL = 209.5e12
MODES = ["fp32", "amp_bf16", "bf16", "bf16_master"]
MODE_LABEL = {"fp32": "fp32", "amp_bf16": "AMP bf16 (fp32 weights)", "bf16": "pure bf16",
              "bf16_master": "bf16 + fp32 master"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def load(name: str) -> list[dict]:
    p = RAW / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(fig, name: str, tight: bool = True) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def act_model_total(r: dict) -> float:
    c = CONFIGS[r["model"]]
    t = activation_bytes_model(c, r["batch"], r["seq"], r["mode"]) + logits_bytes(c, r["batch"], r["seq"])
    if r["mode"] == "amp_bf16":
        t += amp_weight_copy_bytes(c)
    return t


# --------------------------------------------------------------------------- hw
def fig_hw(summary: dict) -> None:
    rows = load("hw")
    if not rows:
        return
    gem = [r for r in rows if r["kind"] == "gemm"]
    summary["gemm_tflops_max"] = max(r["tflops"] for r in gem)
    summary["gemm"] = {r["n"]: r["tflops"] for r in gem}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, lab in (("d2d_copy", "D2D copy (read+write)"), ("reduce_add", "y += x (2R+1W)"),
                      ("h2d_pinned", "H2D pinned (PCIe)"), ("d2h_pinned", "D2H pinned (PCIe)")):
        rr = [r for r in rows if r["kind"] == kind]
        ax[0].loglog([r["bytes"] for r in rr], [r["gbps"] for r in rr], "o-", label=lab, ms=4)
        summary[f"{kind}_gbps_max"] = max(r["gbps"] for r in rr)
        summary[f"{kind}_gbps_1GiB"] = rr[-1]["gbps"]
        summary[f"{kind}_t_small_us"] = rr[0]["time_s"] * 1e6
    ax[0].axhline(1792, color="k", ls=":", lw=1)
    ax[0].text(5e3, 1900, "GDDR7 spec 1.79 TB/s", fontsize=8)
    ax[0].axvline(96 * 2**20, color="gray", ls="--", lw=0.8)
    ax[0].text(1.1e8, 3, "96 MB L2", fontsize=8, color="gray")
    ax[0].set_xlabel("message size (bytes)")
    ax[0].set_ylabel("GB/s")
    ax[0].set_title("Single-GPU bandwidth vs size (CUDA events)")
    ax[0].legend(fontsize=7)
    ring = [r for r in rows if r["kind"] == "ring_1gpu"]
    for P in sorted({r["P"] for r in ring}):
        rr = [r for r in ring if r["P"] == P]
        x = np.array([r["bytes"] for r in rr], float)
        y = np.array([r["time_s"] for r in rr])
        ax[1].loglog(x, y * 1e6, "o", label=f"P={P} measured", ms=4)
        # alpha from small messages (launch-bound), beta from the largest message
        alpha = np.median(y[x <= 2**20]) / (2 * (P - 1))
        bus = np.array([r["model_bus_bytes"] for r in rr])
        beta = (y[-1] - 2 * (P - 1) * alpha) / bus[-1]
        model = 2 * (P - 1) * alpha + bus * beta
        ax[1].loglog(x, model * 1e6, "-", lw=1, color=ax[1].lines[-1].get_color(),
                     label=f"α={alpha*1e6:.1f}µs/step, 1/β={1/beta/1e9:.0f} GB/s")
        summary[f"ring1gpu_P{P}"] = {"alpha_us": alpha * 1e6, "bus_gbps": 1 / beta / 1e9,
                                     "max_err": max(r["max_err"] for r in rr)}
    ax[1].set_xlabel("all-reduce size S (bytes)")
    ax[1].set_ylabel("time (µs)")
    ax[1].set_title("Ring all-reduce emulated on one GPU (P buffers)")
    ax[1].legend(fontsize=7)
    save(fig, "gpu_bandwidth_ring")


# --------------------------------------------------------------------------- memory
def fig_memory(summary: dict) -> None:
    rows = [r for r in load("memory") if r["kind"] == "memory"]
    if not rows:
        return
    models = [m for m in CONFIGS if any(r["model"] == m for r in rows)]
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=False)
    axes = np.atleast_1d(axes)
    comp = [("weights", "weights"), ("grads", "grads"), ("optimizer", "optimizer (+master)"),
            ("activations", "activations (incl. logits)")]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for ax, mname in zip(axes, models):
        rr = {r["mode"]: r for r in rows if r["model"] == mname}
        x = np.arange(len(MODES))
        bottom = np.zeros(len(MODES))
        for (k, lab), col in zip(comp, colors):
            v = np.array([rr[m][k] / GB if m in rr else 0 for m in MODES])
            ax.bar(x, v, bottom=bottom, color=col, label=lab, width=0.65)
            bottom += v
        pk = [rr[m]["peak_total"] / GB for m in MODES]
        ax.plot(x, pk, "k_", ms=22, mew=2, label="measured peak")
        n = rr["fp32"]["params"]
        ax.axhline(16 * n / GB, color="purple", ls="--", lw=1, label="16N (mlsys_calc)")
        ax.set_xticks(x, ["fp32", "AMP", "bf16", "bf16+\nmaster"])
        r0 = rr["fp32"]
        ax.set_title(f"{mname}  (b={r0['batch']}, s={r0['seq']})")
        ax.set_ylabel("GB")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.5, 0.065, "activations = saved after forward; at peak (backward) part of them is already freed while grads "
             "are allocated, so the stacked sum can exceed the measured peak", ha="center", fontsize=7)
    save(fig, "gpu_memory_breakdown", tight=False)
    summary["memory"] = [{k: r[k] for k in ("model", "mode", "params", "batch", "seq", "weights", "grads",
                                            "optimizer", "activations", "peak_total", "bytes_per_param_static",
                                            "step_s", "tokens_per_s", "mfu_nominal")} |
                         {"act_model_total": act_model_total(r)} for r in rows]

    # bytes/param by component vs mlsys_calc and activation model check
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.2
    for i, m in enumerate(MODES):
        rr = [r for r in rows if r["mode"] == m]
        x = np.arange(len(rr)) + (i - 1.5) * width
        ax[0].bar(x, [r["bytes_per_param_static"] for r in rr], width, label=MODE_LABEL[m])
    ax[0].axhline(16, color="purple", ls="--", lw=1)
    ax[0].text(-0.4, 16.4, "mlsys_calc.training_memory_bytes: 2+2+12 = 16 B/param", fontsize=7, color="purple")
    ax[0].set_xticks(range(len(models)), models)
    ax[0].set_ylabel("(weights+grads+optimizer) / N  [bytes]")
    ax[0].set_ylim(0, 19)
    ax[0].set_title("Static training memory per parameter")
    ax[0].legend(fontsize=7, loc="lower right")
    meas = [r["activations"] / GB for r in rows]
    pred = [act_model_total(r) / GB for r in rows]
    for m, mk in zip(MODES, "osD^"):
        sel = [i for i, r in enumerate(rows) if r["mode"] == m]
        ax[1].plot([pred[i] for i in sel], [meas[i] for i in sel], mk, label=MODE_LABEL[m])
    lim = max(meas + pred) * 1.1
    ax[1].plot([0, lim], [0, lim], "k:", lw=1)
    ax[1].set_xlabel("model: L·c·s·b·h + 4·b·s·V (+2N AMP copies)  [GB]")
    ax[1].set_ylabel("measured saved activations [GB]")
    ax[1].set_title("Activation model (c = 64 / 36 / 32 B per s·b·h)")
    ax[1].legend(fontsize=7)
    summary["act_model_ratio"] = {f"{r['model']}/{r['mode']}": r["activations"] / act_model_total(r) for r in rows}
    save(fig, "gpu_bytes_per_param")

    # MFU / throughput
    fig, ax = plt.subplots(figsize=(7, 3.6))
    peak_meas = summary.get("gemm_tflops_max", None)
    for i, m in enumerate(MODES):
        rr = [r for r in rows if r["mode"] == m]
        x = np.arange(len(rr)) + (i - 1.5) * width
        b = ax.bar(x, [r["mfu_nominal"] * 100 for r in rr], width, label=MODE_LABEL[m])
        for xi, r in zip(x, rr):
            ax.text(xi, r["mfu_nominal"] * 100 + 1, f"{r['tokens_per_s']/1e3:.0f}k", ha="center", fontsize=6)
        del b
    if peak_meas:
        ax.axhline(peak_meas * 1e12 / NOMINAL * 100, color="k", ls=":", lw=1)
        ax.text(-0.45, peak_meas * 1e12 / NOMINAL * 100 + 2, f"best bf16 GEMM = {peak_meas:.0f} TFLOP/s", fontsize=7)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("MFU vs 209.5 TFLOP/s nominal [%]")
    ax.set_title("Training-step MFU (6N + 12·L·h·s FLOPs/token); labels = tokens/s")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "gpu_mfu")


def fig_actsweep(summary: dict) -> None:
    rows = [r for r in load("actsweep") if r["kind"] == "actsweep"]
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sel = [r for r in rows if r["model"] == "gpt2-355m" and r["mode"] == "amp_bf16"] or rows
    bs = sorted({r["batch"] for r in sel})
    ss = sorted({r["seq"] for r in sel})
    grid = np.full((len(bs), len(ss)), np.nan)
    for r in sel:
        if not r["oom"]:
            grid[bs.index(r["batch"]), ss.index(r["seq"])] = r["activations"] / GB
    im = ax[0].imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    for i in range(len(bs)):
        for j in range(len(ss)):
            ax[0].text(j, i, "OOM" if np.isnan(grid[i, j]) else f"{grid[i, j]:.1f}", ha="center", va="center",
                       color="w", fontsize=8)
    ax[0].set_xticks(range(len(ss)), ss)
    ax[0].set_yticks(range(len(bs)), bs)
    ax[0].set_xlabel("sequence length s")
    ax[0].set_ylabel("batch b")
    ax[0].set_title(f"{sel[0]['model']} {sel[0]['mode']}: saved activations [GB]")
    fig.colorbar(im, ax=ax[0])
    for (mname, mode), mk in zip([("gpt2-124m", "amp_bf16"), ("gpt2-124m", "bf16"), ("gpt2-355m", "amp_bf16"),
                                  ("gpt2-355m", "bf16")], "os^D"):
        rr = sorted([r for r in rows if r["model"] == mname and r["mode"] == mode and not r["oom"]],
                    key=lambda r: r["tokens"])
        if not rr:
            continue
        tok = np.array([r["tokens"] for r in rr])
        act = np.array([r["activations"] for r in rr])
        ax[1].loglog(tok, act / GB, mk, ms=4, label=f"{mname} {mode}")
        c = CONFIGS[mname]
        per_tok = c.n_layer * {"amp_bf16": 36, "bf16": 32}[mode] * c.n_embd + 4 * c.vocab
        base = amp_weight_copy_bytes(c) if mode == "amp_bf16" else 0
        tt = np.logspace(np.log10(tok.min()), np.log10(tok.max()), 20)
        ax[1].loglog(tt, (base + per_tok * tt) / GB, "-", lw=0.8, color=ax[1].lines[-1].get_color())
        slope = np.polyfit(tok, act, 1)
        summary[f"actsweep_{mname}_{mode}"] = {"fit_bytes_per_token": slope[0], "fit_intercept": slope[1],
                                               "model_bytes_per_token": per_tok, "model_intercept": base}
    ax[1].set_xlabel("tokens per step (b·s)")
    ax[1].set_ylabel("saved activations [GB]")
    ax[1].set_title("Activations are linear in b·s (flash attention: no s² term)")
    ax[1].legend(fontsize=7)
    save(fig, "gpu_activation_sweep")


# --------------------------------------------------------------------------- recompute
def fig_recompute(summary: dict) -> None:
    rows = load("recompute")
    rc = [r for r in rows if r["kind"] == "recompute"]
    mb = [r for r in rows if r["kind"] == "maxbatch"]
    tries = [r for r in rows if r["kind"] == "maxbatch_try"]
    if not rc:
        return
    base = next(r for r in rc if r["ckpt"] == "none")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    offsets = {("full", 2): (-10, 10), ("mlp", 1): (-10, -24), ("attn", 1): (4, 6)}
    for r in rc:
        lab = r["ckpt"] + ("" if r["every"] == 1 else f" every {r['every']}")
        t = r["step_s"] / base["step_s"]
        ax[0].scatter(r["activations"] / GB, t, s=50)
        ax[0].annotate(f"{lab}: {r['peak_total']/GB:.1f} GB peak, x{t:.3f}", (r["activations"] / GB, t), fontsize=7,
                       xytext=offsets.get((r["ckpt"], r["every"]), (4, 4)), textcoords="offset points")
    ax[0].axhline(4 / 3, color="r", ls="--", lw=1, label="theory full recompute (F+B+F)/(F+B) = 4/3")
    ax[0].set_ylim(0.95, 1.4)
    ax[0].set_xlim(0, 21)
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].set_xlabel("saved activations after forward [GB]")
    ax[0].set_ylabel("step time / no-checkpoint step time")
    ax[0].set_title(f"{base['model']} AMP, b={base['batch']}, s={base['seq']}: memory vs time")
    summary["recompute"] = [{"ckpt": r["ckpt"], "every": r["every"], "activations": r["activations"],
                             "peak_total": r["peak_total"], "step_s": r["step_s"],
                             "time_ratio": r["step_s"] / base["step_s"],
                             "act_ratio": r["activations"] / base["activations"], "mfu_nominal": r["mfu_nominal"]}
                            for r in rc]
    if mb:
        labels = [r["ckpt"] + ("" if r["every"] == 1 else f"/{r['every']}") for r in mb]
        x = np.arange(len(mb))
        ax[1].bar(x - 0.2, [r["max_batch"] for r in mb], 0.4, label="max batch (s=1024)")
        ax2 = ax[1].twinx()
        ax2.bar(x + 0.2, [(r["tokens_per_s"] or 0) / 1e3 for r in mb], 0.4, color="#dd8452", label="tokens/s at max batch")
        ax2.set_ylabel("k tokens/s at max batch")
        ax2.grid(False)
        ax[1].set_xticks(x, labels)
        ax[1].set_ylabel("largest batch that fits (ladder)")
        ax[1].set_title("Max batch on 32 GB and throughput there")
        h1, l1 = ax[1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[1].set_ylim(0, max(r["max_batch"] for r in mb) * 1.35)
        ax2.set_ylim(0, max((r["tokens_per_s"] or 0) for r in mb) / 1e3 * 1.35)
        ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", ncol=2)
        summary["maxbatch"] = [{k: r[k] for k in ("ckpt", "every", "max_batch", "step_s", "tokens_per_s")} for r in mb]
        summary["maxbatch_tries"] = [{k: r.get(k) for k in ("ckpt", "every", "batch", "ok", "peak_total", "tokens_per_s")}
                                     for r in tries]
    save(fig, "gpu_recompute")


# --------------------------------------------------------------------------- zero
def fig_zero(summary: dict) -> None:
    rows = load("zero")
    zm = [r for r in rows if r["kind"] == "zero_mem"]
    za = [r for r in rows if r["kind"] == "zero_adam"]
    if not zm:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for st, mk in zip(range(4), "osD^"):
        rr = sorted([r for r in zm if r["stage"] == st], key=lambda r: r["nd"])
        nd = [r["nd"] for r in rr]
        ax[0].loglog(nd, [r["total"] / GB for r in rr], mk, ms=6, label=f"ZeRO-{st} measured")
        ax[0].loglog(nd, [r["calc_total"] / GB for r in rr], "-", lw=1, color=ax[0].lines[-1].get_color())
    ax[0].set_xlabel("data-parallel degree N_d")
    ax[0].set_ylabel("per-rank model state [GB]")
    ax[0].set_title(f"{zm[0]['model']} ({zm[0]['params']/1e6:.0f}M): measured (markers) vs mlsys_calc (lines)")
    ax[0].legend(fontsize=7)
    summary["zero_mem"] = [{k: r[k] for k in ("nd", "stage", "weights", "grads", "optimizer", "total", "calc_total")}
                           for r in zm]
    summary["zero_max_rel_err"] = max(abs(r["total"] - r["calc_total"]) / r["calc_total"] for r in zm)
    if za:
        za = sorted(za, key=lambda r: r["nd"])
        ax[1].loglog([r["nd"] for r in za], [r["rank_step_s_median"] * 1e3 for r in za], "o-", label="measured AdamW shard step")
        t1 = za[0]["rank_step_s_median"]
        ax[1].loglog([r["nd"] for r in za], [t1 * 1e3 / r["nd"] for r in za], "k:", label="ideal 1/N_d")
        for r in za:
            ax[1].annotate(f"err={r['max_abs_err']:.0e}\n{r['bytes_touched_per_rank']/r['rank_step_s_median']/1e9:.0f} GB/s",
                           (r["nd"], r["rank_step_s_median"] * 1e3), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax[1].set_xlabel("N_d")
        ax[1].set_ylabel("optimizer step per rank [ms]")
        ax[1].set_title("ZeRO-1 sharded AdamW: time and equality to full step")
        ax[1].legend(fontsize=7)
        summary["zero_adam"] = za
    save(fig, "gpu_zero")


def fig_zero_comm(summary: dict) -> None:
    """ZeRO stage communication time per step with measured link numbers (PCIe) vs H100 NVLink."""
    mem = [r for r in load("zero") if r["kind"] == "zero_mem"]
    hw = load("hw")
    if not mem or not hw:
        return
    N = mem[0]["params"]
    pcie = max(r["gbps"] for r in hw if r["kind"] == "h2d_pinned") * 1e9
    links = {"PCIe5 x16 (measured H2D)": pcie, "H100 NVLink 450 GB/s": mlsys_calc.GPUS["h100"].link_bw}
    nds = [2, 4, 8, 16, 64]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for (lab, bw), ls in zip(links.items(), ("-", "--")):
        for st, vol in ((0, 2.0), (3, 3.0)):  # all-reduce = RS+AG = 2Ψ ; ZeRO-3 = 3Ψ (bf16 Ψ bytes = 2N)
            t = [vol * (nd - 1) / nd * 2 * N / bw for nd in nds]
            ax.semilogx(nds, np.array(t) * 1e3, ls, marker="o", ms=3,
                        label=f"{'DP/ZeRO-1/2' if st == 0 else 'ZeRO-3'} — {lab}")
    ax.set_xlabel("N_d")
    ax.set_ylabel("gradient/param comm per step [ms]")
    ax.set_title(f"β-only ring model for {N/1e6:.0f}M params in bf16")
    ax.legend(fontsize=7)
    summary["zero_comm_pcie_gbps"] = pcie / 1e9
    summary["zero_comm_ms_nd8_pcie_dp"] = 2 * 7 / 8 * 2 * N / pcie * 1e3
    summary["zero_comm_ms_nd8_pcie_z3"] = 3 * 7 / 8 * 2 * N / pcie * 1e3
    summary["zero_comm_ms_nd8_nvlink_dp"] = 2 * 7 / 8 * 2 * N / links["H100 NVLink 450 GB/s"] * 1e3
    save(fig, "gpu_zero_comm")


# --------------------------------------------------------------------------- pipeline
def fig_pipeline(summary: dict) -> None:
    rows = load("pipeline")
    pl = [r for r in rows if r["kind"] == "pipeline"]
    gantt = [r for r in rows if r["kind"] == "gantt"]
    if not pl:
        return
    g8 = [g for g in gantt if g["m"] == 8] or gantt
    fig, axes = plt.subplots(len(g8), 1, figsize=(10, 1.6 * len(g8) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")
    for ax, g in zip(axes, g8):
        for o in g["ops"]:
            col = cmap(o["mb"] % 10 * 2 + (0 if o["kind"] == "F" else 1))
            ax.barh(o["device"], (o["end"] - o["start"]) * 1e3, left=o["start"] * 1e3, color=col,
                    edgecolor="k", lw=0.3, height=0.8)
            if o["end"] - o["start"] > 0.0015:
                ax.text((o["start"] + o["end"]) / 2 * 1e3, o["device"], f"{o['kind']}{o['mb']}", ha="center",
                        va="center", fontsize=6)
        rec = next(r for r in pl if r["schedule"] == g["schedule"] and r["m"] == g["m"])
        ax.set_yticks(range(g["p"]), [f"stage {d}" for d in range(g["p"])])
        ax.invert_yaxis()
        ax.set_title(f"{g['schedule'].upper()} p={g['p']} m={g['m']}: measured bubble "
                     f"{rec['bubble_measured']:.3f} vs (p−1)/(m+p−1) = {rec['bubble_formula']:.3f}", fontsize=9)
    axes[-1].set_xlabel("time (ms), op durations from CUDA events")
    save(fig, "gpu_pipeline_gantt")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for r in pl:
        if not r["oom"]:
            T = np.array(r["tf_median_s"]) + np.array(r["tb_median_s"])
            mk_model = (r["m"] - 1) * T.max() + T.sum()
            r["makespan_imbalance_model_s"] = mk_model
            r["bubble_imbalance_model"] = 1 - r["m"] * T.sum() / (r["p"] * mk_model)
    for sched, mk in (("gpipe", "o"), ("1f1b", "s")):
        rr = sorted([r for r in pl if r["schedule"] == sched and not r["oom"]], key=lambda r: r["m"])
        ax[0].plot([r["m"] for r in rr], [r["bubble_measured"] for r in rr], mk + "-", label=f"{sched} measured")
        ax[0].plot([r["m"] for r in rr], [r["bubble_imbalance_model"] for r in rr], mk, mfc="none", ms=9,
                   color=ax[0].lines[-1].get_color(), label=f"{sched}: (m−1)·T_max+ΣT_d model")
    mm = np.array(sorted({r["m"] for r in pl}))
    ax[0].plot(mm, [mlsys_calc.pipeline_bubble(4, int(m)) for m in mm], "k:", label="(p−1)/(m+p−1)")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("micro-batches m (p=4)")
    ax[0].set_ylabel("bubble fraction")
    ax[0].set_title("Bubble: measured-duration replay vs formula")
    ax[0].legend(fontsize=7)
    for sched, mk in (("gpipe", "o"), ("1f1b", "s")):
        rr = sorted([r for r in pl if r["schedule"] == sched], key=lambda r: r["m"])
        ok = [r for r in rr if not r["oom"]]
        ax[1].plot([r["m"] for r in ok], [r["peak_alloc_over_weights"] / GB for r in ok], mk + "-",
                   label=f"{sched}: peak allocated (all 4 stages on 1 GPU)")
        ax[1].plot([r["m"] for r in ok], [r["inflight_act_bytes_per_stage"][0] / GB for r in ok], mk + "--",
                   label=f"{sched}: stage-0 in-flight activations")
        for r in rr:
            if r["oom"]:
                ax[1].annotate("OOM", (r["m"], ax[1].get_ylim()[1] * 0.9 if ax[1].lines else 1), color="r", fontsize=8)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("micro-batches m")
    ax[1].set_ylabel("GB")
    ax[1].set_title("Activation memory: GPipe ∝ m, 1F1B ≤ p")
    ax[1].legend(fontsize=6)
    save(fig, "gpu_pipeline_bubble_memory")
    summary["pipeline"] = [{k: r.get(k) for k in ("schedule", "m", "oom", "bubble_measured", "bubble_formula",
                                                    "bubble_imbalance_model", "makespan_imbalance_model_s",
                                                    "makespan_s", "serial_s", "peak_alloc_over_weights",
                                                    "inflight_peak_per_stage", "stage_act_bytes", "tf_median_s",
                                                    "tb_median_s", "tokens_per_s_pipelined")} for r in pl]

    mt = [r for r in rows if r["kind"] == "memtrace" and r["m"] == 8]
    if mt:
        fig, ax = plt.subplots(figsize=(8, 3.4))
        for r in mt:
            a = np.array([t["allocated"] for t in r["trace"]]) / GB
            ax.plot(np.arange(len(a)), a, label=r["schedule"])
        ax.set_xlabel("op index in execution order (F and B of all stages)")
        ax.set_ylabel("allocated [GB]")
        ax.set_title("Memory over one pipelined step, p=4, m=8 (one GPU executes all stages)")
        ax.legend()
        save(fig, "gpu_pipeline_memtrace")


def main() -> int:
    summary: dict = {}
    for f in (fig_hw, fig_memory, fig_actsweep, fig_recompute, fig_zero, fig_zero_comm, fig_pipeline):
        f(summary)
    (RAW / "summary_gpu.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
