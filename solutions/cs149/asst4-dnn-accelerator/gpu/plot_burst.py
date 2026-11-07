"""Plot burst_probe.py output: per-kernel GEMM rate after idle, its distribution, and the effect of idle time.

    python gpu/plot_burst.py [--card rtx6000d]
Reads artifacts/raw/<card>/burst.jsonl (and burst_smi.csv if present); writes burst_summary.csv next to it
and artifacts/figures/<card>/burst.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1]


def rates(r: dict) -> np.ndarray:
    return r["flop"] / np.asarray(r["dur_ms"]) / 1e9


def summarise(r: dict) -> dict:
    tf = rates(r)
    start, dur = np.asarray(r["start_ms"]), np.asarray(r["dur_ms"])
    span = start[-1] + dur[-1]
    first = start + dur <= 2.0
    return dict(dtype=r["dtype"], N=r["N"], idle_s=r["idle_s"], kernels=r["kernels"], span_ms=round(span, 1),
                mean_tflops=round(r["kernels"] * r["flop"] / span / 1e9, 1),
                median_kernel=round(float(np.median(tf)), 1), p90_kernel=round(float(np.percentile(tf, 90)), 1),
                max_kernel=round(float(tf.max()), 1),
                first2ms_tflops=round(float(first.sum() * r["flop"] / max(dur[first].sum(), 1e-9) / 1e9), 1)
                if first.any() else None,
                frac_kernels_above_200=round(float((tf > 200).mean()), 3))


def smi_stats(p: Path) -> dict:
    if not p.exists():
        return {}
    sm, reasons = [], set()
    for rec in csv.reader(open(p, encoding="utf-8")):
        try:
            sm.append(float(rec[1]))
            reasons.add(rec[3].strip())
        except (ValueError, IndexError):
            continue
    return dict(samples=len(sm), sm_min=min(sm), sm_max=max(sm), reasons=sorted(reasons))


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="rtx6000d")
    args = ap.parse_args()
    raw = LAB / "artifacts/raw" / args.card
    rows = [json.loads(x) for x in (raw / "burst.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    gpu = next(r["gpu"] for r in rows if r["kind"] == "run")
    tr = [r for r in rows if r["kind"] == "trace"]
    summ = [summarise(r) for r in tr]
    with open(raw / "burst_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    smi = smi_stats(raw / "burst_smi.csv")
    for s in summ:
        print(s)
    print("smi", smi)

    colors = {2048: "#4C78A8", 4096: "#F58518", 8192: "#54A24B"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
    for r in tr:
        if r["idle_s"] != 3.0 or r["dtype"] != "bf16":
            continue
        tf = rates(r)
        t = np.asarray(r["start_ms"]) + np.asarray(r["dur_ms"])
        col = colors.get(r["N"], "k")
        a.plot(t, tf, ".", ms=1.6, color=col, alpha=0.35)
        win = 10.0
        edges = np.arange(0, t[-1] + win, win)
        idx = np.digitize(t, edges)
        mt = [(edges[i - 1] + win / 2, r["flop"] * (idx == i).sum() / np.asarray(r["dur_ms"])[idx == i].sum() / 1e9)
              for i in np.unique(idx)]
        a.plot(*zip(*mt), "-", color=col, lw=1.6, label=f"N={r['N']} (line = 10 ms mean)")
    a.set_xscale("log")
    a.set_xlabel("time since the first kernel after 3 s idle (ms, log)")
    a.set_ylabel("TFLOP/s per torch.mm (bf16, CUDA events)")
    a.set_ylim(0, 340)
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=7, loc="lower left")
    a.set_title("Single kernels hit ~300; the running mean does not", fontsize=9)
    if smi:
        a.text(0.99, 0.98, f"nvidia-smi every 20 ms: SM {smi['sm_min']:.0f}–{smi['sm_max']:.0f} MHz,\n"
               f"throttle reasons {', '.join(smi['reasons'])} ({smi['samples']} samples)",
               transform=a.transAxes, ha="right", va="top", fontsize=6.5,
               bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    bins = np.linspace(0, 340, 69)
    for r in tr:
        if r["idle_s"] == 3.0 and r["dtype"] == "bf16":
            b.hist(rates(r), bins=bins, histtype="step", lw=1.4, color=colors.get(r["N"], "k"), density=True,
                   label=f"N={r['N']} ({r['kernels']} kernels)")
    b.set_xlabel("TFLOP/s of one kernel")
    b.set_ylabel("density")
    b.set_title("Per-kernel rate is spread, not one throttled level", fontsize=9)
    b.legend(fontsize=7)
    b.grid(True, alpha=0.3)

    for dtype, m in [("bf16", "o"), ("fp16", "s")]:
        sel = sorted((s for s in summ if s["N"] == 4096 and s["dtype"] == dtype), key=lambda s: s["idle_s"])
        x = [max(s["idle_s"], 1e-3) for s in sel]
        c.plot(x, [s["mean_tflops"] for s in sel], marker=m, label=f"{dtype} mean over trace")
        c.plot(x, [s["max_kernel"] for s in sel], marker=m, ls=":", label=f"{dtype} fastest kernel")
    c.set_xscale("log")
    c.set_ylim(0, 340)
    c.set_xlabel("idle before the trace (s, log; 0 plotted at 1 ms)")
    c.set_ylabel("TFLOP/s (N = 4096)")
    c.set_title("Idle time does not buy a burst", fontsize=9)
    c.legend(fontsize=7)
    c.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"{gpu}: per-kernel tensor-core throughput (torch.mm, ~0.4 s of work per trace)", fontsize=10)
    fig.tight_layout()
    out = LAB / "artifacts/figures" / args.card
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "burst.png", dpi=150)
    print("wrote", out / "burst.png")


if __name__ == "__main__":
    main()
