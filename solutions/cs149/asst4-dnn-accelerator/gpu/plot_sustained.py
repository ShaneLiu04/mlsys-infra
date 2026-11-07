"""Plot and summarise sustained_probe.py output: TFLOP/s, SM clock and power on one time axis.

    python gpu/plot_sustained.py [--card rtx6000d]
Reads artifacts/raw/<card>/sustained.jsonl and sustained_smi.csv; writes sustained_summary.csv next to
them and artifacts/figures/<card>/sustained.png. No GPU needed.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median

LAB = Path(__file__).resolve().parents[1]


def load(raw: Path) -> tuple[dict, list[dict], list[dict]]:
    rows = [json.loads(l) for l in (raw / "sustained.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    run = next(r for r in rows if r["kind"] == "run")
    smi = []
    with open(raw / "sustained_smi.csv", encoding="utf-8") as f:
        for rec in csv.reader(f):
            if len(rec) < 6:
                continue
            try:
                ts = datetime.strptime(rec[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                smi.append(dict(ts=ts, sm=float(rec[1]), mem=float(rec[2]), power=float(rec[3]),
                                temp=float(rec[4]), reasons=rec[5].strip()))
            except ValueError:
                continue
    t_first = smi[0]["ts"]
    for s in smi:  # nvidia-smi starts just before the probe's t0, so its first sample is t ~ 0
        s["t_s"] = (s["ts"] - t_first).total_seconds()
    return run, [r for r in rows if r["kind"] in ("phase", "window")], smi


def summarise(rows: list[dict], smi: list[dict]) -> list[dict]:
    phases = [r for r in rows if r["kind"] == "phase"]
    out = []
    for i, p in enumerate(phases):
        end = phases[i + 1]["t_s"] if i + 1 < len(phases) else float("inf")
        w = [r for r in rows if r["kind"] == "window" and r["dtype"] == p["dtype"] and r["N"] == p["N"]]
        tf = [r["tflops"] for r in w]
        t_lo, t_hi = w[0]["t_s"], w[-1]["t_s"]
        s = [x for x in smi if t_lo <= x["t_s"] <= min(t_hi, end)]
        out.append(dict(dtype=p["dtype"], N=p["N"], windows=len(w), first_tflops=tf[0], max_tflops=max(tf),
                        median_tflops=median(tf), last_tflops=tf[-1], last_over_max=tf[-1] / max(tf),
                        median_sm_mhz=median(x["sm"] for x in s) if s else None,
                        min_sm_mhz=min(x["sm"] for x in s) if s else None,
                        median_power_w=median(x["power"] for x in s) if s else None,
                        max_power_w=max(x["power"] for x in s) if s else None,
                        reasons=sorted({x["reasons"] for x in s})))
    return out


def plot(rows: list[dict], smi: list[dict], summary: list[dict], gpu: str, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    a, b, c = axes
    colors = {"bf16": "#4C78A8", "fp16": "#E45756"}
    for p in summary:
        w = [r for r in rows if r["kind"] == "window" and r["dtype"] == p["dtype"] and r["N"] == p["N"]]
        a.plot([r["t_s"] for r in w], [r["tflops"] for r in w], ".-", ms=3, lw=1, color=colors[p["dtype"]])
        a.annotate(f"{p['dtype']} N={p['N']}\nmax {p['max_tflops']:.0f}, last {p['last_tflops']:.0f}",
                   (w[0]["t_s"], p["max_tflops"]), fontsize=6.5, xytext=(0, 4), textcoords="offset points")
    a.set_ylabel("TFLOP/s per 0.25 s window")
    a.set_ylim(0, max(p["max_tflops"] for p in summary) * 1.25)
    a.set_title(f"{gpu}: back-to-back torch.mm for 8 s per shape, 5 s rest between", fontsize=10)
    ts = [x["t_s"] for x in smi]
    b.plot(ts, [x["sm"] for x in smi], lw=0.8, color="k")
    b.set_ylabel("SM clock (MHz)")
    c.plot(ts, [x["power"] for x in smi], lw=0.8, color="#F58518")
    c.axhline(600, color="r", ls="--", lw=0.8, label="600 W limit")
    c.set_ylabel("power (W)")
    c.set_xlabel("time since start (s)")
    c.legend(fontsize=7, loc="upper right")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="rtx6000d")
    args = ap.parse_args()
    raw = LAB / "artifacts" / "raw" / args.card
    run, rows, smi = load(raw)
    summary = summarise(rows, smi)
    with open(raw / "sustained_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for s in summary:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else ";".join(v) if isinstance(v, list) else v)
                        for k, v in s.items()})
    plot(rows, smi, summary, run.get("gpu", args.card), LAB / "artifacts" / "figures" / args.card / "sustained.png")
    for s in summary:
        print(f"{s['dtype']} N={s['N']:>5}: max {s['max_tflops']:6.1f} last {s['last_tflops']:6.1f} "
              f"({s['last_over_max']:.2f})  SM {s['median_sm_mhz']} MHz  {s['median_power_w']} W  {s['reasons']}")


if __name__ == "__main__":
    main()
