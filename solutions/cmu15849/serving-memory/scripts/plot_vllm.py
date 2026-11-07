"""Hand-written engine (continuous / chunked) against vLLM 0.11 (eager / CUDA graph) on one card.

    python scripts/plot_vllm.py [--card rtx6000d]
Reads artifacts/raw/<card>/engine_*.jsonl; writes artifacts/raw/<card>/vllm_compare.csv and
artifacts/figures/<card>/vllm_compare.png.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
POLICIES = [("continuous", "own engine, continuous", "#9ecae1", "o"), ("chunked", "own engine, chunked prefill", "#4292c6", "s"),
            ("vllm_eager", "vLLM 0.11, eager", "#fdae6b", "^"), ("vllm_cudagraph", "vLLM 0.11, CUDA graph", "#E45756", "D")]
METRICS = [("throughput_tok_s", "output tok/s", "Throughput"), ("tpot_mean_ms", "mean TPOT (ms)", "Time per output token"),
           ("ttft_p99_s", "p99 TTFT (s)", "Tail time-to-first-token")]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="rtx6000d")
    args = ap.parse_args()
    raw = LAB / "artifacts/raw" / args.card
    rows = []
    for pol, *_ in POLICIES:
        p = raw / f"engine_{pol}.jsonl"
        if p.exists():
            rows += [r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines()) if r.get("kind") == "engine"]
    keep = ["policy", "rate"] + [m for m, *_ in METRICS] + ["ttft_p50_s", "tpot_p99_ms", "itl_p99_ms"]
    with open(raw / "vllm_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keep, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (m, ylab, title) in zip(axes, METRICS):
        for pol, lab, c, mk in POLICIES:
            sel = sorted((r for r in rows if r["policy"] == pol), key=lambda r: r["rate"])
            if sel:
                ax.plot([r["rate"] for r in sel], [r[m] for r in sel], marker=mk, color=c, label=lab, lw=1.6)
        ax.set_xlabel("Poisson arrival rate (req/s)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([10, 20, 40])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)
    axes[0].legend(fontsize=7)
    fig.suptitle("Qwen2.5-0.5B bf16, same workload and 2 GiB KV budget: own engine vs vLLM (RTX 6000D)", fontsize=10)
    fig.tight_layout()
    out = LAB / "artifacts/figures" / args.card
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "vllm_compare.png", dpi=150)
    print("wrote", out / "vllm_compare.png")


if __name__ == "__main__":
    main()
