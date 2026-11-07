"""Quantization vs model scale: GPT-2 124M / 355M (RTX 5090) against Qwen2.5-7B (RTX 6000D).

Reads artifacts/raw/gpu/quant_ppl_gpt2*.jsonl and artifacts/raw/rtx6000d/quant_ppl_Qwen2.5-7B.jsonl,
writes artifacts/figures/gpu_quant_scale.png and artifacts/raw/rtx6000d/quant_scale_summary.csv.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
FIG = ROOT / "artifacts" / "figures"
SOURCES = [("GPT-2 124M", RAW / "gpu" / "quant_ppl_gpt2.jsonl", "#9ecae1"),
           ("GPT-2 355M", RAW / "gpu" / "quant_ppl_gpt2-medium.jsonl", "#4292c6"),
           ("Qwen2.5-7B", RAW / "rtx6000d" / "quant_ppl_Qwen2.5-7B.jsonl", "#E45756")]
GROUPS = [("weight only, 8 bit", ["w8_tensor", "w8_channel"]),
          ("weight only, 4 bit", ["w4_channel", "gptq_w4_channel", "w4_g128", "awq_w4_g128", "gptq_w4_g128"]),
          ("weight only, 3 bit", ["w3_g128", "awq_w3_g128", "gptq_w3_g128"]),
          ("W8A8 (simulated)", ["w8a8_tensor", "w8a8_token", "smoothquant_tensor", "smoothquant_tensor_a0.8",
                                "smoothquant_token", "smoothquant_token_a0.8", "llm_int8"])]
LABEL = {"w8_tensor": "per-tensor", "w8_channel": "per-channel", "w4_channel": "RTN ch", "gptq_w4_channel": "GPTQ ch",
         "w4_g128": "RTN g128", "awq_w4_g128": "AWQ g128", "gptq_w4_g128": "GPTQ g128", "w3_g128": "RTN g128",
         "awq_w3_g128": "AWQ g128", "gptq_w3_g128": "GPTQ g128", "w8a8_tensor": "per-tensor", "w8a8_token": "per-token",
         "smoothquant_tensor": "SQ α.5 tensor", "smoothquant_tensor_a0.8": "SQ α.8 tensor",
         "smoothquant_token": "SQ α.5 token", "smoothquant_token_a0.8": "SQ α.8 token", "llm_int8": "LLM.int8"}


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def block_index(layer: str) -> int:
    return int(re.search(r"(?:\.h\.|layers\.)(\d+)\.", layer).group(1))


def main() -> None:
    data = {}
    for name, p, _ in SOURCES:
        rows = load(p)
        ppl = {r["method"]: r["ppl"] for r in rows if r.get("part") == "ppl"}
        out = [r for r in rows if r.get("part") == "outliers"]
        if ppl:
            data[name] = (ppl, out)

    summary = []
    for name, (ppl, _) in data.items():
        for m, v in ppl.items():
            summary.append({"model": name, "method": m, "ppl": round(v, 4),
                            "rel_increase_pct": round(100 * (v / ppl["fp16"] - 1), 3)})
    out_csv = RAW / "rtx6000d" / "quant_scale_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.48, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    meths = [m for _, ms in GROUPS for m in ms]
    x = np.arange(len(meths), dtype=float)
    gap = 0.0
    for gi, (_, ms) in enumerate(GROUPS):
        for m in ms:
            x[meths.index(m)] += gap
        gap += 0.8
    wdt = 0.8 / len(SOURCES)
    for i, (name, _, c) in enumerate(SOURCES):
        if name not in data:
            continue
        ppl = data[name][0]
        ys = [100 * (ppl[m] / ppl["fp16"] - 1) if m in ppl else np.nan for m in meths]
        bars = ax.bar(x + (i - 1) * wdt, [max(y, 0.01) if y == y else 0 for y in ys], wdt, color=c, label=name,
                      edgecolor="white", lw=0.4)
        for b, y in zip(bars, ys):
            if y == y:
                ax.text(b.get_x() + b.get_width() / 2, max(y, 0.01) * 1.15, f"{y:.3g}", ha="center", va="bottom",
                        fontsize=5.5, rotation=90, color="#333")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 3e4)
    ax.set_xticks(x, [LABEL[m] for m in meths], rotation=35, ha="right", fontsize=8)
    for gi, (title, ms) in enumerate(GROUPS):
        xs = [x[meths.index(m)] for m in ms]
        ax.text(np.mean(xs), 1.2e4, title, ha="center", fontsize=8.5, weight="bold", color="#444")
        if gi:
            ax.axvline(xs[0] - 0.8, color="#bbb", lw=0.8)
    for yv, t in [(1, "+1%"), (10, "+10%"), (100, "2×")]:
        ax.axhline(yv, color="#999", lw=0.6, ls=":")
        ax.text(x[-1] + 0.55, yv, t, fontsize=7, va="center", color="#777")
    ax.set_ylabel("perplexity increase over fp16 (%, log)")
    ax.set_title("Same quantizers, three model sizes: wikitext-2 perplexity increase over fp16", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3, frameon=False)

    axo = fig.add_subplot(gs[1, 0])
    axr = fig.add_subplot(gs[1, 1])
    for name, _, c in SOURCES:
        if name not in data or not data[name][1]:
            continue
        out = data[name][1]
        nb = max(block_index(r["layer"]) for r in out) + 1
        best = {}
        for r in out:
            b = block_index(r["layer"])
            if r["absmax"] > best.get(b, {"absmax": -1})["absmax"]:
                best[b] = r
        bs = sorted(best)
        depth = [(b + 0.5) / nb for b in bs]
        axo.plot(depth, [best[b]["absmax"] for b in bs], marker=".", color=c, label=name, lw=1.2)
        axr.plot(depth, [best[b]["ratio"] for b in bs], marker=".", color=c, label=name, lw=1.2)
        if name.startswith("Qwen"):
            top = max(bs, key=lambda b: best[b]["absmax"])
            axo.annotate(f"{best[top]['layer'].split('.', 1)[1]}\n|x| = {best[top]['absmax']:.0f}",
                         (depth[bs.index(top)], best[top]["absmax"]), xytext=(-160, -2), textcoords="offset points",
                         fontsize=7, color=c, arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
    axo.set_yscale("log")
    axo.set_ylim(4, 3e4)
    axo.axhline(6, color="#777", lw=0.7, ls="--")
    axo.text(0.01, 6.6, "LLM.int8 threshold 6", fontsize=7, color="#555")
    axo.set_xlabel("relative depth (block / #blocks)")
    axo.set_ylabel("max |x| over input channels (log)")
    axo.set_title("Largest activation entering any Linear, per block", fontsize=9)
    axo.legend(fontsize=7, loc="upper left")
    axr.set_yscale("log")
    axr.set_xlabel("relative depth (block / #blocks)")
    axr.set_ylabel("max |x| / median channel max (log)")
    axr.set_title("Outlier ratio: how far one channel sticks out (= per-tensor A8 step inflation)", fontsize=9)
    axr.legend(fontsize=7, loc="upper left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "gpu_quant_scale.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG / "gpu_quant_scale.pdf", bbox_inches="tight")
    print("wrote", FIG / "gpu_quant_scale.png", out_csv)


if __name__ == "__main__":
    main()
