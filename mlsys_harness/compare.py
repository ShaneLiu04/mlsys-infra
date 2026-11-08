"""Compare the RTX 5090 and RTX 4080 SUPER measurements already in the tree.

Each metric is classified by which reference its 4080/5090 ratio sits nearest,
on a log scale:

* bandwidth: the copy-bandwidth ratio
* compute: the measured bf16 GEMM-peak ratio
* invariant: 1, meaning the card changed and the number did not

The references themselves come from the copy and GEMM-peak rows, so the
classification is a prediction that the other rows then confirm.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from mlsys_harness.paths import read_raw

CLASSES = ("bandwidth", "compute", "invariant")
CLASS_ZH = {"bandwidth": "带宽", "compute": "算力", "invariant": "与卡无关"}


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    plot_label: str
    unit: str
    ada: float
    rtx: float
    expected: str

    @property
    def ratio(self) -> float:
        return self.ada / self.rtx


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _one(path: Path, **where) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    hits = [r for r in _rows(path) if all(r.get(k) == v for k, v in where.items())]
    if len(hits) != 1:
        raise FileNotFoundError(f"{path.name} {where} matched {len(hits)}")
    return hits[0]


def _lab(root: Path, rel: str) -> Path:
    return root / rel


# id -> (label, plot label, unit, expected class); the order is the table order.
METRICS = {
    "copy": ("拷贝带宽", "copy bandwidth", "GB/s", "bandwidth"),
    "scan-cub": ("CUB scan（2^28）", "CUB scan, 2^28", "GB/s", "bandwidth"),
    "fa-8192": ("FlashAttention T=8192", "FlashAttention T=8192", "TFLOP/s", "compute"),
    "cublas-fp16": ("cuBLAS fp16 GEMM N=8192", "cuBLAS fp16 N=8192", "TFLOP/s", "compute"),
    "bf16-peak": ("实测 bf16 GEMM 峰值", "measured bf16 peak", "TFLOP/s", "compute"),
    "scan-frac": ("自写 raking / CUB", "raking / CUB", "比值", "invariant"),
    "vec4-frac": ("vec4 / cuBLAS fp32", "vec4 / cuBLAS fp32", "比值", "invariant"),
    "wmma-frac": ("WMMA bf16 / cuBLAS bf16", "WMMA bf16 / cuBLAS", "比值", "invariant"),
    "spec-g4": ("推测解码 gpt2→xl γ=4 贪心", "speculative decode, gamma=4", "倍", "invariant"),
}
CARDS = {"rtx5090": "RTX 5090", "rtx4080s": "RTX 4080 SUPER", "rtx6000d": "RTX 6000D"}


def card_values(root: Path, slug: str) -> dict[str, float]:
    """Every metric in METRICS for one card, read from that card's raw tree."""
    p1 = "solutions/cs229s/project1-transformer-kernels"
    p2 = "solutions/cs229s/project2-kv-speculative"
    a3 = "solutions/cs149/asst3-scan-render"
    a4 = "solutions/cs149/asst4-dnn-accelerator"

    def raw(rel: str, name: str) -> Path:
        return read_raw(_lab(root, rel), slug) / name

    gem = raw(a4, "gemm_ladder.jsonl")

    def gemm(variant: str) -> float:
        return _one(gem, variant=variant, M=8192, N=8192, K=8192)["gflops"] / 1000.0

    attn = dict(part="fwd", method="triton", dtype="fp16", B=2, H=16, D=64,
                T=8192, causal=False, status="ok")
    scan = raw(a3, "scan.jsonl")
    n = 2**28
    cub = _one(scan, kind="scan", algo="cub", n=n)["eff_gbs"]
    rak = _one(scan, kind="scan", algo="raking", n=n)["eff_gbs"]
    spec = dict(kind="spec", draft="gpt2", target="gpt2-xl", gamma=4, temperature=0.0)
    return {
        "copy": _one(raw(p2, "hw.jsonl"), kind="copy_bw")["GBps"],
        "scan-cub": cub,
        "fa-8192": _one(raw(p1, "attention_fwd.jsonl"), **attn)["tflops"],
        "cublas-fp16": gemm("cublas_fp16"),
        "bf16-peak": json.loads(raw(p1, "peaks.json").read_text(encoding="utf-8"))["peak_bf16_flops"] / 1e12,
        "scan-frac": rak / cub,
        "vec4-frac": gemm("vec4") / gemm("cublas_fp32"),
        "wmma-frac": gemm("wmma_bf16") / gemm("cublas_bf16"),
        "spec-g4": _one(raw(p2, "spec.jsonl"), **spec)["speedup"],
    }


def refs_for(card: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    return {"bandwidth": card["copy"] / base["copy"], "compute": card["bf16-peak"] / base["bf16-peak"],
            "invariant": 1.0}


def collect(root: Path, other: str = "rtx4080s") -> tuple[list[Metric], dict[str, float]]:
    """Metrics for ``other`` against the 5090, and the three reference ratios used to classify them."""
    base, card = card_values(root, "rtx5090"), card_values(root, other)
    metrics = [Metric(mid, label, plot, unit, card[mid], base[mid], expected)
               for mid, (label, plot, unit, expected) in METRICS.items()]
    return metrics, refs_for(card, base)


def predicted_class(ratio: float, refs: dict[str, float]) -> str:
    def dist(name: str) -> float:
        return abs(math.log(ratio) - math.log(refs[name]))
    return min(CLASSES, key=dist)


def render_markdown(metrics: list[Metric], refs: dict[str, float]) -> str:
    def cell(m: Metric) -> str:
        if m.unit == "倍":
            return f"{m.rtx:.3f}", f"{m.ada:.3f}"
        if m.unit == "比值":
            return f"{m.rtx:.3f}", f"{m.ada:.3f}"
        if m.unit == "GB/s":
            return f"{m.rtx:.0f} GB/s", f"{m.ada:.0f} GB/s"
        return f"{m.rtx:.1f}", f"{m.ada:.1f}"

    lines = [
        "同一批实验在两张卡上各跑了一次，表里的比值是 4080 SUPER 除以 5090。"
        "判断一个指标受什么限制，就看它的比值离哪个参照最近："
        f"两张卡的拷贝带宽之比是 {refs['bandwidth']:.2f}，bf16 GEMM 峰值之比是 {refs['compute']:.2f}，"
        "和卡无关的指标比值应该是 1。「最近」按对数距离算。",
        "",
        "| 指标 | 5090 | 4080 SUPER | 比值 | 归类 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for m in metrics:
        a, b = cell(m)
        lines.append(f"| {m.label} | {a} | {b} | {m.ratio:.2f} | {CLASS_ZH[m.expected]} |")
    lines += [
        "",
        "怎么读这张表：FlashAttention 和 cuBLAS fp16 的比值贴近算力之比，说明它们受算力限制；"
        "CUB scan 贴近带宽之比，说明它受带宽限制。"
        "推测解码的加速倍数，以及「自写 kernel 相对厂商库」的比例，在两张卡上几乎一样。"
        "也就是说，换到一张带宽只有四成的卡上，一次验证多个 token 并没有变得更贵。",
        "",
        "一个记录上的瑕疵：4080 的 `peaks.json` 里，`spec_bf16_flops` 和 `spec_bw` 两项仍是 5090 的规格值，"
        "因为当时的 roofline 脚本把规格写死了。实测的 `peak_bf16_flops` 和 `peak_copy_bw` 是这张卡自己的，表里用的也是实测值。"
        "脚本现在按 GPU 名称查 `mlsys_harness.devices`，下一次运行会写进新目录，不覆盖这两份记录。",
        "",
    ]
    return "\n".join(lines)


def available_cards(root: Path) -> list[str]:
    out = []
    for slug in CARDS:
        try:
            card_values(root, slug)
        except (FileNotFoundError, KeyError):
            continue
        out.append(slug)
    return out


def render_cards(root: Path, slugs: list[str]) -> str:
    """One column per card, ratios against the 5090, and each card's own nearest-reference class."""
    vals = {s: card_values(root, s) for s in slugs}
    base = vals["rtx5090"]
    others = [s for s in slugs if s != "rtx5090"]
    refs = {s: refs_for(vals[s], base) for s in others}

    def fmt(unit: str, v: float) -> str:
        if unit in ("倍", "比值"):
            return f"{v:.3f}"
        return f"{v:.0f}" if unit == "GB/s" else f"{v:.1f}"

    head = ["指标", "单位", *(CARDS[s] for s in slugs), *(f"{CARDS[s]} / 5090" for s in others), "预期归类",
            *(f"{CARDS[s]} 实测归类" for s in others)]
    lines = [
        "每张卡的参照比值（相对 5090）：" + "；".join(
            f"{CARDS[s]} 带宽 {refs[s]['bandwidth']:.2f}、算力 {refs[s]['compute']:.2f}" for s in others) + "。",
        "",
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---", "---", *(["---:"] * (len(slugs) + len(others))), "---",
                           *(["---"] * len(others))]) + " |",
    ]
    for mid, (label, _plot_label, unit, expected) in METRICS.items():
        cells = [label, unit, *(fmt(unit, vals[s][mid]) for s in slugs)]
        cells += [f"{vals[s][mid] / base[mid]:.2f}" for s in others]
        cells.append(CLASS_ZH[expected])
        for s in others:
            got = predicted_class(vals[s][mid] / base[mid], refs[s])
            cells.append(CLASS_ZH[got] + ("" if got == expected else "（≠）"))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_outputs(root: Path, out_dir: Path) -> list[Metric]:
    metrics, refs = collect(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = available_cards(root)
    vals = {s: card_values(root, s) for s in cards}
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "unit", *cards, *(f"ratio_{s}" for s in cards if s != "rtx5090"), "class"])
        for mid, (label, _p, unit, expected) in METRICS.items():
            w.writerow([mid, label, unit, *(f"{vals[s][mid]:.6g}" for s in cards),
                        *(f"{vals[s][mid] / vals['rtx5090'][mid]:.4f}" for s in cards if s != "rtx5090"), expected])
    _plot(metrics, refs, out_dir / "comparison.png")
    text = render_markdown(metrics, refs)
    if len(cards) > 2:
        text += "\n## 三张卡\n\n" + render_cards(root, cards)
    (out_dir / "comparison.md").write_text(text, encoding="utf-8", newline="\n")
    return metrics


def update_hardware_doc(root: Path, metrics: list[Metric] | None = None,
                        refs: dict[str, float] | None = None) -> None:
    doc = root / "interview" / "HARDWARE-NUMBERS.md"
    body = doc.read_text(encoding="utf-8")
    if metrics is None or refs is None:
        metrics, refs = collect(root)
    section = render_markdown(metrics, refs)
    cards = available_cards(root)
    if len(cards) > 2:
        section += "\n### 加入第三张卡：RTX 6000D\n\n" + render_cards(root, cards)
    start, end = "<!-- cross-gpu:start -->", "<!-- cross-gpu:end -->"
    block = f"{start}\n## 两张卡：4080 SUPER 相对 5090\n\n{section}{end}\n"
    if start in body and end in body:
        pre, rest = body.split(start, 1)
        _, post = rest.split(end, 1)
        body = pre + block + post.lstrip("\n")
    else:
        body = body.rstrip() + "\n\n" + block
    doc.write_text(body, encoding="utf-8", newline="\n")


def _plot(metrics: list[Metric], refs: dict[str, float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"bandwidth": "#4C78A8", "compute": "#F58518", "invariant": "#54A24B"}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = list(range(len(metrics)))
    ax.barh(ys, [m.ratio for m in metrics], color=[colors[m.expected] for m in metrics], height=0.7)
    ax.axvline(refs["bandwidth"], color=colors["bandwidth"], ls="--", lw=1, label=f"bandwidth {refs['bandwidth']:.2f}")
    ax.axvline(refs["compute"], color=colors["compute"], ls="--", lw=1, label=f"compute {refs['compute']:.2f}")
    ax.axvline(1.0, color=colors["invariant"], ls="--", lw=1, label="unchanged 1.00")
    ax.set_yticks(ys)
    ax.set_yticklabels([m.plot_label for m in metrics])
    ax.set_xlabel("RTX 4080 SUPER / RTX 5090")
    ax.set_xlim(0, 1.35)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Which results move with the card")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = write_outputs(root, root / "artifacts" / "cross_gpu")
    update_hardware_doc(root, metrics)
    bad = [m for m in metrics if predicted_class(m.ratio, collect(root)[1]) != m.expected]
    if bad:
        raise SystemExit("classification mismatch: " + ", ".join(m.id for m in bad))
    print(f"wrote artifacts/cross_gpu ({len(metrics)} metrics)")


if __name__ == "__main__":
    main()
