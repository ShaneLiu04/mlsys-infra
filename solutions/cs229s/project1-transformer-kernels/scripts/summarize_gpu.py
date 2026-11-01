"""Print the key GPU tables (markdown) from artifacts/raw/gpu/*.jsonl; every README/report number comes from here."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "artifacts" / "raw" / "gpu"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 30)


def load(name):
    p = RAW / name
    if not p.exists():
        return None
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return pd.DataFrame([r for r in rows if not str(r.get("part", "")).startswith("meta")])


def show(title, df):
    print(f"\n## {title}\n")
    print(df.to_string(index=False))


def main():
    print(json.loads((RAW / "peaks.json").read_text()))
    f = load("attention_fwd.jsonl")
    if f is not None:
        cols = ["dtype", "D", "causal", "T", "method", "status", "ms", "tflops", "peak_extra_bytes", "max_err"]
        show("attention fwd", f[cols])
    b = load("attention_bwd.jsonl")
    if b is not None:
        show("attention fwd+bwd", b[[c for c in ["D", "causal", "T", "method", "status", "ms", "tflops",
                                                  "peak_extra_bytes", "max_err_dq", "max_err_dk", "max_err_dv"] if c in b]])
    h = load("attention_bh.jsonl")
    if h is not None:
        show("batch*heads", h[["causal", "B", "H", "method", "ms", "tflops"]])
    k = load("attention_blocks.jsonl")
    if k is not None:
        k["implied_TBps"] = k["model_bytes"] / k["ms"] / 1e9
        show("blocks", k[["T", "D", "BLOCK_M", "BLOCK_N", "status", "num_warps", "num_stages", "ms", "tflops",
                          "model_bytes", "implied_TBps", "max_err"]])
    r = load("roofline.jsonl")
    if r is not None:
        show("copy", r[r.part.isin(["copy", "read"])][["part", "buffer_bytes", "ms", "gbps"]])
        show("matmul", r[r.part == "matmul"][["dtype", "M", "ms", "tflops"]])
        show("layer", r[r.part == "layer"][["phase", "B", "T_new", "ctx", "op", "intensity", "ms", "tflops", "gbps",
                                             "roofline_ms", "bound", "roofline_efficiency"]])
    c = load("dwconv.jsonl")
    if c is not None:
        show("dwconv", c[["sweep", "N", "K", "method", "status", "ms", "gflops_direct_equiv", "gbps", "max_err"]])
    for name in ("train_recall.jsonl", "train_mqar.jsonl", "train_throughput.jsonl"):
        d = load(name)
        if d is not None:
            if "part" in d and (d.part == "mqar_curve").any():
                d = d[d.part != "mqar_curve"]
            show(name, d)
    n = load("ncu_attention.jsonl")
    if n is not None:
        show("ncu", n)


if __name__ == "__main__":
    main()
