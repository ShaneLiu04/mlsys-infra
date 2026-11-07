"""How long does a GEMM run at its burst rate after the GPU has been idle, and how fast does that budget refill?

sustained_probe.py sees a flat line from the first 0.25 s window, while bench_roofline.py's 30-rep timing of
N=2048 reports twice that rate. This times every kernel with its own CUDA event pair right after a controlled
idle period, so the transition shows up at kernel resolution.

    python gpu/burst_probe.py
Writes burst.jsonl into the lab's raw directory: one "trace" row per (dtype, N, idle_s) with per-kernel
start offsets and durations in ms.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB.parents[2]))
from mlsys_harness.paths import raw_dir  # noqa: E402

RAW = raw_dir(_LAB)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def trace(a: torch.Tensor, b: torch.Tensor, idle_s: float, work_s: float, cap_tflops: float) -> dict:
    n = a.shape[0]
    flop = 2 * n**3
    k = max(20, int(work_s * cap_tflops * 1e12 / flop))
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(k)]
    torch.cuda.synchronize()
    time.sleep(idle_s)
    for s, e in ev:
        s.record()
        torch.mm(a, b)
        e.record()
    torch.cuda.synchronize()
    t0 = ev[0][0]
    start = [round(t0.elapsed_time(s), 4) for s, _ in ev]
    dur = [round(s.elapsed_time(e), 4) for s, e in ev]
    return dict(N=n, flop=flop, idle_s=idle_s, kernels=k, start_ms=start, dur_ms=dur)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2048,4096,8192")
    ap.add_argument("--idle", default="0,0.002,0.01,0.05,0.2,1,3")
    ap.add_argument("--work", type=float, default=0.4, help="seconds of work per trace at the capped rate")
    ap.add_argument("--cap", type=float, default=144.0, help="rough sustained TFLOP/s, only sizes the trace")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "burst.jsonl"
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="run", gpu=torch.cuda.get_device_name(), torch=torch.__version__)) + "\n")
        for dtype, dt in DTYPES.items():
            for n in (int(s) for s in args.sizes.split(",")):
                a = torch.randn(n, n, device="cuda", dtype=dt)
                b = torch.randn(n, n, device="cuda", dtype=dt)
                for _ in range(3):
                    torch.mm(a, b)
                idles = [float(x) for x in args.idle.split(",")] if n == 4096 else [3.0]
                for idle in idles:
                    r = trace(a, b, idle, args.work, args.cap)
                    tf = [r["flop"] / d / 1e9 for d in r["dur_ms"]]
                    head = sum(tf[:5]) / 5
                    tail = sorted(tf[len(tf) // 2:])[len(tf) // 4]
                    print(f"{dtype} N={n} idle={idle:5.3f}s: first5 {head:6.1f}  median2nd-half {tail:6.1f} TFLOP/s"
                          f"  ({r['kernels']} kernels, {r['start_ms'][-1]:.0f} ms)", flush=True)
                    f.write(json.dumps(dict(kind="trace", dtype=dtype, **r)) + "\n")
                del a, b
                torch.cuda.empty_cache()
    print("wrote", out)


if __name__ == "__main__":
    main()
