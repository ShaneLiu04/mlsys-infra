"""Burst versus sustained tensor-core throughput.

A peak benchmark that times a few large GEMMs can read a clock the card does not hold. This runs
each (dtype, N) back to back for a fixed wall time, records TFLOP/s per 0.25 s window, and samples
`nvidia-smi` (SM clock, power, active clock-event reasons) alongside, so a drop can be attributed
to the clock, the power limit, or neither.

    python gpu/sustained_probe.py [--seconds 8]
Writes sustained.jsonl (one row per window) and sustained_smi.csv into the lab's raw directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

_LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB.parents[2]))
from mlsys_harness.paths import raw_dir  # noqa: E402

RAW = raw_dir(_LAB)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def run(dtype: str, n: int, seconds: float, window: float, t0: float) -> list[dict]:
    a = torch.randn(n, n, device="cuda", dtype=DTYPES[dtype])
    b = torch.randn(n, n, device="cuda", dtype=DTYPES[dtype])
    flop = 2 * n**3
    per = max(1, int(2e12 // flop))  # ~2 TFLOP of work between host checks
    for _ in range(3):
        torch.mm(a, b)
    torch.cuda.synchronize()
    rows, start = [], time.perf_counter()
    w_start, w_iters = start, 0
    while time.perf_counter() - start < seconds:
        for _ in range(per):
            torch.mm(a, b)
        torch.cuda.synchronize()
        w_iters += per
        now = time.perf_counter()
        if now - w_start >= window:
            rows.append(dict(kind="window", dtype=dtype, N=n, t_s=round(now - t0, 3),
                             since_start_s=round(now - start, 3), iters=w_iters,
                             tflops=w_iters * flop / (now - w_start) / 1e12))
            w_start, w_iters = now, 0
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--sizes", default="2048,4096,8192,16384")
    ap.add_argument("--rest", type=float, default=5.0, help="idle seconds between runs so clocks recover")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    smi_path = RAW / "sustained_smi.csv"
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,"
         "clocks_event_reasons.active", "--format=csv,noheader,nounits", "-lms", "100"],
        stdout=open(smi_path, "w"), stderr=subprocess.DEVNULL)
    out = RAW / "sustained.jsonl"
    t0 = time.perf_counter()
    try:
        with out.open("w", encoding="utf-8") as f:
            f.write(json.dumps(dict(kind="run", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                    wall_epoch=time.time())) + "\n")
            for dtype in DTYPES:
                for n in (int(s) for s in args.sizes.split(",")):
                    time.sleep(args.rest)
                    f.write(json.dumps(dict(kind="phase", dtype=dtype, N=n, t_s=round(time.perf_counter() - t0, 3),
                                            wall_epoch=time.time())) + "\n")
                    rows = run(dtype, n, args.seconds, args.window, t0)
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    tf = [r["tflops"] for r in rows]
                    print(f"{dtype} N={n}: first {tf[0]:.1f}  max {max(tf):.1f}  last {tf[-1]:.1f} TFLOP/s", flush=True)
    finally:
        smi.terminate()
    print(f"wrote {out} and {smi_path}")


if __name__ == "__main__":
    main()
