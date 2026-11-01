"""Shared helpers for the GPU experiments: timing, raw-result writers, RTX 5090 numbers."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(LAB.parents[2] / "tools"))

# RTX 5090 (sm_120). Peak numbers are datasheet values; `saxpy_bw` was measured on this box.
RTX5090 = {
    "name": "RTX 5090",
    "sms": 170,
    "hbm_bw": 1.792e12,        # GDDR7 512-bit @ 28 Gbps
    "saxpy_bw": 1.55e12,       # measured streaming bandwidth
    "bf16_flops": 209.5e12,    # dense tensor core, fp32 accumulate (GeForce: half the fp16-accumulate rate)
    "int8_ops": 838e12,        # dense int8 tensor core, int32 accumulate
    "fp8_flops": 419e12,       # dense fp8 e4m3, fp32 accumulate
    "l2_bytes": 96 * 2 ** 20,
    "smem_per_sm": 128 * 1024,
}


def set_threads(n: int = 2) -> None:
    import torch
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))


def cuda_time(fn, iters: int = 50, warmup: int = 10, reps: int = 5) -> float:
    """Median over `reps` of mean seconds per call measured with CUDA events."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / 1e3 / iters)
    return statistics.median(out)


def auto_iters(fn, target_s: float = 0.02) -> int:
    import torch
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    dt = max(time.perf_counter() - t, 1e-6)
    return int(min(2000, max(3, target_s / dt)))


def bench(fn, target_s: float = 0.02) -> float:
    n = auto_iters(fn, target_s)
    return cuda_time(fn, iters=n, warmup=max(2, n // 5), reps=5)


class Writer:
    """Append-only JSONL writer; one file per experiment."""

    def __init__(self, name: str, fresh: bool = True):
        RAW.mkdir(parents=True, exist_ok=True)
        self.path = RAW / f"{name}.jsonl"
        if fresh and self.path.exists():
            self.path.unlink()

    def __call__(self, **row) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


def env_info() -> dict:
    import torch
    import triton
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "cc": f"{p.major}.{p.minor}", "sms": p.multi_processor_count,
            "vram_gb": round(p.total_memory / 2 ** 30, 1), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version()}
