"""Shared helpers for the GPU experiments: CUDA-event timing, peak memory, raw-result writers."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import torch

LAB = Path(__file__).resolve().parents[1]
_REPO = LAB.parents[2]
TOOLS = _REPO / "tools"
for p in (str(_REPO), str(LAB / "code"), str(TOOLS), str(LAB / "gpu")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mlsys_harness.paths import raw_dir  # noqa: E402

RAW_GPU = raw_dir(LAB)


def cuda_time_ms(fn, warmup: int = 3, reps: int = 10, max_ms: float = 1500.0) -> float:
    """Median of per-call CUDA-event times; reps shrink so one measurement stays under ~max_ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    first = s.elapsed_time(e)
    reps = max(1, min(reps, int(max_ms / max(first, 1e-3))))
    times = []
    for _ in range(reps):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def peak_extra_bytes(fn) -> int:
    """Peak allocated bytes during one call of fn, above what was allocated before it."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


class JsonlWriter:
    def __init__(self, name: str):
        RAW_GPU.mkdir(parents=True, exist_ok=True)
        self.path = RAW_GPU / name
        self.f = open(self.path, "w", encoding="utf-8")
        self.n = 0

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        self.n += 1

    def close(self) -> None:
        self.f.close()
        print(f"wrote {self.path.name} ({self.n} rows)", flush=True)


def env_info() -> dict:
    import numpy as np
    info = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "host": platform.node(), "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}
    try:
        import triton
        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu": p.name, "sm_count": p.multi_processor_count, "capability": f"{p.major}.{p.minor}",
                     "vram_gb": round(p.total_memory / 1e9, 2),
                     "l2_mb": round(getattr(p, "L2_cache_size", 0) / 2**20, 1)})
    return info
