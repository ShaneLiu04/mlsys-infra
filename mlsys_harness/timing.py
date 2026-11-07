"""One timing summary for every benchmark.

Two modes, because they answer different questions:

* ``per_call`` times each invocation separately. Use it when the launch itself
  is part of the cost (a single decode step, a short kernel).
* ``batched`` times a loop of calls and divides by the count. Use it when the
  per-call time is close to the timer resolution.

The result always carries p10, p50 and p90, so a report can say which one it
quoted. ``time_cuda`` adds an optional L2 flush: one device-to-device copy of
a buffer larger than the card's L2, so a small kernel is not scored on cached data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeResult:
    mode: str
    samples: int
    p10_s: float
    p50_s: float
    p90_s: float

    def as_dict(self) -> dict:
        return {"mode": self.mode, "samples": self.samples,
                "p10_s": self.p10_s, "p50_s": self.p50_s, "p90_s": self.p90_s}


def percentile(samples: list[float], q: float) -> float:
    """Linear percentile. q is in [0, 1]. One sample returns that sample."""
    if not samples:
        raise ValueError("no samples")
    if not 0.0 <= q <= 1.0:
        raise ValueError(q)
    xs = sorted(samples)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def summarize(samples: list[float], mode: str) -> TimeResult:
    if mode not in ("per_call", "batched"):
        raise ValueError(mode)
    return TimeResult(mode, len(samples), percentile(samples, 0.10),
                      percentile(samples, 0.50), percentile(samples, 0.90))


def time_calls(fn, warmup: int = 5, repeats: int = 20) -> TimeResult:
    """Host-clock per-call timing. Used for CPU work and for tests of the summary."""
    import time
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return summarize(samples, "per_call")


def time_cuda(fn, mode: str = "per_call", warmup: int = 10, repeats: int = 30,
              batch: int = 1, flush_l2_bytes: int = 0) -> TimeResult:
    """CUDA-event timing. ``flush_l2_bytes`` copies that many bytes before every sample."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("time_cuda requires CUDA")
    if mode == "batched" and batch < 1:
        raise ValueError("batch must be >= 1")
    sink = None
    if flush_l2_bytes > 0:
        nbytes = flush_l2_bytes
        sink = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
        src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")

    def once() -> None:
        if sink is not None:
            sink.copy_(src)
        if mode == "batched":
            for _ in range(batch):
                fn()
        else:
            fn()

    for _ in range(warmup):
        once()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(repeats):
        start.record()
        once()
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end) / 1e3
        samples.append(elapsed / batch if mode == "batched" else elapsed)
    return summarize(samples, mode)
