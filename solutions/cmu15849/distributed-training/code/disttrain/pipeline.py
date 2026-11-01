"""Discrete-event simulator for GPipe, 1F1B (PipeDream-flush) and interleaved 1F1B.

Each device executes a fixed op order. ``F(vs, i)`` waits for ``F(vs-1, i)``; ``B(vs, i)``
waits for ``B(vs+1, i)`` (or ``F(last, i)`` on the last virtual stage). Communication is free.
With interleaving, device d holds virtual stages d, d+p, ..., each costing 1/v of a stage.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    device: int
    kind: str  # "F" or "B"
    mb: int
    chunk: int
    start: float
    end: float


def _gpipe(p: int, m: int, d: int) -> list[tuple[str, int, int]]:
    return [("F", i, 0) for i in range(m)] + [("B", i, 0) for i in range(m)]


def _one_f_one_b(p: int, m: int, d: int) -> list[tuple[str, int, int]]:
    warm = min(p - d - 1, m)
    order = [("F", i, 0) for i in range(warm)]
    f, b = warm, 0
    while f < m:
        order += [("F", f, 0), ("B", b, 0)]
        f += 1
        b += 1
    order += [("B", i, 0) for i in range(b, m)]
    return order


def _interleaved(p: int, m: int, d: int, v: int) -> list[tuple[str, int, int]]:
    """Megatron-LM interleaved schedule order (requires m % p == 0)."""
    if m % p:
        raise ValueError("interleaved schedule needs m divisible by p")
    total = m * v

    def fwd(k: int) -> tuple[str, int, int]:
        return "F", (k // (p * v)) * p + k % p, (k // p) % v

    def bwd(k: int) -> tuple[str, int, int]:
        return "B", (k // (p * v)) * p + k % p, v - 1 - (k // p) % v

    warm = min((p - d - 1) * 2 + (v - 1) * p, total)
    order = [fwd(k) for k in range(warm)]
    f, b = warm, 0
    while f < total:
        order += [fwd(f), bwd(b)]
        f += 1
        b += 1
    order += [bwd(k) for k in range(b, total)]
    return order


def schedule_orders(schedule: str, p: int, m: int, v: int = 1) -> list[list[tuple[str, int, int]]]:
    if schedule == "gpipe":
        return [_gpipe(p, m, d) for d in range(p)]
    if schedule == "1f1b":
        return [_one_f_one_b(p, m, d) for d in range(p)]
    if schedule == "interleaved":
        return [_interleaved(p, m, d, v) for d in range(p)]
    raise KeyError(schedule)


def simulate(schedule: str, p: int, m: int, tf: float = 1.0, tb: float = 2.0, v: int = 1) -> list[Op]:
    v = v if schedule == "interleaved" else 1
    orders = schedule_orders(schedule, p, m, v)
    nvs = p * v
    done: dict[tuple[str, int, int], float] = {}  # (kind, virtual stage, mb) -> end
    free = [0.0] * p
    ptr = [0] * p
    ops: list[Op] = []
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for d in range(p):
            while ptr[d] < len(orders[d]):
                kind, mb, chunk = orders[d][ptr[d]]
                vs = chunk * p + d
                if kind == "F":
                    dep = ("F", vs - 1, mb) if vs > 0 else None
                else:
                    dep = ("B", vs + 1, mb) if vs < nvs - 1 else ("F", vs, mb)
                if dep is not None and dep not in done:
                    break
                start = max(free[d], done[dep] if dep else 0.0)
                end = start + (tf if kind == "F" else tb) / v
                done[(kind, vs, mb)] = end
                free[d] = end
                ops.append(Op(d, kind, mb, chunk, start, end))
                ptr[d] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"{schedule} deadlocked")
    return ops


def bubble_fraction(ops: list[Op], p: int) -> float:
    makespan = max(o.end for o in ops)
    busy = sum(o.end - o.start for o in ops)
    return 1.0 - busy / (p * makespan)


def peak_in_flight(ops: list[Op], p: int, v: int = 1) -> list[float]:
    """Max micro-batches whose activations are alive per device (F start .. B end),
    in units of one full stage's activations."""
    peaks = []
    for d in range(p):
        events = [(o.start, 1) if o.kind == "F" else (o.end, -1) for o in ops if o.device == d]
        cur = best = 0
        for _, delta in sorted(events):
            cur += delta
            best = max(best, cur)
        peaks.append(best / v)
    return peaks
