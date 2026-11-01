"""Process-pool Mandelbrot (prog1 analogue; processes instead of threads because of the GIL)."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any

import numpy as np

from .mandelbrot import MandelSpec, dynamic_chunks, interleaved_rows, mandel_rows, static_blocks

_BARRIER: Any = None


def _init_worker(barrier: Any) -> None:
    global _BARRIER
    _BARRIER = barrier


def _task(args: tuple[int, MandelSpec, np.ndarray, bool]) -> tuple[int, np.ndarray, np.ndarray, float, int]:
    k, spec, rows, pinned = args
    if pinned:
        # A worker parked here cannot pick up a second task, so the P static
        # assignments land on P distinct processes, as with one thread per block.
        _BARRIER.wait(timeout=60)
    t0 = time.perf_counter()
    counts = mandel_rows(spec, rows).astype(np.uint16)
    return k, rows, counts, time.perf_counter() - t0, os.getpid()


def _noop(_: int) -> int:
    return os.getpid()


@dataclass
class RunResult:
    image: np.ndarray
    wall_s: float
    worker_busy_s: dict[int, float]
    task_times: list[tuple[int, int, float, int]]
    """(task index, first row, compute seconds, worker pid) per task."""
    n_tasks: int


@dataclass
class WorkerPool:
    pool: Pool
    workers: int

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


def make_pool(workers: int) -> WorkerPool:
    """Create and warm up a pool so process start-up is not timed."""
    barrier = mp.get_context().Barrier(workers)
    pool = Pool(workers, initializer=_init_worker, initargs=(barrier,))
    pool.map(_noop, range(workers * 4), chunksize=1)
    return WorkerPool(pool, workers)


def build_tasks(spec: MandelSpec, schedule: str, workers: int, chunk_rows: int = 8) -> list[np.ndarray]:
    if schedule == "static":
        return static_blocks(spec.height, workers)
    if schedule == "interleaved":
        return interleaved_rows(spec.height, workers)
    if schedule == "dynamic":
        return dynamic_chunks(spec.height, chunk_rows)
    raise ValueError(f"unknown schedule {schedule!r}")


def run_parallel(wp: WorkerPool, spec: MandelSpec, schedule: str, chunk_rows: int = 8) -> RunResult:
    """Render with a row schedule; static schedules run exactly one task on each worker."""
    tasks = build_tasks(spec, schedule, wp.workers, chunk_rows)
    pinned = schedule in ("static", "interleaved")
    image = np.empty((spec.height, spec.width), dtype=np.uint16)
    busy: dict[int, float] = defaultdict(float)
    times: list[tuple[int, int, float, int]] = []
    t0 = time.perf_counter()
    for k, rows, counts, dt, pid in wp.pool.imap_unordered(
        _task, [(k, spec, r, pinned) for k, r in enumerate(tasks)], chunksize=1
    ):
        image[rows] = counts
        busy[pid] += dt
        times.append((k, int(rows[0]), dt, pid))
    wall = time.perf_counter() - t0
    times.sort()
    return RunResult(image=image, wall_s=wall, worker_busy_s=dict(busy), task_times=times, n_tasks=len(tasks))
