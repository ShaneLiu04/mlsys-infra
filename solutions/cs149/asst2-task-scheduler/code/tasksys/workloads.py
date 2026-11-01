"""Runnables used by tests and experiments."""

from __future__ import annotations

import threading
import time

import numpy as np


class NoopRunnable:
    """Empty task body: measures pure scheduling overhead."""

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        return None


class CountingRunnable:
    """Records how many times each task id ran."""

    def __init__(self, n: int) -> None:
        self.counts = [0] * n
        self._lock = threading.Lock()

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        with self._lock:
            self.counts[task_id] += 1


class NumpyRunnable:
    """Each task runs ``reps`` element-wise ufuncs on its own cache-sized array (GIL released inside)."""

    def __init__(self, n_tasks: int, elems: int = 1 << 15, reps: int = 4, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.inputs = [rng.random(elems) for _ in range(n_tasks)]
        self.outputs = [np.empty(elems) for _ in range(n_tasks)]
        self.reps = reps

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        x, y = self.inputs[task_id % len(self.inputs)], self.outputs[task_id % len(self.outputs)]
        for _ in range(self.reps):
            np.sin(x, out=y)
            np.sqrt(y * y + 1.0, out=y)


class PythonLoopRunnable:
    """Pure-Python arithmetic: holds the GIL, so threads cannot run it in parallel."""

    def __init__(self, iters: int = 20_000) -> None:
        self.iters = iters

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        s = 0
        for k in range(self.iters):
            s += k * k
        self.last = s


class SleepRunnable:
    """Models I/O or a blocking call: ``time.sleep`` releases the GIL."""

    def __init__(self, seconds: float = 0.001) -> None:
        self.seconds = seconds

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        time.sleep(self.seconds)
