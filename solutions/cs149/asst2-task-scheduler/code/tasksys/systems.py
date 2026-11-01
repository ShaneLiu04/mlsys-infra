"""Task systems with the CS149 asst2 API.

``run(runnable, n)`` executes ``runnable.run_task(i, n)`` for i in [0, n) and
returns when all are done. ``run_async_with_deps(runnable, n, deps)`` returns a
task id immediately; the bulk launch may start only after every launch in
``deps`` has finished. ``sync()`` waits for all launches issued so far.

As in the official Part A, only the sleeping pool is truly asynchronous; the
other systems implement ``run_async_with_deps`` as a synchronous ``run`` (which
trivially respects dependencies, since every earlier launch already finished).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, Sequence


class IRunnable(Protocol):
    def run_task(self, task_id: int, num_total_tasks: int) -> None: ...


class ITaskSystem:
    name = "base"

    def __init__(self, num_threads: int = 8) -> None:
        self.num_threads = num_threads
        self._next_id = 0

    def run(self, runnable: IRunnable, num_total_tasks: int) -> None:
        raise NotImplementedError

    def run_async_with_deps(self, runnable: IRunnable, num_total_tasks: int, deps: Sequence[int]) -> int:
        self.run(runnable, num_total_tasks)
        tid = self._next_id
        self._next_id += 1
        return tid

    def sync(self) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "ITaskSystem":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TaskSystemSerial(ITaskSystem):
    name = "serial"

    def run(self, runnable: IRunnable, num_total_tasks: int) -> None:
        for i in range(num_total_tasks):
            runnable.run_task(i, num_total_tasks)


class _Counter:
    """Shared ``next task`` counter; dynamic assignment so uneven tasks balance."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.next = 0
        self.lock = threading.Lock()

    def take(self) -> int:
        with self.lock:
            i = self.next
            if i >= self.total:
                return -1
            self.next = i + 1
            return i


class TaskSystemParallelSpawn(ITaskSystem):
    """Creates ``num_threads`` fresh threads on every ``run`` and joins them."""

    name = "spawn"

    def run(self, runnable: IRunnable, num_total_tasks: int) -> None:
        counter = _Counter(num_total_tasks)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                while (i := counter.take()) >= 0:
                    runnable.run_task(i, num_total_tasks)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(min(self.num_threads, max(num_total_tasks, 1)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]


class TaskSystemParallelThreadPoolSpinning(ITaskSystem):
    """Persistent workers that busy-wait for work; ``run`` busy-waits for completion.

    ``yield_in_spin=True`` inserts ``time.sleep(0)`` in every spin iteration (a
    ``sched_yield``), which in CPython also releases the GIL. Without it a spinning
    thread only gives up the GIL when the interpreter's switch interval forces it.
    """

    name = "spinning"

    def __init__(self, num_threads: int = 8, yield_in_spin: bool = False) -> None:
        super().__init__(num_threads)
        self.yield_in_spin = yield_in_spin
        if yield_in_spin:
            self.name = "spinning_yield"
        self._lock = threading.Lock()
        self._runnable: IRunnable | None = None
        self._total = 0
        self._next = 0
        self._done = 0
        self._errors: list[BaseException] = []
        self._stop = False
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(num_threads)]
        for t in self._threads:
            t.start()

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._stop:
                    return
                runnable, total, i = self._runnable, self._total, self._next
                if runnable is not None and i < total:
                    self._next = i + 1
                else:
                    i = -1
            if i < 0:
                if self.yield_in_spin:
                    time.sleep(0)
                continue
            try:
                runnable.run_task(i, total)  # type: ignore[union-attr]
            except BaseException as e:  # noqa: BLE001
                self._errors.append(e)
            with self._lock:
                self._done += 1

    def run(self, runnable: IRunnable, num_total_tasks: int) -> None:
        with self._lock:
            self._runnable, self._total, self._next, self._done = runnable, num_total_tasks, 0, 0
        while True:
            with self._lock:
                if self._done >= self._total:
                    self._runnable = None
                    break
            if self.yield_in_spin:
                time.sleep(0)
        if self._errors:
            err, self._errors = self._errors[0], []
            raise err

    def close(self) -> None:
        with self._lock:
            self._stop = True
        for t in self._threads:
            t.join()


@dataclass
class _Launch:
    tid: int
    runnable: IRunnable
    total: int
    pending_deps: int
    next: int = 0
    done: int = 0
    dependents: list[int] = field(default_factory=list)


class TaskSystemParallelThreadPoolSleeping(ITaskSystem):
    """Persistent workers that sleep on a condition variable; full async task graph.

    One lock protects all scheduler state. Workers pull single task indices from
    the oldest ready launch, so independent launches overlap and no worker idles
    at the tail of one launch while another launch has runnable tasks.
    """

    name = "sleeping"

    def __init__(self, num_threads: int = 8) -> None:
        super().__init__(num_threads)
        self._lock = threading.Lock()
        self._work_cv = threading.Condition(self._lock)
        self._done_cv = threading.Condition(self._lock)
        self._launches: dict[int, _Launch] = {}
        self._ready: deque[int] = deque()
        self._unfinished = 0
        self._errors: list[BaseException] = []
        self._stop = False
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(num_threads)]
        for t in self._threads:
            t.start()

    def _finish_launch_locked(self, launch: _Launch) -> None:
        del self._launches[launch.tid]
        self._unfinished -= 1
        woke = False
        for d in launch.dependents:
            child = self._launches[d]
            child.pending_deps -= 1
            if child.pending_deps == 0:
                self._make_ready_locked(child)
                woke = True
        if woke:
            self._work_cv.notify_all()
        if self._unfinished == 0:
            self._done_cv.notify_all()

    def _make_ready_locked(self, launch: _Launch) -> None:
        if launch.total == 0:
            self._finish_launch_locked(launch)
        else:
            self._ready.append(launch.tid)

    def _worker(self) -> None:
        while True:
            with self._lock:
                while not self._ready and not self._stop:
                    self._work_cv.wait()
                if self._stop:
                    return
                launch = self._launches[self._ready[0]]
                i = launch.next
                launch.next += 1
                if launch.next == launch.total:
                    self._ready.popleft()
            try:
                launch.runnable.run_task(i, launch.total)
            except BaseException as e:  # noqa: BLE001
                with self._lock:
                    self._errors.append(e)
            with self._lock:
                launch.done += 1
                if launch.done == launch.total:
                    self._finish_launch_locked(launch)

    def run_async_with_deps(self, runnable: IRunnable, num_total_tasks: int, deps: Sequence[int]) -> int:
        with self._lock:
            tid = self._next_id
            self._next_id += 1
            live = [d for d in set(deps) if d in self._launches]
            launch = _Launch(tid, runnable, num_total_tasks, pending_deps=len(live))
            self._launches[tid] = launch
            self._unfinished += 1
            for d in live:
                self._launches[d].dependents.append(tid)
            if not live:
                self._make_ready_locked(launch)
                self._work_cv.notify_all()
        return tid

    def run(self, runnable: IRunnable, num_total_tasks: int) -> None:
        self.run_async_with_deps(runnable, num_total_tasks, [])
        self.sync()

    def sync(self) -> None:
        with self._lock:
            while self._unfinished > 0:
                self._done_cv.wait()
            if self._errors:
                err, self._errors = self._errors[0], []
                raise err

    def close(self) -> None:
        with self._lock:
            self._stop = True
            self._work_cv.notify_all()
        for t in self._threads:
            t.join()


def make_system(name: str, num_threads: int = 8) -> ITaskSystem:
    if name == "serial":
        return TaskSystemSerial(num_threads)
    if name == "spawn":
        return TaskSystemParallelSpawn(num_threads)
    if name == "spinning":
        return TaskSystemParallelThreadPoolSpinning(num_threads)
    if name == "spinning_yield":
        return TaskSystemParallelThreadPoolSpinning(num_threads, yield_in_spin=True)
    if name == "sleeping":
        return TaskSystemParallelThreadPoolSleeping(num_threads)
    raise ValueError(f"unknown task system {name!r}")
