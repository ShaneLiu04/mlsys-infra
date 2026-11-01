import os
import random
import threading
import time
from typing import Callable

import numpy as np
import pytest

from tasksys import make_system
from tasksys.workloads import CountingRunnable, NoopRunnable, NumpyRunnable, SleepRunnable

MAX_THREADS = int(os.environ.get("MLSYS_MAX_THREADS", os.cpu_count() or 1))
TIMEOUT_S = 30.0


def nthreads(n: int) -> int:
    return max(1, min(n, MAX_THREADS))


ALL = ["serial", "spawn", "spinning", "spinning_yield", "sleeping"]
if MAX_THREADS < 4:
    ALL = [name for name in ALL if name != "spinning"]


def with_timeout(fn: Callable[[], None], timeout: float = TIMEOUT_S) -> None:
    """Run ``fn`` in a daemon thread; fail (instead of hanging) on deadlock."""
    err: list[BaseException] = []

    def target() -> None:
        try:
            fn()
        except BaseException as e:  # noqa: BLE001
            err.append(e)

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        pytest.fail(f"deadlock: did not finish within {timeout}s")
    if err:
        raise err[0]


class LoggingRunnable:
    """Logs (launch, start, end) timestamps so dependency order can be checked."""

    def __init__(self, launch: int, log: list, lock: threading.Lock, work_s: float = 0.0) -> None:
        self.launch, self.log, self.lock, self.work_s = launch, log, lock, work_s

    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        t0 = time.perf_counter()
        if self.work_s:
            time.sleep(self.work_s * random.random())
        t1 = time.perf_counter()
        with self.lock:
            self.log.append((self.launch, task_id, t0, t1))


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("n", [0, 1, 7, 100, 1000])
def test_run_executes_each_task_once(name, n):
    sys_ = make_system(name, nthreads(4))
    r = CountingRunnable(n)
    try:
        with_timeout(lambda: sys_.run(r, n))
    finally:
        sys_.close()
    assert r.counts == [1] * n


@pytest.mark.parametrize("name", ALL)
def test_many_runs_reuse_pool(name):
    sys_ = make_system(name, nthreads(8))

    def body() -> None:
        for k in range(200):
            r = CountingRunnable(k % 9)
            sys_.run(r, k % 9)
            assert r.counts == [1] * (k % 9)

    try:
        with_timeout(body)
    finally:
        sys_.close()


def random_dag(n_launches: int, rng: random.Random) -> list[list[int]]:
    deps: list[list[int]] = []
    for i in range(n_launches):
        k = rng.randint(0, min(i, 3))
        deps.append(rng.sample(range(i), k) if i else [])
    return deps


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_dag_respects_dependencies(name, seed):
    rng = random.Random(seed)
    dag = random_dag(30, rng)
    sizes = [rng.choice([0, 1, 3, 16]) for _ in dag]
    log: list = []
    lock = threading.Lock()
    sys_ = make_system(name, nthreads(6))

    def body() -> None:
        ids = []
        for i, d in enumerate(dag):
            ids.append(sys_.run_async_with_deps(LoggingRunnable(i, log, lock, 0.002), sizes[i], [ids[j] for j in d]))
        sys_.sync()

    try:
        with_timeout(body)
    finally:
        sys_.close()
    assert len(log) == sum(sizes)
    for i in range(len(dag)):
        assert sorted(t for (l, t, *_ ) in log if l == i) == list(range(sizes[i]))
    start = {i: min((s for (l, _, s, _) in log if l == i), default=None) for i in range(len(dag))}
    end = {i: max((e for (l, _, _, e) in log if l == i), default=None) for i in range(len(dag))}
    for i, d in enumerate(dag):
        for j in d:
            if start[i] is not None and end[j] is not None:
                assert start[i] >= end[j], f"launch {i} started before dep {j} finished"


def test_sleeping_is_actually_async_and_sync_waits():
    sys_ = make_system("sleeping", nthreads(4))
    r = CountingRunnable(8)
    try:
        slow = SleepRunnable(0.05)
        t0 = time.perf_counter()
        a = sys_.run_async_with_deps(slow, 4, [])
        sys_.run_async_with_deps(r, 8, [a])
        issue_time = time.perf_counter() - t0
        assert issue_time < 0.03, "run_async_with_deps must not block"
        assert sum(r.counts) == 0, "dependent launch ran before its dependency"
        with_timeout(sys_.sync)
        assert r.counts == [1] * 8
    finally:
        sys_.close()


def test_sleeping_independent_launches_overlap():
    sys_ = make_system("sleeping", nthreads(8))
    try:
        t0 = time.perf_counter()
        for _ in range(8):
            sys_.run_async_with_deps(SleepRunnable(0.05), 1, [])
        with_timeout(sys_.sync)
        assert time.perf_counter() - t0 < 0.3
    finally:
        sys_.close()


def test_sleeping_chain_is_serialized():
    log: list = []
    lock = threading.Lock()
    sys_ = make_system("sleeping", nthreads(8))
    try:
        prev: list[int] = []
        for i in range(20):
            prev = [sys_.run_async_with_deps(LoggingRunnable(i, log, lock), 5, prev)]
        with_timeout(sys_.sync)
    finally:
        sys_.close()
    for i in range(1, 20):
        assert min(s for (l, _, s, _) in log if l == i) >= max(e for (l, _, _, e) in log if l == i - 1)


def test_sync_with_nothing_pending_returns():
    sys_ = make_system("sleeping", nthreads(2))
    try:
        with_timeout(sys_.sync, 5)
        with_timeout(sys_.sync, 5)
    finally:
        sys_.close()


@pytest.mark.parametrize("name", ["spawn", "spinning_yield", "sleeping"])
def test_stress_many_tiny_tasks(name):
    sys_ = make_system(name, nthreads(8))
    r = CountingRunnable(20_000)
    try:
        with_timeout(lambda: sys_.run(r, 20_000), 60)
    finally:
        sys_.close()
    assert r.counts == [1] * 20_000


def test_stress_async_many_launches():
    sys_ = make_system("sleeping", nthreads(8))
    runs = [CountingRunnable(10) for _ in range(500)]
    rng = random.Random(3)

    def body() -> None:
        ids: list[int] = []
        for r in runs:
            deps = rng.sample(ids[-10:], min(len(ids[-10:]), 2)) if ids else []
            ids.append(sys_.run_async_with_deps(r, 10, deps))
        sys_.sync()

    try:
        with_timeout(body, 60)
    finally:
        sys_.close()
    assert all(r.counts == [1] * 10 for r in runs)


@pytest.mark.parametrize("name", ALL)
def test_numpy_tasks_produce_same_result(name):
    r = NumpyRunnable(16, elems=1000, reps=1)
    sys_ = make_system(name, nthreads(4))
    try:
        with_timeout(lambda: sys_.run(r, 16))
    finally:
        sys_.close()
    for x, y in zip(r.inputs, r.outputs):
        np.testing.assert_allclose(y, np.sqrt(np.sin(x) ** 2 + 1.0))


class Boom:
    def run_task(self, task_id: int, num_total_tasks: int) -> None:
        if task_id == 3:
            raise ValueError("boom")


@pytest.mark.parametrize("name", ALL)
def test_task_exception_propagates_and_system_survives(name):
    sys_ = make_system(name, nthreads(4))
    try:
        with pytest.raises(ValueError):
            sys_.run_async_with_deps(Boom(), 8, [])
            sys_.sync()
        r = CountingRunnable(5)
        with_timeout(lambda: sys_.run(r, 5))
        assert r.counts == [1] * 5
    finally:
        sys_.close()


@pytest.mark.parametrize("name", ["spawn", "sleeping"])
def test_noop_run_is_fast_enough(name):
    sys_ = make_system(name, nthreads(8))
    try:
        t0 = time.perf_counter()
        for _ in range(50):
            sys_.run(NoopRunnable(), 8)
        assert time.perf_counter() - t0 < 10
    finally:
        sys_.close()
