"""Build the C++ asst2 task systems and run every system x every official test at small sizes.

The driver exits non-zero when any test produces a wrong result; each row also carries
"correct". Timeouts catch deadlocks. Skipped when g++ is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
BIN = LAB / "cpp" / "bin"
EXE = ".exe" if os.name == "nt" else ""
MAX_THREADS = int(os.environ.get("MLSYS_MAX_THREADS", os.cpu_count() or 1))
THREADS = sorted({1, min(3, MAX_THREADS), min(8, MAX_THREADS)})
SYSTEMS = ["serial", "spawn", "spinning", "sleeping", "async_spinning", "async_sleeping"]
TESTS = [
    "super_super_light", "super_super_light_async", "super_light", "super_light_async",
    "ping_pong_equal", "ping_pong_equal_async", "ping_pong_unequal", "ping_pong_unequal_async",
    "recursive_fibonacci", "recursive_fibonacci_async",
    "math_operations_in_tight_for_loop", "math_operations_in_tight_for_loop_async",
    "math_operations_in_tight_for_loop_fewer_tasks", "math_operations_in_tight_for_loop_fewer_tasks_async",
    "math_operations_in_tight_for_loop_fan_in", "math_operations_in_tight_for_loop_reduction_tree",
    "spin_between_run_calls", "spin_between_run_calls_async",
    "mandelbrot_chunked", "mandelbrot_chunked_async",
]

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")


@pytest.fixture(scope="module", autouse=True)
def built():
    p = subprocess.run([sys.executable, str(LAB / "scripts" / "build_cpp.py")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def run(name: str, *args: str, timeout: int = 120) -> list[dict]:
    p = subprocess.run([str(BIN / (name + EXE)), *args], capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    assert rows
    return rows


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize("threads", [str(t) for t in THREADS])
def test_every_official_test_correct(system, threads):
    rows = run("tasksys_bench", "--mode", "tests", "--systems", system, "--threads", threads,
               "--scale", "0.05", "--reps", "2")
    assert {r["test"] for r in rows} == set(TESTS)
    bad = [r["test"] for r in rows if r["correct"] != 1]
    assert not bad, bad


def test_async_systems_respect_dependencies_in_every_dag_shape():
    rows = run("tasksys_bench", "--mode", "dag", "--systems", "async_sleeping,async_spinning",
               "--threads", str(min(4, MAX_THREADS)),
               "--launches", "32", "--tasks", "5", "--work", "100", "--reps", "3")
    assert {r["shape"] for r in rows} == {"indep_sync", "indep", "chain", "fanin", "tree", "random"}
    span = {r["shape"]: r["span_launches"] for r in rows}
    assert span["chain"] == 32 and span["indep"] == 1 and span["fanin"] == 2


def test_many_launches_stress_no_deadlock():
    # thousands of tiny launches catch lost-wakeup / generation races in the pools
    stress_threads = sorted({1, 2, min(7, MAX_THREADS)})
    rows = run("tasksys_bench", "--mode", "overhead", "--systems", "spinning,sleeping,async_spinning,async_sleeping",
               "--threads", ",".join(str(t) for t in stress_threads),
               "--tasks", "0,1,3,64", "--launches", "3000", "--reps", "1")
    assert len(rows) == 4 * len(stress_threads) * 4


def test_idle_spinning_burns_cpu_sleeping_does_not():
    rows = {r["system"]: r for r in run("tasksys_bench", "--mode", "idle", "--systems", "spinning,sleeping",
                                       "--threads", "2", "--idle_s", "0.5")}
    assert rows["sleeping"]["cores_burned"] < 0.2
    assert rows["spinning"]["cores_burned"] > 0.5


def test_contention_microbenchmarks_count_correctly():
    rows = run("contention", "--mode", "false_sharing", "--threads", "1,2", "--iters", "100000", "--reps", "1")
    assert all(r["correct"] == 1 for r in rows) and len(rows) == 8
    rows = run("contention", "--mode", "dispatch", "--threads", "1,3", "--n", "100000", "--reps", "1")
    assert all(r["correct"] == 1 for r in rows)
    assert {r["scheme"] for r in rows} == {"mutex", "atomic", "static", "atomic_chunk"}
