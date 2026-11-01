"""Build the C++ asst1 programs and check each against its scalar reference on small inputs.

Every program exits non-zero when any SIMD / threaded result differs from the scalar
reference, and also prints "correct":1 per measurement. Skipped when g++ is missing.
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

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")


@pytest.fixture(scope="module", autouse=True)
def built():
    p = subprocess.run([sys.executable, str(LAB / "scripts" / "build_cpp.py")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def run(name: str, *args: str) -> list[dict]:
    p = subprocess.run([str(BIN / (name + EXE)), *args], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    rows = [json.loads(l) for l in p.stdout.splitlines() if l.startswith("{")]
    assert rows
    return rows


def test_prog1_all_schedules_match_serial():
    rows = run("prog1_mandelbrot_threads", "--reps", "1", "--threads", "1,3,8", "--iters", "64")
    assert len(rows) == 9 and all(r["correct"] == 1 for r in rows)
    assert {r["sched"] for r in rows} == {"block", "interleaved", "dynamic"}
    for r in rows:
        assert sum(r["thread_rows"]) == 1200


@pytest.mark.parametrize("width", [2, 4, 8, 16])
def test_prog2_vector_matches_serial_and_counts(width):
    rows = run(f"prog2_vecintrin_w{width}", "--n", "999")
    assert all(r["correct"] == 1 for r in rows)
    s = next(r for r in rows if r["kind"] == "array_sum")
    assert s["instructions"] == s["expected_instructions"]
    u = {r["dist"]: r["utilization"] for r in rows if r["kind"] == "clamped_exp"}
    assert u["sorted"] >= u["uniform"] >= u["one_slow_per_vector"]


def test_prog3_simd_and_tasks_bit_exact():
    for view in ("1", "2"):
        rows = run("prog3_mandelbrot_simd", "--reps", "1", "--view", view)
        assert rows[0]["correct"] == 1
        assert 0 < rows[0]["util_w16"] <= rows[0]["util_w8"] <= 1
    rows = run("prog3_mandelbrot_simd", "--reps", "1", "--mode", "tasks", "--tasks", "1,7,64", "--threads", "4")
    assert all(r["correct"] == 1 for r in rows)


def test_prog4_sqrt_simd_bit_exact():
    rows = run("prog4_sqrt", "--n", "100003", "--reps", "1", "--threads", "1,3")
    assert all(r["correct"] == 1 for r in rows)
    simd = {r["pattern"]: r for r in rows if r["kind"] == "simd"}
    assert simd["all_one"]["mean_iters"] == 0
    assert simd["all_slow"]["mean_iters"] > simd["random"]["mean_iters"]


def test_prog5_saxpy_results_correct():
    rows = run("prog5_saxpy", "--mode", "one", "--n", "100000", "--threads", "1,2", "--min_s", "0.01")
    assert all(r["correct"] == 1 for r in rows)
    assert {r["kernel"] for r in rows} == {"scalar", "simd", "stream", "read"}


def test_prog6_optimized_versions_agree_with_baseline():
    rows = run("prog6_kmeans", "--m", "20000", "--dims", "16", "--threads", "1,3")
    base = rows[0]
    assert base["ver"] == "baseline"
    for r in rows[1:]:
        assert r["correct"] == 1 and r["iters"] == base["iters"] and r["mismatch"] == 0
