"""Correctness of every BFS variant and of push/pull PageRank. Skips if g++ is missing."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "cpp" / "graphbench.cpp"
EXE = ROOT / "cpp" / "bin" / "graphbench_test.exe"


@pytest.fixture(scope="module")
def binary():
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    EXE.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["g++", "-O3", "-fopenmp", "-std=c++17", "-o", str(EXE), str(SRC)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return EXE


def test_selftest(binary):
    out = subprocess.run([str(binary), "selftest"], check=True, capture_output=True, text=True)
    for name in ("chain", "star", "grid", "er", "rmat"):
        assert f"ok {name} " in out.stdout
    assert "ok all" in out.stdout
    assert "FAIL" not in out.stdout
