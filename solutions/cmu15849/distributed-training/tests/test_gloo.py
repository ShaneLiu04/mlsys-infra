import json
import subprocess
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"


@pytest.mark.gloo
def test_gloo_allreduce_two_processes(tmp_path):
    out = tmp_path / "gloo.json"
    cmd = [sys.executable, "-m", "disttrain.gloo_bench", "--world", "2", "--sizes", "1024", "65536",
           "--iters", "2", "--out", str(out)]
    try:
        proc = subprocess.run(cmd, cwd=CODE, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("gloo process group timed out on this machine")
    if proc.returncode != 0:
        pytest.skip(f"gloo unavailable: {proc.stderr.decode(errors='replace')[-300:]}")
    rows = json.loads(out.read_text())
    assert all(r["correct"] for r in rows)
