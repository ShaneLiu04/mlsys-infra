"""GPU part: checks on the raw CUDA results (run everywhere) and a live build+run (CUDA only)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "gpu"
CORRECT_STAGES = {"allcircles", "binned", "binned_smem", "blockscan_smem"}


def load(name):
    p = RAW / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def test_saxpy_raw_correct_and_ordered():
    rows = [r for r in load("saxpy.jsonl") if r["kind"] == "saxpy"]
    assert rows and all(r["mismatch"] == 0 for r in rows)
    big = max(rows, key=lambda r: r["n"])
    # kernel-only bandwidth is bounded by GDDR7 (~1.79 TB/s), end-to-end by PCIe (<= 64 GB/s)
    assert big["kernel_gbs"] < 1800 and big["pinned_e2e_gbs"] < 64
    assert big["kernel_gbs"] > 10 * big["pinned_e2e_gbs"]
    assert big["pinned_e2e_ms"] < big["pageable_e2e_ms"]


def test_scan_raw_all_correct():
    rows = load("scan.jsonl")
    assert {r["algo"] for r in rows} == {"naive", "smem_nopad", "smem", "raking", "thrust", "cub"}
    assert all(r["mismatch"] == 0 for r in rows)
    assert any(not r["pow2"] for r in rows)
    assert max(r["n"] for r in rows) == 1 << 28


def test_naive_scan_launches_are_2log2m():
    for r in load("scan.jsonl"):
        if r["algo"] == "naive":
            m = r["padded_n"]
            assert r["kernel_launches"] == 2 * (m.bit_length() - 1)


def test_find_repeats_raw_correct():
    rows = load("find_repeats.jsonl")
    assert rows and all(r["correct"] == 1 and r["repeats"] == r["expected"] for r in rows)


def test_render_correct_stages_match_reference():
    rows = [r for r in load("render.jsonl") if r["kind"] == "render"]
    ok = [r for r in rows if r["stage"] in CORRECT_STAGES]
    assert ok
    for r in ok:
        assert 0 <= r["max_abs_diff"] <= 1e-5, r
        assert r["pixels_diff_gt_1e-3"] == 0, r


def test_naive_renderer_is_wrong_on_overlapping_scenes():
    rows = [r for r in load("render.jsonl") if r["kind"] == "render" and r["stage"] == "naive_per_circle"]
    assert rows
    for r in rows:
        if r["scene"] != "rgb":
            assert r["pixels_diff_gt_1e-3"] > 0, r


def _have_cuda():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available() and shutil.which("nvcc") is not None


@pytest.mark.skipif(not _have_cuda(), reason="CUDA GPU / nvcc not available")
def test_live_build_and_run(tmp_path):
    subprocess.run(["bash", str(LAB / "gpu" / "build.sh")], check=True)
    b = LAB / "gpu" / "build"
    subprocess.run([str(b / "scan"), str(tmp_path / "s.jsonl"), str(tmp_path / "f.jsonl"), "18"], check=True)
    rows = [json.loads(x) for x in (tmp_path / "s.jsonl").read_text().splitlines()]
    assert rows and all(r["mismatch"] == 0 for r in rows)
    subprocess.run([str(b / "render"), str(tmp_path / "r.jsonl"), str(tmp_path), "rand10k"], check=True)
    rows = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text().splitlines()]
    rr = [r for r in rows if r["kind"] == "render"]
    assert all(r["max_abs_diff"] == 0 for r in rr if r["stage"] in CORRECT_STAGES)
    assert next(r for r in rr if r["stage"] == "naive_per_circle")["pixels_diff_gt_1e-3"] > 0
