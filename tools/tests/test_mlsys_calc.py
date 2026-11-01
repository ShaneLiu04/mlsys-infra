import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlsys_calc as mc  # noqa: E402


def test_amdahl_limits():
    assert mc.amdahl_speedup(1.0, 8) == pytest.approx(8.0)
    assert mc.amdahl_speedup(0.0, 8) == pytest.approx(1.0)
    assert mc.amdahl_speedup(0.95, 10**9) == pytest.approx(20.0, rel=1e-6)


def test_ridge_points():
    assert mc.GPUS["a100"].ridge == pytest.approx(153, abs=1)
    assert mc.GPUS["h100"].ridge == pytest.approx(295, abs=1)


def test_gemm_intensity_square_and_gemv():
    # square n x n x n in bf16: 2n^3 / (6 n^2) = n/3
    assert mc.gemm_intensity(4096, 4096, 4096) == pytest.approx(4096 / 3)
    # matrix-vector (m=1) is ~1 FLOP/byte in bf16 no matter how large
    assert mc.gemm_intensity(1, 8192, 8192) == pytest.approx(1.0, rel=1e-3)


def test_param_counts_match_published_sizes():
    assert mc.MODELS["llama2-7b"].params() / 1e9 == pytest.approx(6.74, abs=0.02)
    assert mc.MODELS["llama3-8b"].params() / 1e9 == pytest.approx(8.03, abs=0.02)
    assert mc.MODELS["llama3-70b"].params() / 1e9 == pytest.approx(70.6, abs=0.1)
    assert mc.MODELS["gpt2-124m"].params() / 1e6 == pytest.approx(124, abs=1.5)
    assert mc.MODELS["qwen2.5-0.5b"].params() / 1e9 == pytest.approx(0.49, abs=0.01)


def test_kv_cache_per_token():
    assert mc.kv_cache_bytes(mc.MODELS["llama2-7b"], 1) == 512 * 1024
    assert mc.kv_cache_bytes(mc.MODELS["llama3-8b"], 1) == 128 * 1024
    assert mc.kv_cache_bytes(mc.MODELS["llama3-70b"], 1) == 320 * 1024


def test_training_memory_16n_and_zero():
    n = 7e9
    assert mc.training_memory_bytes(n)["total"] == pytest.approx(16 * n)
    z3 = mc.training_memory_bytes(n, dp=8, zero=3)["total"]
    assert z3 == pytest.approx(2 * n)
    z1 = mc.training_memory_bytes(n, dp=8, zero=1)["total"]
    assert z1 == pytest.approx(4 * n + 12 * n / 8)


def test_ring_allreduce_volume_and_time():
    assert mc.allreduce_bus_bytes(1.0, 2) == pytest.approx(1.0)
    assert mc.allreduce_bus_bytes(1.0, 1024) == pytest.approx(2 * 1023 / 1024)
    assert mc.ring_allreduce_time(8e9, 8, 1e9) == pytest.approx(14.0)


def test_pipeline_bubble():
    assert mc.pipeline_bubble(4, 1) == pytest.approx(0.75)
    assert mc.pipeline_bubble(8, 32) == pytest.approx(7 / 39)
    assert mc.pipeline_bubble(8, 32, interleave=2) < mc.pipeline_bubble(8, 32)


def test_speculative_expected_tokens():
    assert mc.speculative_expected_tokens(0.0, 4) == pytest.approx(1.0)
    assert mc.speculative_expected_tokens(1.0, 4) == pytest.approx(5.0)
    alpha, gamma = 0.8, 4
    brute = sum(alpha**i for i in range(gamma + 1))
    assert mc.speculative_expected_tokens(alpha, gamma) == pytest.approx(brute)


def test_decode_is_memory_bound_at_batch_one():
    r = mc.decode_step_time(mc.MODELS["llama3-8b"], mc.GPUS["h100"], seq=1024, batch=1)
    assert r["bound"] == "memory"
    assert r["intensity"] < 2.0
    assert 150 < r["tokens_per_s"] < 250


def test_mfu_roundtrip():
    n, peak = 7e9, 989e12
    tps = 0.5 * peak / (6 * n)
    assert mc.mfu(tps, n, peak) == pytest.approx(0.5)


@pytest.mark.parametrize("cmd", [
    ["hw"],
    ["train-mem", "--params", "7e9", "--dp", "8", "--zero", "2"],
    ["kv", "--model", "llama3-70b", "--seq", "8192", "--batch", "4"],
    ["decode", "--model", "llama3-8b", "--gpu", "a100"],
    ["allreduce", "--bytes", "1e9"],
    ["bubble", "--stages", "4", "--micro", "16"],
    ["spec", "--alpha", "0.7", "--gamma", "3"],
])
def test_cli_runs(cmd, capsys):
    assert mc.main(cmd) == 0
    assert capsys.readouterr().out.strip()
