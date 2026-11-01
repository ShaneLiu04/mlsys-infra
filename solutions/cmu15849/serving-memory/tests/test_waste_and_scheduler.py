import numpy as np
import pytest

import mlsys_calc
from servesim.scheduler import CostModel, simulate_continuous, simulate_static, summarize
from servesim.waste import waste_table
from servesim.workload import MAX_MODEL_LEN, poisson_arrivals, sample_lengths

COST = CostModel(mlsys_calc.MODELS["llama3-8b"], mlsys_calc.GPUS["h100"])


def test_lengths_respect_max_len():
    p, o = sample_lengths(5000)
    assert np.all(p + o <= MAX_MODEL_LEN) and o.min() >= 1 and p.min() >= 8


def test_waste_monotone_in_block_size():
    p, o = sample_lengths(500)
    rows = waste_table(p, o, budget_tokens=100_000)
    paged = [r for r in rows if r["scheme"] == "paged"]
    wastes = [r["waste_frac"] for r in paged]
    assert wastes == sorted(wastes)
    assert rows[0]["waste_frac"] > 0.8 > wastes[-1]
    assert all(r["concurrent_seqs"] > rows[0]["concurrent_seqs"] for r in paged)


def test_decode_step_matches_mlsys_calc():
    seq, batch = 1024, 8
    want = mlsys_calc.decode_step_time(COST.model, COST.gpu, seq, batch)["time_s"]
    assert COST.step_time(seq * batch, batch) == pytest.approx(want, rel=1e-12)


def _check_conservation(res, prompts, outputs):
    for r in res.requests:
        assert r.generated == r.output and r.finish >= r.first_token >= r.arrival
    assert res.decode_tokens + res.prefill_emits == int(outputs.sum())
    assert res.prefill_tokens >= int(prompts.sum())
    assert res.leaked_blocks == 0


@pytest.mark.parametrize("policy", ["static", "continuous", "chunked"])
def test_all_requests_finish_and_tokens_conserved(policy):
    n = 200
    p, o = sample_lengths(n, seed=5)
    arr = poisson_arrivals(n, 20.0, seed=6)
    if policy == "static":
        res = simulate_static(COST, arr, p, o)
        assert res.prefill_tokens == int(p.sum())
    else:
        res = simulate_continuous(COST, arr, p, o, chunked=policy == "chunked")
    _check_conservation(res, p, o)
    s = summarize(res, 1.0, 0.05)
    assert s["throughput_tok_s"] > 0 and 0 <= s["slo_attain"] <= 1


@pytest.mark.parametrize("chunked", [False, True])
def test_preemption_under_tiny_kv_cache(chunked):
    n = 60
    p, o = sample_lengths(n, seed=7)
    arr = np.zeros(n)
    res = simulate_continuous(COST, arr, p, o, chunked=chunked, kv_tokens=8192)
    assert res.preemptions > 0
    _check_conservation(res, p, o)
    assert res.prefill_tokens > int(p.sum())  # recomputed tokens


def test_chunked_bounds_step_time():
    n = 200
    p, o = sample_lengths(n, seed=8)
    arr = poisson_arrivals(n, 30.0, seed=9)
    plain = simulate_continuous(COST, arr, p, o, chunked=False)
    chunk = simulate_continuous(COST, arr, p, o, chunked=True)
    assert max(chunk.step_times) < max(plain.step_times)
