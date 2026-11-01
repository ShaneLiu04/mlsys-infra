import numpy as np
import pytest

from disttrain.zero import Adam, MLPSpec, init_params, make_data, train_single, train_zero

P, STEPS = 4, 5


@pytest.fixture(scope="module")
def setup():
    spec = MLPSpec()
    flat = init_params(spec, P)
    x, y = make_data(spec, 64)
    ref = train_single(spec, flat, x, y, STEPS, Adam())
    return spec, flat, x, y, ref


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_stage_matches_single_process(setup, stage):
    spec, flat, x, y, ref = setup
    res = train_zero(stage, spec, flat, x, y, P, STEPS, Adam())
    assert np.max(np.abs(res.params - ref)) < 1e-12
    assert not np.allclose(res.params, flat)


def test_memory_and_comm_scaling(setup):
    spec, flat, x, y, _ = setup
    n = len(flat)
    res = {s: train_zero(s, spec, flat, x, y, P, 1, Adam()) for s in range(4)}
    assert res[0].mem_bytes["total"] == 32 * n  # float64 params, grads, m, v
    assert res[1].mem_bytes["adam_m"] == 8 * n // P
    assert res[2].mem_bytes["grads"] == 8 * n // P
    assert res[3].mem_bytes["total"] == 32 * n // P
    dp = res[0].comm_bytes_per_step
    assert dp == pytest.approx(2 * (P - 1) / P * 8 * n)
    assert res[1].comm_bytes_per_step == dp and res[2].comm_bytes_per_step == dp
    assert res[3].comm_bytes_per_step == pytest.approx(1.5 * dp)


def test_loss_decreases(setup):
    spec, flat, x, y, _ = setup
    res = train_zero(3, spec, flat, x, y, P, 30, Adam())
    assert res.losses[-1] < 0.5 * res.losses[0]
