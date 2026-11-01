import numpy as np
import pytest
import torch

from disttrain.checkpointing import (best_under_budget, brute_force, evaluate, make_chain,
                                     no_checkpoint, optimal_frontier, run_chain, sqrt_checkpoint)


def test_uniform_chain_formulas():
    n = 16
    ones = [1] * n
    assert no_checkpoint(ones, ones).peak == n and no_checkpoint(ones, ones).recompute == 0
    sq = sqrt_checkpoint(ones, ones)
    assert sq.peak == 4 + 3 and sq.recompute == n - 4


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 5, 9).tolist()
    costs = rng.integers(1, 5, 9).tolist()
    fr = optimal_frontier(sizes, costs)
    for budget in range(min(fr), sum(sizes) + 1):
        plan = best_under_budget(fr, budget)
        assert plan.recompute == brute_force(sizes, costs, budget)
        assert evaluate(sizes, costs, plan.stored).peak <= budget


def test_dp_no_worse_than_sqrt():
    rng = np.random.default_rng(3)
    sizes = rng.integers(1, 5, 25).tolist()
    sq = sqrt_checkpoint(sizes, sizes)
    plan = best_under_budget(optimal_frontier(sizes, sizes), sq.peak)
    assert plan.recompute <= sq.recompute


def test_torch_checkpoint_same_grads_less_memory():
    n, d, b = 16, 64, 32
    model = make_chain(n, d).double()
    x = torch.randn(b, d, dtype=torch.float64)
    loss0, g0, peak0 = run_chain(model, x, None)
    loss1, g1, peak1 = run_chain(model, x, 4)
    assert loss0 == pytest.approx(loss1, rel=1e-14)
    for a, c in zip(g0, g1):
        torch.testing.assert_close(a, c, rtol=0, atol=0)
    act = b * d * 8
    assert peak0 == (n + 1) * act
    assert peak1 < peak0 / 2
