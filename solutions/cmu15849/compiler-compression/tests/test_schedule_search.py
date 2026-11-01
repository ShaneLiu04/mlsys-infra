import numpy as np
import pytest

from compkit.schedule_search import (Schedule, cost, evolutionary_search, grid_search,
                                     random_search, space, tiled_matmul)


@pytest.mark.parametrize("order", ["ijk", "kji", "jki"])
def test_tiled_matmul_correct(order):
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(20, 12)), rng.normal(size=(12, 28))
    np.testing.assert_allclose(tiled_matmul(a, b, Schedule(order, 8, 16, 4)), a @ b, rtol=1e-12)


def test_cost_model_prefers_reuse_and_fit():
    m = n = k = 1024
    good = cost(Schedule("ijk", 64, 64, 64), m, n, k)
    too_big = cost(Schedule("ijk", 512, 512, 512), m, n, k)  # 3 MB working set > 1 MB cache
    tiny = cost(Schedule("ijk", 4, 4, 4), m, n, k)
    assert good < too_big and good < tiny


def test_searches_reach_target():
    m = n = k = 512
    costs = {s: cost(s, m, n, k) for s in space()}
    best = min(costs.values())
    target = 1.05 * best
    rng = np.random.default_rng(0)
    assert grid_search(costs, target) <= len(costs)
    assert random_search(costs, target, rng) <= len(costs)
    assert evolutionary_search(costs, target, rng) <= len(costs)
