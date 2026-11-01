import math

import numpy as np
import pytest

import mlsys_calc
from disttrain.collectives import (ALGORITHMS, SimNetwork, allreduce, closed_form_time,
                                   ring_all_gather, ring_reduce_scatter)

ALPHA, BETA = 5e-6, 1 / 450e9


def _bufs(p, n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(-1000, 1000, size=n).astype(np.int64) for _ in range(p)]


@pytest.mark.parametrize("alg", ALGORITHMS)
@pytest.mark.parametrize("p", [2, 4, 8, 16])
def test_allreduce_exact(alg, p):
    bufs = _bufs(p, 64 * p)
    out, _ = allreduce(alg, bufs)
    want = np.sum(bufs, axis=0)
    for o in out:
        np.testing.assert_array_equal(o, want)


def test_tree_and_naive_non_power_of_two():
    bufs = _bufs(6, 60)
    for alg in ("naive", "tree", "ring"):
        out, _ = allreduce(alg, bufs)
        for o in out:
            np.testing.assert_array_equal(o, np.sum(bufs, axis=0))


@pytest.mark.parametrize("p", [2, 4, 8])
def test_ring_bytes_and_steps(p):
    n = 128 * p
    _, st = allreduce("ring", _bufs(p, n))
    s = n * 8
    assert st.steps == 2 * (p - 1)
    assert np.all(st.bytes_sent == 2 * (p - 1) * s // p)
    assert math.isclose(st.bytes_sent[0], mlsys_calc.allreduce_bus_bytes(s, p))


@pytest.mark.parametrize("alg", ALGORITHMS)
@pytest.mark.parametrize("p", [4, 8, 16])
def test_simulated_time_matches_closed_form(alg, p):
    n = 32 * p
    _, st = allreduce(alg, _bufs(p, n))
    assert math.isclose(st.time(ALPHA, BETA), closed_form_time(alg, n * 8, p, ALPHA, BETA), rel_tol=1e-12)


def test_ring_closed_form_is_mlsys_calc():
    assert closed_form_time("ring", 1e9, 8, ALPHA, BETA) == mlsys_calc.ring_allreduce_time(1e9, 8, 450e9, ALPHA)


def test_halving_doubling_contends_on_ring_topology():
    _, hd = allreduce("halving_doubling", _bufs(8, 256))
    _, ring = allreduce("ring", _bufs(8, 256))
    assert ring.time(0, BETA, "ring") == ring.time(0, BETA, "full")
    assert hd.time(0, BETA, "ring") > hd.time(0, BETA, "full")


def test_reduce_scatter_all_gather_roundtrip():
    p, n = 4, 40
    bufs = _bufs(p, n)
    net = SimNetwork(p)
    shards = ring_reduce_scatter(bufs, net)
    want = np.sum(bufs, axis=0)
    for r in range(p):
        np.testing.assert_array_equal(shards[r], want[r * 10:(r + 1) * 10])
    for full in ring_all_gather(shards, net):
        np.testing.assert_array_equal(full, want)
