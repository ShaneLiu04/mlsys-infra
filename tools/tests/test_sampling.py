import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sampling as s  # noqa: E402


def test_softmax_is_stable_for_large_logits():
    p = s.softmax(np.array([1000.0, 1000.0, -1000.0]))
    assert np.allclose(p, [0.5, 0.5, 0.0])


def test_top_k_keeps_exactly_k_without_ties():
    z = s.top_k_filter(np.array([0.1, 3.0, 2.0, -1.0, 5.0]), 2)
    assert np.isfinite(z).sum() == 2
    assert set(np.flatnonzero(np.isfinite(z))) == {1, 4}


def test_top_p_keeps_token_that_crosses_threshold():
    logits = np.log(np.array([0.5, 0.3, 0.15, 0.05]))
    kept = np.isfinite(s.top_p_filter(logits, 0.7))
    assert kept.tolist() == [True, True, False, False]
    kept = np.isfinite(s.top_p_filter(logits, 0.5))
    assert kept.tolist() == [True, False, False, False]


def test_top_p_tiny_keeps_argmax():
    logits = np.array([[0.0, 2.0, 1.0], [5.0, 0.0, 0.0]])
    kept = np.isfinite(s.top_p_filter(logits, 1e-9))
    assert kept.sum(axis=-1).tolist() == [1, 1]
    assert kept[0, 1] and kept[1, 0]


def test_sample_matches_distribution():
    rng = np.random.default_rng(0)
    probs = np.array([0.6, 0.3, 0.1])
    draws = s.sample(np.tile(np.log(probs), (200_000, 1)), rng)
    freq = np.bincount(draws, minlength=3) / draws.size
    assert np.allclose(freq, probs, atol=0.005)


def test_sample_with_filters_never_picks_dropped_tokens():
    rng = np.random.default_rng(1)
    logits = np.tile(np.array([3.0, 2.9, 0.0, -5.0]), (5000, 1))
    draws = s.sample(logits, rng, temperature=0.7, top_k=3, top_p=0.9)
    assert set(np.unique(draws)) <= {0, 1}


def test_temperature_zero_is_rejected():
    with pytest.raises(ValueError):
        s.softmax(np.zeros(3), temperature=0.0)
