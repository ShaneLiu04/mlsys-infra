"""Exclusive scans match a serial prefix sum, including lengths that are not powers of two."""

import numpy as np

from scanrender.scan import (analytic_stats, blelloch_exclusive, exclusive_scan_sequential,
                             hillis_steele_exclusive)


def _reference(x: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape, dtype=np.float64)
    if x.size:
        out[1:] = np.cumsum(x.astype(np.float64)[:-1])
    return out


def test_scans_match_numpy_on_random_lengths():
    rng = np.random.default_rng(0)
    lengths = [0, 1, 2, 3, 7, 8, 9, 17, 31, 32, 100, 1000]
    lengths += [int(n) for n in rng.integers(0, 300, size=20)]
    for n in lengths:
        x = rng.normal(size=n)
        ref = _reference(x)
        for fn in (exclusive_scan_sequential, hillis_steele_exclusive, blelloch_exclusive):
            out, stats = fn(x)
            np.testing.assert_allclose(out, ref, atol=1e-8)
            assert stats.padded_n >= n


def test_integer_scan_and_closed_form_work():
    x = np.array([3, 1, 4, 1, 5, 9, 2], dtype=np.int64)
    out, stats = blelloch_exclusive(x)
    np.testing.assert_array_equal(out, _reference(x).astype(np.int64))
    assert stats.adds == analytic_stats("blelloch", x.size).adds
    assert analytic_stats("sequential", 10).adds == 9
    assert analytic_stats("hillis_steele", 10).steps == 4
