"""Blocked GEMM stays exact when the shape is not a multiple of the block."""

import numpy as np

from dnnaccel.gemm_cpu import matmul_blocked, matmul_naive


def test_blocked_matches_numpy_off_tile():
    rng = np.random.default_rng(0)
    shapes = [(1, 1, 1, 4), (7, 5, 3, 4), (8, 8, 8, 3), (17, 13, 9, 8), (6, 1, 5, 2)]
    for m, k, n, bs in shapes:
        a = rng.normal(size=(m, k))
        b = rng.normal(size=(k, n))
        np.testing.assert_allclose(matmul_blocked(a, b, bs), a @ b, atol=1e-8)
        if m * n * k <= 4000:
            np.testing.assert_allclose(matmul_naive(a, b), a @ b, atol=1e-6)
