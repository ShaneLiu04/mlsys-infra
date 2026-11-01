import numpy as np
import pytest
import torch

from dnnaccel.gemm_cpu import matmul_blocked, matmul_naive, matmul_rows
from dnnaccel.im2col import blowup_factor, conv2d_direct, conv2d_im2col, im2col
from dnnaccel.systolic import block_cycles, systolic_matmul
from dnnaccel.tiling import (Accel, best_square_tile, fits, intensity_upper_bound,
                             search_tiles, tile_stats, tiled_matmul)

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("m,n,k,tm,tn,tk", [(64, 64, 64, 16, 16, 16), (70, 45, 33, 16, 32, 8),
                                            (128, 96, 200, 128, 32, 64), (5, 7, 3, 8, 8, 8)])
def test_tiled_matmul_matches_and_counts(m, n, k, tm, tn, tk):
    a, b = RNG.standard_normal((m, k)), RNG.standard_normal((k, n))
    c, cnt = tiled_matmul(a, b, tm, tn, tk)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12, atol=1e-12)
    s = tile_stats(m, n, k, tm, tn, tk, Accel(bytes_per_el=1), with_cycles=False)
    assert cnt["loads"] + cnt["stores"] == s.hbm_bytes
    assert cnt["dmas"] == s.dma_count


def test_intensity_grows_with_tile_then_capacity_binds():
    acc = Accel()
    vals = [tile_stats(4096, 4096, 4096, t, t, 128, acc, with_cycles=False).intensity
            for t in (128, 256, 512)]
    assert vals[0] < vals[1] < vals[2]
    assert vals[1] / vals[0] == pytest.approx(2.0, rel=0.05)
    t = best_square_tile(acc)
    assert fits(t, t, 128, acc) and not fits(t + 128, t + 128, 128, acc)


def test_search_respects_capacity_and_beats_small_tiles():
    small = Accel(sbuf_bytes=512 * 1024, psum_bytes=2 * 2**20)
    big = Accel()
    s_small = search_tiles(2048, 2048, 2048, small)
    s_big = search_tiles(2048, 2048, 2048, big)
    assert s_small is not None and s_big is not None
    assert s_small.sbuf_footprint <= small.sbuf_bytes
    assert s_big.psum_footprint <= big.psum_bytes
    assert s_big.intensity > s_small.intensity
    base = tile_stats(2048, 2048, 2048, 128, 128, 128, big, with_cycles=False)
    assert s_small.intensity > base.intensity


def test_intensity_bound_formula():
    s = tile_stats(8192, 8192, 8192, 512, 512, 128, Accel(), with_cycles=False)
    ub = intensity_upper_bound(512, 512, 2)
    assert s.intensity < ub and s.intensity == pytest.approx(ub, rel=0.05)


@pytest.mark.parametrize("m,k,n,p", [(4, 4, 4, 4), (10, 7, 5, 4), (33, 20, 17, 8), (1, 3, 9, 2)])
def test_systolic_correct_and_cycle_count(m, k, n, p):
    a, b = RNG.standard_normal((m, k)), RNG.standard_normal((k, n))
    c, st = systolic_matmul(a, b, p)
    np.testing.assert_allclose(c, a @ b, rtol=1e-12, atol=1e-12)
    blocks = -(-k // p) * -(-n // p)
    assert st.blocks == blocks
    assert st.stream_cycles == blocks * block_cycles(m, p)
    assert st.active_pe_cycles == blocks * m * p * p
    assert 0 < st.utilization() <= st.utilization(pipelined=True) <= 1.0


def test_systolic_utilization_rises_with_m():
    b = RNG.standard_normal((8, 8))
    u = [systolic_matmul(RNG.standard_normal((m, 8)), b, 8)[1].utilization() for m in (8, 64, 512)]
    assert u[0] < u[1] < u[2] < 1.0


def test_cpu_gemm_variants():
    a, b = RNG.standard_normal((24, 17)), RNG.standard_normal((17, 31))
    ref = a @ b
    np.testing.assert_allclose(matmul_naive(a, b), ref, rtol=1e-10)
    np.testing.assert_allclose(matmul_rows(a, b), ref, rtol=1e-10)
    for bs in (4, 8, 64):
        np.testing.assert_allclose(matmul_blocked(a, b, bs), ref, rtol=1e-10)


@pytest.mark.parametrize("shape,f,k,stride,pad", [((2, 3, 9, 11), 4, 3, 1, 1), ((1, 5, 16, 16), 6, 5, 2, 2),
                                                  ((3, 2, 8, 8), 3, 1, 1, 0), ((1, 4, 10, 7), 2, 3, 3, 0)])
def test_conv_im2col_vs_direct_vs_torch(shape, f, k, stride, pad):
    x = RNG.standard_normal(shape)
    w = RNG.standard_normal((f, shape[1], k, k))
    yd = conv2d_direct(x, w, stride, pad)
    yi = conv2d_im2col(x, w, stride, pad)
    yt = torch.nn.functional.conv2d(torch.from_numpy(x), torch.from_numpy(w), stride=stride,
                                    padding=pad).numpy()
    np.testing.assert_allclose(yd, yt, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(yi, yt, rtol=1e-10, atol=1e-10)


def test_im2col_blowup_counts_elements():
    x = RNG.standard_normal((1, 8, 20, 20))
    cols = im2col(x, 3, 3, 1, 1)
    assert cols.size / x.size == pytest.approx(blowup_factor(8, 20, 20, 3, 3, 1, 1))
    assert blowup_factor(8, 20, 20, 3, 3, 1, 1) == pytest.approx(9.0)
    assert blowup_factor(8, 20, 20, 1, 1) == pytest.approx(1.0)
