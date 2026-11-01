import numpy as np
import pytest
import torch

from cpupar.mandelbrot import (
    MandelSpec,
    assignment_work,
    dynamic_chunks,
    interleaved_rows,
    mandel_image,
    mandel_points,
    mandel_points_masked,
    mandel_rows,
    mandel_scalar,
    row_coords,
    static_blocks,
)
from cpupar.parallel import make_pool, run_parallel
from cpupar.saxpy import effective_gbps, saxpy_numpy, saxpy_numpy_threads, saxpy_torch
from cpupar.simd import (
    gang_issued_slots,
    gang_utilization,
    sqrt_inputs,
    sqrt_iterations,
    sqrt_masked,
    sqrt_scalar,
)

SMALL = MandelSpec(width=64, height=48, max_iter=64)


def test_compacted_equals_masked_kernel():
    c_re, c_im = row_coords(SMALL, np.arange(SMALL.height))
    a = mandel_points(c_re, c_im, SMALL.max_iter)
    b = mandel_points_masked(c_re, c_im, SMALL.max_iter)
    np.testing.assert_array_equal(a, b)


def test_vector_kernel_matches_scalar_mostly():
    c_re, c_im = row_coords(SMALL, np.arange(SMALL.height))
    vec = mandel_points(c_re, c_im, SMALL.max_iter)
    ref = np.array([mandel_scalar(float(r), float(i), SMALL.max_iter) for r, i in zip(c_re, c_im)])
    # float32 vs float64 can flip points on the boundary; demand near-total agreement
    assert np.mean(vec == ref) > 0.97


def test_known_points():
    assert mandel_scalar(0.0, 0.0, 100) == 100
    assert mandel_scalar(2.0, 2.0, 100) == 0
    out = mandel_points(np.array([0.0, 2.0], np.float32), np.array([0.0, 2.0], np.float32), 100)
    assert out.tolist() == [100, 0]


@pytest.mark.parametrize("workers", [1, 3, 8])
def test_schedules_partition_rows(workers):
    h = 50
    for parts in (static_blocks(h, workers), interleaved_rows(h, workers), dynamic_chunks(h, 7)):
        allrows = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(allrows, np.arange(h))


def test_interleaved_balances_better_than_static():
    spec = MandelSpec(width=160, height=120, max_iter=128)
    counts = mandel_image(spec)
    st = assignment_work(counts, static_blocks(spec.height, 4))
    il = assignment_work(counts, interleaved_rows(spec.height, 4))
    assert st.max() / st.mean() > il.max() / il.mean()


def test_parallel_matches_serial():
    ref = mandel_image(SMALL)
    wp = make_pool(3)
    try:
        for sched in ("static", "interleaved", "dynamic"):
            res = run_parallel(wp, SMALL, sched, chunk_rows=5)
            np.testing.assert_array_equal(res.image, ref)
            if sched != "dynamic":
                assert len(res.worker_busy_s) == 3, "static blocks must run on distinct workers"
    finally:
        wp.close()


def test_mandel_rows_shape():
    out = mandel_rows(SMALL, np.array([0, 5, 7]))
    assert out.shape == (3, SMALL.width)


def test_gang_utilization_basic():
    assert gang_utilization(np.full(32, 5), 8) == 1.0
    w = np.array([8, 1, 1, 1])
    assert gang_utilization(w, 4) == pytest.approx(11 / 32)
    assert gang_utilization(w, 1) == 1.0
    assert gang_issued_slots(w, 2) == 2 * 8 + 2 * 1


def test_gang_utilization_tail_padding():
    assert gang_utilization(np.array([4, 4, 4]), 4) == pytest.approx(12 / 16)


def test_sqrt_correct_and_matches_scalar():
    x = sqrt_inputs("uniform", 2000, seed=1)
    r, it = sqrt_iterations(x)
    np.testing.assert_allclose(r, np.sqrt(x), rtol=1e-4)
    np.testing.assert_allclose(sqrt_masked(x), np.sqrt(x), rtol=1e-4)
    for i in range(0, 2000, 97):
        _, si = sqrt_scalar(float(x[i]))
        assert abs(si - int(it[i])) <= 1


def test_sqrt_patterns_divergence():
    w = 8
    fast = sqrt_iterations(sqrt_inputs("all_fast", 64))[1]
    slow = sqrt_iterations(sqrt_inputs("all_slow", 64))[1]
    adv = sqrt_iterations(sqrt_inputs("one_slow_per_gang", 64, width=w))[1]
    assert fast.max() == 0 and slow.min() > 10
    assert gang_utilization(slow + 1, w) == 1.0
    assert gang_utilization(adv + 1, w) < 0.25


def test_saxpy_variants_agree():
    rng = np.random.default_rng(0)
    x = rng.random(10_001, dtype=np.float32)
    y = rng.random(10_001, dtype=np.float32)
    ref = 2.5 * x + y
    out = np.empty_like(x)
    saxpy_numpy(2.5, x, y, out)
    np.testing.assert_allclose(out, ref, rtol=1e-6)
    from concurrent.futures import ThreadPoolExecutor

    out2 = np.empty_like(x)
    with ThreadPoolExecutor(3) as pool:
        saxpy_numpy_threads(pool, 3, 2.5, x, y, out2)
    np.testing.assert_allclose(out2, ref, rtol=1e-6)
    t = torch.empty(x.size)
    saxpy_torch(2.5, torch.from_numpy(x), torch.from_numpy(y), t)
    np.testing.assert_allclose(t.numpy(), ref, rtol=1e-6)


def test_effective_gbps():
    assert effective_gbps(1_000_000_000, 12.0) == pytest.approx(1.0)
