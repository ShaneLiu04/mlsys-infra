import numpy as np
import pytest

from scanrender.primitives import (
    find_repeats,
    segmented_exclusive_scan,
    segmented_exclusive_scan_reference,
    stream_compact,
)
from scanrender.renderer import (
    Scene,
    bin_circles,
    brute_force_tests,
    make_scene,
    render_reference,
    render_tiled,
    render_unordered,
)
from scanrender.scan import (
    analytic_stats,
    blelloch_exclusive,
    exclusive_scan_sequential,
    hillis_steele_exclusive,
    next_pow2,
)

SCANS = [exclusive_scan_sequential, hillis_steele_exclusive, blelloch_exclusive]
NAMES = {exclusive_scan_sequential: "sequential", hillis_steele_exclusive: "hillis_steele",
         blelloch_exclusive: "blelloch"}


def ref_exclusive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    out = np.zeros_like(x)
    out[1:] = np.cumsum(x)[:-1]
    return out


@pytest.mark.parametrize("scan", SCANS)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 8, 17, 100, 1000, 1023, 1024, 1025])
def test_scan_matches_cumsum(scan, n):
    x = np.random.default_rng(n).integers(-50, 50, n)
    out, _ = scan(x)
    np.testing.assert_array_equal(out, ref_exclusive(x))


@pytest.mark.parametrize("scan", [hillis_steele_exclusive, blelloch_exclusive])
def test_scan_large_random(scan):
    x = np.random.default_rng(7).integers(0, 10, 100_003)
    np.testing.assert_array_equal(scan(x)[0], ref_exclusive(x))


@pytest.mark.parametrize("scan", SCANS)
def test_int32_input_does_not_overflow(scan):
    x = np.full(3000, 2_000_000_000, dtype=np.int32)
    out, _ = scan(x)
    assert out.dtype == np.int64
    assert out[-1] == 2_000_000_000 * 2999


@pytest.mark.parametrize("scan", SCANS)
def test_bool_and_float_inputs(scan):
    b = np.random.default_rng(1).random(77) < 0.3
    np.testing.assert_array_equal(scan(b)[0], ref_exclusive(b.astype(np.int64)))
    f = np.random.default_rng(2).random(77)
    expect = np.concatenate([[0.0], np.cumsum(f)[:-1]])
    np.testing.assert_allclose(scan(f)[0], expect, rtol=1e-12)


@pytest.mark.parametrize("scan", SCANS)
@pytest.mark.parametrize("n", [1, 2, 7, 64, 100, 1000, 4097])
def test_empirical_work_span_equals_analytic(scan, n):
    _, st = scan(np.ones(n, dtype=np.int64))
    an = analytic_stats(NAMES[scan], n)
    assert (st.adds, st.steps, st.padded_n) == (an.adds, an.steps, an.padded_n)


def test_work_span_orders():
    n = 1 << 16
    s, h, b = (analytic_stats(a, n) for a in ("sequential", "hillis_steele", "blelloch"))
    assert h.steps == 16 and b.steps == 32 and s.steps == n - 1
    assert b.adds == 2 * (n - 1) and h.adds == 16 * n - (n - 1)
    assert next_pow2(1025) == 2048


def test_find_repeats():
    a = np.array([1, 2, 2, 1, 1, 1, 3, 5, 3, 3])
    np.testing.assert_array_equal(find_repeats(a), [1, 3, 4, 8])
    assert find_repeats(np.array([1])).size == 0
    assert find_repeats(np.array([1, 2, 3])).size == 0
    rng = np.random.default_rng(0)
    for n in (2, 31, 1000, 4097):
        a = rng.integers(0, 4, n)
        expect = np.nonzero(a[:-1] == a[1:])[0]
        np.testing.assert_array_equal(find_repeats(a), expect)


def test_stream_compact():
    rng = np.random.default_rng(3)
    v = rng.normal(size=1001)
    np.testing.assert_array_equal(stream_compact(v, v > 0), v[v > 0])
    assert stream_compact(v, np.zeros(1001, bool)).size == 0
    assert stream_compact(v[:0], np.zeros(0, bool)).size == 0


@pytest.mark.parametrize("n", [1, 2, 9, 100, 1025])
def test_segmented_scan(n):
    rng = np.random.default_rng(n)
    v = rng.integers(-5, 10, n)
    heads = rng.random(n) < 0.2
    np.testing.assert_array_equal(segmented_exclusive_scan(v, heads), segmented_exclusive_scan_reference(v, heads))


def test_segmented_scan_all_heads_and_none():
    v = np.arange(10)
    np.testing.assert_array_equal(segmented_exclusive_scan(v, np.ones(10, bool)), np.zeros(10))
    np.testing.assert_array_equal(segmented_exclusive_scan(v, np.zeros(10, bool)), np.concatenate([[0], np.cumsum(v)[:-1]]))


def tiny_scene() -> Scene:
    f = np.float32
    return Scene(np.array([[10, 10], [14, 10]], f), np.array([6, 6], f),
                 np.array([[1, 0, 0], [0, 0, 1]], f), np.array([0.5, 0.5], f))


def test_over_blend_is_order_dependent():
    s = tiny_scene()
    a, _ = render_reference(s, 24, 24)
    b, _ = render_reference(s.permuted(np.array([1, 0])), 24, 24)
    assert np.abs(a - b).max() > 0.1


@pytest.mark.parametrize("kind", ["few_big", "many_small", "mixed"])
@pytest.mark.parametrize("tile", [8, 16, 32])
def test_tiled_equals_reference(kind, tile):
    s = make_scene(kind, 128, seed=1)
    ref, _ = render_reference(s, 128, 128)
    out, _ = render_tiled(s, 128, 128, tile)
    assert np.abs(out - ref).max() <= 1e-6


def test_tiled_non_multiple_image_size():
    s = make_scene("mixed", 100, seed=2)
    ref, _ = render_reference(s, 90, 100)
    out, _ = render_tiled(s, 90, 100, 16)
    assert np.abs(out - ref).max() <= 1e-6


def test_unordered_is_wrong():
    s = make_scene("few_big", 128, seed=0)
    ref, _ = render_reference(s, 128, 128)
    bad = render_unordered(s, 128, 128, seed=5)
    assert np.abs(bad - ref).max() > 0.05


def test_bins_preserve_order_and_cover():
    s = make_scene("mixed", 128, seed=4)
    lists, st = bin_circles(s, 128, 128, 16)
    assert st.bbox_tests == s.n * len(lists)
    for lst in lists:
        assert np.all(np.diff(lst) > 0)


def test_binning_saves_tests_for_small_circles():
    s = make_scene("many_small", 256, seed=0)
    _, st = render_tiled(s, 256, 256, 16)
    assert st.circle_pixel_tests + st.bbox_tests < brute_force_tests(s, 256, 256) / 20
