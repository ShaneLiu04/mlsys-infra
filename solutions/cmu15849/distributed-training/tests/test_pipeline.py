import pytest

import mlsys_calc
from disttrain.pipeline import bubble_fraction, peak_in_flight, simulate


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
@pytest.mark.parametrize("p,m", [(2, 4), (4, 8), (4, 16), (8, 8), (8, 32)])
def test_bubble_matches_formula(schedule, p, m):
    ops = simulate(schedule, p, m)
    assert len(ops) == 2 * p * m
    assert bubble_fraction(ops, p) == pytest.approx(mlsys_calc.pipeline_bubble(p, m), abs=1e-12)


@pytest.mark.parametrize("p,m", [(4, 8), (8, 16)])
def test_peak_activations(p, m):
    assert peak_in_flight(simulate("gpipe", p, m), p) == [m] * p
    assert peak_in_flight(simulate("1f1b", p, m), p) == [p - d for d in range(p)]


@pytest.mark.parametrize("p,m,v", [(4, 8, 2), (4, 16, 2), (4, 8, 4)])
def test_interleaved_reduces_bubble(p, m, v):
    ops = simulate("interleaved", p, m, v=v)
    assert len(ops) == 2 * p * m * v
    b = bubble_fraction(ops, p)
    assert b < bubble_fraction(simulate("1f1b", p, m), p)
    assert b == pytest.approx(mlsys_calc.pipeline_bubble(p, m, v), abs=1e-9)


def test_dependencies_respected():
    ops = simulate("1f1b", 4, 8)
    end = {(o.kind, o.device, o.mb): o.end for o in ops}
    for o in ops:
        if o.kind == "F" and o.device > 0:
            assert o.start >= end[("F", o.device - 1, o.mb)]
        if o.kind == "B" and o.device < 3:
            assert o.start >= end[("B", o.device + 1, o.mb)]
