import numpy as np
import pytest

from compkit.graph_ir import (Graph, const, constant_fold, cse, evaluate, fuse_elementwise,
                              inp, kernel_cost, merge_shared_matmuls, op, qkv_projection,
                              transformer_mlp)

RNG = np.random.default_rng(0)


def _feeds(g):
    return {n.attrs["name"]: RNG.normal(size=n.attrs["shape"]) for n in g.nodes() if n.op == "input"}


def test_constant_folding():
    x = inp("x", (3, 4))
    c = op("mul", op("add", const(np.ones(4)), const(np.full(4, 2.0))), const(np.full(4, 0.5)))
    g = Graph([op("add", x, c)])
    f = constant_fold(g)
    assert len(f.kernels()) == 1
    feeds = _feeds(g)
    np.testing.assert_array_equal(evaluate(f, feeds)[0], evaluate(g, feeds)[0])


def test_cse_dedupes():
    x = inp("x", (3, 4))
    w = const(RNG.normal(size=(4, 4)))
    a = op("relu", op("matmul", x, w))
    b = op("relu", op("matmul", x, w))
    g = Graph([op("add", a, b)])
    c = cse(g)
    assert len(g.kernels()) == 5 and len(c.kernels()) == 3
    feeds = _feeds(g)
    np.testing.assert_array_equal(evaluate(c, feeds)[0], evaluate(g, feeds)[0])


def test_fusion_groups_elementwise_chain():
    g = transformer_mlp(16, 8, 32, RNG)
    f = fuse_elementwise(g)
    ops = [n.attrs.get("ops", [n.op]) for n in f.kernels()]
    assert ["add", "gelu"] in ops and ["add", "add"] in ops
    assert len(f.kernels()) == 5
    e = fuse_elementwise(g, epilogue=True)
    assert [n.attrs.get("ops", [n.op]) for n in e.kernels()] == [
        ["layernorm"], ["matmul", "add", "gelu"], ["matmul", "add", "add"]]
    feeds = _feeds(g)
    ref = evaluate(g, feeds)[0]
    np.testing.assert_array_equal(evaluate(f, feeds)[0], ref)
    np.testing.assert_array_equal(evaluate(e, feeds)[0], ref)


def test_fusion_does_not_fuse_multi_consumer_intermediate():
    x = inp("x", (4, 4))
    a = op("exp", x)
    g = Graph([op("add", op("relu", a), op("mul", a, a))])
    f = fuse_elementwise(g)
    feeds = _feeds(g)
    np.testing.assert_array_equal(evaluate(f, feeds)[0], evaluate(g, feeds)[0])
    assert any(n.op == "exp" for n in f.kernels())  # exp stays materialised


def test_merge_shared_matmuls_qkv():
    g = qkv_projection(8, 16, 4, RNG)
    m = merge_shared_matmuls(g)
    assert sum(n.op == "matmul" for n in g.kernels()) == 3
    assert sum(n.op == "matmul" for n in m.kernels()) == 1
    feeds = _feeds(g)
    for a, b in zip(evaluate(m, feeds), evaluate(g, feeds)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)


def test_fusion_reduces_traffic_at_realistic_shape():
    g = transformer_mlp(4096, 4096, 16384)
    base = sum(r["bytes"] for r in kernel_cost(g))
    fused = sum(r["bytes"] for r in kernel_cost(fuse_elementwise(g)))
    epi = sum(r["bytes"] for r in kernel_cost(fuse_elementwise(g, epilogue=True)))
    assert epi < fused < base
    # elementwise fusion removes write+read of the [T, d_ff] intermediate (bias->gelu) and [T, d]
    t, d, f = 4096, 4096, 16384
    assert base - fused == pytest.approx(2 * 2 * (t * f + t * d))
