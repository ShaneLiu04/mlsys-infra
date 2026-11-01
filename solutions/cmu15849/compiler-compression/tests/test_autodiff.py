import numpy as np
import pytest
import torch

from compkit.autodiff import Tensor, numerical_grad


def _mlp_loss(x, w1, b1, w2, labels):
    h = (x @ w1 + b1).relu()
    return (h @ w2).softmax_cross_entropy(labels)


def test_mlp_grads_match_finite_differences_and_torch():
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(5, 4)))
    w1 = Tensor(rng.normal(size=(4, 6)))
    b1 = Tensor(rng.normal(size=(6,)))
    w2 = Tensor(rng.normal(size=(6, 3)))
    labels = np.array([0, 2, 1, 1, 0])
    loss = _mlp_loss(x, w1, b1, w2, labels)
    loss.backward()

    for p in (x, w1, b1, w2):
        num = numerical_grad(lambda: float(_mlp_loss(x, w1, b1, w2, labels).data), p)
        np.testing.assert_allclose(p.grad, num, rtol=1e-6, atol=1e-8)

    tx, tw1, tb1, tw2 = (torch.tensor(t.data, requires_grad=True) for t in (x, w1, b1, w2))
    tl = torch.nn.functional.cross_entropy(torch.relu(tx @ tw1 + tb1) @ tw2, torch.tensor(labels))
    tl.backward()
    assert float(loss.data) == pytest.approx(tl.item(), rel=1e-12)
    for ours, theirs in ((x, tx), (w1, tw1), (b1, tb1), (w2, tw2)):
        np.testing.assert_allclose(ours.grad, theirs.grad.numpy(), rtol=1e-10, atol=1e-12)


def test_exp_log_mul_sum_and_reuse():
    rng = np.random.default_rng(1)
    a = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)))
    b = Tensor(rng.normal(size=(1, 4)))

    def f():
        return ((a * b).exp() + a.log() * a).sum(axis=1).sum()

    out = f()
    out.backward()
    for p in (a, b):
        np.testing.assert_allclose(p.grad, numerical_grad(lambda: float(f().data), p), rtol=1e-6, atol=1e-8)


def test_deep_chain_no_recursion_limit():
    x = Tensor(np.array([1.0]))
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.sum().backward()
    assert x.grad[0] == pytest.approx(1.0001 ** 5000)
