import pytest
import torch

from disttrain.tensor_parallel import (MLP, Attention, CommLog, ParallelAttention, ParallelMLP,
                                       unshard_grads)


def _check(ref, par, kind, x):
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    y_ref = ref(x1)
    y_par = par(x2)
    torch.testing.assert_close(y_par, y_ref, rtol=1e-10, atol=1e-12)
    g = torch.randn_like(y_ref)
    y_ref.backward(g)
    y_par.backward(g)
    torch.testing.assert_close(x2.grad, x1.grad, rtol=1e-10, atol=1e-12)
    grads = unshard_grads(par, kind)
    for name, p in ref.named_parameters():
        torch.testing.assert_close(grads[name], p.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("t", [1, 2, 4])
def test_mlp_forward_backward_equal(t):
    torch.manual_seed(0)
    ref = MLP(16, 64).double()
    log = CommLog()
    par = ParallelMLP(ref, t, log)
    _check(ref, par, "mlp", torch.randn(2, 5, 16, dtype=torch.float64))
    assert log.count("forward") == 1 and log.count("backward") == 1
    assert log.nbytes("forward") == 2 * 5 * 16 * 8


@pytest.mark.parametrize("t", [1, 2, 4])
def test_attention_forward_backward_equal(t):
    torch.manual_seed(1)
    ref = Attention(32, 4).double()
    log = CommLog()
    par = ParallelAttention(ref, t, log)
    _check(ref, par, "attn", torch.randn(2, 6, 32, dtype=torch.float64))
    assert log.count("forward") == 1 and log.count("backward") == 1


def test_heads_must_divide():
    with pytest.raises(ValueError):
        ParallelAttention(Attention(32, 4), 3, CommLog())
