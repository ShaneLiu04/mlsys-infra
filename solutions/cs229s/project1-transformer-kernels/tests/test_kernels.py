import numpy as np
import pytest
import torch

from tfkernels.attention import (IOCounter, flash_backward, flash_forward, naive_attention,
                                 naive_attention_backward, online_attention, online_softmax)
from tfkernels.conv1d import conv_direct, conv_fft, conv_loop, conv_windows
from tfkernels.io_model import decode_attention_intensity, flash_blocks, flash_elements, naive_elements
from tfkernels.recall import RecallTask, build_hand_coded, recall_accuracy

TOL = 1e-5

SHAPES = [  # (batch dims, Tq, Tk, d, br, bc)
    ((), 16, 16, 8, 4, 4),
    ((), 37, 37, 16, 8, 16),
    ((2, 3), 50, 50, 8, 16, 7),
    ((), 1, 1, 4, 64, 64),
    ((2,), 129, 129, 32, 32, 64),
]


def _qkv(rng, bdims, tq, tk, d):
    return (rng.standard_normal(bdims + (tq, d)), rng.standard_normal(bdims + (tk, d)),
            rng.standard_normal(bdims + (tk, d)))


def _torch_ref(q, k, v, do, causal):
    qt, kt, vt = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
    o = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    o.backward(torch.tensor(do))
    return o.detach().numpy(), qt.grad.numpy(), kt.grad.numpy(), vt.grad.numpy()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("bdims,tq,tk,d,br,bc", SHAPES)
def test_flash_forward_backward_match_torch(bdims, tq, tk, d, br, bc, causal):
    rng = np.random.default_rng(tq * 7 + d)
    q, k, v = _qkv(rng, bdims, tq, tk, d)
    do = rng.standard_normal(q.shape)
    o_ref, dq_ref, dk_ref, dv_ref = _torch_ref(q, k, v, do, causal)
    o, lse = flash_forward(q, k, v, causal, br, bc)
    np.testing.assert_allclose(o, o_ref, atol=TOL, rtol=TOL)
    _, lse_naive = naive_attention(q, k, v, causal)
    np.testing.assert_allclose(lse, lse_naive, atol=TOL)
    dq, dk, dv = flash_backward(q, k, v, o, do, lse, causal, br, bc)
    for got, ref in ((dq, dq_ref), (dk, dk_ref), (dv, dv_ref)):
        np.testing.assert_allclose(got, ref, atol=TOL, rtol=TOL)


def test_non_square_cross_attention():
    rng = np.random.default_rng(1)
    q, k, v = _qkv(rng, (2,), 23, 41, 8)
    do = rng.standard_normal(q.shape)
    o_ref, dq_ref, dk_ref, dv_ref = _torch_ref(q, k, v, do, False)
    o, lse = flash_forward(q, k, v, False, 5, 9)
    np.testing.assert_allclose(o, o_ref, atol=TOL)
    for got, ref in zip(flash_backward(q, k, v, o, do, lse, False, 5, 9), (dq_ref, dk_ref, dv_ref)):
        np.testing.assert_allclose(got, ref, atol=TOL)


@pytest.mark.parametrize("causal", [False, True])
def test_naive_and_online_match_torch(causal):
    rng = np.random.default_rng(2)
    q, k, v = _qkv(rng, (3,), 33, 33, 16)
    do = rng.standard_normal(q.shape)
    o_ref, dq_ref, dk_ref, dv_ref = _torch_ref(q, k, v, do, causal)
    np.testing.assert_allclose(naive_attention(q, k, v, causal)[0], o_ref, atol=TOL)
    np.testing.assert_allclose(online_attention(q, k, v, causal), o_ref, atol=TOL)
    for got, ref in zip(naive_attention_backward(q, k, v, do, causal), (dq_ref, dk_ref, dv_ref)):
        np.testing.assert_allclose(got, ref, atol=TOL)


def test_online_softmax_stable_for_large_logits():
    x = np.array([[1000.0, 1001.0, 999.0], [-5.0, 0.0, 5.0]])
    ref = torch.softmax(torch.tensor(x), -1).numpy()
    np.testing.assert_allclose(online_softmax(x), ref, atol=1e-12)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("t,d,br,bc", [(64, 8, 8, 16), (100, 16, 16, 16), (37, 4, 4, 8)])
def test_io_counter_matches_model(t, d, br, bc, causal):
    rng = np.random.default_rng(3)
    q, k, v = _qkv(rng, (), t, t, d)
    io = IOCounter()
    flash_forward(q, k, v, causal, br, bc, io=io)
    assert io.total == flash_elements(t, d, br, bc, causal)
    if causal:
        assert io.blocks_skipped > 0 or t <= bc


def test_io_ratio_grows_with_t():
    br, bc = flash_blocks(100 * 1024, 64)
    assert (br, bc) == (64, 200)
    r = [naive_elements(t, 64) / flash_elements(t, 64, br, bc) for t in (1024, 4096, 16384)]
    assert r[0] < r[1] < r[2]


def test_decode_intensity_formula():
    rows = {r["variant"]: r for r in decode_attention_intensity()}
    assert rows["MHA"]["intensity_flop_per_byte"] == pytest.approx(1.0)
    assert rows["GQA-8"]["intensity_flop_per_byte"] == pytest.approx(4.0)
    assert rows["MQA"]["intensity_flop_per_byte"] == pytest.approx(32.0)
    assert all(r["bound"] == "memory" for r in rows.values())


@pytest.mark.parametrize("c,n,k", [(3, 20, 5), (4, 7, 7), (2, 5, 9), (1, 64, 64), (5, 33, 1)])
def test_conv_variants_agree(c, n, k):
    rng = np.random.default_rng(4)
    x, h = rng.standard_normal((c, n)), rng.standard_normal((c, k))
    ref = conv_loop(x, h)
    tconv = torch.nn.functional.conv1d(torch.tensor(np.pad(x, ((0, 0), (k - 1, 0))))[None],
                                       torch.tensor(h[:, ::-1].copy())[:, None], groups=c)[0].numpy()
    np.testing.assert_allclose(ref, tconv, atol=1e-10)
    for fn in (conv_direct, conv_windows, conv_fft):
        np.testing.assert_allclose(fn(x, h), ref, atol=1e-9)


def test_conv_is_causal():
    rng = np.random.default_rng(5)
    x, h = rng.standard_normal((2, 30)), rng.standard_normal((2, 30))
    y0 = conv_fft(x, h)
    x2 = x.copy()
    x2[:, 20:] += 100.0
    np.testing.assert_allclose(conv_fft(x2, h)[:, :20], y0[:, :20], atol=1e-8)


@pytest.mark.parametrize("pairs,keys", [(4, 8), (8, 16), (16, 16), (24, 32)])
def test_hand_coded_recall_is_exact(pairs, keys):
    task = RecallTask(n_keys=keys, n_values=16, n_pairs=pairs)
    assert recall_accuracy(build_hand_coded(task), task, 2000, seed=pairs) == 1.0


def test_recall_attention_patterns():
    task = RecallTask(n_pairs=6)
    model = build_hand_coded(task)
    seq, _ = task.sample(4, np.random.default_rng(0))
    _, (a1, a2) = model(seq, return_attn=True)
    t = task.seq_len
    assert torch.all(a1[:, torch.arange(1, t), torch.arange(0, t - 1)] > 0.99)
    q = seq[:, -1]
    hit = (seq[:, :-1] == q[:, None]).int().argmax(1) + 1
    assert torch.all(a2[torch.arange(4), -1, hit] > 0.99)
