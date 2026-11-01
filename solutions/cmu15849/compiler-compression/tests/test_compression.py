import numpy as np
import pytest
import torch
from torch import nn

from compkit.lora import LoRALinear, fit_low_rank_update, trainable_params
from compkit.pruning import magnitude_mask, nm_mask
from compkit.quantization import (gptq, kmeans_quantize, llm_int8, quantize_weight, rel_err, smooth,
                                  synthetic_activations, synthetic_weight, w8a8)

RNG = np.random.default_rng(0)


def test_granularity_ordering():
    w = synthetic_weight(512, 256, np.random.default_rng(1))
    errs = {g: rel_err(w, quantize_weight(w, 4, g)) for g in ("tensor", "channel", "group")}
    assert errs["group"] < errs["channel"] < errs["tensor"]
    assert rel_err(w, quantize_weight(w, 8, "tensor")) < errs["group"]


def test_quantize_levels():
    w = synthetic_weight(256, 64, np.random.default_rng(2))
    q = quantize_weight(w, 4, "channel")
    for j in range(w.shape[1]):
        assert len(np.unique(q[:, j])) <= 15


def test_outliers_break_per_tensor_w8a8_and_fixes_work():
    rng = np.random.default_rng(3)
    w = synthetic_weight(512, 256, rng)
    x_clean, _ = synthetic_activations(256, 512, rng)
    x_out, _ = synthetic_activations(256, 512, rng, n_outlier=4, outlier_scale=50)
    clean = rel_err(x_clean @ w, w8a8(x_clean, w, "tensor", "channel"))
    bad = rel_err(x_out @ w, w8a8(x_out, w, "tensor", "channel"))
    assert bad > 3 * clean
    y_int8, n_out = llm_int8(x_out, w)
    assert n_out == 4 and rel_err(x_out @ w, y_int8) < clean
    xs, ws = smooth(x_out, w, 0.5)
    np.testing.assert_allclose(xs @ ws, x_out @ w, rtol=1e-10, atol=1e-10)
    assert rel_err(x_out @ w, w8a8(xs, ws, "tensor", "channel")) < bad / 2


def test_gptq_beats_rtn_on_correlated_inputs():
    rng = np.random.default_rng(4)
    w = synthetic_weight(128, 64, rng)
    x_cal, _ = synthetic_activations(1024, 128, rng, rank_decay=1.0)
    rtn = rel_err(x_cal @ w, x_cal @ quantize_weight(w, 4, "channel"))
    assert rel_err(x_cal @ w, x_cal @ gptq(w, x_cal, 4)) < rtn


def test_kmeans_16_centroids():
    w = synthetic_weight(256, 64, np.random.default_rng(5))
    q, c = kmeans_quantize(w, 16)
    assert len(np.unique(q)) <= 16 and len(c) == 16
    assert rel_err(w, q) < rel_err(w, quantize_weight(w, 4, "tensor"))


def test_pruning_masks():
    w = RNG.normal(size=(64, 32))
    m = magnitude_mask(w, 0.7)
    assert m.sum() == round(0.3 * w.size)
    assert np.abs(w[~m]).max() <= np.abs(w[m]).min()
    nm = nm_mask(w)
    assert np.all(nm.reshape(16, 4, 32).sum(1) == 2)


def test_lora_init_params_and_merge():
    torch.manual_seed(0)
    base = nn.Linear(64, 48, dtype=torch.float64)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(5, 64, dtype=torch.float64)
    torch.testing.assert_close(lora(x), base(x), rtol=0, atol=0)
    assert trainable_params(lora) == 4 * (64 + 48)
    with torch.no_grad():
        lora.B.normal_()
    torch.testing.assert_close(lora.merged()(x), lora(x), rtol=1e-12, atol=1e-12)


def test_lora_fits_low_rank_update():
    r = fit_low_rank_update(d=32, true_rank=4, r=4, steps=400)
    assert r["rel_err_dW"] < 0.05
    r2 = fit_low_rank_update(d=32, true_rank=4, r=2, steps=400)
    assert r2["rel_err_dW"] >= r2["eckart_young_bound"] - 1e-6
