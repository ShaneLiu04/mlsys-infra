"""Tests for the GPU part. Algorithmic pieces (GPTQ, N:M masks, SmoothQuant identity) run on CPU;
kernel tests skip without CUDA/Triton."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))

from llm_utils import ActQuantLinear, quant_asym, quant_sym  # noqa: E402
from prune_bench import nm_mask, row_mask, unstructured_mask  # noqa: E402
from quant_llm import awq_quantize, gptq_quantize  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _correlated(n=2048, d=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    basis = torch.randn(d, d, generator=g)
    spec = torch.arange(1, d + 1, dtype=torch.float32) ** -1.0
    return torch.randn(n, d, generator=g) @ (basis * spec[:, None])


def _proxy(dw, H):
    return float(torch.einsum("oi,ij,oj->", dw, H, dw))


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    x = _correlated()
    H = x.t() @ x / x.shape[0]
    w = torch.randn(128, 256)
    for bits, g in [(4, 128), (3, 128), (4, None)]:
        rtn = quant_asym(w, bits, g)
        q = gptq_quantize(w, H, bits, g)
        assert _proxy(q - w, H) < 0.5 * _proxy(rtn - w, H)


def test_gptq_output_on_grid():
    torch.manual_seed(1)
    x = _correlated(512, 128)
    w = torch.randn(16, 128)
    q = gptq_quantize(w, x.t() @ x, 4, None)
    for row in q:
        assert torch.unique(row).numel() <= 16  # 4-bit grid per output channel


def test_awq_not_worse_than_rtn():
    x = _correlated(1024, 256, seed=2)
    x[:, :4] *= 30  # salient input channels
    H = x.t() @ x / x.shape[0]
    w = torch.randn(64, 256)
    a = awq_quantize(w, H, torch.sqrt(torch.diag(H)), 4, 128)
    assert _proxy(a - w, H) <= _proxy(quant_asym(w, 4, 128) - w, H) + 1e-6


def test_rtn_granularity_ordering():
    torch.manual_seed(0)
    w = torch.distributions.StudentT(4.0).sample((256, 512))
    e = {g: float((quant_sym(w, 8, g) - w).norm()) for g in ["tensor", "channel", "group"]}
    assert e["group"] < e["channel"] < e["tensor"]


def test_masks():
    torch.manual_seed(0)
    s = torch.rand(8, 16)
    m = nm_mask(s, 2, 4)
    assert torch.all(m.view(8, 4, 4).sum(-1) == 2)
    assert abs(float((~unstructured_mask(s, 0.5)).float().mean()) - 0.5) < 1e-6
    assert torch.all((~row_mask(s, 0.25)).sum(1) == 4)


def test_smoothquant_identity_without_quant_noise():
    """(x / s) (W diag(s))^T == x W^T: with act='none' only the weight rounding remains."""
    torch.manual_seed(0)
    w = torch.randn(32, 64, dtype=torch.float64)
    x = torch.randn(10, 64, dtype=torch.float64)
    x[:, 3] *= 50
    s = x.abs().amax(0) ** 0.5 / w.abs().amax(0) ** 0.5
    q = ActQuantLinear(w, None, "none", s)
    ref = x @ w.t()
    assert float((q(x) - ref).norm() / ref.norm()) < 0.02


def test_llm_int8_beats_per_tensor_with_outliers():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 0.05
    x = torch.randn(256, 128)
    x[:, [5, 77]] *= 40
    ref = x @ w.t()
    err = {a: float((ActQuantLinear(w, None, a)(x) - ref).norm() / ref.norm()) for a in ["tensor", "llm_int8"]}
    assert err["llm_int8"] < 0.3 * err["tensor"]


# --------------------------------------------------------------------------- CUDA / Triton kernels

@cuda
def test_triton_rmsnorm_and_bias_gelu():
    tk = pytest.importorskip("triton_kernels")
    x = torch.randn(64, 1000, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1000, device="cuda", dtype=torch.bfloat16)
    assert float((tk.rmsnorm(x, w).float() - tk.rmsnorm_ref(x, w).float()).abs().max()) < 0.05
    y = tk.bias_gelu(x, w)
    ref = torch.nn.functional.gelu((x + w).float())
    assert float((y.float() - ref).abs().max()) < 0.05


@cuda
def test_triton_matmul_and_autotune():
    tk = pytest.importorskip("triton_kernels")
    a = torch.randn(300, 200, device="cuda", dtype=torch.float16)
    b = torch.randn(200, 260, device="cuda", dtype=torch.float16)
    ref = (a.float() @ b.float())
    for c in [tk.matmul(a, b, 64, 64, 32, 4, 2), tk.matmul_autotuned(a, b)]:
        assert float((c.float() - ref).abs().max() / ref.abs().max()) < 1e-2


@cuda
def test_weight_only_gemv():
    tk = pytest.importorskip("triton_kernels")
    w = torch.randn(256, 512, device="cuda", dtype=torch.float16)
    x = torch.randn(512, device="cuda", dtype=torch.float16)
    q8, s8 = tk.quant_w8(w)
    ref8 = (q8.float() * s8.float()[:, None]) @ x.float()
    assert float((tk.gemv_w8(x, q8, s8).float() - ref8).norm() / ref8.norm()) < 1e-2
    q4, s4 = tk.quant_w4(w)
    ref4 = tk.dequant_w4(q4, s4) @ x.float()
    assert float((tk.gemv_w4(x, q4, s4).float() - ref4).norm() / ref4.norm()) < 1e-2
    assert float((tk.dequant_w4(q4, s4) - w.float()).norm() / w.float().norm()) < 0.15


@cuda
def test_lora_merge_equivalence():
    pytest.importorskip("triton_kernels")
    from lora_ft import LoRALinear
    base = torch.nn.Linear(64, 32).cuda()
    m = LoRALinear(base, 4, 8.0).float()
    with torch.no_grad():
        m.B.normal_()
    x = torch.randn(5, 64, device="cuda")
    y_merged = x @ m.merged_weight().t() + m.b.float()
    assert float((m(x) - y_merged).abs().max()) < 1e-4
