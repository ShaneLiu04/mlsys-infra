"""GPU-part tests: Triton FlashAttention fwd/bwd, CUDA depthwise conv, tiny GPT. Skipped without CUDA."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)
pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))

from flash_triton import flash_attention, flash_attn_fwd, naive_attention, reference_fp32  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("b,h,t,d", [(1, 2, 128, 64), (2, 3, 300, 64), (1, 2, 257, 128), (1, 1, 1024, 128)])
def test_flash_forward_matches_fp32(b, h, t, d, causal, dtype):
    torch.manual_seed(t + d)
    q, k, v = (torch.randn(b, h, t, d, device="cuda", dtype=dtype) for _ in range(3))
    o, lse = flash_attn_fwd(q, k, v, causal)
    o_ref, lse_ref = reference_fp32(q, k, v, causal)
    tol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    assert (o.float() - o_ref).abs().max().item() < tol
    assert (lse - lse_ref).abs().max().item() < 1e-3


@pytest.mark.parametrize("bm,bn", [(16, 16), (32, 64), (64, 16), (128, 32), (64, 128)])
def test_flash_forward_block_sizes(bm, bn):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 384, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    for causal in (False, True):
        o, _ = flash_attn_fwd(q, k, v, causal, block_m=bm, block_n=bn, num_stages=1)
        o_ref, _ = reference_fp32(q, k, v, causal)
        assert (o.float() - o_ref).abs().max().item() < 4e-3


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("t,d", [(256, 64), (200, 64), (384, 128)])
def test_flash_backward_matches_fp32_autograd(t, d, causal):
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 2, t, d, device="cuda", dtype=torch.float16, requires_grad=True) for _ in range(3))
    do = torch.randn(2, 2, t, d, device="cuda", dtype=torch.float16)
    flash_attention(q, k, v, causal).backward(do)
    got = [x.grad.float().clone() for x in (q, k, v)]
    qf, kf, vf = (x.detach().float().requires_grad_(True) for x in (q, k, v))
    torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, is_causal=causal).backward(do.float())
    for g, r in zip(got, (qf.grad, kf.grad, vf.grad)):
        assert (g - r).abs().max().item() < 2e-2 * max(1.0, r.abs().max().item())


def test_naive_matches_sdpa():
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (naive_attention(q, k, v, True) - ref).abs().max().item() < 4e-3


@pytest.fixture(scope="module")
def dwconv():
    from dwconv_cuda import load_ext
    return load_ext()


@pytest.mark.parametrize("n,k", [(1000, 3), (4096, 4), (777, 64), (2048, 1024), (64, 300)])
def test_dwconv_kernels_match_reference(dwconv, n, k):
    from dwconv_cuda import conv_fft, conv_torch, run_kernel
    torch.manual_seed(n + k)
    x = torch.randn(2, 5, n, device="cuda")
    h = torch.randn(5, k, device="cuda") / np.sqrt(k)
    ref = conv_torch(x.double(), h.double()).float()
    for name in ("naive", "smem", "regblock"):
        assert (run_kernel(dwconv, name, x, h) - ref).abs().max().item() < 1e-4, name
    assert (conv_fft(x, h) - ref).abs().max().item() < 1e-3


def test_dwconv_is_causal(dwconv):
    from dwconv_cuda import run_kernel
    x = torch.randn(1, 3, 512, device="cuda")
    h = torch.randn(3, 33, device="cuda")
    y0 = run_kernel(dwconv, "regblock", x, h)
    x2 = x.clone()
    x2[..., 300:] += 100.0
    y1 = run_kernel(dwconv, "regblock", x2, h)
    assert torch.equal(y0[..., :300], y1[..., :300])


def test_tiny_gpt_attention_impls_agree():
    from tiny_gpt import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab=64, seq_len=32, d_model=64, n_layers=2, n_heads=4)).cuda()
    idx = torch.randint(0, 64, (2, 32), device="cuda")
    outs = {}
    for impl in ("naive", "sdpa", "triton"):
        m.set_attn_impl(impl)
        with torch.autocast("cuda", dtype=torch.float16):
            outs[impl] = m(idx).float()
    assert (outs["naive"] - outs["sdpa"]).abs().max().item() < 2e-2
    assert (outs["naive"] - outs["triton"]).abs().max().item() < 2e-2
