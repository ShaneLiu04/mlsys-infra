"""GPU part (gpu/gpt2_fast.py). Skips without CUDA or without the GPT-2 weights."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from gpt2_fast import (MODEL_ROOT, GPT2, SpecEngine, accept_reject_torch, generate_graph,  # noqa: E402
                       generate_kv, generate_nocache, load_state_dict)

if not os.path.exists(os.path.join(MODEL_ROOT, "gpt2", "model.safetensors")):
    pytest.skip("GPT-2 weights not downloaded", allow_module_level=True)

torch.backends.cuda.matmul.allow_tf32 = False
PROMPT = [464, 3139, 286, 4881, 318, 6342, 13, 383, 3139, 286, 4486, 318]


@pytest.fixture(scope="module")
def small():
    return GPT2("gpt2", torch.float32)


def test_logits_match_hf(small):
    transformers = pytest.importorskip("transformers")
    hf = transformers.GPT2LMHeadModel.from_pretrained(os.path.join(MODEL_ROOT, "gpt2"), dtype=torch.float32,
                                                      attn_implementation="eager").cuda().eval()
    x = torch.randint(0, 50257, (2, 96), device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    with torch.no_grad():
        ref = hf(x).logits
    assert (small.forward(x) - ref).abs().max().item() < 1e-3


def test_greedy_identical_across_variants(small):
    x = torch.tensor([PROMPT], device="cuda")
    a = generate_nocache(small, x, 40)
    assert torch.equal(a, generate_kv(small, x, 40))
    assert torch.equal(a, generate_graph(small, x, 40))
    small.attn_impl = "bmm"
    try:
        assert torch.equal(a, generate_graph(small, x, 40))
    finally:
        small.attn_impl = "sdpa"


@pytest.mark.parametrize("use_graph", [False, True])
def test_speculative_greedy_equals_target(small, use_graph):
    target = GPT2("gpt2-medium", torch.float32) if os.path.exists(
        os.path.join(MODEL_ROOT, "gpt2-medium", "model.safetensors")) else small
    eng = SpecEngine(target, small, 4, 128, use_graph=use_graph)
    ref, _ = eng.target_only(PROMPT, 48)
    out, st = eng.generate(PROMPT, 48)
    assert out == ref
    assert st.accepted + st.rounds >= 48


def test_accept_reject_torch_preserves_distribution():
    from scipy import stats
    V, g, n = 6, 3, 40000
    gen = torch.Generator("cuda").manual_seed(0)
    p = torch.tensor([0.30, 0.25, 0.15, 0.15, 0.10, 0.05], device="cuda")
    q = torch.tensor([0.05, 0.10, 0.40, 0.25, 0.10, 0.10], device="cuda")
    P, Q = p.expand(g + 1, V), q.expand(g, V)
    first = np.zeros(V)
    for _ in range(n):
        d = torch.multinomial(Q, 1, generator=gen)[:, 0]
        n_acc, nxt, _ = accept_reject_torch(P, Q, d, gen)
        k = int(n_acc)
        first[int(d[0]) if k > 0 else int(nxt)] += 1
    assert stats.chisquare(first, p.cpu().numpy() * n).pvalue > 1e-3
    assert stats.chisquare(first, q.cpu().numpy() * n).pvalue < 1e-12
