import numpy as np
import pytest
import torch
from scipy import stats

from kvspec.data import Grammar, decode, encode
from kvspec.model import GPT, GPTConfig, KVCache, generate_kv, generate_nocache
from kvspec.speculative import (accept_reject_greedy, accept_reject_sample, speculative_generate,
                                target_only_kv)

torch.set_num_threads(2)


def _model(n_layer: int, n_embd: int, seed: int, block: int = 128) -> GPT:
    torch.manual_seed(seed)
    m = GPT(GPTConfig(vocab_size=28, block_size=block, n_layer=n_layer, n_head=4, n_embd=n_embd)).double()
    with torch.no_grad():  # sharpen random logits so greedy paths are non-trivial
        m.wte.weight.mul_(25.0)
    return m.eval()


@pytest.fixture(scope="module")
def target() -> GPT:
    return _model(3, 64, 0)


@pytest.fixture(scope="module")
def draft() -> GPT:
    return _model(1, 32, 1)


@torch.no_grad()
def test_kv_cache_matches_full_forward(target):
    idx = torch.randint(0, 28, (2, 40), generator=torch.Generator().manual_seed(0))
    full = target(idx)
    cache = KVCache(target.cfg, 2, torch.float64)
    parts = [target(idx[:, :17], cache), target(idx[:, 17:18], cache), target(idx[:, 18:], cache)]
    np.testing.assert_allclose(torch.cat(parts, 1).numpy(), full.detach().numpy(), atol=1e-10)
    assert cache.length == 40


def test_generate_kv_equals_nocache_greedy_and_logits(target):
    prompt = torch.tensor([encode("abc de")])
    a, la = generate_nocache(target, prompt, 50, return_logits=True)
    b, lb = generate_kv(target, prompt, 50, return_logits=True)
    assert torch.equal(a, b)
    np.testing.assert_allclose(torch.stack(la).numpy(), torch.stack(lb).numpy(), atol=1e-9)


def test_generate_kv_equals_nocache_float32():
    torch.manual_seed(3)
    m = GPT(GPTConfig(vocab_size=28, block_size=128, n_layer=2, n_head=2, n_embd=32)).eval()
    prompt = torch.tensor([[1, 2, 3]])
    _, la = generate_nocache(m, prompt, 30, return_logits=True)
    _, lb = generate_kv(m, prompt, 30, return_logits=True)
    np.testing.assert_allclose(torch.stack(la).numpy(), torch.stack(lb).numpy(), atol=1e-5)


def test_block_size_enforced(target):
    with pytest.raises(ValueError):
        target(torch.zeros(1, 129, dtype=torch.long))


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_greedy_speculative_equals_target_greedy(target, draft, gamma):
    prompt = encode("hello wor")
    ref = target_only_kv(target, prompt, 60)
    out, st = speculative_generate(target, draft, prompt, 60, gamma)
    assert out == ref
    assert st.target_calls == st.rounds and st.accepted <= st.proposed


def test_greedy_speculative_with_self_draft_accepts_everything(target):
    prompt = encode("xyz")
    out, st = speculative_generate(target, target, prompt, 40, 4)
    assert out == target_only_kv(target, prompt, 40)
    assert st.alpha == 1.0 and st.target_calls == 8


def test_sampling_speculative_runs_and_self_draft_accepts_all(target):
    out, st = speculative_generate(target, target, encode("ab"), 30, 3, temperature=1.0, seed=0)
    assert len(out) == 32 and st.alpha == 1.0
    assert st.alpha_overlap == pytest.approx(1.0)


def test_accept_reject_matches_target_distribution():
    """Chi-square: tokens emitted by the speculative sampler ~ p, even though drafts ~ q."""
    p = np.array([0.40, 0.25, 0.15, 0.10, 0.07, 0.03])
    q = np.array([0.05, 0.10, 0.20, 0.30, 0.25, 0.10])
    gamma, trials = 3, 60000
    rng = np.random.default_rng(0)
    first = np.zeros(6, dtype=np.int64)
    all_tok = np.zeros(6, dtype=np.int64)
    accepted = 0
    for _ in range(trials):
        drafts = rng.choice(6, gamma, p=q)
        n, nxt = accept_reject_sample(np.tile(p, (gamma + 1, 1)), np.tile(q, (gamma, 1)), drafts, rng)
        emitted = list(drafts[:n]) + [nxt]
        first[emitted[0]] += 1
        np.add.at(all_tok, emitted, 1)
        accepted += n
    for counts in (first, all_tok):
        chi2 = stats.chisquare(counts, counts.sum() * p)
        assert chi2.pvalue > 1e-3
        assert 0.5 * np.abs(counts / counts.sum() - p).sum() < 0.01
    assert stats.chisquare(first, first.sum() * q).pvalue < 1e-12
    alpha = np.minimum(p, q).sum()
    mean_acc = accepted / trials
    expected = sum(alpha ** i for i in range(1, gamma + 1))
    assert mean_acc == pytest.approx(expected, rel=0.02)


def test_accept_reject_edge_cases():
    rng = np.random.default_rng(1)
    p = np.array([[0.2, 0.3, 0.5]] * 4)
    assert accept_reject_sample(p, p[:3], np.array([0, 1, 2]), rng)[0] == 3
    p0 = np.array([[0.0, 0.5, 0.5]] * 3)
    q0 = np.array([[1.0, 0.0, 0.0]] * 2)
    for _ in range(50):
        n, nxt = accept_reject_sample(p0, q0, np.array([0, 0]), rng)
        assert n == 0 and nxt in (1, 2)
    n, nxt = accept_reject_sample(np.array([[0.0, 1.0, 0.0]]), np.zeros((0, 3)), np.array([], dtype=int), rng)
    assert (n, nxt) == (0, 1)
    with pytest.raises(ValueError):
        accept_reject_sample(p, np.array([[0.0, 0.5, 0.5]]), np.array([0]), rng)


def test_accept_reject_greedy_cases():
    assert accept_reject_greedy(np.array([3, 1, 4, 1]), np.array([3, 1, 4])) == (3, 1)
    assert accept_reject_greedy(np.array([3, 1, 4, 1]), np.array([3, 2, 4])) == (1, 1)
    assert accept_reject_greedy(np.array([3, 1]), np.array([0])) == (0, 3)
    assert accept_reject_greedy(np.array([7]), np.array([], dtype=int)) == (0, 7)


def test_alpha_estimator_ignores_untested_drafts():
    from kvspec.speculative import SpecStats
    st = SpecStats(rounds=2, target_calls=2, proposed=8, accepted=5, rejections=1)
    assert st.alpha == pytest.approx(5 / 6)
    assert st.accept_fraction == pytest.approx(5 / 8)


def test_grammar_roundtrip():
    g = Grammar()
    s = g.text(500, seed=0)
    assert len(s) == 500 and decode(encode(s)) == s
    assert len(g.words) == 64
