"""Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) with KV caches and rollback."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model import GPT, KVCache


def accept_reject_sample(p: np.ndarray, q: np.ndarray, drafts: np.ndarray,
                         rng: np.random.Generator) -> tuple[int, int]:
    """p: (g+1, V) target probs, q: (g, V) draft probs, drafts: (g,) tokens drawn from q.

    Accept draft i with prob min(1, p_i(x)/q_i(x)); on the first rejection sample from
    norm(max(p_i - q_i, 0)) and stop; if all g accepted, sample a bonus token from p_g.
    Returns (n_accepted, next_token). The emitted tokens are distributed exactly as p.
    """
    g = len(drafts)
    for i in range(g):
        x = int(drafts[i])
        if q[i, x] <= 0.0:
            raise ValueError("draft token has zero draft probability")
        if rng.random() < min(1.0, p[i, x] / q[i, x]):
            continue
        resid = np.maximum(p[i] - q[i], 0.0)
        tot = resid.sum()
        resid = p[i] / p[i].sum() if tot <= 0.0 else resid / tot
        return i, int(rng.choice(len(resid), p=resid))
    last = p[g] / p[g].sum()
    return g, int(rng.choice(len(last), p=last))


def accept_reject_greedy(target_argmax: np.ndarray, drafts: np.ndarray) -> tuple[int, int]:
    """target_argmax: (g+1,) argmax of the target at each draft position (+ bonus)."""
    for i, x in enumerate(drafts):
        if int(x) != int(target_argmax[i]):
            return i, int(target_argmax[i])
    return len(drafts), int(target_argmax[len(drafts)])


@dataclass
class SpecStats:
    rounds: int = 0
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0
    rejections: int = 0
    overlap_sum: float = 0.0  # sum over tested positions of sum_x min(p, q) = 1 - TV(p, q)
    accepted_per_round: list[int] = field(default_factory=list)

    @property
    def tested(self) -> int:
        """Drafts actually judged: those after the first rejection in a round never are."""
        return self.accepted + self.rejections

    @property
    def alpha(self) -> float:
        """Per-token acceptance estimate (geometric MLE). accepted/proposed would be biased low."""
        return self.accepted / max(self.tested, 1)

    @property
    def accept_fraction(self) -> float:
        return self.accepted / max(self.proposed, 1)

    @property
    def alpha_overlap(self) -> float:
        return self.overlap_sum / max(self.tested, 1)

    @property
    def tokens_per_target_call(self) -> float:
        return (self.accepted + self.rounds) / max(self.target_calls, 1)


def _probs(logits: torch.Tensor, temperature: float) -> np.ndarray:
    return torch.softmax(logits.double() / temperature, dim=-1).numpy()


@torch.no_grad()
def speculative_generate(target: GPT, draft: GPT, prompt: list[int], n_new: int, gamma: int,
                         temperature: float = 0.0, seed: int = 0) -> tuple[list[int], SpecStats]:
    """Batch-1 speculative decoding; temperature 0 = greedy variant.

    Cache invariant at the start of a round: each cache holds a prefix of seq[:-1];
    the draft catches up on whatever it is missing in one forward.
    """
    if len(prompt) + n_new + gamma > min(target.cfg.block_size, draft.cfg.block_size):
        raise ValueError("prompt + n_new + gamma must fit in block_size")
    rng = np.random.default_rng(seed)
    dt = next(target.parameters()).dtype
    tc, dc = KVCache(target.cfg, 1, dt), KVCache(draft.cfg, 1, next(draft.parameters()).dtype)
    seq = list(prompt)
    if len(seq) > 1:
        target(torch.tensor([seq[:-1]]), tc)
    st = SpecStats()
    while len(seq) - len(prompt) < n_new:
        g = min(gamma, n_new - (len(seq) - len(prompt)))
        drafts, qs = [], []
        logits = draft(torch.tensor([seq[dc.length:]]), dc)[0, -1]
        st.draft_calls += 1
        for i in range(g):
            if temperature == 0.0:
                x = int(logits.argmax())
            else:
                qi = _probs(logits, temperature)
                x = int(rng.choice(len(qi), p=qi))
                qs.append(qi)
            drafts.append(x)
            if i < g - 1:
                logits = draft(torch.tensor([[x]]), dc)[0, -1]
                st.draft_calls += 1
        tl = target(torch.tensor([seq[tc.length:] + drafts]), tc)[0, -(g + 1):]
        st.target_calls += 1
        if temperature == 0.0:
            n_acc, nxt = accept_reject_greedy(tl.argmax(-1).numpy(), np.array(drafts))
        else:
            p = _probs(tl, temperature)
            q = np.stack(qs)
            n_acc, nxt = accept_reject_sample(p, q, np.array(drafts), rng)
            n_tested = min(n_acc + 1, g)
            st.overlap_sum += float(np.minimum(p[:n_tested], q[:n_tested]).sum())
        seq += drafts[:n_acc] + [nxt]
        tc.length = len(seq) - 1
        dc.length = min(dc.length, len(seq) - 1)
        st.rounds += 1
        st.proposed += g
        st.accepted += n_acc
        st.rejections += int(n_acc < g)
        st.accepted_per_round.append(n_acc)
    return seq[:len(prompt) + n_new], st


@torch.no_grad()
def target_only_kv(target: GPT, prompt: list[int], n_new: int, temperature: float = 0.0,
                   seed: int = 0) -> list[int]:
    """Baseline autoregressive decoding with KV cache, same sampler as speculative_generate."""
    rng = np.random.default_rng(seed)
    cache = KVCache(target.cfg, 1, next(target.parameters()).dtype)
    seq = list(prompt)
    logits = target(torch.tensor([seq]), cache)[0, -1]
    for i in range(n_new):
        if temperature == 0.0:
            x = int(logits.argmax())
        else:
            pi = _probs(logits, temperature)
            x = int(rng.choice(len(pi), p=pi))
        seq.append(x)
        if i < n_new - 1:
            logits = target(torch.tensor([[x]]), cache)[0, -1]
    return seq
