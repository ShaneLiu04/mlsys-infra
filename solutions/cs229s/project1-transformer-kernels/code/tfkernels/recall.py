"""Associative recall solved by a 2-layer attention-only transformer with hand-set weights.

Sequence: [BOS, k1, v1, ..., kn, vn, q], q = some k_i, target = v_i. Keys are distinct.
Residual stream (one-hot subspaces): [tok | pos | prev | out], width 3V + T_max.

Layer 1 (previous-token head): query = beta1 * e_pos(p), key = e_pos(j+1), so position p
attends to p-1; value copies tok(j) into the prev subspace.
Layer 2 (induction/match head): query = beta2 * tok(p), key = prev(j); the last position
(token q) attends to the position whose previous token was q, i.e. v_i; value copies tok(j)
into the out subspace. Unembedding reads out and argmaxes over value tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RecallTask:
    n_keys: int = 16
    n_values: int = 16
    n_pairs: int = 8

    @property
    def vocab(self) -> int:
        return 1 + self.n_keys + self.n_values

    @property
    def seq_len(self) -> int:
        return 2 * self.n_pairs + 2

    def value_ids(self) -> range:
        return range(1 + self.n_keys, self.vocab)

    def sample(self, batch: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self.n_pairs > self.n_keys:
            raise ValueError("need n_pairs <= n_keys for distinct keys")
        keys = np.argsort(rng.random((batch, self.n_keys)), axis=1)[:, :self.n_pairs] + 1
        vals = rng.integers(0, self.n_values, (batch, self.n_pairs)) + 1 + self.n_keys
        seq = np.zeros((batch, self.seq_len), dtype=np.int64)
        seq[:, 1:-1:2] = keys
        seq[:, 2:-1:2] = vals
        pick = rng.integers(0, self.n_pairs, batch)
        seq[:, -1] = keys[np.arange(batch), pick]
        return torch.from_numpy(seq), torch.from_numpy(vals[np.arange(batch), pick])


class AttnOnlyTransformer(torch.nn.Module):
    """Plain causal attention-only transformer: x += softmax(QK^T + mask) V W_O per layer."""

    def __init__(self, vocab: int, t_max: int, n_layers: int = 2):
        super().__init__()
        self.vocab, self.t_max = vocab, t_max
        self.d = 3 * vocab + t_max
        self.tok_emb = torch.nn.Parameter(torch.zeros(vocab, self.d), requires_grad=False)
        self.pos_emb = torch.nn.Parameter(torch.zeros(t_max, self.d), requires_grad=False)
        self.wq = torch.nn.Parameter(torch.zeros(n_layers, self.d, self.d), requires_grad=False)
        self.wk = torch.nn.Parameter(torch.zeros(n_layers, self.d, self.d), requires_grad=False)
        self.wv = torch.nn.Parameter(torch.zeros(n_layers, self.d, self.d), requires_grad=False)
        self.wo = torch.nn.Parameter(torch.zeros(n_layers, self.d, self.d), requires_grad=False)
        self.unembed = torch.nn.Parameter(torch.zeros(self.d, vocab), requires_grad=False)

    def forward(self, idx: torch.Tensor, return_attn: bool = False):
        t = idx.shape[1]
        x = self.tok_emb[idx] + self.pos_emb[:t]
        mask = torch.ones(t, t, dtype=torch.bool, device=idx.device).tril()
        attns = []
        for layer in range(self.wq.shape[0]):
            q, k, v = x @ self.wq[layer], x @ self.wk[layer], x @ self.wv[layer]
            s = (q @ k.transpose(-1, -2)).masked_fill(~mask, float("-inf"))
            a = torch.softmax(s, dim=-1)
            attns.append(a)
            x = x + (a @ v) @ self.wo[layer]
        logits = x @ self.unembed
        return (logits, attns) if return_attn else logits


def build_hand_coded(task: RecallTask, beta1: float = 30.0, beta2: float = 30.0) -> AttnOnlyTransformer:
    vsz, tm = task.vocab, task.seq_len
    model = AttnOnlyTransformer(vsz, tm)
    tok = slice(0, vsz)
    pos = slice(vsz, vsz + tm)
    prev = slice(vsz + tm, 2 * vsz + tm)
    out = slice(2 * vsz + tm, 3 * vsz + tm)
    eye_v = torch.eye(vsz)
    with torch.no_grad():
        model.tok_emb[:, tok] = eye_v
        model.pos_emb[:, pos] = torch.eye(tm)
        # layer 1: score(p, j) = beta1 * [j + 1 == p]
        model.wq[0, pos, :tm] = beta1 * torch.eye(tm)
        model.wk[0, pos, :tm] = torch.diag(torch.ones(tm - 1), diagonal=1)
        model.wv[0, tok, :vsz] = eye_v
        model.wo[0, :vsz, prev] = eye_v
        # layer 2: score(p, j) = beta2 * <tok(p), prev(j)>
        model.wq[1, tok, :vsz] = beta2 * eye_v
        model.wk[1, prev, :vsz] = eye_v
        model.wv[1, tok, :vsz] = eye_v
        model.wo[1, :vsz, out] = eye_v
        vals = list(task.value_ids())
        model.unembed[out.start + vals[0]:out.start + vals[-1] + 1, vals[0]:vals[-1] + 1] = torch.eye(len(vals))
    return model


@torch.no_grad()
def recall_accuracy(model: AttnOnlyTransformer, task: RecallTask, n: int, seed: int) -> float:
    """Fraction of sequences whose argmax over value tokens at the last position is correct."""
    seq, target = task.sample(n, np.random.default_rng(seed))
    v0 = task.value_ids().start
    pred = model(seq)[:, -1, v0:].argmax(-1) + v0
    return float((pred == target).float().mean())
