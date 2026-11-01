"""Synthetic word-level grammar rendered as characters.

Spelling inside a word is (nearly) deterministic, word-to-word transitions are a sparse random
bigram chain. A small draft model learns spelling quickly, so acceptance is high mid-word and
lower at word boundaries -- qualitatively like real text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

ALPHABET = " .abcdefghijklmnopqrstuvwxyz"


def encode(s: str) -> list[int]:
    return [ALPHABET.index(c) for c in s]


def decode(ids: list[int]) -> str:
    return "".join(ALPHABET[i] for i in ids)


@dataclass
class Grammar:
    n_words: int = 64
    successors: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        words: set[str] = set()
        while len(words) < self.n_words:
            n = int(rng.integers(2, 8))
            words.add("".join(rng.choice(list(ALPHABET[2:]), n)))
        self.words = sorted(words)
        self.next_ids = np.stack([rng.choice(self.n_words, self.successors, replace=False)
                                  for _ in range(self.n_words)])
        probs = np.array([0.55, 0.25, 0.15, 0.05])[:self.successors]
        self.next_p = probs / probs.sum()

    def text(self, n_chars: int, seed: int) -> str:
        rng = np.random.default_rng(seed)
        out, total, w = [], 0, int(rng.integers(self.n_words))
        while total < n_chars:
            word = self.words[w]
            end = "." if rng.random() < 0.1 else " "
            out.append(word + end)
            total += len(word) + 1
            w = int(self.next_ids[w, rng.choice(self.successors, p=self.next_p)])
        return "".join(out)[:n_chars]


def batches(data: torch.Tensor, block: int, batch: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x, y
