"""Synthetic request lengths and Poisson arrivals (fixed seeds)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_MODEL_LEN = 4096


@dataclass(frozen=True)
class LengthDist:
    prompt_median: float = 256.0
    prompt_sigma: float = 1.0
    output_median: float = 128.0
    output_sigma: float = 0.9
    prompt_clip: tuple[int, int] = (8, 3072)
    output_clip: tuple[int, int] = (4, 1024)


def sample_lengths(n: int, dist: LengthDist = LengthDist(), seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = np.exp(rng.normal(np.log(dist.prompt_median), dist.prompt_sigma, n))
    o = np.exp(rng.normal(np.log(dist.output_median), dist.output_sigma, n))
    p = np.clip(np.round(p), *dist.prompt_clip).astype(np.int64)
    o = np.clip(np.round(o), *dist.output_clip).astype(np.int64)
    o = np.minimum(o, MAX_MODEL_LEN - p)
    return p, o


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(1.0 / rate, n))
