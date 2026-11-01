"""Reference implementations of LLM sampling, sized for a whiteboard.

All functions take logits of shape [..., V] and work on the last axis.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0; use greedy() for argmax")
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits; ties at the k-th value are all kept."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = np.partition(logits, -k, axis=-1)[..., -k][..., None]
    return np.where(logits < kth, -np.inf, logits)


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest prefix of sorted tokens whose mass reaches p.

    The token that crosses p is kept, so at least one token always survives.
    """
    if p >= 1.0:
        return logits
    order = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, order, axis=-1)
    probs = softmax(sorted_logits)
    mass_before = np.cumsum(probs, axis=-1) - probs
    drop_sorted = mass_before >= p
    drop = np.empty_like(drop_sorted)
    np.put_along_axis(drop, order, drop_sorted, axis=-1)
    return np.where(drop, -np.inf, logits)


def greedy(logits: np.ndarray) -> np.ndarray:
    return logits.argmax(axis=-1)


def sample(logits: np.ndarray, rng: np.random.Generator, temperature: float = 1.0,
           top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    """Temperature first, then top-k, then top-p (the order used by HF and vLLM)."""
    z = logits / temperature
    z = top_k_filter(z, top_k)
    z = top_p_filter(z, top_p)
    probs = softmax(z)
    flat = probs.reshape(-1, probs.shape[-1])
    u = rng.random((flat.shape[0], 1))
    idx = (flat.cumsum(axis=-1) < u).sum(axis=-1)
    return np.minimum(idx, flat.shape[-1] - 1).reshape(probs.shape[:-1])
