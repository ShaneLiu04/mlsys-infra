"""Magnitude pruning: unstructured and N:M (e.g. 2:4) along the reduction dimension."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def magnitude_mask(w: np.ndarray, sparsity: float) -> np.ndarray:
    k = int(round(sparsity * w.size))
    if k == 0:
        return np.ones_like(w, dtype=bool)
    thresh = np.partition(np.abs(w).ravel(), k - 1)[k - 1]
    mask = np.abs(w) > thresh
    # break ties so exactly k weights are removed
    ties = np.flatnonzero((np.abs(w) == thresh).ravel())
    need = w.size - k - int(mask.sum())
    flat = mask.ravel()
    flat[ties[:need]] = True
    return flat.reshape(w.shape)


def nm_mask(w: np.ndarray, n: int = 2, m: int = 4) -> np.ndarray:
    """Keep the n largest of every m consecutive weights along axis 0 (input dim of X @ W)."""
    d_in, d_out = w.shape
    if d_in % m:
        raise ValueError("d_in must be divisible by m")
    g = np.abs(w).reshape(d_in // m, m, d_out)
    order = np.argsort(-g, axis=1, kind="stable")
    mask = np.zeros_like(g, dtype=bool)
    np.put_along_axis(mask, order[:, :n], True, axis=1)
    return mask.reshape(d_in, d_out)


# --------------------------------------------------------------------------- tiny MLP study

def teacher_data(n: int, d: int = 32, classes: int = 10, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, 64, generator=g) * (2.0 / d ** 0.5)  # keeps tanh out of full saturation
    b = torch.randn(64, classes, generator=g)
    gx = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(n, d, generator=gx)
    return x, (torch.tanh(x @ a) @ b).argmax(1)


def make_mlp(d: int = 32, h: int = 256, classes: int = 10, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, classes))


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int, lr: float = 2e-3,
          masks: dict[str, torch.Tensor] | None = None, seed: int = 0) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if masks:
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in masks:
                            p.mul_(masks[name])


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(x).argmax(1) == y).float().mean())


def prune_model(model: nn.Module, kind: str, sparsity: float = 0.0) -> dict[str, torch.Tensor]:
    """Prune every Linear weight in place; returns masks. torch weight is [out, in]."""
    masks = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not name.endswith("weight"):
                continue
            w = p.detach().numpy().T  # [in, out] to match X @ W
            if kind == "magnitude":
                m = magnitude_mask(w, sparsity)
            elif kind == "2:4":
                m = nm_mask(w, 2, 4)
            else:
                raise KeyError(kind)
            mask = torch.from_numpy(m.T.copy()).to(p.dtype)
            p.mul_(mask)
            masks[name] = mask
    return masks
