"""LoRA (Hu et al. 2021): y = W0 x + (alpha / r) B A x with W0 frozen and B initialised to zero."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=base.weight.dtype))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def delta(self) -> torch.Tensor:
        return self.scaling * self.B @ self.A

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        dtype=self.base.weight.dtype)
        out.weight.copy_(self.base.weight + self.delta())
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out


def trainable_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def fit_low_rank_update(d: int, true_rank: int, r: int, steps: int, seed: int = 0,
                        lr: float = 1e-2) -> dict:
    """Learn y = (W0 + dW) x with a LoRA adapter on frozen W0; dW has rank ``true_rank``.

    With x ~ N(0, I) the expected loss is ||scaling*BA - dW||_F^2 / d_out, so the best
    achievable error is the Eckart-Young tail of dW's singular values.
    """
    g = torch.Generator().manual_seed(seed)
    base = nn.Linear(d, d, bias=False, dtype=torch.float64)
    with torch.no_grad():
        base.weight.copy_(torch.randn(d, d, generator=g, dtype=torch.float64) / math.sqrt(d))
    sv = torch.linspace(1.0, 0.2, true_rank, dtype=torch.float64)
    u, _ = torch.linalg.qr(torch.randn(d, true_rank, generator=g, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(d, true_rank, generator=g, dtype=torch.float64))
    dw = u @ torch.diag(sv) @ v.T
    target = base.weight.detach() + dw
    torch.manual_seed(seed)
    lora = LoRALinear(base, r, alpha=2 * r)
    opt = torch.optim.Adam([lora.A, lora.B], lr=lr)
    for _ in range(steps):
        x = torch.randn(256, d, generator=g, dtype=torch.float64)
        loss = ((lora(x) - x @ target.T) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    err = float(torch.linalg.norm(lora.delta().detach() - dw) / torch.linalg.norm(dw))
    tail = float(torch.sqrt((sv[r:] ** 2).sum()) / torch.linalg.norm(dw)) if r < true_rank else 0.0
    return {"d": d, "true_rank": true_rank, "r": r, "steps": steps, "rel_err_dW": err,
            "eckart_young_bound": tail, "trainable_params": trainable_params(lora),
            "full_ft_params": d * d}
