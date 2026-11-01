"""Short CPU training of target / draft models on the synthetic grammar."""

from __future__ import annotations

import time

import torch

from .data import batches
from .model import GPT, GPTConfig


def train(cfg: GPTConfig, data: torch.Tensor, steps: int, block: int = 128, batch: int = 32,
          lr: float = 3e-3, seed: int = 0, log_every: int = 100) -> tuple[GPT, list[dict]]:
    torch.manual_seed(seed)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed)
    log, t0 = [], time.perf_counter()
    model.train()
    for step in range(steps):
        x, y = batches(data, block, batch, gen)
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(loss), "elapsed_s": time.perf_counter() - t0})
    model.eval()
    return model, log


@torch.no_grad()
def eval_loss(model: GPT, data: torch.Tensor, block: int = 128, batch: int = 64, n_batches: int = 8,
              seed: int = 123) -> float:
    gen = torch.Generator().manual_seed(seed)
    tot = 0.0
    for _ in range(n_batches):
        x, y = batches(data, block, batch, gen)
        tot += float(torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten()))
    return tot / n_batches
