"""Q5: fine-tune GPT-2 on wikitext-2: full FT vs LoRA (r=4..64) vs QLoRA-like (frozen int8 / int4 base).

Reports trainable params, peak memory, step time, train/val loss curves, and a merged-weight check.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from common import Writer, env_info, set_threads
from llm_utils import linear_modules, load_model, weight_oi, wikitext_ids, windows
import triton_kernels as tk


class _QLinearFn(torch.autograd.Function):
    """y = x @ deq(Wq)^T; saves only the quantized weight, re-dequantizes in backward."""

    @staticmethod
    def forward(ctx, x, deq, *qt):
        ctx.deq = deq
        ctx.save_for_backward(*qt)
        return F.linear(x, deq(*qt).to(x.dtype))

    @staticmethod
    def backward(ctx, gy):
        w = ctx.deq(*ctx.saved_tensors).to(gy.dtype)
        return (gy @ w, None) + (None,) * len(ctx.saved_tensors)


def _deq8(q, s):
    return q.to(torch.bfloat16) * s[:, None]


def _deq4(q, s):
    return tk.dequant_w4(q, s).to(torch.bfloat16)


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, r: int, alpha: float, quant: str | None = None):
        super().__init__()
        w = weight_oi(base).detach().float()
        out_f, in_f = w.shape
        self.quant = quant
        if quant == "int8":
            q, s = tk.quant_w8(w)
            self.register_buffer("q", q)
            self.register_buffer("s", s.to(torch.bfloat16))
        elif quant == "int4":
            q, s = tk.quant_w4(w)
            self.register_buffer("q", q)
            self.register_buffer("s", s)
        else:
            self.register_buffer("w", w.to(torch.bfloat16))
        self.register_buffer("b", base.bias.detach().to(torch.bfloat16) if base.bias is not None else None)
        self.r, self.scale = r, alpha / r
        self.A = torch.nn.Parameter(torch.randn(r, in_f, device=w.device) / math.sqrt(in_f))
        self.B = torch.nn.Parameter(torch.zeros(out_f, r, device=w.device))

    def base_weight(self) -> torch.Tensor:
        if self.quant == "int8":
            return _deq8(self.q, self.s)
        if self.quant == "int4":
            return _deq4(self.q, self.s)
        return self.w

    def forward(self, x):
        if self.quant == "int8":
            y = _QLinearFn.apply(x, _deq8, self.q, self.s)
        elif self.quant == "int4":
            y = _QLinearFn.apply(x, _deq4, self.q, self.s)
        else:
            y = F.linear(x, self.w)
        y = y + (x @ self.A.t()) @ self.B.t() * self.scale
        return y + self.b if self.b is not None else y

    def merged_weight(self) -> torch.Tensor:
        return self.base_weight().float() + self.scale * (self.B.float() @ self.A.float())


def add_lora(model, r: int, alpha: float, quant: str | None):
    for p in model.parameters():
        p.requires_grad_(False)
    for n, m in linear_modules(model).items():
        parent = model.get_submodule(n.rsplit(".", 1)[0])
        setattr(parent, n.rsplit(".", 1)[1], LoRALinear(m, r, alpha, quant))
    model.to(torch.bfloat16)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.data = m.A.data.float()
            m.B.data = m.B.data.float()
    return model


def lm_loss(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x).logits
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))


@torch.no_grad()
def val_loss(model, val):
    model.eval()
    ls = [float(lm_loss(model, val[i:i + 8].cuda())) for i in range(0, val.shape[0], 8)]
    model.train()
    return sum(ls) / len(ls)


def run(cfg: dict, args, w: Writer, train_ids, val):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, _ = load_model(args.model, dtype=torch.float32)
    base_params = sum(p.numel() for p in model.parameters())
    if cfg["kind"] == "full":
        lr = args.lr_full
    else:
        model = add_lora(model, cfg["r"], 2 * cfg["r"], cfg.get("quant"))
        lr = args.lr_lora
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = base_params + (trainable if cfg["kind"] != "full" else 0)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    nwin = train_ids.numel() // args.seq
    data = train_ids[: nwin * args.seq].view(nwin, args.seq)
    mem_static = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(args.steps):
        x = data[torch.randint(0, nwin, (args.bs,), generator=g)].cuda()
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss = lm_loss(model, x)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        row = dict(part="curve", config=cfg["name"], step=step, loss=float(loss))
        if step % args.val_every == 0 or step == args.steps - 1:
            row["val_loss"] = val_loss(model, val)
        w(**row)
    peak = torch.cuda.max_memory_allocated()
    step_s = sorted(times[10:])[len(times[10:]) // 2]
    res = dict(part="summary", config=cfg["name"], kind=cfg["kind"], r=cfg.get("r", 0),
               quant=cfg.get("quant") or "bf16", trainable=trainable, total_params=total,
               trainable_frac=trainable / total, peak_mem_gb=peak / 2 ** 30,
               static_mem_gb=mem_static / 2 ** 30, step_s=step_s,
               tokens_per_s=args.bs * args.seq / step_s, final_val=val_loss(model, val))
    if cfg["kind"] == "lora" and cfg.get("quant") is None and cfg["r"] == 8:
        res.update(merge_check(model, val))
    w(**res)
    del model, opt
    torch.cuda.empty_cache()


@torch.no_grad()
def merge_check(model, val) -> dict:
    """Unmerged LoRA forward vs plain model with W' = W + (alpha/r) B A, both in fp32."""
    model = model.float().eval()
    x = val[:4].cuda()
    y_lora = model(x).logits
    lin = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    for n, m in lin.items():
        wm = m.merged_weight()
        lin_new = torch.nn.Linear(wm.shape[1], wm.shape[0], bias=m.b is not None, device=wm.device)
        lin_new.weight.data.copy_(wm)
        if m.b is not None:
            lin_new.bias.data.copy_(m.b.float())
        parent = model.get_submodule(n.rsplit(".", 1)[0])
        setattr(parent, n.rsplit(".", 1)[1], lin_new)
    y_merged = model(x).logits
    d = (y_lora - y_merged).abs()
    return {"merge_max_abs_diff": float(d.max()), "merge_logit_absmax": float(y_lora.abs().max()),
            "merge_argmax_agree": float((y_lora.argmax(-1) == y_merged.argmax(-1)).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--lr-full", type=float, default=3e-5)
    ap.add_argument("--lr-lora", type=float, default=1e-3)
    ap.add_argument("--configs", default="all")
    a = ap.parse_args()
    set_threads(2)
    w = Writer("lora_ft")
    w(part="env", **env_info(), model=a.model, steps=a.steps, bs=a.bs, seq=a.seq)
    _, tok = load_model(a.model)
    train_ids = wikitext_ids(tok, "train")
    val = windows(wikitext_ids(tok, "validation"), a.seq, 32, seed=0)
    cfgs = [{"name": "full", "kind": "full"}]
    cfgs += [{"name": f"lora_r{r}", "kind": "lora", "r": r} for r in [4, 8, 16, 32, 64]]
    cfgs += [{"name": f"qlora_{q}_r16", "kind": "lora", "r": 16, "quant": q} for q in ["int8", "int4"]]
    if a.configs != "all":
        cfgs = [c for c in cfgs if c["name"] in a.configs.split(",")]
    for c in cfgs:
        run(c, a, w, train_ids, val)


if __name__ == "__main__":
    main()
