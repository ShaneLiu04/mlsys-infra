"""GPT-2 loading, wikitext-2 data, perplexity, and a swappable linear wrapper for quant/pruning."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

HF = Path(os.environ.get("HF_HOME", "/root/autodl-tmp/hf"))
MODELS = HF / "models"
WIKI = HF / "data" / "wikitext2"


def load_model(name: str = "gpt2", dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = MODELS / name
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
    return model.cuda().eval(), tok


def wikitext_ids(tok, split: str = "test") -> torch.Tensor:
    import pandas as pd
    texts = pd.read_parquet(WIKI / f"{split}.parquet")["text"].tolist()
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    return ids


def windows(ids: torch.Tensor, seq: int = 1024, n: int | None = None, seed: int | None = None):
    k = ids.numel() // seq
    w = ids[: k * seq].view(k, seq)
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        w = w[torch.randperm(k, generator=g)]
    return w[:n] if n else w


@torch.no_grad()
def perplexity(model, wins: torch.Tensor, bs: int = 8) -> float:
    nll, cnt = 0.0, 0
    for i in range(0, wins.shape[0], bs):
        x = wins[i:i + bs].cuda()
        logits = model(x).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                               reduction="sum")
        nll += float(loss)
        cnt += x[:, 1:].numel()
    return math.exp(nll / cnt)


def linear_modules(model) -> dict[str, torch.nn.Module]:
    """Every projection inside the transformer blocks (Conv1D in GPT-2, or nn.Linear)."""
    out = {}
    for name, m in model.named_modules():
        if ".h." in f".{name}" or name.startswith("transformer.h."):
            if type(m).__name__ in ("Conv1D", "Linear") and "lm_head" not in name:
                out[name] = m
    return out


def weight_oi(m: torch.nn.Module) -> torch.Tensor:
    """Weight as [out, in] regardless of Conv1D ([in, out]) or Linear ([out, in])."""
    w = m.weight.detach()
    return w.t() if type(m).__name__ == "Conv1D" else w


def set_weight_oi(m: torch.nn.Module, w_oi: torch.Tensor) -> None:
    w = w_oi.t() if type(m).__name__ == "Conv1D" else w_oi
    m.weight.data.copy_(w.to(m.weight.dtype))


class ActQuantLinear(torch.nn.Module):
    """y = Q_a(x / s) @ Q_w(W diag(s))^T + b with simulated int8 activations.

    act: 'none' | 'tensor' (per-tensor dynamic absmax) | 'token' (per-token) | 'llm_int8'
    (columns with any |x| > threshold stay fp16, the rest per-token int8 x per-channel int8 W).
    """

    def __init__(self, w_oi: torch.Tensor, bias: torch.Tensor | None, act: str,
                 smooth: torch.Tensor | None = None, threshold: float = 6.0):
        super().__init__()
        self.act, self.threshold = act, threshold
        self.register_buffer("w_fp", w_oi.clone())
        ws = w_oi.float() * (smooth[None, :] if smooth is not None else 1.0)
        self.register_buffer("w_q", quant_sym(ws, 8, "channel").to(w_oi.dtype))
        self.register_buffer("smooth", smooth.to(w_oi.dtype) if smooth is not None else None)
        self.register_buffer("bias", bias.clone() if bias is not None else None)
        self.outlier_cols = []

    def forward(self, x):
        if self.smooth is not None:
            x = x / self.smooth
        if self.act == "llm_int8":
            x2 = x.reshape(-1, x.shape[-1])
            cols = (x2.abs() > self.threshold).any(0)
            self.outlier_cols.append(int(cols.sum()))
            xr = torch.where(cols[None, :], torch.zeros_like(x2), x2)
            y = F.linear(quant_act(xr, "token"), self.w_q)
            if cols.any():
                y = y + F.linear(x2[:, cols], self.w_fp[:, cols])
            y = y.view(*x.shape[:-1], -1)
        else:
            y = F.linear(quant_act(x, self.act) if self.act != "none" else x, self.w_q)
        return y + self.bias if self.bias is not None else y


def quant_act(x: torch.Tensor, mode: str) -> torch.Tensor:
    xf = x.float()
    if mode == "tensor":
        s = xf.abs().amax().clamp_min(1e-8) / 127
    else:
        s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return (torch.round(xf / s).clamp(-127, 127) * s).to(x.dtype)


def quant_sym(w: torch.Tensor, bits: int, gran: str, g: int = 128) -> torch.Tensor:
    """Symmetric absmax RTN on [out, in]; gran in {'tensor', 'channel', 'group'}."""
    qmax = 2 ** (bits - 1) - 1
    wf = w.float()
    if gran == "tensor":
        s = wf.abs().amax().clamp_min(1e-8) / qmax
        return torch.round(wf / s).clamp(-qmax, qmax) * s
    if gran == "channel":
        s = wf.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
        return torch.round(wf / s).clamp(-qmax, qmax) * s
    o, i = wf.shape
    wg = wf.view(o, i // g, g)
    s = wg.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (torch.round(wg / s).clamp(-qmax, qmax) * s).view(o, i)


def asym_params(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-max asymmetric params along the last dim (rows of a [out, cols] slice)."""
    qmax = 2 ** bits - 1
    lo = w.amin(-1, keepdim=True).clamp(max=0)
    hi = w.amax(-1, keepdim=True).clamp(min=0)
    s = ((hi - lo) / qmax).clamp_min(1e-8)
    z = torch.round(-lo / s)
    return s, z


def asym_q(w: torch.Tensor, s: torch.Tensor, z: torch.Tensor, bits: int) -> torch.Tensor:
    return (torch.clamp(torch.round(w / s) + z, 0, 2 ** bits - 1) - z) * s


def quant_asym(w: torch.Tensor, bits: int, g: int | None = 128) -> torch.Tensor:
    """Asymmetric min-max RTN, group-wise along the input dim (g=None: per output channel)."""
    wf = w.float()
    o, i = wf.shape
    gg = g or i
    wg = wf.view(o, i // gg, gg)
    s, z = asym_params(wg, bits)
    return asym_q(wg, s, z, bits).view(o, i)
