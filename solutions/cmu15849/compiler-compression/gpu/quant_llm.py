"""Q3a: post-training quantization of GPT-2 on wikitext-2 -- perplexity vs bits.

RTN (int8 per-tensor / per-channel; int4/int3 per-channel and group-128), GPTQ (own implementation,
block-sequential Hessians from 128x1024 calibration tokens), AWQ-style activation-aware scaling,
and simulated W8A8: naive per-tensor / per-token activations, SmoothQuant (alpha=0.5), LLM.int8.
Also per-layer activation outlier statistics.
"""

from __future__ import annotations

import argparse
import copy
import math
import time

import torch

from common import Writer, env_info, set_threads
from llm_utils import (ActQuantLinear, asym_params, asym_q, linear_modules, load_model, perplexity,
                       quant_asym, quant_sym, set_weight_oi, weight_oi, wikitext_ids, windows)


# --------------------------------------------------------------------------- calibration helpers

def run_calib(model, calib: torch.Tensor, bs: int = 8):
    with torch.no_grad():
        for i in range(0, calib.shape[0], bs):
            model(calib[i:i + bs].cuda())


def collect(model, mods: dict, calib: torch.Tensor, want_h: bool = True):
    """Hessian H = sum x x^T (fp32) and per-channel absmax of the inputs of the given modules."""
    stats = {n: {"H": None, "amax": None, "n": 0} for n in mods}
    hooks = []
    for n, m in mods.items():
        def hook(mod, inp, out, n=n):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            st = stats[n]
            am = x.abs().amax(0)
            st["amax"] = am if st["amax"] is None else torch.maximum(st["amax"], am)
            if want_h:
                h = x.t() @ x
                st["H"] = h if st["H"] is None else st["H"] + h
            st["n"] += x.shape[0]
        hooks.append(m.register_forward_hook(hook))
    run_calib(model, calib)
    for h in hooks:
        h.remove()
    return stats


def block_groups(mods: dict) -> list[list[str]]:
    groups: dict[int, list[str]] = {}
    for n in mods:
        idx = int(n.split(".h.")[1].split(".")[0])
        groups.setdefault(idx, []).append(n)
    return [groups[k] for k in sorted(groups)]


# --------------------------------------------------------------------------- GPTQ

def gptq_quantize(w: torch.Tensor, H: torch.Tensor, bits: int, g: int | None = 128,
                  block: int = 128, damp: float = 0.01) -> torch.Tensor:
    """GPTQ (Frantar et al. 2023): column-wise OBQ with lazy block updates, asymmetric min-max grid."""
    W = w.float().clone()
    H = H.clone()
    cols = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    if g is None:
        s, z = asym_params(W, bits)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            if g is not None and (i1 + i) % g == 0:
                s, z = asym_params(W1[:, i:i + g] if i + g <= W1.shape[1] else W[:, i1 + i:i1 + i + g], bits)
            wcol = W1[:, i:i + 1]
            q = asym_q(wcol, s, z, bits)
            Q1[:, i:i + 1] = q
            err = (wcol - q) / Hi[i, i]
            W1[:, i:] -= err @ Hi[i:i + 1, i:]
            E1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def awq_quantize(w: torch.Tensor, H: torch.Tensor, amean: torch.Tensor, bits: int, g: int = 128):
    """AWQ-style: scale input channels by s = mean|x|^a before group RTN, pick a by the proxy
    output error tr(dW H dW^T); returns the effective (de-scaled) weight."""
    wf = w.float()
    best, best_err = None, math.inf
    for a in [i / 10 for i in range(11)]:
        s = amean.clamp_min(1e-5) ** a
        s = s / (s.max() * s.min()).sqrt()
        wq = quant_asym(wf * s[None, :], bits, g) / s[None, :]
        d = wq - wf
        err = float(torch.einsum("oi,ij,oj->", d, H, d).detach())
        if err < best_err:
            best, best_err = wq, err
    return best


# --------------------------------------------------------------------------- methods

WEIGHT_METHODS = {
    "w8_tensor": dict(kind="rtn_sym", bits=8, gran="tensor"),
    "w8_channel": dict(kind="rtn_sym", bits=8, gran="channel"),
    "w4_channel": dict(kind="rtn_asym", bits=4, g=None),
    "w4_g128": dict(kind="rtn_asym", bits=4, g=128),
    "w3_g128": dict(kind="rtn_asym", bits=3, g=128),
    "awq_w4_g128": dict(kind="awq", bits=4, g=128),
    "awq_w3_g128": dict(kind="awq", bits=3, g=128),
    "gptq_w4_channel": dict(kind="gptq", bits=4, g=None),
    "gptq_w4_g128": dict(kind="gptq", bits=4, g=128),
    "gptq_w3_g128": dict(kind="gptq", bits=3, g=128),
}
ACT_METHODS = {
    "w8a8_tensor": dict(act="tensor"),
    "w8a8_token": dict(act="token"),
    "smoothquant_tensor": dict(act="tensor", alpha=0.5),
    "smoothquant_token": dict(act="token", alpha=0.5),
    "smoothquant_tensor_a0.8": dict(act="tensor", alpha=0.8),
    "smoothquant_token_a0.8": dict(act="token", alpha=0.8),
    "llm_int8": dict(act="llm_int8"),
}


def apply_weight_method(model, cfg: dict, calib: torch.Tensor) -> dict:
    mods = linear_modules(model)
    info = {"layers": len(mods)}
    if cfg["kind"].startswith("rtn"):
        for m in mods.values():
            w = weight_oi(m)
            wq = quant_sym(w, cfg["bits"], cfg["gran"]) if cfg["kind"] == "rtn_sym" \
                else quant_asym(w, cfg["bits"], cfg["g"])
            set_weight_oi(m, wq)
        return info
    for names in block_groups(mods):  # block-sequential: Hessians see already-quantized earlier blocks
        sub = {n: mods[n] for n in names}
        st = collect(model, sub, calib, want_h=True)
        for n in names:
            w = weight_oi(mods[n])
            H = st[n]["H"] / st[n]["n"]
            if cfg["kind"] == "gptq":
                wq = gptq_quantize(w, H, cfg["bits"], cfg["g"])
            else:
                amean = torch.sqrt(torch.diag(H))  # RMS of each input channel
                wq = awq_quantize(w, H, amean, cfg["bits"], cfg["g"])
            set_weight_oi(mods[n], wq)
    return info


def apply_act_method(model, cfg: dict, calib: torch.Tensor) -> dict:
    mods = linear_modules(model)
    st = collect(model, mods, calib, want_h=False) if "alpha" in cfg else None
    wrapped = {}
    for n, m in mods.items():
        w = weight_oi(m).detach()
        smooth = None
        if st is not None:
            xa = st[n]["amax"].clamp_min(1e-5)
            wa = w.float().abs().amax(0).clamp_min(1e-5)
            smooth = (xa ** cfg["alpha"] / wa ** (1 - cfg["alpha"])).clamp_min(1e-5)
        q = ActQuantLinear(w, m.bias.detach() if m.bias is not None else None, cfg["act"], smooth)
        parent = model.get_submodule(n.rsplit(".", 1)[0])
        setattr(parent, n.rsplit(".", 1)[1], q)
        wrapped[n] = q
    return {"layers": len(mods), "wrapped": wrapped}


@torch.no_grad()
def outlier_stats(model, w: Writer, model_name: str, calib: torch.Tensor):
    mods = linear_modules(model)
    st = collect(model, mods, calib[:16], want_h=False)
    for n, s in st.items():
        am = s["amax"]
        med = float(am.median())
        top = torch.topk(am, 3)
        w(part="outliers", model=model_name, layer=n, absmax=float(am.max()), median_absmax=med,
          ratio=float(am.max()) / max(med, 1e-8), n_gt6=int((am > 6).sum()), channels=am.numel(),
          top_channels=top.indices.tolist(), top_values=[round(float(v), 2) for v in top.values])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--eval-windows", type=int, default=0)
    ap.add_argument("--calib", type=int, default=128)
    a = ap.parse_args()
    set_threads(2)
    torch.manual_seed(0)
    w = Writer(f"quant_ppl_{a.model}")
    w(part="env", **env_info())
    base, tok = load_model(a.model)
    test = windows(wikitext_ids(tok, "test"), 1024, a.eval_windows or None)
    calib = windows(wikitext_ids(tok, "train"), 1024, a.calib, seed=0)
    nparams = sum(p.numel() for p in base.parameters())
    w(part="meta", model=a.model, params=nparams, eval_tokens=int(test.numel()), calib_tokens=int(calib.numel()))
    t = time.perf_counter()
    w(part="ppl", model=a.model, method="fp16", bits=16, ppl=perplexity(base, test),
      quant_s=0.0, eval_s=time.perf_counter() - t)
    outlier_stats(base, w, a.model, calib)
    methods = list(WEIGHT_METHODS) + list(ACT_METHODS) if a.methods == "all" else a.methods.split(",")
    for name in methods:
        model = copy.deepcopy(base)
        t0 = time.perf_counter()
        if name in WEIGHT_METHODS:
            cfg = WEIGHT_METHODS[name]
            info = apply_weight_method(model, cfg, calib)
            bits = cfg["bits"]
        else:
            cfg = ACT_METHODS[name]
            info = apply_act_method(model, cfg, calib)
            bits = 8
        torch.cuda.synchronize()
        qs = time.perf_counter() - t0
        t1 = time.perf_counter()
        ppl = perplexity(model, test)
        extra = {}
        if name == "llm_int8":
            oc = [c for q in info["wrapped"].values() for c in q.outlier_cols]
            extra = {"outlier_cols_mean": sum(oc) / len(oc), "outlier_cols_max": max(oc)}
        w(part="ppl", model=a.model, method=name, bits=bits, ppl=ppl, quant_s=qs,
          eval_s=time.perf_counter() - t1, **extra)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
