"""Q4: pruning on GPU -- perplexity of GPT-2 under unstructured / N:M pruning, and kernel speed of
2:4 semi-structured sparsity (cuSPARSELt / CUTLASS) and CSR sparse vs dense cuBLAS."""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F

from common import Writer, bench, env_info, set_threads
from llm_utils import linear_modules, load_model, perplexity, set_weight_oi, weight_oi, wikitext_ids, windows
from quant_llm import collect


def unstructured_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    k = int(score.numel() * sparsity)
    if k == 0:
        return torch.ones_like(score, dtype=torch.bool)
    thr = score.flatten().kthvalue(k).values
    return score > thr


def row_mask(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-output-row unstructured mask (Wanda compares weights within a row)."""
    k = int(score.shape[1] * sparsity)
    idx = torch.topk(score, k, dim=1, largest=False).indices
    m = torch.ones_like(score, dtype=torch.bool)
    m.scatter_(1, idx, False)
    return m


def nm_mask(score: torch.Tensor, n: int, m: int) -> torch.Tensor:
    """Keep the n largest of every m consecutive input-dim weights."""
    o, i = score.shape
    g = score.view(o, i // m, m)
    idx = torch.topk(g, n, dim=-1).indices
    mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    return mask.view(o, i)


METHODS = {
    "magnitude_0.3": ("mag", "layer", 0.3), "magnitude_0.5": ("mag", "layer", 0.5),
    "magnitude_0.7": ("mag", "layer", 0.7), "magnitude_2:4": ("mag", "nm", (2, 4)),
    "magnitude_4:8": ("mag", "nm", (4, 8)), "wanda_0.5": ("wanda", "row", 0.5),
    "wanda_2:4": ("wanda", "nm", (2, 4)),
}


def ppl_part(w: Writer, model_name: str, eval_windows: int | None):
    base, tok = load_model(model_name)
    test = windows(wikitext_ids(tok, "test"), 1024, eval_windows)
    calib = windows(wikitext_ids(tok, "train"), 1024, 64, seed=0)
    w(part="prune_ppl", model=model_name, method="dense", sparsity=0.0, ppl=perplexity(base, test))
    mods = linear_modules(base)
    st = collect(base, mods, calib, want_h=True)
    xnorm = {n: torch.sqrt(torch.diag(s["H"])) for n, s in st.items()}
    del st
    for name, (score_kind, kind, arg) in METHODS.items():
        model = copy.deepcopy(base)
        zeros, total = 0, 0
        for n, m in linear_modules(model).items():
            wt = weight_oi(m).float()
            score = wt.abs() * (xnorm[n][None, :] if score_kind == "wanda" else 1.0)
            if kind == "layer":
                mask = unstructured_mask(score, arg)
            elif kind == "row":
                mask = row_mask(score, arg)
            else:
                mask = nm_mask(score, *arg)
            set_weight_oi(m, wt * mask)
            zeros += int((~mask).sum())
            total += mask.numel()
        w(part="prune_ppl", model=model_name, method=name, sparsity=zeros / total,
          ppl=perplexity(model, test))
        del model
        torch.cuda.empty_cache()


def speed_part(w: Writer, ms, nks):
    from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
    for n, k in nks:
        torch.manual_seed(0)
        wd = torch.randn(n, k, device="cuda", dtype=torch.float16)
        w24 = wd * nm_mask(wd.abs(), 2, 4)
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            flops = 2 * m * n * k
            t_dense = bench(lambda: F.linear(x, w24))
            w(part="sparse24", N=n, K=k, M=m, impl="dense_cublas", time_s=t_dense,
              tflops=flops / t_dense / 1e12, ok=True)
            ref = F.linear(x, w24).float()
            for backend in ["cusparselt", "cutlass"]:
                try:
                    SparseSemiStructuredTensor._FORCE_CUTLASS = backend == "cutlass"
                    ws = to_sparse_semi_structured(w24)
                    y = F.linear(x, ws)
                    err = float((y.float() - ref).norm() / ref.norm())
                    t = bench(lambda: F.linear(x, ws))
                    w(part="sparse24", N=n, K=k, M=m, impl=backend, time_s=t,
                      tflops=flops / t / 1e12, speedup=t_dense / t, rel_err=err, ok=True)
                except Exception as e:  # noqa: BLE001
                    w(part="sparse24", N=n, K=k, M=m, impl=backend, ok=False,
                      error=type(e).__name__ + ": " + str(e)[:200])
    n = k = 4096
    for m in [1, 32, 512]:
        x = torch.randn(k, m, device="cuda", dtype=torch.float16)
        for sp in [0.5, 0.9, 0.95, 0.99]:
            wd = torch.randn(n, k, device="cuda", dtype=torch.float16)
            wp = wd * unstructured_mask(wd.abs().float(), sp)
            t_dense = bench(lambda: wp @ x)
            try:
                csr = wp.to_sparse_csr()
                err = float(((csr @ x).float() - (wp @ x).float()).norm() / (wp @ x).float().norm())
                t = bench(lambda: csr @ x)
                w(part="csr", N=n, K=k, M=m, sparsity=sp, dense_s=t_dense, csr_s=t,
                  speedup=t_dense / t, rel_err=err, ok=True)
            except Exception as e:  # noqa: BLE001
                w(part="csr", N=n, K=k, M=m, sparsity=sp, dense_s=t_dense, ok=False, error=str(e)[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["ppl", "speed"], required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--eval-windows", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    set_threads(2)
    w = Writer(f"prune_{a.part}" + (f"_{a.model}" if a.part == "ppl" else ""))
    w(part="env", **env_info())
    if a.part == "ppl":
        ppl_part(w, a.model, a.eval_windows or None)
    elif a.quick:
        speed_part(w, [64], [(1024, 1024)])
    else:
        speed_part(w, [32, 128, 512, 2048, 8192], [(4096, 4096), (16384, 4096), (4096, 16384)])


if __name__ == "__main__":
    main()
