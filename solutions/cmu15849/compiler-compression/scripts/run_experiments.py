"""Generate every CSV/figure for the compiler-compression lab (CPU, < 5 minutes)."""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT.parents[2] / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mlsys_calc  # noqa: E402
from compkit.autodiff import Tensor, numerical_grad  # noqa: E402
from compkit.graph_ir import (Graph, evaluate, fuse_elementwise, kernel_cost,  # noqa: E402
                              merge_shared_matmuls, qkv_projection, transformer_mlp, cse, constant_fold)
from compkit.lora import LoRALinear, fit_low_rank_update  # noqa: E402
from compkit.pruning import (accuracy, magnitude_mask, make_mlp, nm_mask, prune_model,  # noqa: E402
                             teacher_data, train)
from compkit.quantization import (gptq, kmeans_quantize, llm_int8, quantize_act,  # noqa: E402
                                  quantize_weight, rel_err, smooth, synthetic_activations,
                                  synthetic_weight, w8a8)
from compkit.schedule_search import (Schedule, cost, evolutionary_search, grid_search,  # noqa: E402
                                     random_search, space)

RAW = ROOT / "artifacts" / "raw"
FIG = ROOT / "artifacts" / "figures"


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- autodiff

def exp_autodiff() -> None:
    rng = np.random.default_rng(0)
    x = Tensor(rng.normal(size=(8, 5)))
    w1, b1, w2 = (Tensor(rng.normal(size=s) * 0.5) for s in ((5, 16), (16,), (16, 4)))
    labels = rng.integers(0, 4, 8)

    def loss_fn():
        return ((x @ w1 + b1).relu() @ w2).softmax_cross_entropy(labels)

    loss = loss_fn()
    loss.backward()
    tx, tw1, tb1, tw2 = (torch.tensor(t.data, requires_grad=True) for t in (x, w1, b1, w2))
    tl = torch.nn.functional.cross_entropy(torch.relu(tx @ tw1 + tb1) @ tw2, torch.tensor(labels))
    tl.backward()
    rows = []
    for name, ours, theirs in (("x", x, tx), ("W1", w1, tw1), ("b1", b1, tb1), ("W2", w2, tw2)):
        num = numerical_grad(lambda: float(loss_fn().data), ours)
        rows.append({"param": name, "max_abs_diff_vs_torch": float(np.abs(ours.grad - theirs.grad.numpy()).max()),
                     "max_abs_diff_vs_finite_diff": float(np.abs(ours.grad - num).max()),
                     "grad_max_abs": float(np.abs(ours.grad).max())})
    write_csv("autodiff_check.csv", rows)


# --------------------------------------------------------------------------- graph IR

def exp_graph() -> dict:
    rng = np.random.default_rng(0)
    # correctness of every pass on small random-weight graphs
    checks = []
    g = transformer_mlp(32, 16, 64, rng)
    feeds = {"x": rng.normal(size=(32, 16))}
    ref = evaluate(g, feeds)[0]
    for name, gg in (("constant_fold+cse", cse(constant_fold(g))), ("fuse_elementwise", fuse_elementwise(g)),
                     ("fuse_epilogue", fuse_elementwise(g, epilogue=True))):
        checks.append({"graph": "mlp", "pass": name, "kernels_before": len(g.kernels()),
                       "kernels_after": len(gg.kernels()), "max_abs_diff": float(np.abs(evaluate(gg, feeds)[0] - ref).max())})
    q = qkv_projection(32, 16, 4, rng)
    qm = fuse_elementwise(merge_shared_matmuls(q))
    diffs = [float(np.abs(a - b).max()) for a, b in zip(evaluate(qm, feeds), evaluate(q, feeds))]
    checks.append({"graph": "qkv", "pass": "merge_shared_matmuls", "kernels_before": len(q.kernels()),
                   "kernels_after": len([k for k in qm.kernels() if k.op != "slice"]), "max_abs_diff": max(diffs)})
    write_csv("graph_pass_checks.csv", checks)

    tokens, d, dff = 4096, 4096, 16384
    base = transformer_mlp(tokens, d, dff)
    variants = {"unfused": base, "elementwise_fused": fuse_elementwise(base),
                "epilogue_fused": fuse_elementwise(base, epilogue=True)}
    krows, totals = [], []
    for name, gg in variants.items():
        rows = kernel_cost(gg)
        for r in rows:
            krows.append({"variant": name, **r})
        totals.append({"variant": name, "kernels": len(rows), "hbm_GB": sum(r["bytes"] for r in rows) / 1e9,
                       "time_us_roofline": sum(r["time_us"] for r in rows),
                       "non_matmul_GB": sum(r["bytes"] for r in rows if "matmul" not in r["kernel"]) / 1e9})
    for t in totals:
        t["hbm_vs_unfused"] = t["hbm_GB"] / totals[0]["hbm_GB"]
        t["time_vs_unfused"] = t["time_us_roofline"] / totals[0]["time_us_roofline"]
    write_csv("fusion_kernels.csv", krows)
    write_csv("fusion_totals.csv", totals)

    m = mlsys_calc.MODELS["llama3-8b"]
    qrows = []
    for tok in (1, 4096):
        g0 = qkv_projection(tok, m.d_model, m.n_kv_heads * m.head_dim)
        g1 = merge_shared_matmuls(g0)
        for name, gg in (("separate", g0), ("merged", g1)):
            rows = kernel_cost(gg)
            mm = [r for r in rows if "matmul" in r["kernel"]]
            qrows.append({"tokens": tok, "variant": name, "matmul_kernels": len(mm),
                          "matmul_hbm_MB": sum(r["bytes"] for r in mm) / 1e6,
                          "matmul_time_us": sum(r["time_us"] for r in mm),
                          "bound": "/".join(sorted({r["bound"] for r in mm}))})
    write_csv("qkv_merge.csv", qrows)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(variants)
    for i, name in enumerate(names):
        rows = [r for r in krows if r["variant"] == name]
        bottom = 0.0
        for r in rows:
            gb = r["bytes"] / 1e9
            ax.bar(i, gb, bottom=bottom, edgecolor="k", lw=0.5,
                   color="tab:blue" if "matmul" in r["kernel"] else "tab:orange")
            if gb > 0.08:
                ax.text(i, bottom + gb / 2, r["kernel"], ha="center", va="center", fontsize=6)
            bottom += gb
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("HBM traffic per MLP block (GB, bf16)")
    ax.set_title(f"T={tokens}, d={d}, d_ff={dff}; blue = kernels containing a matmul", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fusion_traffic.png", dpi=140)
    plt.close(fig)
    return {t["variant"]: t["hbm_GB"] for t in totals}


# --------------------------------------------------------------------------- schedule search

def exp_schedule() -> dict:
    m = n = k = 2048
    sp = space()
    costs = {s: cost(s, m, n, k) for s in sp}
    best_s = min(costs, key=costs.get)
    best = costs[best_s]
    target = 1.05 * best
    n_ok = sum(c <= target for c in costs.values())
    runs = []
    for seed in range(200):
        runs.append({"method": "random", "seed": seed,
                     "evals": random_search(costs, target, np.random.default_rng(seed))})
        runs.append({"method": "evolutionary", "seed": seed,
                     "evals": evolutionary_search(costs, target, np.random.default_rng(10_000 + seed))})
    runs.append({"method": "grid", "seed": -1, "evals": grid_search(costs, target)})
    write_csv("schedule_runs.csv", runs)
    stats = []
    for meth in ("grid", "random", "evolutionary"):
        ev = np.array([r["evals"] for r in runs if r["method"] == meth])
        stats.append({"method": meth, "space_size": len(sp), "configs_within_5pct": n_ok,
                      "evals_median": float(np.median(ev)), "evals_mean": float(ev.mean()),
                      "evals_p90": float(np.percentile(ev, 90)), "runs": len(ev)})
    write_csv("schedule_search.csv", stats)
    top = sorted(costs.items(), key=lambda kv: kv[1])[:10]
    worst = max(costs.values())
    naive = costs[Schedule("ijk", 4, 4, 4)]
    write_csv("schedule_top.csv", [{"rank": i + 1, "order": s.order, "tm": s.tm, "tn": s.tn, "tk": s.tk,
                                    "time_ms": c * 1e3, "gflops": 2 * m * n * k / c / 1e9}
                                   for i, (s, c) in enumerate(top)]
              + [{"rank": "tiles4", "order": "ijk", "tm": 4, "tn": 4, "tk": 4, "time_ms": naive * 1e3,
                  "gflops": 2 * m * n * k / naive / 1e9},
                 {"rank": "worst", "order": "", "tm": "", "tn": "", "tk": "", "time_ms": worst * 1e3,
                  "gflops": 2 * m * n * k / worst / 1e9}])

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for meth in ("random", "evolutionary"):
        ev = np.sort([r["evals"] for r in runs if r["method"] == meth])
        ax.step(ev, np.arange(1, len(ev) + 1) / len(ev), where="post", label=meth)
    ax.axvline(stats[0]["evals_median"], color="k", ls="--", label="grid (enumeration order)")
    ax.set_xscale("log")
    ax.set_xlabel("cost-model evaluations to reach within 5% of best")
    ax.set_ylabel("fraction of 200 runs")
    ax.set_title(f"matmul {m}^3 tiling space ({len(sp)} schedules, {n_ok} within 5%)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "schedule_search.png", dpi=140)
    plt.close(fig)
    from compkit.schedule_search import Machine
    mach = Machine()
    return {"best": f"{best_s}", "best_ms": best * 1e3,
            "compute_floor_ms": 2 * m * n * k / mach.peak_flops * 1e3,
            "ridge_flop_per_byte": mach.peak_flops / mach.bw, "cache_bytes": mach.cache_bytes}


# --------------------------------------------------------------------------- quantization

def exp_quant() -> None:
    rows = []
    for dist, df in (("gaussian", None), ("student_t4", 4.0)):
        rng = np.random.default_rng(0)
        wd = synthetic_weight(1024, 1024, rng, df=df)
        x, _ = synthetic_activations(512, 1024, rng)
        yd = x @ wd
        for bits in (8, 4):
            for gran in ("tensor", "channel", "group"):
                wq = quantize_weight(wd, bits, gran)
                rows.append({"weights": dist, "bits": bits, "granularity": gran if gran != "group" else "group128",
                             "weight_rel_err": rel_err(wd, wq), "output_rel_err": rel_err(yd, x @ wq)})
        wk, _ = kmeans_quantize(wd, 16)
        rows.append({"weights": dist, "bits": 4, "granularity": "kmeans16_shared",
                     "weight_rel_err": rel_err(wd, wk), "output_rel_err": rel_err(yd, x @ wk)})
    write_csv("quant_weight.csv", rows)

    rng = np.random.default_rng(0)
    w = synthetic_weight(1024, 1024, rng)

    orows = []
    for scale in (1, 5, 10, 20, 50, 100):
        xo, idx = synthetic_activations(512, 1024, np.random.default_rng(1), n_outlier=8, outlier_scale=scale)
        yo = xo @ w
        rest = np.setdiff1d(np.arange(1024), idx)
        xq = quantize_act(xo, 8, "tensor")
        wq = quantize_weight(w, 8, "channel")
        y8, n_out = llm_int8(xo, w)
        xs, ws = smooth(xo, w, 0.5)
        orows.append({
            "outlier_scale": scale, "n_outlier_channels": 8, "d": 1024,
            "W8A8_tensor_tensor": rel_err(yo, w8a8(xo, w, "tensor", "tensor")),
            "W8A8_actTensor_wChannel": rel_err(yo, w8a8(xo, w, "tensor", "channel")),
            "W8A8_actToken_wChannel": rel_err(yo, w8a8(xo, w, "token", "channel")),
            "LLM.int8_threshold6": rel_err(yo, y8), "llm_int8_fp_channels": n_out,
            "SmoothQuant0.5_actTensor_wChannel": rel_err(yo, w8a8(xs, ws, "tensor", "channel")),
            "SmoothQuant0.5_actTensor_wTensor": rel_err(yo, w8a8(xs, ws, "tensor", "tensor")),
            "normal_channel_part_err_actTensor": rel_err(xo[:, rest] @ w[rest], xq[:, rest] @ wq[rest]),
        })
    write_csv("quant_outliers.csv", orows)

    grows = []
    for decay in (0.0, 0.5, 1.0):
        r = np.random.default_rng(2)
        wg = synthetic_weight(512, 512, r)
        xc, _ = synthetic_activations(4096, 512, r, rank_decay=decay)
        perm = r.permutation(len(xc))
        cal, test = xc[perm[:2048]], xc[perm[2048:]]
        rtn = quantize_weight(wg, 4, "channel")
        gq = gptq(wg, cal, 4)
        grows.append({"input_spectrum_decay": decay, "bits": 4, "granularity": "channel",
                      "rtn_output_err_test": rel_err(test @ wg, test @ rtn),
                      "gptq_output_err_test": rel_err(test @ wg, test @ gq),
                      "rtn_output_err_cal": rel_err(cal @ wg, cal @ rtn),
                      "gptq_output_err_cal": rel_err(cal @ wg, cal @ gq),
                      "rtn_weight_err": rel_err(wg, rtn), "gptq_weight_err": rel_err(wg, gq)})
    write_csv("gptq.csv", grows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, lab in (("W8A8_actTensor_wChannel", "W8A8 per-tensor act"),
                     ("W8A8_actToken_wChannel", "W8A8 per-token act"),
                     ("SmoothQuant0.5_actTensor_wChannel", "SmoothQuant a=0.5"),
                     ("LLM.int8_threshold6", "LLM.int8 (t=6)"),
                     ("normal_channel_part_err_actTensor", "per-tensor: non-outlier part")):
        ax.plot([r["outlier_scale"] for r in orows], [r[key] for r in orows], marker="o", label=lab,
                ls="--" if "part" in key else "-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("outlier channel magnitude (x normal), 8 of 1024 channels")
    ax.set_ylabel("relative error of Y = XW")
    ax.set_title("int8 activation outliers (weights per-channel int8)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "quant_outliers.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- pruning

def exp_pruning_matrix() -> None:
    rng = np.random.default_rng(0)
    w = synthetic_weight(1024, 1024, rng)
    x, _ = synthetic_activations(512, 1024, rng)
    y = x @ w
    rows = []
    for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        m = magnitude_mask(w, s)
        rows.append({"method": "unstructured", "sparsity": s, "weight_rel_err": rel_err(w, w * m),
                     "output_rel_err": rel_err(y, x @ (w * m))})
    for n_, m_ in ((2, 4), (4, 8), (1, 4), (2, 8)):
        m = nm_mask(w, n_, m_)
        rows.append({"method": f"{n_}:{m_}", "sparsity": 1 - n_ / m_, "weight_rel_err": rel_err(w, w * m),
                     "output_rel_err": rel_err(y, x @ (w * m))})
    write_csv("pruning_matrix.csv", rows)


def _mlp_job(args: tuple[str, float]) -> dict:
    kind, s = args
    torch.set_num_threads(1)
    x, y = teacher_data(25_000)
    xtr, ytr, xte, yte = x[:20_000], y[:20_000], x[20_000:], y[20_000:]
    model = make_mlp()
    train(model, xtr, ytr, epochs=15)
    dense = accuracy(model, xte, yte)
    if kind == "dense":  # same 3 extra epochs as the pruned fine-tune, for a fair baseline
        train(model, xtr, ytr, epochs=3, lr=1e-3, seed=1)
        return {"method": "dense", "sparsity": 0.0, "acc_no_ft": dense,
                "acc_ft": accuracy(model, xte, yte), "dense_acc": dense}
    masks = prune_model(model, kind, s)
    no_ft = accuracy(model, xte, yte)
    train(model, xtr, ytr, epochs=3, lr=1e-3, masks=masks, seed=1)
    real = float(sum((m == 0).sum() for m in masks.values()) / sum(m.numel() for m in masks.values()))
    return {"method": kind, "sparsity": round(real, 4), "acc_no_ft": no_ft,
            "acc_ft": accuracy(model, xte, yte), "dense_acc": dense}


def exp_pruning_mlp() -> list[dict]:
    jobs = [("dense", 0.0)] + [("magnitude", s) for s in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)] + [("2:4", 0.5)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_mlp_job, jobs))
    write_csv("pruning_mlp.csv", rows)
    return rows


def plot_pruning(mlp_rows: list[dict]) -> None:
    import csv as _csv
    with open(RAW / "pruning_matrix.csv", encoding="utf-8") as f:
        mat = list(_csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    un = [r for r in mat if r["method"] == "unstructured"]
    axes[0].plot([float(r["sparsity"]) for r in un], [float(r["output_rel_err"]) for r in un], "o-", label="unstructured")
    for r in mat:
        if r["method"] != "unstructured":
            axes[0].plot(float(r["sparsity"]), float(r["output_rel_err"]), "s", label=r["method"])
    axes[0].set_xlabel("sparsity")
    axes[0].set_ylabel("relative error of Y = XW")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    mg = [r for r in mlp_rows if r["method"] == "magnitude"]
    axes[1].plot([r["sparsity"] for r in mg], [r["acc_no_ft"] for r in mg], "o-", label="magnitude, no FT")
    axes[1].plot([r["sparsity"] for r in mg], [r["acc_ft"] for r in mg], "o-", label="magnitude + 3 ep FT")
    for r in mlp_rows:
        if r["method"] == "2:4":
            axes[1].plot(r["sparsity"], r["acc_no_ft"], "s", color="tab:red", label="2:4 no FT")
            axes[1].plot(r["sparsity"], r["acc_ft"], "^", color="tab:red", label="2:4 + FT")
    axes[1].axhline(mlp_rows[0]["dense_acc"], color="k", ls="--", lw=0.8, label="dense")
    axes[1].set_xlabel("sparsity")
    axes[1].set_ylabel("test accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pruning.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- LoRA

def _lora_job(r: int) -> dict:
    torch.set_num_threads(1)
    return fit_low_rank_update(d=256, true_rank=8, r=r, steps=1500)


def exp_lora() -> None:
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_lora_job, (1, 2, 4, 8, 16)))
    write_csv("lora_fit.csv", rows)

    torch.manual_seed(0)
    base = torch.nn.Linear(512, 512, dtype=torch.float64)
    lora = LoRALinear(base, 8, 16)
    x = torch.randn(64, 512, dtype=torch.float64)
    init_diff = float((lora(x) - base(x)).abs().max())
    with torch.no_grad():
        lora.B.normal_(0, 0.1)
    merge_diff = float((lora.merged()(x) - lora(x)).abs().max())

    m = mlsys_calc.MODELS["llama3-8b"]
    kv = m.n_kv_heads * m.head_dim
    prow = []
    for r in (8, 16, 64):
        qv = m.layers * r * ((m.d_model + m.d_model) + (m.d_model + kv))
        allp = m.layers * r * (2 * (m.d_model + m.d_model) + 2 * (m.d_model + kv)
                               + 3 * (m.d_model + m.d_ff))
        prow.append({"model": m.name, "r": r, "lora_qv_params": qv, "lora_all_linear_params": allp,
                     "full_params": m.params(), "qv_frac": qv / m.params(), "all_frac": allp / m.params(),
                     "init_output_max_diff": init_diff, "merge_output_max_diff": merge_diff})
    write_csv("lora_params.csv", prow)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    summary: dict = {}
    exp_autodiff()
    summary["fusion_hbm_GB"] = exp_graph()
    summary["schedule"] = exp_schedule()
    exp_quant()
    exp_pruning_matrix()
    mlp_rows = exp_pruning_mlp()
    plot_pruning(mlp_rows)
    exp_lora()
    summary["wall_s"] = time.perf_counter() - t0
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
