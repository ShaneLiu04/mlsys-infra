"""Generate every CSV/figure for the distributed-training lab (CPU, < 5 minutes)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
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
from disttrain.checkpointing import (best_under_budget, make_chain, no_checkpoint,  # noqa: E402
                                     optimal_frontier, run_chain, sqrt_checkpoint)
from disttrain.collectives import ALGORITHMS, allreduce, closed_form_time  # noqa: E402
from disttrain.pipeline import bubble_fraction, peak_in_flight, simulate  # noqa: E402
from disttrain.tensor_parallel import (MLP, Attention, CommLog, ParallelAttention,  # noqa: E402
                                       ParallelMLP, unshard_grads)
from disttrain.zero import Adam, MLPSpec, init_params, make_data, train_single, train_zero  # noqa: E402

RAW = ROOT / "artifacts" / "raw"
FIG = ROOT / "artifacts" / "figures"
H100 = mlsys_calc.GPUS["h100"]
ALPHA = 5e-6  # assumed per-message latency (NCCL-like order of magnitude)
BETA = 1.0 / H100.link_bw


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- collectives

def exp_collectives() -> dict:
    p = 8
    rng = np.random.default_rng(0)
    rows, stats_at = [], {}
    sim_sizes = [2 ** k for k in range(6, 23, 2)]
    for nbytes in sim_sizes:
        bufs = [rng.standard_normal(nbytes // 4).astype(np.float32) for _ in range(p)]
        for alg in ALGORITHMS:
            out, st = allreduce(alg, bufs)
            want = np.sum(np.stack(bufs).astype(np.float64), 0)
            err = max(float(np.max(np.abs(o - want))) for o in out)
            stats_at[alg] = (nbytes, st)
            rows.append({"alg": alg, "P": p, "bytes": nbytes, "source": "simulated", "steps": st.steps,
                         "max_bytes_sent_per_rank": int(st.bytes_sent.max()),
                         "time_full_s": st.time(ALPHA, BETA, "full"),
                         "time_ring_topo_s": st.time(ALPHA, BETA, "ring"),
                         "closed_form_full_s": closed_form_time(alg, nbytes, p, ALPHA, BETA),
                         "max_abs_err_fp32": err})
    for nbytes in [2 ** k for k in range(24, 31, 2)]:
        for alg in ALGORITHMS:
            s0, st = stats_at[alg]
            sc = nbytes / s0
            rows.append({"alg": alg, "P": p, "bytes": nbytes, "source": "trace_scaled", "steps": st.steps,
                         "max_bytes_sent_per_rank": int(st.bytes_sent.max() * sc),
                         "time_full_s": st.time(ALPHA, BETA, "full", sc),
                         "time_ring_topo_s": st.time(ALPHA, BETA, "ring", sc),
                         "closed_form_full_s": closed_form_time(alg, nbytes, p, ALPHA, BETA),
                         "max_abs_err_fp32": ""})
    write_csv("allreduce_sweep.csv", rows)

    # crossover where ring beats tree / halving-doubling (closed form, full topology)
    grid = np.logspace(2, 10, 2000)
    cross = {}
    for other in ("tree", "halving_doubling"):
        for topo in ("full", "ring"):
            def t(alg, s):
                if topo == "full":
                    return closed_form_time(alg, s, p, ALPHA, BETA)
                s0, st = stats_at[alg]
                return st.time(ALPHA, BETA, "ring", s / s0)
            win = [s for s in grid if t("ring", s) < t(other, s)]
            cross[f"ring_beats_{other}_{topo}_from_bytes"] = float(win[0]) if win else None

    ptab = []
    for pp in (4, 8, 16, 32, 64):
        for nbytes in (1024, 2 ** 30):
            r = {"P": pp, "bytes": nbytes}
            for alg in ALGORITHMS:
                r[f"{alg}_s"] = closed_form_time(alg, nbytes, pp, ALPHA, BETA)
            ptab.append(r)
    write_csv("allreduce_vs_P.csv", ptab)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "time_full_s", "full topology (switch)"),
                           (axes[1], "time_ring_topo_s", "physical ring (link contention)")):
        for alg in ALGORITHMS:
            sel = [r for r in rows if r["alg"] == alg]
            x = [r["bytes"] for r in sel]
            y = [r[key] for r in sel]
            line, = ax.plot(x, y, label=alg)
            sim = [r for r in sel if r["source"] == "simulated"]
            ax.plot([r["bytes"] for r in sim], [r[key] for r in sim], "o", ms=3, color=line.get_color())
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("message bytes per rank")
        ax.set_title(f"P=8, alpha={ALPHA * 1e6:.0f}us, 1/beta=450GB/s: {title}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("modelled all-reduce time (s)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "allreduce_time_vs_size.png", dpi=140)
    plt.close(fig)
    return cross


def exp_gloo() -> dict:
    sizes = [2 ** k for k in range(10, 25, 2)]
    all_rows = []
    for world in (2, 4):
        out = RAW / f"gloo_w{world}.json"
        cmd = [sys.executable, "-m", "disttrain.gloo_bench", "--world", str(world), "--iters", "20",
               "--out", str(out), "--sizes", *map(str, sizes)]
        try:
            proc = subprocess.run(cmd, cwd=ROOT / "code", capture_output=True, timeout=240)
            if proc.returncode != 0:
                return {"gloo": "failed", "stderr": proc.stderr.decode(errors="replace")[-500:]}
        except subprocess.TimeoutExpired:
            return {"gloo": "timeout"}
        all_rows += json.loads(out.read_text())
        out.unlink()
    fits = {}
    for world in (2, 4):
        sel = [r for r in all_rows if r["world"] == world]
        a = np.array([[2 * (world - 1), 2 * (world - 1) / world * r["bytes"]] for r in sel])
        y = np.array([r["min_s"] for r in sel])  # medians are polluted by OS scheduling noise
        (alpha, beta), *_ = np.linalg.lstsq(a / y[:, None], np.ones_like(y), rcond=None)
        fits[world] = (float(alpha), float(beta))
        for r in sel:
            r["ring_fit_s"] = float(a[sel.index(r)] @ [alpha, beta])
            r["fit_alpha_s"], r["fit_bw_GBps"] = float(alpha), float(1 / beta / 1e9)
    write_csv("gloo_allreduce.csv", all_rows)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for world in (2, 4):
        sel = [r for r in all_rows if r["world"] == world]
        line, = ax.plot([r["bytes"] for r in sel], [r["min_s"] for r in sel], "o-", label=f"world={world} min")
        ax.plot([r["bytes"] for r in sel], [r["median_s"] for r in sel], "x:", color=line.get_color(),
                label=f"world={world} median")
        ax.plot([r["bytes"] for r in sel], [r["ring_fit_s"] for r in sel], "--", color="gray")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("bytes per rank")
    ax.set_ylabel("all_reduce time (s), measured")
    ax.set_title("torch.distributed gloo, 8-core Windows CPU\n(dashed: alpha-beta ring fit on min)", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "gloo_allreduce.png", dpi=140)
    plt.close(fig)
    return {"gloo": "ok", **{f"w{w}_alpha_us": f[0] * 1e6 for w, f in fits.items()},
            **{f"w{w}_bw_GBps": 1 / f[1] / 1e9 for w, f in fits.items()}}


# --------------------------------------------------------------------------- tensor parallel

def exp_tensor_parallel() -> None:
    rows = []
    b, s, d, h = 2, 8, 64, 8
    for t in (1, 2, 4, 8):
        for kind in ("mlp", "attn"):
            torch.manual_seed(0)
            ref = (MLP(d, 4 * d) if kind == "mlp" else Attention(d, h)).double()
            log = CommLog()
            par = ParallelMLP(ref, t, log) if kind == "mlp" else ParallelAttention(ref, t, log)
            x1 = torch.randn(b, s, d, dtype=torch.float64, requires_grad=True)
            x2 = x1.detach().clone().requires_grad_(True)
            y1, y2 = ref(x1), par(x2)
            g = torch.randn_like(y1)
            y1.backward(g)
            y2.backward(g)
            grads = unshard_grads(par, kind)
            gerr = max(float((grads[n] - p.grad).abs().max()) for n, p in ref.named_parameters())
            rows.append({"block": kind, "T": t, "fwd_max_abs_err": float((y1 - y2).abs().max()),
                         "param_grad_max_abs_err": gerr,
                         "input_grad_max_abs_err": float((x1.grad - x2.grad).abs().max()),
                         "allreduce_fwd": log.count("forward"), "allreduce_bwd": log.count("backward"),
                         "allreduce_bytes_fwd_fp64": log.nbytes("forward"),
                         "allreduce_bytes_bwd_fp64": log.nbytes("backward")})
    write_csv("tp_equivalence.csv", rows)

    m = mlsys_calc.MODELS["llama3-8b"]
    real = []
    for tp in (2, 4, 8):
        for seq in (2048, 8192):
            msg = 1 * seq * m.d_model * 2  # bf16 activations, batch 1
            per_layer = 4  # (attn + mlp) x (fwd + bwd), counted above
            t_ar = mlsys_calc.ring_allreduce_time(msg, tp, H100.link_bw, ALPHA)
            real.append({"model": m.name, "TP": tp, "seq": seq, "allreduce_msg_MB": msg / 1e6,
                         "allreduces_per_layer_fwd_bwd": per_layer,
                         "ring_time_per_allreduce_ms": t_ar * 1e3,
                         "comm_ms_per_step_all_layers": per_layer * m.layers * t_ar * 1e3})
    write_csv("tp_comm_llama3_8b.csv", real)


# --------------------------------------------------------------------------- ZeRO

def exp_zero() -> None:
    p, steps = 4, 20
    spec = MLPSpec()
    flat = init_params(spec, p)
    x, y = make_data(spec, 64)
    ref = train_single(spec, flat, x, y, steps, Adam())
    rows = []
    for stage in range(4):
        t0 = time.perf_counter()
        r = train_zero(stage, spec, flat, x, y, p, steps, Adam())
        rows.append({"stage": stage, "P": p, "params_padded": len(flat), "steps": steps,
                     "max_abs_diff_vs_single": float(np.max(np.abs(r.params - ref))),
                     "final_loss": r.losses[-1], **{f"mem_{k}_B": v for k, v in r.mem_bytes.items()},
                     "comm_bytes_per_step_per_rank": r.comm_bytes_per_step,
                     "wall_s": time.perf_counter() - t0})
    base_m, base_c = rows[0]["mem_total_B"], rows[0]["comm_bytes_per_step_per_rank"]
    for r in rows:
        r["mem_vs_stage0"] = r["mem_total_B"] / base_m
        r["comm_vs_stage0"] = r["comm_bytes_per_step_per_rank"] / base_c
    write_csv("zero_stages.csv", rows)

    paper = []
    for stage in range(4):
        mm = mlsys_calc.training_memory_bytes(7.5e9, dp=64, zero=stage)
        paper.append({"stage": stage, "params": 7.5e9, "dp": 64,
                      **{f"{k}_GB": v / 1e9 for k, v in mm.items()}})
    write_csv("zero_paper_scale.csv", paper)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cats = ["params", "grads", "adam_m", "adam_v"]
    bottom = np.zeros(4)
    for c in cats:
        vals = np.array([r[f"mem_{c}_B"] for r in rows]) / 1e3
        axes[0].bar([f"ZeRO-{s}" for s in range(4)], vals, bottom=bottom, label=c)
        bottom += vals
    axes[0].set_ylabel("persistent KB per rank (fp64, P=4)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("measured on simulated ranks", fontsize=9)
    axes[1].bar([f"ZeRO-{s}" for s in range(4)], [r["comm_vs_stage0"] for r in rows], color="tab:gray")
    axes[1].set_ylabel("bytes sent per rank per step / DP")
    axes[1].set_title("communication volume (ring RS/AG)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "zero_memory_comm.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- pipeline

def exp_pipeline() -> None:
    rows, peaks = [], []
    for p in (4, 8):
        for m in (1, 2, 4, 8, 16, 32, 64):
            for sched, v in (("gpipe", 1), ("1f1b", 1), ("interleaved", 2)):
                if sched == "interleaved" and m % p:
                    continue
                ops = simulate(sched, p, m, v=v)
                pk = peak_in_flight(ops, p, v)
                rows.append({"schedule": sched, "p": p, "m": m, "v": v,
                             "makespan_tf_units": max(o.end for o in ops),
                             "bubble_sim": bubble_fraction(ops, p),
                             "bubble_formula": mlsys_calc.pipeline_bubble(p, m, v),
                             "peak_act_stage0": pk[0], "peak_act_last": pk[-1]})
                if m == 16:
                    for d, val in enumerate(pk):
                        peaks.append({"schedule": sched, "p": p, "m": m, "v": v, "device": d,
                                      "peak_microbatch_activations": val})
    write_csv("pipeline_bubble.csv", rows)
    write_csv("pipeline_peak_activations.csv", peaks)

    p, m = 4, 8
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    cmap = plt.get_cmap("tab10")
    for ax, (sched, v) in zip(axes, (("gpipe", 1), ("1f1b", 1), ("interleaved", 2))):
        ops = simulate(sched, p, m, v=v)
        for o in ops:
            color = cmap(o.mb % 10)
            ax.barh(o.device, o.end - o.start, left=o.start, color=color,
                    alpha=1.0 if o.kind == "F" else 0.5, edgecolor="k", lw=0.3)
            label = f"{o.mb}" if v == 1 else f"{o.mb}.{o.chunk}"
            ax.text((o.start + o.end) / 2, o.device, label, ha="center", va="center", fontsize=5)
        ax.set_yticks(range(p))
        ax.set_yticklabels([f"dev{d}" for d in range(p)])
        ax.invert_yaxis()
        ax.set_title(f"{sched} (p={p}, m={m}, v={v}, tb=2tf): bubble={bubble_fraction(ops, p):.3f}",
                     fontsize=9)
    axes[-1].set_xlabel("time (units of one full-stage forward)")
    fig.tight_layout()
    fig.savefig(FIG / "pipeline_gantt.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for sched, v, mk in (("1f1b", 1, "o"), ("interleaved", 2, "s")):
        for p in (4, 8):
            sel = [r for r in rows if r["schedule"] == sched and r["p"] == p]
            ax.plot([r["m"] for r in sel], [r["bubble_sim"] for r in sel], mk, label=f"{sched} p={p} (sim)")
            ax.plot([r["m"] for r in sel], [r["bubble_formula"] for r in sel], "-", color="gray", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("micro-batches m")
    ax.set_ylabel("bubble fraction")
    ax.set_title("simulated (markers) vs (p-1)/v / (m+(p-1)/v) (lines)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pipeline_bubble.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- checkpointing

def exp_checkpointing() -> None:
    rows = []
    n = 64
    ones = [1] * n
    fr = optimal_frontier(ones, ones)
    for peak, plan in fr.items():
        rows.append({"chain": "uniform64", "policy": "dp_optimal", "peak": plan.peak,
                     "recompute": plan.recompute, "n_stored": len(plan.stored)})
    for name, plan in (("none", no_checkpoint(ones, ones)), ("sqrt_n", sqrt_checkpoint(ones, ones))):
        rows.append({"chain": "uniform64", "policy": name, "peak": plan.peak,
                     "recompute": plan.recompute, "n_stored": len(plan.stored)})
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 9, 32).tolist()
    costs = rng.integers(1, 9, 32).tolist()
    frh = optimal_frontier(sizes, costs)
    for peak, plan in frh.items():
        rows.append({"chain": "hetero32", "policy": "dp_optimal", "peak": plan.peak,
                     "recompute": plan.recompute, "n_stored": len(plan.stored)})
    sq = sqrt_checkpoint(sizes, costs)
    for name, plan in (("none", no_checkpoint(sizes, costs)), ("sqrt_n", sq)):
        rows.append({"chain": "hetero32", "policy": name, "peak": plan.peak,
                     "recompute": plan.recompute, "n_stored": len(plan.stored)})
    dp_at_sq = best_under_budget(frh, sq.peak)
    rows.append({"chain": "hetero32", "policy": "dp_at_sqrt_budget", "peak": dp_at_sq.peak,
                 "recompute": dp_at_sq.recompute, "n_stored": len(dp_at_sq.stored)})
    write_csv("ckpt_frontier.csv", rows)

    trows = []
    nl, d, b = 16, 256, 64
    model = make_chain(nl, d)
    x = torch.randn(b, d)
    ref_loss, ref_g, _ = run_chain(model, x, None)
    for seg in (None, 1, 2, 4, 8, 16):
        run_chain(model, x, seg)
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            loss, g, peak = run_chain(model, x, seg)
            times.append(time.perf_counter() - t0)
        dt = float(np.median(times))
        trows.append({"layers": nl, "d": d, "batch": b, "segment": seg or 0,
                      "peak_saved_bytes": peak, "peak_in_activations": peak / (b * d * 4),
                      "grad_max_abs_diff": max(float((a - c).abs().max()) for a, c in zip(g, ref_g)),
                      "fwd_bwd_median_ms": dt * 1e3})
    base = trows[0]["fwd_bwd_median_ms"]
    for r in trows:
        r["time_vs_no_ckpt"] = r["fwd_bwd_median_ms"] / base
    write_csv("ckpt_torch.csv", trows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, chain in zip(axes, ("uniform64", "hetero32")):
        sel = [r for r in rows if r["chain"] == chain and r["policy"] == "dp_optimal"]
        ax.step([r["peak"] for r in sel], [r["recompute"] for r in sel], where="post", label="DP optimal")
        for pol, mk in (("none", "ks"), ("sqrt_n", "r^")):
            r = next(r for r in rows if r["chain"] == chain and r["policy"] == pol)
            ax.plot(r["peak"], r["recompute"], mk, ms=8, label=pol)
        ax.set_xlabel("peak activation memory (units)")
        ax.set_ylabel("recompute cost (units)")
        ax.set_title(chain, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ckpt_frontier.png", dpi=140)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    summary = {"alpha_s": ALPHA, "link_bw_Bps": H100.link_bw}
    t0 = time.perf_counter()
    summary["collectives"] = exp_collectives()
    summary["gloo"] = exp_gloo()
    exp_tensor_parallel()
    exp_zero()
    exp_pipeline()
    exp_checkpointing()
    summary["wall_s"] = time.perf_counter() - t0
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
