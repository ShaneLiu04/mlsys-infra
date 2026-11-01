"""Generate every CSV/figure for the serving-memory lab (CPU, < 5 minutes)."""

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

import mlsys_calc  # noqa: E402
from servesim.scheduler import CostModel, simulate_continuous, simulate_static, summarize  # noqa: E402
from servesim.waste import waste_table  # noqa: E402
from servesim.workload import LengthDist, poisson_arrivals, sample_lengths  # noqa: E402

RAW = ROOT / "artifacts" / "raw"
FIG = ROOT / "artifacts" / "figures"
COST = CostModel(mlsys_calc.MODELS["llama3-8b"], mlsys_calc.GPUS["h100"])
TTFT_SLO, TPOT_SLO = 1.0, 0.05
N_REQ = 3000
RATES = (2, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160)
POLICIES = ("static", "continuous", "chunked")


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def exp_waste() -> list[dict]:
    p, o = sample_lengths(10_000, seed=0)
    budget = COST.kv_budget_tokens()
    rows = waste_table(p, o, budget)
    for r in rows:
        r["kv_budget_tokens"] = budget
        r["kv_budget_GB"] = budget * COST.kv_per_token / 1e9
        r["kv_bytes_per_token"] = COST.kv_per_token
    write_csv("kv_waste.csv", rows)
    dist = LengthDist()
    stats = {"prompt_mean": float(p.mean()), "prompt_p50": float(np.median(p)), "prompt_p99": float(np.percentile(p, 99)),
             "output_mean": float(o.mean()), "output_p50": float(np.median(o)), "output_p99": float(np.percentile(o, 99)),
             "dist": dist.__dict__}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["contig 4096" if r["scheme"].startswith("contiguous_max") else
              "contig exact" if r["scheme"] == "contiguous_exact_final_len" else
              f"paged {r['block']}" if r["scheme"] == "paged" else "ideal" for r in rows]
    axes[0].bar(labels, [100 * r["waste_frac"] for r in rows], color="tab:red")
    axes[0].set_ylabel("KV memory wasted (%), lifetime avg")
    axes[1].bar(labels, [r["concurrent_seqs"] for r in rows], color="tab:blue")
    axes[1].set_ylabel(f"concurrent sequences in {rows[0]['kv_budget_GB']:.1f} GB KV")
    for ax in axes:
        ax.tick_params(axis="x", rotation=40, labelsize=8)
    fig.suptitle("Llama-3-8B on H100 80GB (0.9 util), lognormal lengths, 10k requests", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "kv_waste.png", dpi=140)
    plt.close(fig)
    return [stats]


def _run(args: tuple[str, float]) -> dict:
    policy, rate = args
    p, o = sample_lengths(N_REQ, seed=1)
    arr = poisson_arrivals(N_REQ, rate, seed=2)
    t0 = time.perf_counter()
    if policy == "static":
        res = simulate_static(COST, arr, p, o)
    else:
        res = simulate_continuous(COST, arr, p, o, chunked=policy == "chunked")
    s = summarize(res, TTFT_SLO, TPOT_SLO)
    assert res.leaked_blocks == 0 and all(r.generated == r.output for r in res.requests)
    return {"policy": policy, "rate_req_s": rate, **s, "sim_wall_s": time.perf_counter() - t0}


def exp_scheduler() -> list[dict]:
    jobs = [(pol, r) for pol in POLICIES for r in RATES]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_run, jobs))
    write_csv("scheduler_sweep.csv", rows)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    keys = (("throughput_tok_s", "output tokens/s"), ("ttft_p99_s", "P99 TTFT (s)"),
            ("tpot_p99_ms", "P99 TPOT (ms)"), ("goodput_req_s", f"goodput req/s (TTFT<={TTFT_SLO}s, TPOT<={TPOT_SLO * 1e3:.0f}ms)"))
    for ax, (k, lab) in zip(axes, keys):
        for pol in POLICIES:
            sel = [r for r in rows if r["policy"] == pol]
            ax.plot([r["rate_req_s"] for r in sel], [r[k] for r in sel], "o-", label=pol, ms=3)
        ax.set_xlabel("arrival rate (req/s)")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.3)
        if k in ("ttft_p99_s", "tpot_p99_ms"):
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle("modelled Llama-3-8B / H100 roofline step times (not GPU measurements)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "latency_vs_load.png", dpi=140)
    plt.close(fig)
    return rows


def _run_pressure(args: tuple[str, int]) -> dict:
    policy, kv_tokens = args
    rate = 60.0
    p, o = sample_lengths(N_REQ, seed=1)
    arr = poisson_arrivals(N_REQ, rate, seed=2)
    res = simulate_continuous(COST, arr, p, o, chunked=policy == "chunked", kv_tokens=kv_tokens)
    s = summarize(res, TTFT_SLO, TPOT_SLO)
    assert res.leaked_blocks == 0
    return {"policy": policy, "rate_req_s": rate, "kv_tokens": kv_tokens,
            "kv_GB": kv_tokens * COST.kv_per_token / 1e9,
            "recomputed_prefill_tokens": res.prefill_tokens - int(p.sum()),
            **{k: s[k] for k in ("throughput_tok_s", "ttft_p99_s", "tpot_p99_ms", "goodput_req_s", "preemptions")}}


def exp_kv_pressure() -> None:
    jobs = [(pol, kv) for pol in ("continuous", "chunked")
            for kv in (COST.kv_budget_tokens(), 100_000, 50_000, 25_000, 12_500)]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_run_pressure, jobs))
    write_csv("kv_pressure.csv", rows)


def exp_step_time() -> None:
    rows = []
    for batch in (1, 8, 32, 64, 128, 256):
        for seq in (512, 2048):
            d = mlsys_calc.decode_step_time(COST.model, COST.gpu, seq, batch)
            rows.append({"kind": "decode", "batch": batch, "ctx": seq, "tokens": batch,
                         "step_ms": COST.step_time(seq * batch, batch) * 1e3, "bound": d["bound"]})
    for tok in (128, 512, 2048, 8192):
        t = COST.step_time(0, 0, [(0, tok)])
        rows.append({"kind": "prefill", "batch": 1, "ctx": 0, "tokens": tok, "step_ms": t * 1e3,
                     "bound": mlsys_calc.roofline_time(2 * COST.model.params() * tok, COST.weight_bytes, COST.gpu)[1]})
    write_csv("step_time_model.csv", rows)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stats = exp_waste()
    exp_step_time()
    exp_scheduler()
    exp_kv_pressure()
    summary = {"model": COST.model.name, "gpu": COST.gpu.name, "kv_bytes_per_token": COST.kv_per_token,
               "weights_GB": COST.weight_bytes / 1e9, "kv_budget_tokens": COST.kv_budget_tokens(),
               "lengths": stats[0], "ttft_slo_s": TTFT_SLO, "tpot_slo_s": TPOT_SLO,
               "n_requests": N_REQ, "wall_s": time.perf_counter() - t0}
    (RAW / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
