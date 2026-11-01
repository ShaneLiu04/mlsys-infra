"""Train target/draft briefly, then produce every CSV/figure cited in the README (~3-4 min CPU)."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))
sys.path.insert(0, str(LAB.parents[2] / "tools"))

import mlsys_calc as calc  # noqa: E402
from kvspec.data import Grammar, decode, encode  # noqa: E402
from kvspec.model import GPT, GPTConfig, KVCache, generate_kv, generate_nocache  # noqa: E402
from kvspec.speculative import speculative_generate, target_only_kv  # noqa: E402
from kvspec.train import eval_loss, train  # noqa: E402

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
TARGET_CFG = GPTConfig(n_layer=4, n_head=4, n_embd=128, block_size=1024)
DRAFT_CFG = GPTConfig(n_layer=1, n_head=2, n_embd=48, block_size=1024)
TARGET_STEPS, DRAFT_STEPS = 300, 400
PROMPT_LEN, N_NEW, N_PROMPTS = 16, 104, 16
GAMMAS = (1, 2, 4, 8)


def write_csv(name: str, rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {name} ({len(rows)} rows)")


def train_models(grammar: Grammar) -> tuple[GPT, GPT, list[str]]:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    train_data = torch.tensor(encode(grammar.text(1_000_000, seed=0)))
    val_text = grammar.text(200_000, seed=1)
    val_data = torch.tensor(encode(val_text))
    log_rows, info = [], {}
    models = {}
    for name, cfg, steps in (("target", TARGET_CFG, TARGET_STEPS), ("draft", DRAFT_CFG, DRAFT_STEPS)):
        t0 = time.perf_counter()
        model, log = train(cfg, train_data, steps, seed=0)
        dt = time.perf_counter() - t0
        log_rows += [{"model": name, **r} for r in log]
        info[name] = {"params": model.n_params(), "layers": cfg.n_layer, "d_model": cfg.n_embd,
                      "steps": steps, "train_s": round(dt, 1), "val_loss": eval_loss(model, val_data)}
        models[name] = model
        print(f"  trained {name}: {info[name]}")
    write_csv("train_log.csv", log_rows)
    (RAW / "models.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    starts = np.random.default_rng(5).integers(0, len(val_text) - PROMPT_LEN, N_PROMPTS)
    prompts = [val_text[s:s + PROMPT_LEN] for s in starts]
    return models["target"], models["draft"], prompts


def exp_kv(target: GPT, prompts: list[str]) -> None:
    rows = []
    for p in prompts[:5]:
        idx = torch.tensor([encode(p)])
        a, la = generate_nocache(target, idx, 200, return_logits=True)
        b, lb = generate_kv(target, idx, 200, return_logits=True)
        rows.append({"prompt": p, "new_tokens": 200, "greedy_identical": bool(torch.equal(a, b)),
                     "max_abs_logit_diff": float((torch.stack(la) - torch.stack(lb)).abs().max()),
                     "sample": decode(b[0, len(p):len(p) + 60].tolist())})
    write_csv("kv_equivalence.csv", rows)

    prompt = torch.tensor([encode(prompts[0][:8])])
    trows = []
    for total in (64, 128, 256, 512, 1024):
        n = total - 8
        res = {}
        for name, fn in (("nocache", generate_nocache), ("kv", generate_kv)):
            best = float("inf")
            for _ in range(1 if (name == "nocache" and total >= 512) else 2):
                t0 = time.perf_counter()
                fn(target, prompt, n)
                best = min(best, time.perf_counter() - t0)
            res[name] = best
        trows.append({"total_len": total, "new_tokens": n, "nocache_s": res["nocache"], "kv_s": res["kv"],
                      "nocache_tok_s": n / res["nocache"], "kv_tok_s": n / res["kv"],
                      "speedup": res["nocache"] / res["kv"]})
        print(f"  kv timing T={total}: {trows[-1]}")
    write_csv("kv_timing.csv", trows)

    step_rows = []
    with torch.no_grad():
        idx = prompt.clone()
        cache = KVCache(target.cfg)
        target(idx, cache)
        nxt = torch.tensor([[1]])
        for t in range(8, 1024):
            t0 = time.perf_counter()
            target(nxt, cache)
            t1 = time.perf_counter()
            idx = torch.cat([idx, nxt], 1)
            if t % 16 == 0:
                t2 = time.perf_counter()
                target(idx)
                step_rows.append({"position": t, "kv_ms": (t1 - t0) * 1e3,
                                  "nocache_ms": (time.perf_counter() - t2) * 1e3})
    write_csv("kv_step_latency.csv", step_rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pos = np.array([r["position"] for r in step_rows])
    for key, lab in (("nocache_ms", "no cache (full prefix)"), ("kv_ms", "KV cache (1 token)")):
        ax[0].plot(pos, [r[key] for r in step_rows], ".-", lw=0.8, label=lab)
    ax[0].set(xlabel="position t", ylabel="ms per decode step", title="Per-step latency (target, batch 1)")
    ax[0].legend(fontsize=8)
    L = [r["total_len"] for r in trows]
    ax[1].loglog(L, [r["nocache_s"] for r in trows], "o-", base=2, label="no cache")
    ax[1].loglog(L, [r["kv_s"] for r in trows], "s-", base=2, label="KV cache")
    ax[1].loglog(L, [trows[0]["nocache_s"] * (x / L[0]) ** 2 for x in L], "k:", base=2, label="~T^2")
    ax[1].loglog(L, [trows[0]["kv_s"] * (x / L[0]) for x in L], "k--", base=2, label="~T")
    ax[1].set(xlabel="total sequence length T", ylabel="total generate time (s)", title="Total cost")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "kv_cache_timing.png", dpi=130)
    plt.close(fig)


def _time_per_token(model: GPT, n: int = 200) -> float:
    with torch.no_grad():
        cache = KVCache(model.cfg)
        model(torch.tensor([[1] * 16]), cache)
        x = torch.tensor([[2]])
        t0 = time.perf_counter()
        for _ in range(n):
            model(x, cache)
        return (time.perf_counter() - t0) / n


def _time_forward(model: GPT, n_tok: int, prefix: int = 64, n: int = 200) -> float:
    """Cached forward of n_tok new tokens after `prefix` cached tokens (cache rolled back each time)."""
    with torch.no_grad():
        cache = KVCache(model.cfg)
        model(torch.tensor([[1] * prefix]), cache)
        x = torch.tensor([[2] * n_tok])
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(n):
                cache.length = prefix
                model(x, cache)
            best = min(best, (time.perf_counter() - t0) / n)
        return best


def exp_spec(target: GPT, draft: GPT, prompts: list[str]) -> None:
    ids = [encode(p) for p in prompts]
    c = min(_time_per_token(draft) for _ in range(3)) / min(_time_per_token(target) for _ in range(3))
    t1 = _time_forward(target, 1)
    verify = {g: _time_forward(target, g + 1) / t1 for g in GAMMAS}
    rows = []
    for mode, temp in (("greedy", 0.0), ("sample", 1.0)):
        base_best = float("inf")
        for _ in range(2):
            t0 = time.perf_counter()
            base = [target_only_kv(target, p, N_NEW, temp, seed=i) for i, p in enumerate(ids)]
            base_best = min(base_best, time.perf_counter() - t0)
        for gamma in GAMMAS:
            best = float("inf")
            for _ in range(2):
                t0 = time.perf_counter()
                outs = [speculative_generate(target, draft, p, N_NEW, gamma, temp, seed=i)
                        for i, p in enumerate(ids)]
                best = min(best, time.perf_counter() - t0)
            sts = [s for _, s in outs]
            acc, prop = sum(s.accepted for s in sts), sum(s.proposed for s in sts)
            tested = sum(s.tested for s in sts)
            rounds = sum(s.rounds for s in sts)
            full = [n for s in sts for n in s.accepted_per_round]
            alpha = acc / tested
            overlap = sum(s.overlap_sum for s in sts) / tested if temp else None
            row = {"mode": mode, "gamma": gamma, "prompts": len(ids), "new_tokens_each": N_NEW,
                   "alpha_measured": alpha, "accepted_over_proposed": acc / prop,
                   "alpha_overlap_1_minus_tv": overlap if temp else "",
                   "target_calls": rounds, "tokens_per_target_call": (acc + rounds) / rounds,
                   "formula_tokens_per_call": calc.speculative_expected_tokens(alpha, gamma),
                   "formula_tokens_overlap": calc.speculative_expected_tokens(overlap, gamma) if temp else "",
                   "mean_accepted_per_round": float(np.mean(full)),
                   "cost_ratio_c": c, "predicted_speedup": calc.speculative_speedup(alpha, gamma, c),
                   "verify_cost_vs_1tok": verify[gamma],
                   "predicted_speedup_with_verify_cost": calc.speculative_expected_tokens(alpha, gamma)
                   / (gamma * c + verify[gamma]),
                   "baseline_s": base_best, "spec_s": best, "measured_speedup": base_best / best,
                   "greedy_identical": (sum(o == b for (o, _), b in zip(outs, base)) if temp == 0.0 else "")}
            rows.append(row)
            print(f"  spec {mode} g={gamma}: alpha={alpha:.3f} tok/call={row['tokens_per_target_call']:.2f} "
                  f"formula={row['formula_tokens_per_call']:.2f} speedup={row['measured_speedup']:.2f}")
    write_csv("speculative.csv", rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mode, st in (("greedy", "o"), ("sample", "s")):
        rr = [r for r in rows if r["mode"] == mode]
        g = [r["gamma"] for r in rr]
        ax[0].plot(g, [r["tokens_per_target_call"] for r in rr], st + "-", label=f"{mode} measured")
        ax[0].plot(g, [r["formula_tokens_per_call"] for r in rr], st + "--", label=f"{mode} (1-a^(g+1))/(1-a)")
        ax[1].plot(g, [r["measured_speedup"] for r in rr], st + "-", label=f"{mode} measured")
        ax[1].plot(g, [r["predicted_speedup"] for r in rr], st + "--", label=f"{mode} predicted (c={c:.2f})")
        ax[1].plot(g, [r["predicted_speedup_with_verify_cost"] for r in rr], st + ":",
                   label=f"{mode} predicted, measured verify cost")
    ax[0].set(xlabel="gamma (draft tokens)", ylabel="tokens per target forward", title="Acceptance")
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].set(xlabel="gamma", ylabel="wall-clock speedup vs target+KV", title="CPU speedup (batch 1)")
    for a in ax:
        a.set_xscale("log", base=2)
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "speculative.png", dpi=130)
    plt.close(fig)


def exp_inference_math() -> None:
    m, g = calc.MODELS["llama3-8b"], calc.GPUS["h100"]
    rows = []
    for seq in (4096, 32768):
        for b in (1, 8, 64, 256):
            r = calc.decode_step_time(m, g, seq, b)
            rows.append({"model": m.name, "gpu": g.name, "seq": seq, "batch": b,
                         "kv_bytes_per_token_kib": calc.kv_cache_bytes(m, 1) / 1024,
                         "weight_gb": r["weight_gb"], "kv_gb": r["kv_gb"],
                         "fits_80gb": r["weight_gb"] + r["kv_gb"] <= g.hbm_bytes / calc.GB,
                         "intensity_flop_per_byte": r["intensity"], "ridge": g.ridge,
                         "step_ms": r["time_s"] * 1e3, "tokens_per_s": r["tokens_per_s"], "bound": r["bound"],
                         "critical_batch_no_kv": calc.critical_batch(m, g)})
    write_csv("inference_math.csv", rows)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    torch.manual_seed(0)
    exp_inference_math()
    target, draft, prompts = train_models(Grammar())
    torch.set_num_threads(1)
    exp_kv(target, prompts)
    exp_spec(target, draft, prompts)
    meta = {"machine": platform.processor(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "torch": torch.__version__, "train_threads": min(8, os.cpu_count() or 1), "decode_threads": 1,
            "target_cfg": TARGET_CFG.__dict__, "draft_cfg": DRAFT_CFG.__dict__,
            "elapsed_s": round(time.perf_counter() - t0, 1)}
    (RAW / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done in {meta['elapsed_s']} s")


if __name__ == "__main__":
    main()
