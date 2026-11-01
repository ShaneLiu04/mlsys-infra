"""Training on GPU.

--part mqar       : learning curves on multi-query associative recall (MQAR) for 1- vs 2-layer GPTs and for
                    {bf16+SDPA, bf16+Triton flash, fp32+naive attention} with the same seed.
--part throughput : GPT-2-small-shaped model (12L, d=768, T=1024): tokens/s, peak memory and MFU
                    (6N rule from tools/mlsys_calc.py) for precision x attention implementation, plus a
                    sequence-length sweep at fixed tokens/step.
--part recall     : the hand-coded 2-layer associative-recall transformer evaluated on GPU vs CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import RAW_GPU, JsonlWriter, env_info, is_oom
from tiny_gpt import GPT, GPTConfig

import mlsys_calc as calc  # noqa: E402

def mqar_batch(bs: int, pairs: int, gen: torch.Generator, n_keys: int = 64, n_vals: int = 64):
    """[k1 v1 ... kP vP | q1 a1 ... qP aP]; queries are a permutation of the keys; loss on the answers."""
    dev = "cuda"
    keys = torch.rand(bs, n_keys, device=dev, generator=gen).argsort(1)[:, :pairs] + 1
    vals = torch.randint(0, n_vals, (bs, pairs), device=dev, generator=gen) + 1 + n_keys
    perm = torch.rand(bs, pairs, device=dev, generator=gen).argsort(1)
    qk, qv = keys.gather(1, perm), vals.gather(1, perm)
    seq = torch.empty(bs, 4 * pairs, dtype=torch.long, device=dev)
    seq[:, 0:2 * pairs:2], seq[:, 1:2 * pairs:2] = keys, vals
    seq[:, 2 * pairs::2], seq[:, 2 * pairs + 1::2] = qk, qv
    return seq, qv


def mqar_loss(model, seq, qv, pairs):
    logits = model(seq)[:, 2 * pairs::2].float()
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), qv.reshape(-1))
    acc = (logits.argmax(-1) == qv).float().mean()
    return loss, acc


def lr_at(step: int, total: int, peak: float, warmup: int = 100) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return 0.5 * peak * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def run_mqar(w: JsonlWriter, name: str, layers: int, precision: str, impl: str, steps: int,
             pairs: int = 16, bs: int = 256, d: int = 128, heads: int = 4, lr: float = 3e-3,
             n_keys: int = 64, n_vals: int = 64, wd: float = 0.0, part: str = "mqar") -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=1 + n_keys + n_vals, seq_len=4 * pairs, d_model=d, n_layers=layers, n_heads=heads)
    model = GPT(cfg).cuda()
    model.set_attn_impl(impl)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98), fused=True)
    kv = {"n_keys": n_keys, "n_vals": n_vals}
    gen = torch.Generator(device="cuda").manual_seed(1)
    amp = precision == "bf16"
    t_train = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        seq, qv = mqar_batch(bs, pairs, gen, **kv)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss, acc = mqar_loss(model, seq, qv, pairs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            torch.cuda.synchronize()
            t_train += time.perf_counter() - t0
            w.write({"part": f"{part}_curve", "run": name, "layers": layers, "precision": precision, "impl": impl,
                     "step": step, "loss": loss.item(), "acc": acc.item(), "lr": opt.param_groups[0]["lr"],
                     "elapsed_s": t_train})
            if step % 500 == 0:
                print(f"  {name} step {step} loss {loss.item():.3f} acc {acc.item():.3f}", flush=True)
            t0 = time.perf_counter()
    torch.cuda.synchronize()
    t_train += time.perf_counter() - t0
    model.eval()
    eval_gen = torch.Generator(device="cuda").manual_seed(12345)
    accs, losses = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for _ in range(16):
            seq, qv = mqar_batch(512, pairs, eval_gen, **kv)
            loss, acc = mqar_loss(model, seq, qv, pairs)
            accs.append(acc.item())
            losses.append(loss.item())
    tokens = steps * bs * 4 * pairs
    row = {"part": f"{part}_final", "run": name, "layers": layers, "precision": precision, "impl": impl,
           "steps": steps, "batch": bs, "pairs": pairs, "seq_len": 4 * pairs, "d_model": d, "heads": heads,
           "lr": lr, "weight_decay": wd, **kv,
           "params": model.n_params(), "train_s": t_train, "tokens_per_s": tokens / t_train,
           "eval_acc": float(np.mean(accs)), "eval_loss": float(np.mean(losses)), "eval_sequences": 16 * 512}
    w.write(row)
    print(f"mqar {name:<22} acc={row['eval_acc']:.4f} loss={row['eval_loss']:.3f} "
          f"{row['tokens_per_s'] / 1e6:.2f} Mtok/s {t_train:.1f}s", flush=True)


def chosen_mqar_config() -> dict:
    """Hardest sweep setting (n_keys * pairs) whose 2-layer model reached >= 90% eval accuracy."""
    p = RAW_GPU / "train_mqar_sweep.jsonl"
    best = {"n_keys": 64, "pairs": 4, "lr": 1e-3, "weight_decay": 0.0}
    if p.exists():
        rows = [json.loads(x) for x in p.read_text().splitlines() if '"sweep_final"' in x]
        ok = [r for r in rows if r["eval_acc"] >= 0.9]
        if not ok and rows:
            ok = [max(rows, key=lambda r: r["eval_acc"])]
        if ok:
            r = max(ok, key=lambda r: (r["n_keys"] * r["pairs"], r["eval_acc"]))
            best = {"n_keys": r["n_keys"], "pairs": r["pairs"], "lr": r["lr"], "weight_decay": r["weight_decay"]}
    return best


def part_mqar(w: JsonlWriter, quick: bool) -> None:
    steps = 200 if quick else 6000
    c = chosen_mqar_config()
    kw = dict(pairs=c["pairs"], lr=c["lr"], n_keys=c["n_keys"], n_vals=c["n_keys"], wd=c["weight_decay"])
    print("mqar config", c, flush=True)
    run_mqar(w, "L2_bf16_sdpa", 2, "bf16", "sdpa", steps, **kw)
    run_mqar(w, "L1_bf16_sdpa", 1, "bf16", "sdpa", steps, **kw)
    run_mqar(w, "L2_bf16_triton", 2, "bf16", "triton", steps, **kw)
    run_mqar(w, "L2_fp32_naive", 2, "fp32", "naive", steps, **kw)


def part_mqar_sweep(w: JsonlWriter, quick: bool) -> None:
    """Find a setting that crosses the induction-head phase transition within ~2k steps."""
    steps = 100 if quick else 6000
    for nk, pairs, lr, wd in ((64, 4, 1e-3, 0.0), (64, 8, 1e-3, 0.0), (64, 8, 3e-3, 0.1), (64, 16, 1e-3, 0.1),
                              (128, 16, 1e-3, 0.1)):
        run_mqar(w, f"k{nk}_P{pairs}_lr{lr:g}_wd{wd:g}", 2, "bf16", "sdpa", steps, pairs=pairs, lr=lr,
                 n_keys=nk, n_vals=nk, wd=wd, part="sweep")


def peak_flops() -> tuple[float, str]:
    p = RAW_GPU / "peaks.json"
    if p.exists():
        return json.loads(p.read_text())["peak_bf16_flops"], "measured_bf16_matmul"
    return 209.5e12, "spec_dense_bf16"


def throughput(w: JsonlWriter, precision: str, impl: str, seq_len: int, batch: int, sweep: str,
               steps: int = 10, warmup: int = 3) -> None:
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=32768, seq_len=seq_len, d_model=768, n_layers=12, n_heads=12)
    row = {"part": "throughput", "sweep": sweep, "precision": precision, "impl": impl, "seq_len": seq_len,
           "batch": batch, "tokens_per_step": batch * seq_len, "layers": 12, "d_model": 768, "heads": 12,
           "vocab": cfg.vocab}
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"
    model = opt = None
    try:
        model = GPT(cfg).cuda()
        model.set_attn_impl(impl)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
        idx = torch.randint(0, cfg.vocab, (batch, seq_len + 1), device="cuda")
        amp = precision == "bf16"

        def step():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(idx[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, cfg.vocab), idx[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return loss
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(steps):
            loss = step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / steps
        n = model.n_params()
        peak, peak_src = peak_flops()
        tps = batch * seq_len / dt
        attn_flops_tok = 12 * cfg.n_layers * seq_len * cfg.d_model  # PaLM appendix: 6N + 12*L*T*d
        row.update({"status": "ok", "step_ms": dt * 1e3, "tokens_per_s": tps, "params": n,
                    "train_flops_per_token_6N": calc.train_flops(n, 1), "attn_flops_per_token": attn_flops_tok,
                    "mfu_6N": calc.mfu(tps, n, peak), "mfu_with_attn": (6 * n + attn_flops_tok) * tps / peak,
                    "peak_flops": peak, "peak_source": peak_src,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "final_loss": loss.item()})
    except Exception as exc:  # noqa: BLE001
        row["status"] = "oom" if is_oom(exc) else "error"
        row["error"] = str(exc).splitlines()[0][:200]
    del model, opt
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    w.write(row)
    print(f"tput {sweep:<5} {precision:<5} {impl:<7} T={seq_len:<5} B={batch:<3} {row['status']:<5} "
          f"{row.get('tokens_per_s', 0) / 1e3:8.1f} ktok/s MFU={row.get('mfu_6N', 0):.3f} "
          f"mem={row.get('peak_mem_gb', 0):.1f}GB", flush=True)


def part_throughput(w: JsonlWriter, quick: bool) -> None:
    for precision in ("fp32", "tf32", "bf16"):
        for impl in ("naive", "sdpa", "triton"):
            if impl == "triton" and precision != "bf16":
                continue
            throughput(w, precision, impl, 1024, 8, "main", steps=3 if quick else 10)
    for t in (512, 1024, 2048, 4096, 8192):
        for impl in ("naive", "sdpa", "triton"):
            throughput(w, "bf16", impl, t, max(1, 16384 // t), "seqlen", steps=3 if quick else 8)


def part_recall(w: JsonlWriter, quick: bool) -> None:
    from tfkernels.recall import RecallTask, build_hand_coded
    for pairs in (4, 8, 16, 32):
        task = RecallTask(n_keys=max(16, pairs), n_values=16, n_pairs=pairs)
        for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 30.0):
            model = build_hand_coded(task, beta, beta)
            seq, target = task.sample(5000, np.random.default_rng(100 + pairs))
            v0 = task.value_ids().start
            with torch.no_grad():
                cpu_pred = model(seq)[:, -1, v0:].argmax(-1) + v0
                mg = model.cuda()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                gpu_logits = mg(seq.cuda())[:, -1, v0:]
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                gpu_pred = gpu_logits.argmax(-1).cpu() + v0
            w.write({"part": "recall_gpu", "n_pairs": pairs, "beta": beta, "n_test": 5000,
                     "acc_cpu": float((cpu_pred == target).float().mean()),
                     "acc_gpu": float((gpu_pred == target).float().mean()),
                     "pred_mismatch_cpu_vs_gpu": int((cpu_pred != gpu_pred).sum()), "gpu_ms": dt * 1e3})
        print(f"recall pairs={pairs} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["mqar", "mqar_sweep", "throughput", "recall"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(2)
    w = JsonlWriter(f"train_{a.part}.jsonl")
    w.write({"part": "meta", **env_info()})
    t0 = time.time()
    {"mqar": part_mqar, "mqar_sweep": part_mqar_sweep, "throughput": part_throughput,
     "recall": part_recall}[a.part](w, a.quick)
    w.write({"part": "meta_end", "elapsed_s": round(time.time() - t0, 1)})
    w.close()


if __name__ == "__main__":
    main()
