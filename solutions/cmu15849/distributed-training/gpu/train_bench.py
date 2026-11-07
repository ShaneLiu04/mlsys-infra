"""Single-GPU measurements for the distributed-training lab.

python gpu/train_bench.py <cmd> [--quick]
  hw        bf16 GEMM peak, D2D copy / reduction / pinned H2D-D2H bandwidth, one-GPU ring all-reduce
  memory    per-component training memory (weights/grads/optimizer/activations), fp32 / amp / bf16 / bf16+fp32-master
  actsweep  activation bytes vs batch x seq
  recompute torch.utils.checkpoint modes: memory saved, time overhead, max batch
  zero      per-rank memory of ZeRO stages from real sharded flat tensors + sharded AdamW correctness/time
  pipeline  4-stage GPT executed with GPipe / 1F1B order, CUDA-event timings -> measured Gantt
Each command appends JSON lines to artifacts/raw/gpu/<cmd>.jsonl (file is rewritten per run).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB.parents[2]), str(HERE), str(LAB / "code"), str(LAB.parents[2] / "tools")]

import mlsys_calc  # noqa: E402
from gpt import CONFIGS, GPT, Stage, activation_bytes_model, flops_per_token, init_seed, logits_bytes  # noqa: E402
from mlsys_harness.devices import get  # noqa: E402
from mlsys_harness.paths import raw_dir  # noqa: E402

RAW = raw_dir(LAB)
MiB = 2**20
_D5090 = get("rtx5090")
NOMINAL_5090_BF16 = _D5090.spec_bf16_flops  # dense bf16 tensor, fp32 accumulate
NOMINAL_5090_BW = _D5090.spec_hbm_bw


class Sink:
    def __init__(self, name: str):
        RAW.mkdir(parents=True, exist_ok=True)
        self.f = open(RAW / f"{name}.jsonl", "w", encoding="utf-8")

    def __call__(self, **row) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        print(json.dumps(row), flush=True)


def alloc() -> int:
    return torch.cuda.memory_allocated()


def sync() -> None:
    torch.cuda.synchronize()


def env_row() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"kind": "env", "gpu": p.name, "sms": p.multi_processor_count, "vram_bytes": p.total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version()}


def cuda_time(fn, iters: int = 20, warmup: int = 3) -> float:
    """Median seconds per call measured with CUDA events."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


# --------------------------------------------------------------------------- hw

def cmd_hw(sink: Sink, quick: bool) -> None:
    sink(**env_row())
    for n in ([4096] if quick else [2048, 4096, 8192, 16384]):
        a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
        t = cuda_time(lambda: a @ b, iters=10)
        sink(kind="gemm", n=n, dtype="bf16", time_s=t, tflops=2 * n**3 / t / 1e12)
        del a, b
    sizes = [2**k for k in range(12, 31, 2)] if not quick else [2**20, 2**26]
    for nbytes in sizes:
        n = nbytes // 4
        x = torch.empty(n, device="cuda")
        y = torch.empty(n, device="cuda")
        t = cuda_time(lambda: y.copy_(x), iters=20)
        sink(kind="d2d_copy", bytes=nbytes, time_s=t, gbps=2 * nbytes / t / 1e9)  # read + write
        t = cuda_time(lambda: y.add_(x), iters=20)
        sink(kind="reduce_add", bytes=nbytes, time_s=t, gbps=3 * nbytes / t / 1e9)  # 2 reads + 1 write
        h = torch.empty(n, pin_memory=True)
        t = cuda_time(lambda: x.copy_(h, non_blocking=True), iters=10)
        sink(kind="h2d_pinned", bytes=nbytes, time_s=t, gbps=nbytes / t / 1e9)
        t = cuda_time(lambda: h.copy_(x, non_blocking=True), iters=10)
        sink(kind="d2h_pinned", bytes=nbytes, time_s=t, gbps=nbytes / t / 1e9)
        del x, y, h
    # One-GPU ring all-reduce over P buffers: every "link" is an HBM copy + add, so this measures
    # the step-count (alpha) and HBM-side beta, not a real interconnect.
    for P in ([4] if quick else [2, 4, 8]):
        for nbytes in ([2**20, 2**26] if quick else [2**k for k in range(12, 29, 2)]):
            n = nbytes // 4 // P * P
            bufs = [torch.randn(n, device="cuda") for _ in range(P)]
            ref = torch.stack(bufs).sum(0)
            work = [b.clone() for b in bufs]

            def ring():
                for w, b in zip(work, bufs):
                    w.copy_(b)
                ch = [w.view(P, -1) for w in work]
                for s in range(P - 1):  # reduce-scatter
                    for r in range(P):
                        c = (r - s - 1) % P
                        ch[r][c].add_(ch[(r - 1) % P][c])
                for s in range(P - 1):  # all-gather
                    for r in range(P):
                        c = (r - s) % P
                        ch[r][c].copy_(ch[(r - 1) % P][c])

            ring()
            err = max(float((w - ref).abs().max()) for w in work)
            t_all = cuda_time(ring, iters=10)
            t_init = cuda_time(lambda: [w.copy_(b) for w, b in zip(work, bufs)], iters=10)
            t = t_all - t_init
            sink(kind="ring_1gpu", P=P, bytes=n * 4, time_s=t, steps=2 * (P - 1), launches=2 * (P - 1) * P,
                 max_err=err, model_bus_bytes=mlsys_calc.allreduce_bus_bytes(n * 4, P))
            del bufs, work, ref
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- training step helpers

def build(cfg_name: str, mode: str):
    """Returns model, optimizer, list of (model_param, master_param) for bf16_master."""
    cfg = CONFIGS[cfg_name]
    init_seed(0)
    model = GPT(cfg).cuda()
    if mode in ("bf16", "bf16_master"):
        model.to(torch.bfloat16)
    return model


def make_opt(model, mode: str):
    if mode == "bf16_master":
        master = [p.detach().float().clone() for p in model.parameters()]
        opt = torch.optim.AdamW(master, lr=1e-4, fused=True)
        return opt, list(zip(model.parameters(), master))
    return torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True), None


def train_step(model, opt, pairs, mode, x, y, rec: dict | None = None):
    m0 = alloc()
    if rec is not None:
        torch.cuda.reset_peak_memory_stats()
    if mode == "amp_bf16":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
    else:
        loss = model(x, y)
    if rec is not None:
        sync()
        rec["act"] = alloc() - m0
        rec["peak_fwd"] = torch.cuda.max_memory_allocated() - m0
    loss.backward()
    if rec is not None:
        sync()
        rec["grads"] = alloc() - m0
        rec["peak_bwd"] = torch.cuda.max_memory_allocated() - m0
    if pairs is not None:
        for p, mp in pairs:
            mp.grad = p.grad.float()
            p.grad = None
    m2 = alloc()
    opt.step()
    if pairs is not None:
        with torch.no_grad():
            for p, mp in pairs:
                p.copy_(mp)
                mp.grad = None
    if rec is not None:
        sync()
        rec["opt_step_delta"] = alloc() - m2
    opt.zero_grad(set_to_none=True)
    return loss


def batch(cfg, b, s):
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randint(0, cfg.vocab, (b, s), device="cuda", generator=g)
    return x, torch.roll(x, -1, 1)


def step_time(model, opt, pairs, mode, x, y, iters: int) -> float:
    for _ in range(2):
        train_step(model, opt, pairs, mode, x, y)
    sync()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        train_step(model, opt, pairs, mode, x, y)
        sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def free_all():
    import gc
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- memory

def cmd_memory(sink: Sink, quick: bool) -> None:
    sink(**env_row())
    combos = [("gpt2-124m", 8, 1024)] if quick else [("gpt2-124m", 8, 1024), ("gpt2-355m", 8, 1024), ("gpt2-774m", 4, 1024)]
    for cfg_name, b, s in combos:
        cfg = CONFIGS[cfg_name]
        for mode in ("fp32", "amp_bf16", "bf16", "bf16_master"):
            free_all()
            base = alloc()
            torch.cuda.reset_peak_memory_stats()
            model = build(cfg_name, mode)
            w = alloc() - base
            opt, pairs = make_opt(model, mode)
            master = alloc() - base - w
            x, y = batch(cfg, b, s)
            rec: dict = {}
            train_step(model, opt, pairs, mode, x, y, rec)
            static_after = alloc() - base
            opt_states = static_after - w  # master copy (if any) + exp_avg + exp_avg_sq
            torch.cuda.reset_peak_memory_stats()
            m_before = alloc()
            t = step_time(model, opt, pairs, mode, x, y, iters=5 if quick else 10)
            peak = torch.cuda.max_memory_allocated() - base
            n = model.num_params()
            calc = mlsys_calc.training_memory_bytes(n)
            act_bytes = 4 if mode == "fp32" else 2
            fpt = flops_per_token(cfg, s)
            tok_s = b * s / t
            sink(kind="memory", model=cfg_name, mode=mode, params=n, batch=b, seq=s,
                 weights=w, master_alloc_at_init=master, grads=rec["grads"], optimizer=opt_states,
                 activations=rec["act"], peak_fwd_over_static=rec["peak_fwd"], peak_bwd_over_static=rec["peak_bwd"],
                 peak_total=peak, static_before_step=m_before - base,
                 bytes_per_param_static=(w + rec["grads"] + opt_states) / n,
                 calc_16N=calc["total"], calc_weights=calc["weights"], calc_grads=calc["grads"],
                 calc_optimizer=calc["optimizer"],
                 act_model=activation_bytes_model(cfg, b, s, mode), logits_model=logits_bytes(cfg, b, s, act_bytes),
                 step_s=t, tokens_per_s=tok_s, flops_per_token=fpt,
                 mfu_nominal=tok_s * fpt / NOMINAL_5090_BF16)
            del model, opt, pairs, x, y
    free_all()


def cmd_actsweep(sink: Sink, quick: bool) -> None:
    sink(**env_row())
    models = ["gpt2-124m"] if quick else ["gpt2-124m", "gpt2-355m"]
    bs = [1, 4] if quick else [1, 2, 4, 8, 16]
    ss = [256, 1024] if quick else [128, 256, 512, 1024]
    for cfg_name in models:
        cfg = CONFIGS[cfg_name]
        for mode in ("amp_bf16", "bf16"):
            model = build(cfg_name, mode)
            opt, pairs = make_opt(model, mode)
            for b in bs:
                for s in ss:
                    free_all()
                    x, y = batch(cfg, b, s)
                    rec: dict = {}
                    try:
                        train_step(model, opt, pairs, mode, x, y, rec)
                        oom = False
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                        opt.zero_grad(set_to_none=True)
                    ab = 2
                    sink(kind="actsweep", model=cfg_name, mode=mode, batch=b, seq=s, tokens=b * s, oom=oom,
                         activations=rec.get("act"), peak_bwd_over_static=rec.get("peak_bwd"),
                         act_model=activation_bytes_model(cfg, b, s, mode),
                         logits_model=logits_bytes(cfg, b, s, ab),
                         embed_model=b * s * cfg.n_embd * (4 if mode == "amp_bf16" else 2))
                    del x, y
            del model, opt, pairs
    free_all()


# --------------------------------------------------------------------------- recompute

RECOMP_MODES = [("none", 1), ("full", 1), ("full", 2), ("mlp", 1), ("attn", 1)]


def cmd_recompute(sink: Sink, quick: bool) -> None:
    sink(**env_row())
    cfg_name = "gpt2-124m" if quick else "gpt2-774m"
    cfg = CONFIGS[cfg_name]
    s = 1024
    b0 = 2 if quick else 8
    mode = "amp_bf16"
    model = build(cfg_name, mode)
    opt, pairs = make_opt(model, mode)
    for ck, every in RECOMP_MODES:
        model.set_ckpt(ck, every)
        free_all()
        x, y = batch(cfg, b0, s)
        rec: dict = {}
        train_step(model, opt, pairs, mode, x, y, rec)
        static = alloc()
        torch.cuda.reset_peak_memory_stats()
        t = step_time(model, opt, pairs, mode, x, y, iters=3 if quick else 8)
        peak = torch.cuda.max_memory_allocated()
        tok_s = b0 * s / t
        sink(kind="recompute", model=cfg_name, ckpt=ck, every=every, batch=b0, seq=s,
             activations=rec["act"], peak_bwd_over_static=rec["peak_bwd"], static=static, peak_total=peak,
             step_s=t, tokens_per_s=tok_s, mfu_nominal=tok_s * flops_per_token(cfg, s) / NOMINAL_5090_BF16)
        del x, y
    # max batch that fits (geometric ladder; stop at first OOM)
    ladder = [2, 4, 8] if quick else [4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
    for ck, every in ([("none", 1), ("full", 1)] if quick else [("none", 1), ("full", 1), ("mlp", 1), ("full", 2)]):
        model.set_ckpt(ck, every)
        best = None
        for b in ladder:
            free_all()
            x, y = batch(cfg, b, s)
            try:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                train_step(model, opt, pairs, mode, x, y)
                sync()
                t0 = time.perf_counter()
                train_step(model, opt, pairs, mode, x, y)
                sync()
                dt = time.perf_counter() - t0
                best = (b, dt, torch.cuda.max_memory_allocated())
                sink(kind="maxbatch_try", model=cfg_name, ckpt=ck, every=every, batch=b, ok=True, step_s=dt,
                     peak_total=best[2], tokens_per_s=b * s / dt)
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                sink(kind="maxbatch_try", model=cfg_name, ckpt=ck, every=every, batch=b, ok=False)
                del x, y
                break
            del x, y
        sink(kind="maxbatch", model=cfg_name, ckpt=ck, every=every, seq=s, max_batch=best[0] if best else 0,
             step_s=best[1] if best else None, tokens_per_s=best[0] * s / best[1] if best else None)
    free_all()


# --------------------------------------------------------------------------- zero

def cmd_zero(sink: Sink, quick: bool) -> None:
    """Rank-0 memory of ZeRO-0..3 with Nd ranks, built from real flat tensors of a GPT-2 model."""
    sink(**env_row())
    cfg_name = "gpt2-124m" if quick else "gpt2-774m"
    init_seed(0)
    model = GPT(CONFIGS[cfg_name]).cuda().to(torch.bfloat16)
    flat_p = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    N = flat_p.numel()
    del model
    free_all()
    g = torch.Generator(device="cuda").manual_seed(2)
    flat_g = (torch.randn(N, device="cuda", generator=g) * 1e-3).to(torch.bfloat16)
    nds = [1, 4] if quick else [1, 2, 4, 8, 16, 64]
    for nd in nds:
        pad = (-N) % nd
        shard = (N + pad) // nd
        for stage in range(4):
            free_all()
            base = alloc()
            w = torch.empty(N if stage < 3 else shard, device="cuda", dtype=torch.bfloat16)
            m_w = alloc() - base
            gr = torch.empty(N if stage < 2 else shard, device="cuda", dtype=torch.bfloat16)
            m_g = alloc() - base - m_w
            n_opt = N if stage < 1 else shard
            master = torch.empty(n_opt, device="cuda")
            ea = torch.zeros(n_opt, device="cuda")
            es = torch.zeros(n_opt, device="cuda")
            m_o = alloc() - base - m_w - m_g
            calc = mlsys_calc.training_memory_bytes(N, dp=nd, zero=stage)
            sink(kind="zero_mem", model=cfg_name, params=N, nd=nd, stage=stage, weights=m_w, grads=m_g,
                 optimizer=m_o, total=m_w + m_g + m_o, calc_weights=calc["weights"], calc_grads=calc["grads"],
                 calc_optimizer=calc["optimizer"], calc_total=calc["total"], shard_elems=shard, pad=pad)
            del w, gr, master, ea, es
        # sharded AdamW: run each rank's shard step sequentially, compare with one full step
        if nd in (1, 4, 16) or quick:
            free_all()
            pp = F.pad(flat_p.float(), (0, pad))
            gg = F.pad(flat_g.float(), (0, pad))
            full = pp.clone()
            ref_opt = torch.optim.AdamW([full], lr=1e-3, fused=True)
            full.grad = gg
            ref_opt.step()
            ref_opt.step()
            ref = full.detach().clone()
            del ref_opt, full
            free_all()
            out = torch.empty_like(pp)
            times = []
            for r in range(nd):  # two steps per rank; the second (state already allocated) is timed
                sl = slice(r * shard, (r + 1) * shard)
                sp = pp[sl].clone()
                o = torch.optim.AdamW([sp], lr=1e-3, fused=True)
                sp.grad = gg[sl].clone()
                o.step()
                a, bb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                o.step()
                bb.record()
                bb.synchronize()
                times.append(a.elapsed_time(bb) / 1e3)
                out[sl] = sp.detach()
                del o, sp
            err = float((out - ref).abs().max())
            sink(kind="zero_adam", model=cfg_name, params=N, nd=nd, max_abs_err=err,
                 bitwise_equal=bool(torch.equal(out, ref)), rank_step_s_median=statistics.median(times),
                 bytes_touched_per_rank=16 * shard)
            del pp, gg, out, ref
    free_all()


# --------------------------------------------------------------------------- pipeline

def replay(orders, dur):
    """Event-driven start/end times from per-op durations (same dependency rules as disttrain.pipeline)."""
    p = len(orders)
    done: dict = {}
    free = [0.0] * p
    ptr = [0] * p
    ops = []
    remaining = sum(len(o) for o in orders)
    while remaining:
        progressed = False
        for d in range(p):
            while ptr[d] < len(orders[d]):
                kind, mb, _ = orders[d][ptr[d]]
                dep = (("F", d - 1, mb) if d > 0 else None) if kind == "F" else \
                    (("B", d + 1, mb) if d < p - 1 else ("F", d, mb))
                if dep is not None and dep not in done:
                    break
                start = max(free[d], done[dep] if dep else 0.0)
                end = start + dur[(kind, d, mb)]
                done[(kind, d, mb)] = end
                free[d] = end
                ops.append({"device": d, "kind": kind, "mb": mb, "start": start, "end": end})
                ptr[d] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("deadlock")
    return ops


def run_pipeline(stages, orders, xs, ys, record_mem: bool):
    """Execute ops in a serialisation consistent with the (unit-cost) parallel schedule."""
    p, m = len(stages), len(xs)
    unit = {(k, d, i): (1.0 if k == "F" else 2.0) for d in range(p) for (k, i, _) in orders[d]}
    serial = sorted(replay(orders, unit), key=lambda o: (o["start"], o["device"]))
    acts, grads = {}, {}
    ev = {}
    mem_trace = []
    act_bytes = {}
    for o in serial:
        k, d, i = o["kind"], o["device"], o["mb"]
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        m0 = alloc()
        a.record()
        if k == "F":
            inp = xs[i] if d == 0 else acts[(d - 1, i)][1].detach().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = stages[d](inp, ys[i] if d == p - 1 else None)
            if d == p - 1:
                out = out / m
            acts[(d, i)] = (inp, out)
        else:
            inp, out = acts.pop((d, i))
            if d == p - 1:
                out.backward()
            else:
                out.backward(grads.pop((d, i)))
            if d > 0:
                grads[(d - 1, i)] = inp.grad
            del inp, out
        b.record()
        ev[(k, d, i)] = (a, b)
        if record_mem:
            sync()
            if k == "F":
                act_bytes[(d, i)] = alloc() - m0
            mem_trace.append({"kind": k, "device": d, "mb": i, "allocated": alloc()})
    sync()
    dur = {key: a.elapsed_time(b) / 1e3 for key, (a, b) in ev.items()}
    return dur, mem_trace, act_bytes


def cmd_pipeline(sink: Sink, quick: bool) -> None:
    from disttrain.pipeline import schedule_orders
    sink(**env_row())
    cfg_name = "tiny" if quick else "gpt2-355m"
    cfg = CONFIGS[cfg_name]
    p, b, s = 4, (2 if quick else 4), (64 if quick else 512)
    init_seed(0)
    model = GPT(cfg).cuda()
    stages = [Stage(model, d, p) for d in range(p)]
    for sched in ("gpipe", "1f1b"):
        for m in ([1, 4] if quick else [1, 2, 4, 8, 16, 32]):
            free_all()
            g = torch.Generator(device="cuda").manual_seed(3)
            xs = [torch.randint(0, cfg.vocab, (b, s), device="cuda", generator=g) for _ in range(m)]
            ys = [torch.roll(x, -1, 1) for x in xs]
            orders = schedule_orders(sched, p, m)
            try:
                run_pipeline(stages, orders, xs, ys, record_mem=False)  # warm-up
                model.zero_grad(set_to_none=True)
                free_all()
                base = alloc()
                torch.cuda.reset_peak_memory_stats()
                dur, _, _ = run_pipeline(stages, orders, xs, ys, record_mem=False)
                peak = torch.cuda.max_memory_allocated() - base
                model.zero_grad(set_to_none=True)
                free_all()
                _, trace, act_b = run_pipeline(stages, orders, xs, ys, record_mem=True)
                model.zero_grad(set_to_none=True)
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                sink(kind="pipeline", model=cfg_name, schedule=sched, p=p, m=m, mb_batch=b, seq=s, oom=True)
                del xs, ys
                continue
            ops = replay(orders, dur)
            makespan = max(o["end"] for o in ops)
            busy = sum(o["end"] - o["start"] for o in ops)
            serial_time = sum(dur.values())
            # in-flight micro-batches per device: F start .. B end
            peaks = []
            for d in range(p):
                evs = sorted([(o["start"], 1) if o["kind"] == "F" else (o["end"], -1) for o in ops if o["device"] == d])
                cur = best = 0
                for _, dl in evs:
                    cur += dl
                    best = max(best, cur)
                peaks.append(best)
            stage_act = [statistics.median([v for (dd, _), v in act_b.items() if dd == d]) for d in range(p)]
            tf = [statistics.median([dur[("F", d, i)] for i in range(m)]) for d in range(p)]
            tb = [statistics.median([dur[("B", d, i)] for i in range(m)]) for d in range(p)]
            sink(kind="pipeline", model=cfg_name, schedule=sched, p=p, m=m, mb_batch=b, seq=s, oom=False,
                 makespan_s=makespan, busy_s=busy, bubble_measured=1 - busy / (p * makespan),
                 bubble_formula=mlsys_calc.pipeline_bubble(p, m), serial_s=serial_time,
                 peak_alloc_over_weights=peak, inflight_peak_per_stage=peaks,
                 stage_act_bytes=stage_act, inflight_act_bytes_per_stage=[pk * a for pk, a in zip(peaks, stage_act)],
                 tf_median_s=tf, tb_median_s=tb,
                 tokens_per_s_pipelined=m * b * s / makespan)
            if m in (4, 8) or quick:
                sink(kind="gantt", model=cfg_name, schedule=sched, p=p, m=m, ops=ops)
                sink(kind="memtrace", model=cfg_name, schedule=sched, p=p, m=m, trace=trace)
            del xs, ys
    free_all()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hw", "memory", "actsweep", "recompute", "zero", "pipeline"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None, help="raw file stem (default: cmd)")
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    sink = Sink(a.out or a.cmd)
    t0 = time.time()
    globals()[f"cmd_{a.cmd}"](sink, a.quick)
    sink(kind="done", cmd=a.cmd, wall_s=time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
