"""Single-GPU measurements for the serving-memory lab (Qwen2.5-0.5B, bf16, RTX 5090).

python gpu/serve_bench.py <cmd> [--quick] [--policy P]
  verify    own Qwen2 forward vs HF transformers (logits error, greedy agreement); paged kernel vs fp32 reference
  roofline  prefill / decode step time vs batch and context (eager and CUDA graph)
  paged     paged (Triton, shuffled blocks) vs contiguous-table vs torch gather vs padded SDPA, block 8..64
  waste     real GPU allocations: max-length / exact-length contiguous vs paged pool; allocator churn
  engine    continuous-batching engine vs static batching vs chunked prefill under Poisson load
Raw rows go to artifacts/raw/gpu/<cmd>[_<policy>].jsonl.
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

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB.parents[2]), str(HERE), str(LAB / "code"), str(LAB.parents[2] / "tools")]

import mlsys_calc  # noqa: E402
from mlsys_harness.paths import raw_dir  # noqa: E402
from servesim.block_manager import BlockManager  # noqa: E402
from servesim.waste import waste_table  # noqa: E402
from servesim.workload import MAX_MODEL_LEN, poisson_arrivals, sample_lengths  # noqa: E402

RAW = raw_dir(LAB)


class Sink:
    def __init__(self, name: str):
        RAW.mkdir(parents=True, exist_ok=True)
        self.f = open(RAW / f"{name}.jsonl", "w", encoding="utf-8")

    def __call__(self, echo: bool = True, **row) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        if echo:
            print(json.dumps(row)[:400], flush=True)


def env_row() -> dict:
    p = torch.cuda.get_device_properties(0)
    import triton
    return {"kind": "env", "gpu": p.name, "sms": p.multi_processor_count, "vram_bytes": p.total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
            "python": platform.python_version()}


def cuda_time(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / 1e3)
    return statistics.median(ts)


def load_model():
    from qwen import Qwen2, find_checkpoint
    return Qwen2(find_checkpoint())


# --------------------------------------------------------------------------- verify

def cmd_verify(sink: Sink, quick: bool, _a) -> None:
    from paged_attn import build_paged, gather_decode, paged_decode, reference_decode
    from qwen import ContiguousCache, find_checkpoint
    sink(**env_row())
    m = load_model()
    sink(kind="model", params=m.num_params(), weight_bytes=m.weight_bytes(), kv_bytes_per_token=m.kv_bytes_per_token(),
         layers=m.L, d=m.d, H=m.H, Hk=m.Hk, D=m.D, ff=m.ff, calc_params=mlsys_calc.MODELS["qwen2.5-0.5b"].params())
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(str(find_checkpoint()), torch_dtype=torch.bfloat16).cuda().eval()
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, 32000, (2, 48), generator=g).cuda()
    with torch.no_grad():
        ref = hf(prompt).logits.float()
        cache = ContiguousCache(m, 2, 128)
        mine = m.forward(prompt.reshape(-1), torch.arange(48, device="cuda").repeat(2), cache.prefill_attn(48)).float()
        mine = mine.view(2, 48, -1)
        err = (mine - ref).abs()
        sink(kind="verify_logits", max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
             logit_scale=float(ref.abs().mean()), argmax_agree=float((mine.argmax(-1) == ref.argmax(-1)).float().mean()))
        # greedy decode 32 tokens: own engine-style decode with contiguous cache vs HF generate
        cache = ContiguousCache(m, 2, 128)
        cache.len = 0
        lg = m.forward(prompt.reshape(-1), torch.arange(48, device="cuda").repeat(2), cache.prefill_attn(48))
        nxt = lg.view(2, 48, -1)[:, -1].argmax(-1)
        cache.len = 48
        toks = [nxt]
        for s in range(31):
            lg = m.forward(nxt, torch.full((2,), 48 + s, device="cuda"), cache.decode_attn(48 + s))
            nxt = lg.argmax(-1)
            toks.append(nxt)
        mine_gen = torch.stack(toks, 1)
        hf_gen = hf.generate(prompt, max_new_tokens=32, do_sample=False, min_new_tokens=32)[:, 48:]
        sink(kind="verify_greedy", tokens=32, agree_frac=float((mine_gen == hf_gen).float().mean()),
             first_mismatch=[int((mine_gen[b] != hf_gen[b]).nonzero()[0]) if (mine_gen[b] != hf_gen[b]).any() else -1
                             for b in range(2)])
    del hf
    torch.cuda.empty_cache()
    # fp32 on both sides separates implementation bugs from bf16 rounding
    from qwen import Qwen2
    m32 = Qwen2(find_checkpoint(), dtype=torch.float32)
    hf32 = AutoModelForCausalLM.from_pretrained(str(find_checkpoint()), torch_dtype=torch.float32).cuda().eval()
    with torch.no_grad():
        ref = hf32(prompt).logits
        cache = ContiguousCache(m32, 2, 128)
        mine = m32.forward(prompt.reshape(-1), torch.arange(48, device="cuda").repeat(2), cache.prefill_attn(48)).view(2, 48, -1)
        err = (mine - ref).abs()
        sink(kind="verify_logits_fp32", max_abs_err=float(err.max()), mean_abs_err=float(err.mean()),
             logit_scale=float(ref.abs().mean()), argmax_agree=float((mine.argmax(-1) == ref.argmax(-1)).float().mean()))
    del m32, hf32
    torch.cuda.empty_cache()
    # paged kernel correctness
    lens = sample_lengths(64, seed=1)[0].tolist()
    for bs in (8, 16, 32, 64):
        for (H, Hk, D) in ((14, 2, 64), (32, 8, 128)):
            kp, vp, bt, ln = build_paged(lens[:16], bs, Hk, D, seed=bs)
            q = torch.randn(16, H, D, device="cuda", dtype=torch.bfloat16)
            r = reference_decode(q, kp, vp, bt, ln)
            o1 = paged_decode(q, kp, vp, bt, ln).float()
            o2 = gather_decode(q, kp, vp, bt, ln).float()
            sink(kind="verify_paged", block=bs, H=H, Hk=Hk, D=D, triton_max_err=float((o1 - r).abs().max()),
                 gather_max_err=float((o2 - r).abs().max()), bf16_eps=2**-8)


# --------------------------------------------------------------------------- roofline

def cmd_roofline(sink: Sink, quick: bool, _a) -> None:
    from qwen import ContiguousCache
    sink(**env_row())
    m = load_model()
    W = m.weight_bytes()
    kvt = m.kv_bytes_per_token()
    sink(kind="model", params=m.num_params(), weight_bytes=W, kv_bytes_per_token=kvt)
    # bandwidth / compute probes
    x = torch.empty(2**28 // 2, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    t = cuda_time(lambda: y.copy_(x))
    sink(kind="probe", what="d2d_copy_512MiB", gbps=2 * x.numel() * 2 / t / 1e9)
    del x, y
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t = cuda_time(lambda: a @ a, iters=10)
    sink(kind="probe", what="gemm_8192_bf16", tflops=2 * 8192**3 / t / 1e12)
    del a
    torch.cuda.empty_cache()
    with torch.no_grad():
        # prefill: one sequence of s tokens, and batched
        for B, s in ([(1, 512), (4, 512)] if quick else
                     [(1, s) for s in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)] + [(8, 512), (16, 512), (4, 2048)]):
            cache = ContiguousCache(m, B, s)
            ids = torch.randint(0, 32000, (B * s,), device="cuda")
            pos = torch.arange(s, device="cuda").repeat(B)
            fn = lambda: m.forward(ids, pos, cache.prefill_attn(s), logits_idx=torch.arange(s - 1, B * s, s, device="cuda"))  # noqa: E731
            t = cuda_time(fn, iters=5 if quick else 10)
            tok = B * s
            flops = 2 * m.num_params() * tok + 4 * m.L * m.d * B * s * (s + 1) / 2
            sink(kind="prefill", batch=B, seq=s, time_s=t, tokens_per_s=tok / t, tflops=flops / t / 1e12)
            del cache
            torch.cuda.empty_cache()
        # decode: B sequences each with ctx tokens; eager and CUDA graph
        Bs = [1, 8, 64] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        ctxs = [512] if quick else [128, 512, 2048, 8192]
        for ctx in ctxs:
            for B in Bs:
                if B * (ctx + 1) * kvt > 12e9:
                    continue
                cache = ContiguousCache(m, B, ctx + 1)
                ids = torch.randint(0, 32000, (B,), device="cuda")
                pos = torch.full((B,), ctx, device="cuda")
                attn = cache.decode_attn(ctx)
                fn = lambda: m.forward(ids, pos, attn)  # noqa: E731
                t_eager = cuda_time(fn, iters=10)
                # wall-clock eager (includes Python / launch overhead not hidden by async queueing)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    fn()
                    torch.cuda.synchronize()
                t_wall = (time.perf_counter() - t0) / 10
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(2):
                        fn()
                torch.cuda.current_stream().wait_stream(s)
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    fn()
                t_graph = cuda_time(gph.replay, iters=20)
                kv_bytes = B * (ctx + 1) * kvt
                flops = 2 * m.num_params() * B + 4 * m.L * m.d * B * (ctx + 1)
                sink(kind="decode", batch=B, ctx=ctx, t_eager_s=t_eager, t_wall_s=t_wall, t_graph_s=t_graph,
                     tok_s_graph=B / t_graph, tok_s_wall=B / t_wall, weight_bytes=W, kv_bytes=kv_bytes,
                     flops=flops, achieved_gbps_graph=(W + kv_bytes) / t_graph / 1e9,
                     intensity=flops / (W + kv_bytes))
                del cache, gph
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- paged

def cmd_paged(sink: Sink, quick: bool, _a) -> None:
    from paged_attn import build_paged, gather_decode, padded_decode, paged_decode
    sink(**env_row())
    p, o = sample_lengths(4096, seed=0)
    rng = np.random.default_rng(0)
    ctx_all = (p + rng.integers(0, np.maximum(o, 1))).astype(int)  # live length at a random decode step
    shapes = [(14, 2, 64)] if quick else [(14, 2, 64), (32, 8, 128)]
    for (H, Hk, D) in shapes:
        for B in ([64] if quick else [16, 64, 256]):
            lens = ctx_all[:B].tolist()
            Lmax = MAX_MODEL_LEN
            q = torch.randn(B, H, D, device="cuda", dtype=torch.bfloat16)
            kv_read = sum(lens) * Hk * D * 2 * 2
            # contiguous max-length preallocation [B,Hk,4096,D]
            kc = torch.randn(B, Hk, Lmax, D, device="cuda", dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            ln = torch.tensor(lens, dtype=torch.int32, device="cuda")
            t_pad = cuda_time(lambda: padded_decode(q, kc, vc, ln))
            kc2, vc2 = kc[:, :, : max(lens)].contiguous(), vc[:, :, : max(lens)].contiguous()
            t_padmax = cuda_time(lambda: padded_decode(q, kc2, vc2, ln))
            sink(kind="paged_time", H=H, Hk=Hk, D=D, batch=B, impl="sdpa_padded_4096", block=0, time_s=t_pad,
                 kv_read_bytes=kv_read, eff_gbps=kv_read / t_pad / 1e9, alloc_bytes=kc.numel() * 2 * 2)
            sink(kind="paged_time", H=H, Hk=Hk, D=D, batch=B, impl="sdpa_padded_batchmax", block=0, time_s=t_padmax,
                 kv_read_bytes=kv_read, eff_gbps=kv_read / t_padmax / 1e9, alloc_bytes=kc2.numel() * 2 * 2)
            del kc, vc, kc2, vc2
            for bs in (8, 16, 32, 64):
                for shuffle in (True, False):
                    kp, vp, bt, ln = build_paged(lens, bs, Hk, D, shuffle=shuffle, seed=bs)
                    t_tr = cuda_time(lambda: paged_decode(q, kp, vp, bt, ln))
                    sink(kind="paged_time", H=H, Hk=Hk, D=D, batch=B, block=bs,
                         impl="triton_paged" if shuffle else "triton_contig_table", time_s=t_tr,
                         kv_read_bytes=kv_read, eff_gbps=kv_read / t_tr / 1e9, alloc_bytes=kp.numel() * 2 * 2)
                    if shuffle:
                        for tpi in (16, 32, 128):
                            if tpi >= bs:
                                t2 = cuda_time(lambda: paged_decode(q, kp, vp, bt, ln, tokens_per_iter=tpi))
                                sink(kind="paged_tile", H=H, Hk=Hk, D=D, batch=B, block=bs, tokens_per_iter=tpi,
                                     time_s=t2, eff_gbps=kv_read / t2 / 1e9)
                        t_g = cuda_time(lambda: gather_decode(q, kp, vp, bt, ln))
                        sink(kind="paged_time", H=H, Hk=Hk, D=D, batch=B, block=bs, impl="torch_gather",
                             time_s=t_g, kv_read_bytes=kv_read, eff_gbps=kv_read / t_g / 1e9,
                             alloc_bytes=kp.numel() * 2 * 2)
                    del kp, vp, bt
                    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- waste

def cmd_waste(sink: Sink, quick: bool, _a) -> None:
    sink(**env_row())
    from qwen import find_checkpoint
    c = json.loads((find_checkpoint() / "config.json").read_text())
    kvt = 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * (c["hidden_size"] // c["num_attention_heads"]) * 2
    budget = 4 * 2**30  # 4 GiB of KV
    budget_tok = budget // kvt
    n = 2000 if quick else 10000
    p, o = sample_lengths(n, seed=0)
    for r in waste_table(p, o, budget_tok):
        sink(kind="waste_model", kv_bytes_per_token=kvt, budget_tokens=int(budget_tok), **r)
    rng = np.random.default_rng(0)
    # steady-state snapshot: a request is resident for o steps, so draw requests with probability
    # proportional to o (size-biased, as in the lifetime average of waste.py), then a uniform step
    pick = rng.choice(n, size=n, p=o / o.sum())
    p, o = p[pick], o[pick]
    live = (p + rng.integers(0, np.maximum(o, 1))).astype(int)  # live length at a random decode step
    final = (p + o).astype(int)
    # (a) contiguous max-length reservation: one tensor per sequence until the budget is used
    for scheme, sizes in (("contiguous_max4096", np.full(n, MAX_MODEL_LEN)), ("contiguous_exact_final_len", final)):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        held, used, k = [], 0, 0
        while k < n and torch.cuda.memory_allocated() - base + int(sizes[k]) * kvt <= budget:
            held.append(torch.empty(int(sizes[k]) * kvt // 2, device="cuda", dtype=torch.bfloat16))
            used += int(live[k]) * kvt
            k += 1
        alloc = torch.cuda.memory_allocated() - base
        sink(kind="waste_measured", scheme=scheme, block=0, concurrent_seqs=k, alloc_bytes=alloc,
             live_bytes=used, waste_frac=1 - used / alloc, reserved_bytes=torch.cuda.memory_reserved())
        del held
    # (b) paged: one pool tensor of `budget` bytes, lab BlockManager hands out blocks
    for bs in (8, 16, 32, 64):
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        nblocks = budget // (bs * kvt)
        pool = torch.empty(nblocks * bs * kvt // 2, device="cuda", dtype=torch.bfloat16)
        mgr = BlockManager(int(nblocks), bs)
        k, used = 0, 0
        while k < n and mgr.can_allocate(int(live[k])):
            mgr.add(k)
            mgr.append_many(k, int(live[k]))
            used += int(live[k]) * kvt
            k += 1
        blocks_used = nblocks - mgr.gpu.num_free
        pool.view(int(nblocks), -1)[: int(blocks_used), 0] = 1  # touch the used blocks
        torch.cuda.synchronize()
        sink(kind="waste_measured", scheme="paged", block=bs, concurrent_seqs=k,
             alloc_bytes=int(blocks_used) * bs * kvt, pool_bytes=torch.cuda.memory_allocated() - base,
             live_bytes=used, waste_frac=1 - used / (int(blocks_used) * bs * kvt))
        del pool
    # (c) allocator churn: exact-length per-request tensors through PyTorch's caching allocator
    torch.cuda.empty_cache()
    slots = 64
    alloc0 = torch.cuda.memory_allocated()
    live_t: dict[int, tuple[torch.Tensor, int]] = {}
    arr = rng.permutation(n)
    clock = 0
    trace = []
    nxt = 0
    ends: list[tuple[int, int]] = []
    while nxt < (2000 if quick else 6000):
        while len(live_t) < slots:
            i = int(arr[nxt % n])
            live_t[nxt] = (torch.empty(int(final[i]) * kvt // 2, device="cuda", dtype=torch.bfloat16), clock + int(o[i]))
            ends.append((clock + int(o[i]), nxt))
            nxt += 1
        ends.sort()
        clock, rid = ends.pop(0)
        del live_t[rid]
        if rid % 50 == 0:
            a = torch.cuda.memory_allocated() - alloc0
            r = torch.cuda.memory_reserved()
            trace.append((nxt, a, r))
    st = torch.cuda.memory_stats()
    sink(kind="churn", slots=slots, requests=nxt, trace=trace,
         num_alloc_retries=st.get("num_alloc_retries", 0),
         inactive_split_bytes=st.get("inactive_split_bytes.all.current", 0),
         segments=st.get("segment.all.current", 0))
    live_t.clear()


# --------------------------------------------------------------------------- engine

def cmd_engine(sink: Sink, quick: bool, a) -> None:
    from engine import Engine, Req, summarize
    sink(**env_row())
    m = load_model()
    n = 60 if quick else a.n
    kv_bytes = 2 * 2**30
    rates = [float(x) for x in a.rates.split(",")]
    policy = a.policy
    # warm-up: compile Triton, cuBLAS heuristics
    eng = Engine(m, kv_bytes, policy="chunked")
    pw, ow = sample_lengths(24, seed=99)
    eng.run([Req(i, 0.0, int(x), int(min(y, 32))) for i, (x, y) in enumerate(zip(pw, ow))])
    del eng
    torch.cuda.empty_cache()
    p, o = sample_lengths(n, seed=0)
    if a.long_frac > 0:
        idx = np.random.default_rng(7).choice(n, int(round(a.long_frac * n)), replace=False)
        p = np.array(p).copy()
        p[idx] = a.long_prompt
        o = np.minimum(np.array(o), MAX_MODEL_LEN - a.long_prompt)
    for rate in rates:
        arr = poisson_arrivals(n, rate, seed=1)
        reqs = [Req(i, float(t), int(x), int(y)) for i, (t, x, y) in enumerate(zip(arr, p, o))]
        eng = Engine(m, kv_bytes, policy=policy)
        t0 = time.time()
        reqs, steps, peak_blocks = eng.run(reqs)
        wall = time.time() - t0
        s = summarize(reqs, steps, ttft_slo=1.0, tpot_slo=0.05)
        sink(kind="engine", policy=policy, rate=rate, n=n, long_frac=a.long_frac, long_prompt=a.long_prompt,
             kv_bytes=kv_bytes, num_blocks=eng.num_blocks,
             block=eng.bs, static_batch=eng.static_batch if policy == "static" else None,
             peak_blocks=peak_blocks, wall_s=wall, **s)
        sink(echo=False, kind="engine_requests", policy=policy, rate=rate,
             cols=["arrival", "prompt", "output", "ttft", "e2e", "tpot", "max_itl"],
             rows=[[r.arrival, r.prompt, r.output, r.first - r.arrival, r.finish - r.arrival,
                    (r.finish - r.first) / max(r.output - 1, 1),
                    float(np.max(np.diff(r.emits))) if len(r.emits) > 1 else 0.0] for r in reqs])
        sink(echo=False, kind="engine_steps", policy=policy, rate=rate,
             cols=["t", "dt", "n_dec", "n_prefill", "n_prefill_seqs", "ctx_sum", "blocks_used",
                   "prefill_pairs", "prefill_kv"],
             rows=[[st["t"], st["dt"], st["n_dec"], st["n_prefill"], st["n_prefill_seqs"], st["ctx_sum"],
                    st["blocks_used"], sum(n * c0 + n * (n + 1) / 2 for c0, n in st["prefill_ctx"]),
                    sum(c0 + n for c0, n in st["prefill_ctx"])] for st in steps])
        del eng
        torch.cuda.empty_cache()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "roofline", "paged", "waste", "engine"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--policy", default="continuous")
    ap.add_argument("--rates", default="4,8,16,32,48")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--long-frac", type=float, default=0.0, help="fraction of prompts replaced by --long-prompt")
    ap.add_argument("--long-prompt", type=int, default=3072)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    name = a.cmd + (f"_{a.policy}" + ("_long" if a.long_frac > 0 else "") if a.cmd == "engine" else "")
    sink = Sink(name)
    t0 = time.time()
    globals()[f"cmd_{a.cmd}"](sink, a.quick, a)
    sink(kind="done", cmd=a.cmd, wall_s=time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
