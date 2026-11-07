"""GPU experiments for Project 2 (RTX 5090). Each subcommand appends rows to artifacts/raw/gpu/<cmd>.jsonl.

    python gpu/run_gpu.py hw | verify | latency | batch | memory | spec | spec_exact | spec_cost | dist
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpt2_fast import (MODEL_ROOT, GPT2, GraphStep, SpecEngine, StaticKV, generate_graph,  # noqa: E402
                       generate_kv, generate_nocache, load_state_dict)

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parents[2]))
from mlsys_harness.paths import raw_dir  # noqa: E402

OUT = raw_dir(LAB, "P2_OUT")
DATA = Path(os.environ.get("P2_DATA", "/root/autodl-tmp/p2data"))
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def emit(name: str, row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(name, row, flush=True)


def cuda_ms(fn, reps: int = 30, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained(os.path.join(MODEL_ROOT, "gpt2"))


_IDS = None


def corpus_ids() -> list[int]:
    global _IDS
    if _IDS is None:
        import pandas as pd
        text = "".join(pd.read_parquet(DATA / "wikitext2_test.parquet")["text"])
        _IDS = tokenizer()(text)["input_ids"]
    return _IDS


def prompts(n: int, P: int, seed: int = 0) -> list[list[int]]:
    ids = corpus_ids()
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.choice(len(ids) - P, n, replace=False))
    return [ids[s:s + P] for s in starts]


def first_diff(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


# --------------------------------------------------------------------------- hardware

def cmd_hw(_):
    x = torch.empty(512 * 1024 * 1024, dtype=torch.float32, device="cuda")  # 2 GiB
    y = torch.empty_like(x)
    ms = cuda_ms(lambda: y.copy_(x), reps=20)
    emit("hw", {"kind": "copy_bw", "bytes_moved": 2 * x.numel() * 4, "ms": ms,
                "GBps": 2 * x.numel() * 4 / ms / 1e6})
    del x, y
    for n in (4096, 8192):
        a = torch.randn(n, n, dtype=torch.float16, device="cuda")
        b = torch.randn(n, n, dtype=torch.float16, device="cuda")
        ms = cuda_ms(lambda: a @ b, reps=20)
        emit("hw", {"kind": "gemm_fp16", "n": n, "ms": ms, "TFLOPs": 2 * n ** 3 / ms / 1e9})
    p = torch.cuda.get_device_properties(0)
    emit("hw", {"kind": "device", "name": p.name, "sms": p.multi_processor_count,
                "mem_GB": p.total_memory / 1e9, "torch": torch.__version__, "cuda": torch.version.cuda})


# --------------------------------------------------------------------------- correctness

def cmd_verify(a):
    from transformers import GPT2LMHeadModel
    import transformers
    ps = prompts(5, 64, seed=1)
    for name in a.models:
        sd = load_state_dict(name)
        hf = GPT2LMHeadModel.from_pretrained(os.path.join(MODEL_ROOT, name), dtype=torch.float32,
                                             attn_implementation="eager").cuda().eval()
        m = GPT2(name, torch.float32, sd=sd)
        ids = torch.tensor(prompts(2, 128, seed=2), device="cuda")
        with torch.no_grad():
            ref = hf(ids).logits.float()
            mine = m.forward(ids)
            cache = StaticKV(m.cfg, 2, 128, torch.float32)
            parts = [m.forward(ids[:, :64], cache)]
            for t in range(64, 128):
                parts.append(m.forward(ids[:, t:t + 1], cache))
            kv = torch.cat(parts, 1)
        emit("verify", {"kind": "logits", "model": name, "transformers": transformers.__version__,
                        "nocache_vs_hf_maxabs": (mine - ref).abs().max().item(),
                        "kv_vs_hf_maxabs": (kv - ref).abs().max().item(),
                        "kv_vs_nocache_maxabs": (kv - mine).abs().max().item(),
                        "nocache_vs_hf_meanabs": (mine - ref).abs().mean().item(),
                        "rel_err": ((mine - ref).abs().max() / ref.abs().max()).item(),
                        "prob_maxabs": (mine.softmax(-1) - ref.softmax(-1)).abs().max().item(),
                        "logit_absmax": ref.abs().max().item(),
                        "argmax_agree": (mine.argmax(-1) == ref.argmax(-1)).float().mean().item()})
        hf_out = []
        with torch.no_grad():
            for p in ps:
                g = hf.generate(torch.tensor([p], device="cuda"), max_new_tokens=64, do_sample=False,
                                pad_token_id=50256)
                hf_out.append(g[0].tolist())
        del hf
        free()
        for dtype in (torch.float32, torch.float16):
            md = m if dtype == torch.float32 else GPT2(name, dtype, sd=sd)
            for i, p in enumerate(ps):
                x = torch.tensor([p], device="cuda")
                o_nc = generate_nocache(md, x, 64)[0].tolist()
                o_kv = generate_kv(md, x, 64)[0].tolist()
                o_gr = generate_graph(md, x, 64)[0].tolist()
                emit("verify", {"kind": "greedy", "model": name, "dtype": str(dtype).split(".")[-1],
                                "prompt": i, "n_new": 64,
                                "kv_eq_nocache": o_kv == o_nc, "graph_eq_nocache": o_gr == o_nc,
                                "hf_eq_nocache": hf_out[i] == o_nc,
                                "first_diff_kv": first_diff(o_kv[64:], o_nc[64:]),
                                "first_diff_graph": first_diff(o_gr[64:], o_nc[64:]),
                                "first_diff_hf": first_diff(hf_out[i][64:], o_nc[64:])})
            del md
            free()
        del m, sd
        free()
    tok = tokenizer()
    emit("verify", {"kind": "sample_prompt", "prompt": tok.decode(ps[0])})


# --------------------------------------------------------------------------- latency

def cmd_latency(a):
    ids = torch.tensor(prompts(1, 1024, seed=3), device="cuda")
    for name in a.models:
        m = GPT2(name, torch.float16)
        tok = ids[:, :1]
        cache = StaticKV(m.cfg, 1, 1024, torch.float16)
        full = GraphStep(m, cache, 1, t_attn=1024)
        for ctx in (16, 32, 64, 128, 256, 512, 768, 1024):
            nc = cuda_ms(lambda: m.forward(ids[:, :ctx], n_last=1))

            def kv_step():
                cache.length = ctx - 1
                m.forward(tok, cache)
            kv = cuda_ms(kv_step)
            gs = GraphStep(m, cache, 1, t_attn=ctx)
            gr = cuda_ms(lambda: gs(tok, ctx - 1))
            gf = cuda_ms(lambda: full(tok, ctx - 1))
            m.attn_impl = "bmm"
            gb = GraphStep(m, cache, 1, t_attn=ctx)
            m.attn_impl = "sdpa"
            grb = cuda_ms(lambda: gb(tok, ctx - 1))
            del gs, gb
            emit("latency", {"kind": "step", "model": name, "params": m.n_params(),
                             "weight_bytes": m.weight_bytes(), "ctx": ctx, "nocache_ms": nc,
                             "kv_eager_ms": kv, "kv_graph_ms": gr, "kv_graph_full1024_ms": gf,
                             "kv_graph_bmm_ms": grb})
        del full, cache
        free()
        for P, N in ((128, 128), (512, 128), (896, 128)):
            x = ids[:, :P]
            for variant in ("nocache", "kv_eager", "kv_graph"):
                if variant == "nocache" and P > 512:
                    continue
                cache = StaticKV(m.cfg, 1, P + N, torch.float16)
                gs = GraphStep(m, cache, 1) if variant == "kv_graph" else None
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if variant == "nocache":
                    nxt = m.forward(x, n_last=1)[:, -1].argmax(-1, keepdim=True)
                else:
                    cache.length = 0
                    nxt = m.forward(x, cache, n_last=1)[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                seq = torch.cat([x, nxt], 1)
                for i in range(N - 1):
                    if variant == "nocache":
                        nxt = m.forward(seq, n_last=1)[:, -1].argmax(-1, keepdim=True)
                        seq = torch.cat([seq, nxt], 1)
                    elif variant == "kv_eager":
                        nxt = m.forward(nxt, cache)[:, -1].argmax(-1, keepdim=True)
                    else:
                        nxt = gs(nxt, P + i)[:, -1].argmax(-1, keepdim=True)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                emit("latency", {"kind": "generate", "model": name, "variant": variant, "prompt": P,
                                 "n_new": N, "prefill_ms": (t1 - t0) * 1e3, "decode_ms": (t2 - t1) * 1e3,
                                 "decode_tok_s": (N - 1) / (t2 - t1)})
                del gs, cache
                free()
        del m
        free()


def cmd_batch(a):
    ctx = a.ctx
    for name in a.models:
        m = GPT2(name, torch.float16)
        kv_per_tok = 2 * m.cfg.n_layer * m.cfg.n_embd * 2
        for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            kvb = kv_per_tok * ctx * B
            if kvb + m.weight_bytes() > 26e9:
                emit("batch", {"model": name, "B": B, "ctx": ctx, "skipped": "oom", "kv_bytes": kvb})
                continue
            cache = StaticKV(m.cfg, B, ctx, torch.float16)
            tok = torch.zeros(B, 1, dtype=torch.long, device="cuda")
            gs = GraphStep(m, cache, 1, t_attn=ctx)
            gr = cuda_ms(lambda: gs(tok, ctx - 1), reps=20)

            def kv_step():
                cache.length = ctx - 1
                m.forward(tok, cache)
            kv = cuda_ms(kv_step, reps=20)
            del gs
            m.attn_impl = "bmm"
            gb = GraphStep(m, cache, 1, t_attn=ctx)
            m.attn_impl = "sdpa"
            grb = cuda_ms(lambda: gb(tok, ctx - 1), reps=20)
            emit("batch", {"model": name, "B": B, "ctx": ctx, "params": m.n_params(),
                           "weight_bytes": m.weight_bytes(), "kv_bytes": kvb, "graph_ms": gr,
                           "eager_ms": kv, "graph_bmm_ms": grb, "tok_s_graph": B / gr * 1e3,
                           "tok_s_eager": B / kv * 1e3, "tok_s_graph_bmm": B / grb * 1e3})
            del gb, cache
            free()
        del m
        free()


def cmd_memory(a):
    for name in a.models:
        m = GPT2(name, torch.float16)
        c = m.cfg
        for T in (128, 256, 512, 1024):
            for B in (1, 8, 32, 64):
                formula = 2 * c.n_layer * B * c.n_head * c.hd * T * 2
                if formula > 16e9:
                    continue
                free()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                cache = StaticKV(c, B, T, torch.float16)
                alloc = torch.cuda.memory_allocated() - base
                torch.cuda.reset_peak_memory_stats()
                tok = torch.zeros(B, 1, dtype=torch.long, device="cuda")
                cache.length = T - 1
                m.forward(tok, cache)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - base
                emit("memory", {"model": name, "T": T, "B": B, "formula_bytes": formula,
                                "allocated_bytes": alloc, "decode_peak_bytes": peak,
                                "reserved_bytes": torch.cuda.memory_reserved(),
                                "weight_bytes": m.weight_bytes()})
                del cache, tok
        del m
        free()


# --------------------------------------------------------------------------- speculative

def cmd_spec(a):
    target, draft = GPT2(a.target, torch.float16), GPT2(a.draft, torch.float16)
    ps = prompts(a.n_prompts, a.prompt_len, seed=4)
    tmax = a.prompt_len + a.n_new + 16
    base_eng = SpecEngine(target, draft, 1, tmax)
    tag = {"target": a.target, "draft": a.draft, "n_prompts": a.n_prompts, "prompt_len": a.prompt_len,
           "n_new": a.n_new}
    ref, margins, base_s = {}, {}, {}
    for T in a.temps:
        out = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i, p in enumerate(ps):
            gen = torch.Generator(device="cuda").manual_seed(1000 + i)
            o, _ = base_eng.target_only(p, a.n_new, T, gen)
            out.append(o)
        torch.cuda.synchronize()
        base_s[T] = time.perf_counter() - t0
        ref[T] = out
        emit("spec", {**tag, "kind": "baseline", "temperature": T, "wall_s": base_s[T],
                      "tok_s": a.n_prompts * a.n_new / base_s[T]})
    tok1 = torch.tensor([[ps[0][-1]]], device="cuda")
    for gamma in a.gammas:
        eng = SpecEngine(target, draft, gamma, tmax)
        P = a.prompt_len
        eng.prefill(ps[0])
        t1 = cuda_ms(lambda: eng.t_step[1](tok1, P), reps=20)
        tg = cuda_ms(lambda: eng.t_step[gamma + 1](tok1.repeat(1, gamma + 1), P), reps=20)
        d1 = cuda_ms(lambda: eng.d_step[1](tok1, P), reps=20)
        for T in a.temps:
            from gpt2_fast import SpecStats
            tot = SpecStats()
            same, diffs = 0, []
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i, p in enumerate(ps):
                gen = torch.Generator(device="cuda").manual_seed(2000 + i)
                o, st = eng.generate(p, a.n_new, T, gen)
                tot.add(st)
                if T == 0.0:
                    if o == ref[0.0][i]:
                        same += 1
                    else:
                        diffs.append({"prompt": i, "pos": first_diff(o[P:], ref[0.0][i][P:])})
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            for d in diffs:
                if d["prompt"] not in margins:
                    margins[d["prompt"]] = base_eng.target_only(ps[d["prompt"]], a.n_new,
                                                                record_margin=True)[1]
                d["target_margin"] = margins[d["prompt"]][d["pos"]]
            hist = np.bincount(tot.accepted_per_round, minlength=gamma + 1).tolist()
            emit("spec", {**tag, "kind": "spec", "gamma": gamma, "temperature": T, "wall_s": wall,
                          "tok_s": a.n_prompts * a.n_new / wall, "speedup": base_s[T] / wall,
                          "rounds": tot.rounds, "proposed": tot.proposed, "accepted": tot.accepted,
                          "rejections": tot.rejections, "alpha": tot.alpha,
                          "accept_fraction": tot.accepted / tot.proposed,
                          "alpha_overlap": tot.overlap_sum / max(tot.accepted + tot.rejections, 1),
                          "tokens_per_target_call": tot.tokens_per_target_call,
                          "accepted_hist": hist, "t_target1_ms": t1, "t_target_gamma1_ms": tg,
                          "t_draft1_ms": d1, "greedy_identical": same if T == 0.0 else None,
                          "greedy_diffs": diffs if T == 0.0 else None})
        del eng
        free()


def cmd_spec_exact(a):
    """fp32 + eager: greedy speculative output must equal target greedy exactly."""
    target, draft = GPT2(a.target, torch.float32), GPT2(a.draft, torch.float32)
    ps = prompts(a.n_prompts, a.prompt_len, seed=4)
    for use_graph in (False, True):
        for gamma in a.gammas:
            eng = SpecEngine(target, draft, gamma, a.prompt_len + a.n_new + 16, use_graph=use_graph)
            same, diffs = 0, []
            for i, p in enumerate(ps):
                ref, mg = eng.target_only(p, a.n_new, 0.0, record_margin=True)
                o, _ = eng.generate(p, a.n_new, 0.0)
                if o == ref:
                    same += 1
                else:
                    j = first_diff(o[a.prompt_len:], ref[a.prompt_len:])
                    diffs.append({"prompt": i, "pos": j, "target_margin": mg[j]})
            emit("spec_exact", {"target": a.target, "draft": a.draft, "dtype": "float32",
                                "graph": use_graph, "gamma": gamma, "n_prompts": a.n_prompts,
                                "n_new": a.n_new, "identical": same, "diffs": diffs})
            del eng
            free()


def cmd_spec_cost(a):
    """Batched cost of verification vs single-token decode -> where speculation stops paying."""
    draft = GPT2(a.draft, torch.float16)
    ctx = a.ctx
    for tname in a.targets:
        target = GPT2(tname, torch.float16)
        for B in (1, 2, 4, 8, 16, 32, 64, 128):
            tc = StaticKV(target.cfg, B, ctx + 16, torch.float16)
            dc = StaticKV(draft.cfg, B, ctx + 16, torch.float16)
            tok = torch.zeros(B, 1, dtype=torch.long, device="cuda")
            row = {"target": tname, "draft": a.draft, "B": B, "ctx": ctx}
            g = GraphStep(draft, dc, 1)
            row["t_draft1_ms"] = cuda_ms(lambda: g(tok, ctx), reps=20)
            del g
            for t in (1, 2, 3, 5, 9):
                g = GraphStep(target, tc, t)
                row[f"t_target{t}_ms"] = cuda_ms(lambda: g(tok.repeat(1, t), ctx), reps=20)
                del g
            emit("spec_cost", row)
            del tc, dc
            free()
        del target
        free()


def cmd_dist(a):
    """Sampling speculative decoding must reproduce the target distribution (fp32, fixed prompt)."""
    from scipy import stats
    target, draft = GPT2(a.target, torch.float32), GPT2(a.draft, torch.float32)
    P = a.prompt_len
    p = prompts(1, P, seed=5)[0]
    for T in a.temps:
        eng = SpecEngine(target, draft, a.gamma, P + a.n_new + 16)
        eng.prefill(p)
        x = torch.tensor([p], device="cuda")
        exact_p = torch.softmax(target.forward(x)[0, -1].double() / T, -1).cpu().numpy()
        exact_q = torch.softmax(draft.forward(x)[0, -1].double() / T, -1).cpu().numpy()
        gen = torch.Generator(device="cuda").manual_seed(7)
        spec, base = [], []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        acc = 0
        for _ in range(a.trials):
            o, st = eng.generate(p, a.n_new, T, gen, prefill=False)
            spec.append(o[P:])
            acc += st.accepted
        t_spec = time.perf_counter() - t0
        for _ in range(a.trials):
            o, _ = eng.target_only(p, a.n_new, T, gen, prefill=False)
            base.append(o[P:])
        spec, base = np.array(spec), np.array(base)
        V = exact_p.shape[0]
        res = {"target": a.target, "draft": a.draft, "temperature": T, "gamma": a.gamma,
               "trials": a.trials, "n_new": a.n_new, "prompt_len": P, "spec_wall_s": t_spec,
               "tv_exact_pq": 0.5 * float(np.abs(exact_p - exact_q).sum())}
        # token 1 against the exact target distribution (bins with expected >= 5, tail merged)
        for label, sample in (("spec", spec[:, 0]), ("base", base[:, 0])):
            cnt = np.bincount(sample, minlength=V)
            for ref_name, ref in (("p", exact_p), ("q", exact_q)):
                e = ref * a.trials
                keep = e >= 5
                obs = np.append(cnt[keep], cnt[~keep].sum())
                exp = np.append(e[keep], e[~keep].sum())
                if exp[-1] < 1e-9:
                    obs, exp = obs[:-1], exp[:-1]
                exp = exp * obs.sum() / exp.sum()
                chi = stats.chisquare(obs, exp)
                res[f"tok1_{label}_vs_{ref_name}_pvalue"] = float(chi.pvalue)
                res[f"tok1_{label}_vs_{ref_name}_tv"] = 0.5 * float(np.abs(cnt / a.trials - ref).sum())
        # later tokens: two-sample chi-square, spec vs target-only
        for j in range(a.n_new):
            cs, cb = np.bincount(spec[:, j], minlength=V), np.bincount(base[:, j], minlength=V)
            keep = (cs + cb) >= 10
            tab = np.array([np.append(cs[keep], cs[~keep].sum()), np.append(cb[keep], cb[~keep].sum())])
            tab = tab[:, tab.sum(0) > 0]
            chi = stats.chi2_contingency(tab)
            res[f"tok{j + 1}_two_sample_pvalue"] = float(chi.pvalue)
            res[f"tok{j + 1}_two_sample_tv"] = 0.5 * float(np.abs(cs - cb).sum() / a.trials)
        top = np.argsort(-exact_p)[:30]
        res["top_tokens"] = top.tolist()
        res["top_exact_p"] = exact_p[top].tolist()
        res["top_exact_q"] = exact_q[top].tolist()
        res["top_spec_freq"] = (np.bincount(spec[:, 0], minlength=V)[top] / a.trials).tolist()
        res["top_base_freq"] = (np.bincount(base[:, 0], minlength=V)[top] / a.trials).tolist()
        res["alpha_mean_accepted"] = acc / a.trials
        emit("dist", res)
        del eng
        free()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--target", default="gpt2-xl")
    ap.add_argument("--targets", nargs="+", default=["gpt2-large", "gpt2-xl"])
    ap.add_argument("--draft", default="gpt2")
    ap.add_argument("--gammas", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.7, 1.0])
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--prompt_len", type=int, default=64)
    ap.add_argument("--n_new", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--trials", type=int, default=4000)
    a = ap.parse_args()
    {"hw": cmd_hw, "verify": cmd_verify, "latency": cmd_latency, "batch": cmd_batch,
     "memory": cmd_memory, "spec": cmd_spec, "spec_exact": cmd_spec_exact,
     "spec_cost": cmd_spec_cost, "dist": cmd_dist}[a.cmd](a)


if __name__ == "__main__":
    main()
