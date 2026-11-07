"""vLLM under the exact workload `serve_bench.py engine` gives the hand-written engine.

Same Qwen2.5-0.5B bf16 checkpoint, same length distribution and Poisson arrivals (same seeds), same
2 GiB KV budget, max 128 sequences and 8192 batched tokens, outputs forced to the sampled length.
Prompts are random token ids, so prefix caching has nothing to hit and is turned off.

  <vllm venv>/bin/python gpu/vllm_bench.py --rates 10,20,40 [--eager]

vLLM runs in its own venv; this file only needs numpy, vllm and the repo's servesim / mlsys_harness.
V1 always chunks long prefills, so the closest own-engine policy is `chunked`.
Rows go to artifacts/raw/<card>/engine_vllm_<mode>.jsonl in the engine schema.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path[:0] = [str(LAB.parents[2]), str(LAB / "code")]

from mlsys_harness.paths import raw_dir  # noqa: E402
from servesim.workload import MAX_MODEL_LEN, poisson_arrivals, sample_lengths  # noqa: E402

RAW = raw_dir(LAB)
KV_BYTES = 2 * 2**30


def checkpoint() -> str:
    hf = os.environ.get("HF_HOME", "/root/autodl-tmp/hf")
    hits = sorted(glob.glob(f"{hf}/hub/models--Qwen--Qwen2.5-0.5B/snapshots/*/config.json"))
    if not hits:
        raise SystemExit("Qwen2.5-0.5B not under HF_HOME/hub")
    return str(Path(hits[0]).parent)


def summarize(arr, prompt, out, first, finish, emits, ttft_slo=1.0, tpot_slo=0.05) -> dict:
    arr, first, finish = map(np.asarray, (arr, first, finish))
    out = np.asarray(out)
    span = finish.max() - arr.min()
    ttft = first - arr
    tpot = (finish - first) / np.maximum(out - 1, 1)
    e2e = finish - arr
    itl = np.concatenate([np.diff(e) for e in emits if len(e) > 1])
    ok = (ttft <= ttft_slo) & (tpot <= tpot_slo)
    return {"requests": int(len(arr)), "makespan_s": float(span),
            "throughput_tok_s": float(out.sum() / span), "throughput_req_s": float(len(arr) / span),
            "ttft_mean_s": float(ttft.mean()), "ttft_p50_s": float(np.median(ttft)),
            "ttft_p99_s": float(np.percentile(ttft, 99)),
            "tpot_mean_ms": float(tpot.mean() * 1e3), "tpot_p99_ms": float(np.percentile(tpot, 99) * 1e3),
            "itl_p50_ms": float(np.median(itl) * 1e3), "itl_p99_ms": float(np.percentile(itl, 99) * 1e3),
            "itl_max_ms": float(itl.max() * 1e3), "e2e_p50_s": float(np.median(e2e)),
            "e2e_p99_s": float(np.percentile(e2e, 99)),
            "slo_attain": float(ok.mean()), "goodput_req_s": float(ok.sum() / span)}


async def run_rate(engine, rate: float, n: int, p, o, vocab: int, tag: str):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    arr = poisson_arrivals(n, rate, seed=1)
    rng = np.random.default_rng(2)
    prompts = [rng.integers(100, vocab - 100, int(x)).tolist() for x in p]
    first = np.zeros(n)
    finish = np.zeros(n)
    emits: list[list[float]] = [[] for _ in range(n)]
    t0 = time.perf_counter()

    async def one(i: int):
        await asyncio.sleep(max(0.0, arr[i] - (time.perf_counter() - t0)))
        sp = SamplingParams(max_tokens=int(o[i]), min_tokens=int(o[i]), ignore_eos=True, temperature=0.0,
                            output_kind=RequestOutputKind.DELTA)
        got = 0
        async for res in engine.generate(TokensPrompt(prompt_token_ids=prompts[i]), sp, f"{tag}-{rate}-{i}"):
            now = time.perf_counter() - t0
            k = len(res.outputs[0].token_ids)
            if k and got == 0:
                first[i] = now
            emits[i].extend([now] * k)
            got += k
        finish[i] = time.perf_counter() - t0
        assert got == int(o[i]), (i, got, o[i])

    await asyncio.gather(*(one(i) for i in range(n)))
    return arr, first, finish, emits


async def main_async(a) -> None:
    from vllm import AsyncEngineArgs, __version__
    from vllm.v1.engine.async_llm import AsyncLLM

    mode = "eager" if a.eager else "cudagraph"
    args = AsyncEngineArgs(model=checkpoint(), dtype="bfloat16", max_model_len=MAX_MODEL_LEN, max_num_seqs=128,
                           max_num_batched_tokens=8192, enable_prefix_caching=False, enforce_eager=a.eager,
                           kv_cache_memory_bytes=KV_BYTES, seed=0, disable_log_stats=True)
    engine = AsyncLLM.from_engine_args(args)
    vocab = 151000
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"engine_vllm_{mode}.jsonl"
    import torch
    with open(path, "w", encoding="utf-8") as f:
        def sink(**row):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(json.dumps(row)[:400], flush=True)

        p0 = torch.cuda.get_device_properties(0)
        sink(kind="env", gpu=p0.name, sms=p0.multi_processor_count, vram_bytes=p0.total_memory,
             torch=torch.__version__, cuda=torch.version.cuda, vllm=__version__, mode=mode, kv_bytes=KV_BYTES)
        pw, ow = sample_lengths(24, seed=99)
        await run_rate(engine, 1000.0, 24, pw, np.minimum(ow, 32), vocab, "warm")
        p, o = sample_lengths(a.n, seed=0)
        for rate in [float(x) for x in a.rates.split(",")]:
            t = time.time()
            arr, first, finish, emits = await run_rate(engine, rate, a.n, p, o, vocab, mode)
            s = summarize(arr, p, o, first, finish, emits)
            sink(kind="engine", policy=f"vllm_{mode}", rate=rate, n=a.n, kv_bytes=KV_BYTES, wall_s=time.time() - t, **s)
            sink(kind="engine_requests", policy=f"vllm_{mode}", rate=rate,
                 cols=["arrival", "prompt", "output", "ttft", "e2e", "tpot", "max_itl"],
                 rows=[[float(arr[i]), int(p[i]), int(o[i]), float(first[i] - arr[i]), float(finish[i] - arr[i]),
                        float((finish[i] - first[i]) / max(int(o[i]) - 1, 1)),
                        float(np.max(np.diff(emits[i]))) if len(emits[i]) > 1 else 0.0] for i in range(a.n)])
    engine.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="10,20,40")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--eager", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
