"""Discrete-time serving simulator: static vs continuous (Orca) vs chunked prefill (Sarathi).

Time advances one engine iteration at a time. Iteration cost is a roofline bound from
``mlsys_calc``: weights are read once per iteration, every context token's KV is read once,
FLOPs = 2·N per processed token + 4·L·d per (query, context) pair. No kernel launch or
CPU overheads are modelled, so absolute latencies are optimistic lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import mlsys_calc

from .block_manager import BlockManager
from .workload import MAX_MODEL_LEN


@dataclass(frozen=True)
class CostModel:
    model: mlsys_calc.Model
    gpu: mlsys_calc.GPU
    bytes_per_el: int = 2

    @property
    def weight_bytes(self) -> float:
        return self.model.params() * self.bytes_per_el

    @property
    def kv_per_token(self) -> int:
        return mlsys_calc.kv_cache_bytes(self.model, 1, 1, self.bytes_per_el)

    def kv_budget_tokens(self, mem_util: float = 0.9) -> int:
        return int((mem_util * self.gpu.hbm_bytes - self.weight_bytes) // self.kv_per_token)

    def step_time(self, decode_ctx_sum: float, n_decode: int,
                  prefill: list[tuple[int, int]] = ()) -> float:
        """``prefill`` holds (already-cached context, new tokens) per prefilling sequence."""
        m = self.model
        n_tok = n_decode + sum(l for _, l in prefill)
        pairs = decode_ctx_sum + sum(l * c0 + l * (l + 1) / 2 for c0, l in prefill)
        flops = 2 * m.params() * n_tok + 4 * m.layers * m.d_model * pairs
        kv_tokens = decode_ctx_sum + sum(c0 + l for c0, l in prefill)
        nbytes = self.weight_bytes + self.kv_per_token * kv_tokens
        return mlsys_calc.roofline_time(flops, nbytes, self.gpu)[0]


@dataclass
class Request:
    rid: int
    arrival: float
    prompt: int
    output: int
    generated: int = 0
    prefilled: int = 0
    prefill_target: int = 0
    first_token: float = -1.0
    finish: float = -1.0
    preemptions: int = 0

    def __post_init__(self) -> None:
        self.prefill_target = self.prompt


@dataclass
class SimResult:
    requests: list[Request]
    steps: int
    prefill_tokens: int
    decode_tokens: int
    preemptions: int
    leaked_blocks: int = 0
    step_times: list[float] = field(default_factory=list)
    prefill_emits: int = 0  # tokens sampled at the end of a (re)prefill


def summarize(res: SimResult, ttft_slo: float, tpot_slo: float) -> dict:
    """Throughput, TTFT/TPOT statistics and goodput (requests/s meeting both SLOs)."""
    rq = res.requests
    t0 = min(r.arrival for r in rq)
    span = max(r.finish for r in rq) - t0
    ttft = np.array([r.first_token - r.arrival for r in rq])
    tpot = np.array([(r.finish - r.first_token) / max(r.output - 1, 1) for r in rq])
    ok = (ttft <= ttft_slo) & (tpot <= tpot_slo)
    return {"requests": len(rq), "makespan_s": span,
            "throughput_tok_s": sum(r.output for r in rq) / span,
            "throughput_req_s": len(rq) / span,
            "ttft_mean_s": float(ttft.mean()), "ttft_p99_s": float(np.percentile(ttft, 99)),
            "tpot_mean_ms": float(tpot.mean() * 1e3), "tpot_p99_ms": float(np.percentile(tpot, 99) * 1e3),
            "slo_attain": float(ok.mean()), "goodput_req_s": float(ok.sum() / span),
            "preemptions": res.preemptions, "steps": res.steps,
            "max_step_ms": float(max(res.step_times) * 1e3)}


def _make_requests(arrivals: np.ndarray, prompts: np.ndarray, outputs: np.ndarray) -> list[Request]:
    return [Request(i, float(a), int(p), int(o)) for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]


def simulate_static(cost: CostModel, arrivals, prompts, outputs, max_batch: int | None = None) -> SimResult:
    """Request-level batching: pad to the longest prompt, run until the longest output ends."""
    budget = cost.kv_budget_tokens()
    cap = budget // MAX_MODEL_LEN
    max_batch = min(max_batch or cap, cap)
    reqs = _make_requests(arrivals, prompts, outputs)
    t, i, steps, pf_tok, dec_tok = 0.0, 0, 0, 0, 0
    times = []
    while i < len(reqs):
        if reqs[i].arrival > t:
            t = reqs[i].arrival
        j = i
        while j < len(reqs) and j - i < max_batch and reqs[j].arrival <= t:
            j += 1
        batch = reqs[i:j]
        i = j
        b = len(batch)
        maxp = max(r.prompt for r in batch)
        maxo = max(r.output for r in batch)
        dt = cost.step_time(0, 0, [(0, maxp)] * b)
        t += dt
        times.append(dt)
        steps += 1
        pf_tok += sum(r.prompt for r in batch)
        for r in batch:
            r.prefilled, r.generated, r.first_token = r.prompt, 1, t
            if r.output == 1:
                r.finish = t
        for s in range(1, maxo):
            dt = cost.step_time(b * (maxp + s), b)
            t += dt
            times.append(dt)
            steps += 1
            for r in batch:
                if r.generated < r.output:
                    r.generated += 1
                    dec_tok += 1
                    if r.generated == r.output:
                        r.finish = t
    return SimResult(reqs, steps, pf_tok, dec_tok, 0, 0, times, len(reqs))


def simulate_continuous(cost: CostModel, arrivals, prompts, outputs, chunked: bool,
                        block_size: int = 16, max_num_seqs: int = 256,
                        max_batched_tokens: int = 8192, chunk_tokens: int = 512,
                        watermark_frac: float = 0.01, kv_tokens: int | None = None) -> SimResult:
    """Iteration-level scheduling with a paged KV cache and recompute preemption.

    Without chunking (Orca/vLLM-v0 style) newly admitted prompts are prefilled whole in the
    same iteration as running decodes, up to ``max_batched_tokens``. With chunking (Sarathi)
    each iteration has a token budget ``chunk_tokens``: decodes first, then prefill chunks.
    """
    kv_tokens = kv_tokens or cost.kv_budget_tokens()
    mgr = BlockManager(kv_tokens // block_size, block_size)
    watermark = int(watermark_frac * mgr.gpu.num_blocks)
    reqs = _make_requests(arrivals, prompts, outputs)
    waiting: list[Request] = []
    running: list[Request] = []  # admission order; newest last
    nxt, done, prefill_emits = 0, 0, 0
    t, steps, pf_tok, dec_tok, n_pre = 0.0, 0, 0, 0, 0
    times = []

    def preempt(victim: Request) -> None:
        nonlocal n_pre
        running.remove(victim)
        mgr.preempt_recompute(victim.rid)
        victim.prefill_target = victim.prompt + victim.generated
        victim.prefilled = 0
        victim.preemptions += 1
        n_pre += 1
        waiting.insert(0, victim)

    while done < len(reqs):
        while nxt < len(reqs) and reqs[nxt].arrival <= t:
            waiting.append(reqs[nxt])
            nxt += 1
        if not running and not waiting:
            t = reqs[nxt].arrival
            continue

        decodes = [r for r in running if r.prefilled == r.prefill_target]
        # reserve one slot per decode; preempt newest running request until it fits
        while True:
            need = sum(mgr.slots_needed_for_append(r.rid) for r in decodes)
            if need <= mgr.gpu.num_free:
                break
            victim = running[-1]
            preempt(victim)
            if victim in decodes:
                decodes.remove(victim)

        budget = (chunk_tokens - len(decodes)) if chunked else max_batched_tokens
        plan: list[tuple[Request, int]] = []
        if chunked:
            for r in running:
                if r.prefilled < r.prefill_target and budget > 0:
                    n = min(budget, r.prefill_target - r.prefilled)
                    plan.append((r, n))
                    budget -= n
        while waiting and budget > 0 and len(running) < max_num_seqs:
            r = waiting[0]
            if not mgr.can_allocate(r.prefill_target, watermark + need):
                break
            n = r.prefill_target if not chunked else min(budget, r.prefill_target)
            if not chunked and n > budget and plan:
                break
            waiting.pop(0)
            mgr.add(r.rid)
            mgr.append_many(r.rid, r.prefill_target)  # reserve the whole prompt up front
            running.append(r)
            plan.append((r, n))
            budget -= n
        if not decodes and not plan:
            if nxt < len(reqs):
                t = max(t, reqs[nxt].arrival)
                continue
            raise RuntimeError("scheduler stalled: request cannot fit in KV cache")

        # invariant while decoding: cached KV = prompt + generated - 1 (last token not yet fed)
        ctx_sum = sum(mgr.lengths[r.rid] + 1 for r in decodes)
        dt = cost.step_time(ctx_sum, len(decodes), [(r.prefilled, n) for r, n in plan])
        t += dt
        times.append(dt)
        steps += 1

        finished: list[Request] = []
        for r in decodes:
            mgr.append(r.rid)
            r.generated += 1
            dec_tok += 1
            if r.generated >= r.output:
                finished.append(r)
        for r, n in plan:
            r.prefilled += n
            pf_tok += n
            if r.prefilled == r.prefill_target:  # a completed (re)prefill samples one token
                r.generated += 1
                prefill_emits += 1
                if r.first_token < 0:
                    r.first_token = t
                if r.generated >= r.output:
                    finished.append(r)
        for r in finished:
            r.finish = t
            running.remove(r)
            mgr.free(r.rid)
            done += 1
    leaked = mgr.gpu.num_blocks - mgr.gpu.num_free
    return SimResult(reqs, steps, pf_tok, dec_tok, n_pre, leaked, times, prefill_emits)
