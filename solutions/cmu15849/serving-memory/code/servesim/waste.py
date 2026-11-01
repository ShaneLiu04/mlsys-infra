"""KV-cache waste: contiguous max-length reservation vs paged blocks.

Both metrics are averaged over each request's lifetime (every decode step), because the
concurrent batch at a fixed budget is set by memory held while the request is running.
"""

from __future__ import annotations

import numpy as np

from .workload import MAX_MODEL_LEN


def lifetime_tokens(prompt: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    """Mean over requests of (sum over steps of used tokens, number of steps)."""
    # during step j (0..o-1) the sequence holds p + j tokens
    steps = output.astype(np.float64)
    used = prompt * steps + steps * (steps - 1) / 2
    return float(used.sum()), float(steps.sum())


def allocated_paged(prompt: np.ndarray, output: np.ndarray, block: int) -> float:
    total = 0.0
    for p, o in zip(prompt.tolist(), output.tolist()):
        lens = np.arange(p, p + o)
        total += float((-(-lens // block) * block).sum())
    return total


def waste_table(prompt: np.ndarray, output: np.ndarray, budget_tokens: int,
                blocks: tuple[int, ...] = (8, 16, 32, 64)) -> list[dict]:
    used, steps = lifetime_tokens(prompt, output)
    mean_used = used / steps  # token-weighted mean live length
    rows = []
    reserved = MAX_MODEL_LEN * steps
    rows.append({"scheme": f"contiguous_max{MAX_MODEL_LEN}", "block": MAX_MODEL_LEN,
                 "waste_frac": 1 - used / reserved,
                 "mean_alloc_tokens_per_seq": reserved / steps,
                 "concurrent_seqs": budget_tokens // MAX_MODEL_LEN})
    final = prompt + output
    exact = float((prompt + output).astype(np.float64) @ output) / steps
    rows.append({"scheme": "contiguous_exact_final_len", "block": 0,
                 "waste_frac": 1 - mean_used / exact, "mean_alloc_tokens_per_seq": exact,
                 "concurrent_seqs": int(budget_tokens / exact)})
    for b in blocks:
        alloc = allocated_paged(prompt, output, b)
        rows.append({"scheme": "paged", "block": b, "waste_frac": 1 - used / alloc,
                     "mean_alloc_tokens_per_seq": alloc / steps,
                     "concurrent_seqs": int(budget_tokens / (alloc / steps))})
    rows.append({"scheme": "ideal_no_waste", "block": 1, "waste_frac": 0.0,
                 "mean_alloc_tokens_per_seq": mean_used,
                 "concurrent_seqs": int(budget_tokens / mean_used)})
    for r in rows:
        r["final_len_mean"] = float(final.mean())
    return rows
