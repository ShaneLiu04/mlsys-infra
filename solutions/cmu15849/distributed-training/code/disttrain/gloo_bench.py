"""Real all-reduce over torch.distributed (gloo, CPU processes on one machine).

Run as a module so that spawned workers can import it:
    python -m disttrain.gloo_bench --world 4 --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world: int, init_file: str, sizes: list[int], iters: int, out: str) -> None:
    dist.init_process_group("gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    rows = []
    for nbytes in sizes:
        n = nbytes // 4
        t = torch.full((n,), float(rank + 1))
        dist.all_reduce(t)
        ok = bool(torch.all(t == world * (world + 1) / 2))
        times = []
        for _ in range(iters):
            t.fill_(1.0)
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(t)
            times.append(time.perf_counter() - t0)
        rows.append({"world": world, "bytes": nbytes, "median_s": statistics.median(times),
                     "min_s": min(times), "correct": ok})
    if rank == 0:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


def run(world: int, sizes: list[int], iters: int, out: str) -> list[dict]:
    init_file = os.path.join(tempfile.mkdtemp(), "pg").replace("\\", "/")
    mp.spawn(_worker, args=(world, init_file, sizes, iters, out), nprocs=world, join=True)
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=[2 ** k for k in range(10, 25, 2)])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.world, a.sizes, a.iters, a.out)


if __name__ == "__main__":
    main()
