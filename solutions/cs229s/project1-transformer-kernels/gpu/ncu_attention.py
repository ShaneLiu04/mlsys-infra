"""Measured DRAM traffic of one attention forward call with Nsight Compute.

Driver: python gpu/ncu_attention.py           -> runs `ncu` once per config, writes ncu_attention.jsonl
Child : python gpu/ncu_attention.py --one ... -> warm up (compile), then one profiled call between
        cudaProfilerStart/Stop (ncu --profile-from-start off), so only that call's kernels are counted.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys

import torch

from common import JsonlWriter

METRICS = ["dram__bytes_read.sum", "dram__bytes_write.sum", "lts__t_bytes.sum", "gpu__time_duration.sum"]


def child(a) -> None:
    from bench_attention import fwd_fn
    from flash_triton import flash_attn_fwd
    q, k, v = (torch.randn(a.b, a.h, a.t, a.d, device="cuda", dtype=torch.float16) for _ in range(3))
    causal = bool(a.causal)
    if a.method == "triton" and a.bm:
        fn = lambda: flash_attn_fwd(q, k, v, causal, block_m=a.bm, block_n=a.bn, num_stages=a.stages,  # noqa: E731
                                    num_warps=8 if a.bm * a.d >= 128 * 128 or a.bm >= 256 else 4)
    else:
        fn = fwd_fn(a.method, q, k, v, causal)
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def parse(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    totals: dict[str, float] = {}
    kernels = set()
    for r in csv.DictReader(io.StringIO("\n".join(lines))):
        name = r.get("Metric Name")
        if name not in METRICS:
            continue
        val = float(r["Metric Value"].replace(",", ""))
        totals[name] = totals.get(name, 0.0) + val
        kernels.add((r["ID"], r["Kernel Name"][:60]))
    totals["n_kernels"] = len(kernels)
    totals["kernels"] = sorted({kname for _, kname in kernels})
    return totals


def driver() -> None:
    w = JsonlWriter("ncu_attention.jsonl")
    b, h, d = 2, 16, 64
    configs = []
    for t in (1024, 2048, 4096, 8192, 16384):
        if t <= 8192:
            configs.append(dict(method="naive", t=t, bm=0, bn=0))
        configs.append(dict(method="sdpa_flash", t=t, bm=0, bn=0))
        for bm in (16, 32, 64, 128):
            configs.append(dict(method="triton", t=t, bm=bm, bn=64 if bm >= 64 else 32))
    for cfg in configs:
        cmd = ["ncu", "--profile-from-start", "off", "--csv", "--print-units", "base", "--metrics", ",".join(METRICS),
               sys.executable, __file__, "--one", "--method", cfg["method"], "--b", str(b), "--h", str(h),
               "--t", str(cfg["t"]), "--d", str(d), "--bm", str(cfg["bm"]), "--bn", str(cfg["bn"])]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        row = {"method": cfg["method"], "B": b, "H": h, "T": cfg["t"], "D": d, "causal": False,
               "BLOCK_M": cfg["bm"], "BLOCK_N": cfg["bn"], "returncode": p.returncode}
        if p.returncode == 0:
            row.update(parse(p.stdout))
            row["dram_bytes"] = row.get("dram__bytes_read.sum", 0) + row.get("dram__bytes_write.sum", 0)
        else:
            row["stderr"] = (p.stderr + p.stdout)[-400:]
        if "dram__bytes_read.sum" not in row:
            row["stderr"] = (p.stderr + p.stdout)[-600:]
        w.write(row)
        print(f"ncu {cfg['method']:<11} T={cfg['t']:<6} BM={cfg['bm']:<3} rc={p.returncode} "
              f"dram={row.get('dram_bytes', 0) / 1e9:.3f} GB L2={row.get('lts__t_bytes.sum', 0) / 1e9:.3f} GB "
              f"k={row.get('n_kernels')}", flush=True)
        if "ERR_NVGPUCTRPERM" in row.get("stderr", ""):
            print("no permission for GPU performance counters; stopping", flush=True)
            break
    w.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--method", default="triton")
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--h", type=int, default=16)
    ap.add_argument("--t", type=int, default=4096)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--bm", type=int, default=0)
    ap.add_argument("--bn", type=int, default=0)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--causal", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    child(args) if args.one else driver()
