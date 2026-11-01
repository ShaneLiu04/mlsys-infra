"""Run the C++ asst1 experiments and store one JSON line per measurement.

Usage:
  python scripts/run_cpp_experiments.py --platform server --cores 0,1,2,... [--smt-offset 64]
      [--remote-cores 32,33,...] [--only prog1,prog5]

--cores        physical cores to pin to, in order (thread t -> cores[t % len])
--smt-offset   logical id of a core's SMT sibling = core + offset (omit if unknown)
--remote-cores cores on the other socket, for the NUMA experiments
Output: artifacts/raw/cpp/<platform>/<experiment>.jsonl, plus meta.json.
Timing experiments run one at a time on purpose: concurrent runs would contend for
the same cores and memory bandwidth and corrupt each other's numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
BIN = LAB / "cpp" / "bin"
EXE = ".exe" if os.name == "nt" else ""


def run(exe: str, args: list[str], out: Path, extra: dict | None = None) -> int:
    cmd = [str(BIN / (exe + EXE)), *args]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for line in p.stdout.splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                if extra:
                    rec.update(extra)
                f.write(json.dumps(rec) + "\n")
                n += 1
    status = "ok" if p.returncode == 0 else f"rc={p.returncode}"
    print(f"[{time.time() - t0:7.1f}s] {status:6} {exe} {' '.join(args)} -> {n} rows", flush=True)
    if p.returncode != 0:
        print(p.stderr[-2000:], flush=True)
    return p.returncode


def csv(xs) -> str:
    return ",".join(str(x) for x in xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--smt-offset", type=int, default=None)
    ap.add_argument("--remote-cores", default="")
    ap.add_argument("--only", default="")
    ap.add_argument("--max-threads", type=int, default=None)
    args = ap.parse_args()

    cores = [int(c) for c in args.cores.split(",")]
    remote = [int(c) for c in args.remote_cores.split(",")] if args.remote_cores else []
    P = args.max_threads or len(cores)
    only = set(args.only.split(",")) if args.only else None
    out = LAB / "artifacts" / "raw" / "cpp" / args.platform
    out.mkdir(parents=True, exist_ok=True)

    def want(name: str) -> bool:
        return only is None or name in only

    def fresh(name: str) -> Path:
        f = out / f"{name}.jsonl"
        f.unlink(missing_ok=True)
        return f

    meta = {"platform": args.platform, "cores": cores, "smt_offset": args.smt_offset, "remote_cores": remote,
            "max_threads": P, "uname": platform.platform(), "python": sys.version.split()[0],
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    threads = sorted({t for t in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16] if t <= P} | {P})
    pin = ["--cpus", csv(cores)]
    rc = 0

    if want("prog1"):
        f = fresh("prog1_threads")
        for view in (1, 2):
            rc |= run("prog1_mandelbrot_threads", ["--view", str(view), "--reps", "7", "--threads", csv(threads + [2 * P]), *pin], f)
        f = fresh("prog1_chunk")
        for view in (1, 2):
            for chunk in (1, 2, 4, 8, 16, 32, 1200 // P):
                rc |= run("prog1_mandelbrot_threads", ["--view", str(view), "--reps", "7", "--sched", "dynamic",
                                                       "--threads", str(P), "--chunk", str(chunk), *pin], f)
        f = fresh("prog1_rows")
        for view in (1, 2):
            rc |= run("prog1_mandelbrot_threads", ["--view", str(view), "--reps", "1", "--mode", "counts"], f)
        if args.smt_offset is not None:
            f = fresh("prog1_smt")
            k = P // 2
            sib = lambda cs: [c for x in cs for c in (x, x + args.smt_offset)]
            layouts = {
                f"{k}T_on_{k}cores": (k, cores[:k]),
                f"{k}T_on_{k // 2}cores_x2smt": (k, sib(cores[: k // 2])),
                f"{2 * k}T_on_{k}cores_x2smt": (2 * k, sib(cores[:k])),
                f"{2 * k}T_on_{2 * k}cores": (2 * k, cores[: 2 * k]),
            }
            for name, (T, cpus) in layouts.items():
                for view in (1, 2):
                    rc |= run("prog1_mandelbrot_threads", ["--view", str(view), "--reps", "7", "--sched", "dynamic",
                                                           "--threads", str(T), "--cpus", csv(cpus)], f, {"layout": name})
                    for isa in ("avx512", "avx512x2"):
                        rc |= run("prog3_mandelbrot_simd", ["--view", str(view), "--reps", "7", "--mode", "tasks", "--isa", isa,
                                                            "--threads", str(T), "--tasks", "300", "--cpus", csv(cpus)], f, {"layout": name})

    if want("prog2"):
        f = fresh("prog2_vecintrin")
        for w in (2, 4, 8, 16):
            rc |= run(f"prog2_vecintrin_w{w}", ["--n", "10000"], f)

    if want("prog3"):
        f = fresh("prog3_simd")
        for view in (1, 2):
            rc |= run("prog3_mandelbrot_simd", ["--view", str(view), "--reps", "7", "--mode", "simd"], f)
        f = fresh("prog3_tasks")
        for view in (1, 2):
            for isa in ("avx2", "avx512", "avx512x2"):
                rc |= run("prog3_mandelbrot_simd", ["--view", str(view), "--reps", "7", "--mode", "tasks", "--isa", isa,
                                                    "--threads", str(P), "--tasks", "1,2,4,8,16,32,64,128,300,600,1200", *pin], f)
            for T in threads:
                rc |= run("prog3_mandelbrot_simd", ["--view", str(view), "--reps", "7", "--mode", "tasks",
                                                    "--threads", str(T), "--tasks", "300", *pin], f, {"scaling": 1})
        # Is a single worker slower than the main thread because of where it is pinned?
        f = fresh("prog3_pin")
        for name, extra in {"cpu0": ["--cpus", "0"], "cpu_first_core": ["--cpus", str(cores[1])], "unpinned": []}.items():
            for view in (1, 2):
                rc |= run("prog3_mandelbrot_simd", ["--view", str(view), "--reps", "9", "--mode", "tasks",
                                                    "--threads", "1", "--tasks", "1", *extra], f, {"pin": name})

    if want("prog4"):
        f = fresh("prog4_sqrt")
        rc |= run("prog4_sqrt", ["--reps", "3", "--threads", csv(sorted({1, 2, 4, 8, P} & set(range(1, P + 1)))), *pin], f)
        f = fresh("prog4_iters")
        rc |= run("prog4_sqrt", ["--xsweep"], f)

    if want("prog5"):
        f = fresh("prog5_sizes")
        rc |= run("prog5_saxpy", ["--mode", "sizes", "--max_log2", "27", "--threads", csv(sorted({1, 4, P})), *pin], f)
        f = fresh("prog5_threads")
        rc |= run("prog5_saxpy", ["--mode", "one", "--n", str(1 << 26), "--threads", csv(threads), *pin], f)
        if remote:
            f = fresh("prog5_numa")
            k = min(P, len(remote))
            for name, cpus in {"local": cores[:k], "remote": remote[:k],
                               "split": cores[: k // 2] + remote[: k // 2]}.items():
                rc |= run("prog5_saxpy", ["--mode", "one", "--n", str(1 << 26), "--threads", str(k), "--cpus", csv(cpus)],
                          f, {"placement": name})

    if want("prog6"):
        f = fresh("prog6_kmeans")
        rc |= run("prog6_kmeans", ["--threads", csv(sorted({1, 2, 4, 8, P} & set(range(1, P + 1)))), *pin], f)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_file = out / "meta.json"
    old = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    old.setdefault("runs", []).append(meta)
    meta_file.write_text(json.dumps(old, indent=2))
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
