"""Run the C++ asst2 experiments and store one JSON line per measurement.

Usage:
  python scripts/run_cpp_experiments.py --platform server12c --cores 0,1,2,... [--smt-offset 64]
      [--extra-cores 24,27,29,30] [--only tests,overhead]

--cores        physical cores to pin to (worker t -> cores[t % len]); the process is also
               restricted to these cores with taskset on Linux by the caller
--smt-offset   logical id of a core's SMT sibling = core + offset (for the SMT false-sharing run)
Output: artifacts/raw/cpp/<platform>/<experiment>.jsonl, plus meta.json. Every binary
repeats each measurement 3 times internally (--reps 3) and stores all repetitions.
Experiments run one at a time so they do not contend for the same cores.
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

COMPUTE_TESTS = "ping_pong_equal,ping_pong_unequal,recursive_fibonacci,mandelbrot_chunked"
SYSTEMS = "serial,spawn,spinning,sleeping,async_spinning,async_sleeping"


def run(exe: str, args: list[str], out: Path, extra: dict | None = None, timeout: int = 900) -> int:
    cmd = [str(BIN / (exe + EXE)), *args]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        stdout, rc, err = p.stdout, p.returncode, p.stderr
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = "timeout", ""
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for line in stdout.splitlines():
            if line.startswith("{"):
                rec = json.loads(line)
                if extra:
                    rec.update(extra)
                f.write(json.dumps(rec) + "\n")
                n += 1
    status = "ok" if rc == 0 else f"rc={rc}"
    print(f"[{time.time() - t0:7.1f}s] {status:6} {exe} {' '.join(args)} -> {n} rows", flush=True)
    if rc != 0:
        print(err[-2000:], flush=True)
    return 0 if rc == 0 else 1


def csv(xs) -> str:
    return ",".join(str(x) for x in xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True)
    ap.add_argument("--cores", required=True)
    ap.add_argument("--smt-offset", type=int, default=None)
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    cores = [int(c) for c in args.cores.split(",")]
    P = len(cores)
    only = set(args.only.split(",")) if args.only else None
    out = LAB / "artifacts" / "raw" / "cpp" / args.platform
    out.mkdir(parents=True, exist_ok=True)
    R = ["--reps", str(args.reps)]
    pin = ["--cpus", csv(cores)]
    threads_sweep = sorted({t for t in (1, 2, 4, 6, 8, 12, 16) if t <= P} | {P})

    def want(name: str) -> bool:
        return only is None or name in only

    def fresh(name: str) -> Path:
        f = out / f"{name}.jsonl"
        f.unlink(missing_ok=True)
        return f

    meta = {"platform": args.platform, "cores": cores, "smt_offset": args.smt_offset,
            "machine": platform.platform(), "processor": platform.processor(), "python": sys.version.split()[0],
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        meta["gxx"] = subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        pass
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                meta["cpu_model"] = line.split(":", 1)[1].strip()
                break
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    fails = 0
    B = "tasksys_bench"

    if want("tests"):  # (2) system x test matrix at P threads; also gives CPU time per test
        f = fresh("tests")
        for t in ["all"]:
            fails += run(B, ["--mode", "tests", "--systems", SYSTEMS, "--tests", t, "--threads", str(P), *pin, *R], f)

    if want("overhead"):  # (1) launch overhead vs tasks and threads (empty tasks)
        f = fresh("overhead")
        fails += run(B, ["--mode", "overhead", "--systems", SYSTEMS, "--threads", csv(threads_sweep),
                         "--tasks", "1,4,16,64,256,1024,4096,16384", "--launches", "2000", *pin, *R], f)

    if want("scaling"):  # (3) thread scaling for compute-heavy tests
        f = fresh("scaling")
        fails += run(B, ["--mode", "tests", "--systems", "spawn,spinning,sleeping,async_sleeping",
                         "--tests", COMPUTE_TESTS, "--threads", csv(threads_sweep), *pin, *R], f)
        fails += run(B, ["--mode", "tests", "--systems", "serial", "--tests", COMPUTE_TESTS, "--threads", "1", *pin, *R], f)

    if want("idle"):  # (4) idle CPU burn of a pool with no work
        f = fresh("idle")
        fails += run(B, ["--mode", "idle", "--systems", SYSTEMS, "--threads", csv(threads_sweep), "--idle_s", "1", *pin], f)

    if want("oversub"):  # (5) more workers than cores
        f = fresh("oversub")
        for mult in (1, 2, 4):
            fails += run(B, ["--mode", "tests", "--systems", "spinning,sleeping,async_spinning,async_sleeping",
                             "--tests", "math_operations_in_tight_for_loop,ping_pong_equal,recursive_fibonacci,spin_between_run_calls",
                             "--threads", str(P * mult), "--scale", "0.25", *pin, *R], f, {"cores": P, "mult": mult},
                         timeout=600)

    if want("granularity"):  # (6) work per task vs number of tasks at fixed total work
        f = fresh("granularity")
        fails += run(B, ["--mode", "granularity", "--systems", "spinning,sleeping,async_sleeping,spawn",
                         "--threads", str(P), "--total", "200000000",
                         "--tasks", "1,2,4,8,12,16,24,32,64,128,256,1024,4096,16384,65536,262144", *pin, *R], f)

    if want("dag"):  # (7) async DAG shapes; serial run gives T1
        f = fresh("dag")
        dag = ["--mode", "dag", "--launches", "64", "--tasks", str(P + 1), "--work", "200000"]
        fails += run(B, [*dag, "--systems", "async_sleeping,async_spinning,sleeping", "--threads", str(P), *pin, *R], f)
        fails += run(B, [*dag, "--systems", "serial", "--threads", "1", "--shape", "indep", *pin, *R], f)
        g = fresh("dag_tasks")  # sync-per-launch vs async as tasks-per-launch varies
        for n in (1, P // 2, P - 1, P, P + 1, 2 * P - 1, 2 * P + 1, 4 * P + 1):
            fails += run(B, ["--mode", "dag", "--launches", "64", "--tasks", str(n), "--work", "200000",
                             "--systems", "async_sleeping", "--threads", str(P), "--shape", "indep_sync,indep",
                             *pin, *R], g)

    if want("false_sharing"):  # (8) packed vs padded counters
        f = fresh("false_sharing")
        fails += run("contention", ["--mode", "false_sharing", "--threads", csv(threads_sweep),
                                    "--iters", "20000000", *pin, *R], f, {"placement": "cores"})
        if args.smt_offset is not None:
            smt = [cores[0], cores[0] + args.smt_offset]
            fails += run("contention", ["--mode", "false_sharing", "--threads", "2", "--iters", "20000000",
                                        "--cpus", csv(smt), *R], f, {"placement": "smt_siblings"})

    if want("dispatch"):  # (9) mutex vs atomic vs chunked vs static task counters
        f = fresh("dispatch")
        for w in (0, 50):
            fails += run("contention", ["--mode", "dispatch", "--threads", csv(threads_sweep), "--n", "10000000",
                                        "--work", str(w), "--chunks", "8,64", *pin, *R], f)

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("fails:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
