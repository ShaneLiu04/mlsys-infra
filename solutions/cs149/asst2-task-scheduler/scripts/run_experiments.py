"""Run all asst2 experiments; writes artifacts/raw/*.csv and artifacts/figures/*.png."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from tasksys import ITaskSystem, make_system  # noqa: E402
from tasksys.workloads import NoopRunnable, NumpyRunnable, PythonLoopRunnable, SleepRunnable  # noqa: E402

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
SYSTEMS = ["serial", "spawn", "spinning", "spinning_yield", "sleeping"]
THREADS = 8


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def timed_launches(sys_: ITaskSystem, runnable, n_tasks: int, budget_s: float, max_launches: int) -> tuple[float, int]:
    """Mean seconds per synchronous ``run`` launch within a time budget (after one warm-up)."""
    sys_.run(runnable, n_tasks)
    k = 0
    t0 = time.perf_counter()
    while k < max_launches and (k < 3 or time.perf_counter() - t0 < budget_s):
        sys_.run(runnable, n_tasks)
        k += 1
    return (time.perf_counter() - t0) / k, k


def exp_launch_overhead() -> list[dict]:
    print("[overhead] empty tasks, 8 threads")
    rows: list[dict] = []
    for name in SYSTEMS:
        sys_ = make_system(name, THREADS)
        try:
            for n in (1, 8, 64, 512):
                per, k = timed_launches(sys_, NoopRunnable(), n, 0.4, 2000)
                rows.append({"system": name, "threads": THREADS, "tasks_per_launch": n, "launches": k,
                             "us_per_launch": round(per * 1e6, 2), "us_per_task": round(per / n * 1e6, 3)})
                print(f"  {name:15s} n={n:4d} {per * 1e6:10.1f} us/launch ({k} launches)")
        finally:
            sys_.close()
    write_csv("launch_overhead.csv", rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in SYSTEMS:
        rs = [r for r in rows if r["system"] == name]
        ax.plot([r["tasks_per_launch"] for r in rs], [r["us_per_launch"] for r in rs], "o-", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("empty tasks per launch")
    ax.set_ylabel("µs per run() call")
    ax.set_title("Launch overhead, 8 threads (CPython)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "launch_overhead.png", dpi=120)
    plt.close(fig)
    return rows


def exp_throughput() -> None:
    print("[throughput] work types vs threads")
    n_tasks = 128
    works = {
        "numpy_ufunc": NumpyRunnable(n_tasks, elems=1 << 15, reps=4),
        "python_loop": PythonLoopRunnable(20_000),
        "sleep_1ms": SleepRunnable(0.001),
    }
    rows: list[dict] = []
    serial_t: dict[str, float] = {}
    for wname, r in works.items():
        sys_ = make_system("serial", 1)
        serial_t[wname], _ = timed_launches(sys_, r, n_tasks, 0.5, 20)
        rows.append({"work": wname, "system": "serial", "threads": 1, "s_per_launch": round(serial_t[wname], 5),
                     "speedup_vs_serial": 1.0, "task_ms": round(serial_t[wname] / n_tasks * 1e3, 4)})
        configs = [("sleeping", t) for t in (1, 2, 4, 8, 16)] + [(s, THREADS) for s in ("spawn", "spinning_yield")]
        for sname, t in configs:
            sys_ = make_system(sname, t)
            try:
                per, _ = timed_launches(sys_, r, n_tasks, 0.5, 20)
            finally:
                sys_.close()
            rows.append({"work": wname, "system": sname, "threads": t, "s_per_launch": round(per, 5),
                         "speedup_vs_serial": round(serial_t[wname] / per, 3),
                         "task_ms": round(serial_t[wname] / n_tasks * 1e3, 4)})
            print(f"  {wname:12s} {sname:15s} t={t:2d} speedup={serial_t[wname] / per:6.2f}")
    write_csv("throughput.csv", rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for wname in works:
        rs = [r for r in rows if r["work"] == wname and r["system"] == "sleeping"]
        ax.plot([r["threads"] for r in rs], [r["speedup_vs_serial"] for r in rs], "o-", label=wname)
    ax.plot([1, 16], [1, 16], "k:", label="linear")
    ax.axvline(4, color="gray", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("threads (sleeping pool)")
    ax.set_ylabel("speedup vs serial")
    ax.set_title("128 tasks/launch: who escapes the GIL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "throughput.png", dpi=120)
    plt.close(fig)


def exp_cpu_burn() -> None:
    print("[cpu burn] idle and waiting CPU time")
    rows: list[dict] = []
    for name in SYSTEMS:
        sys_ = make_system(name, THREADS)
        try:
            sys_.run(NoopRunnable(), 8)
            c0, w0 = time.process_time(), time.perf_counter()
            time.sleep(1.0)
            idle_cpu, idle_wall = time.process_time() - c0, time.perf_counter() - w0
            r = SleepRunnable(0.02)
            c0, w0 = time.process_time(), time.perf_counter()
            for _ in range(10):
                sys_.run(r, THREADS)
            wait_cpu, wait_wall = time.process_time() - c0, time.perf_counter() - w0
        finally:
            sys_.close()
        rows.append({"system": name, "threads": THREADS,
                     "idle_wall_s": round(idle_wall, 3), "idle_cpu_s": round(idle_cpu, 3),
                     "idle_cores_burned": round(idle_cpu / idle_wall, 3),
                     "sleeptask_wall_s": round(wait_wall, 3), "sleeptask_cpu_s": round(wait_cpu, 3),
                     "sleeptask_cores_burned": round(wait_cpu / wait_wall, 3)})
        print(f"  {name:15s} idle {idle_cpu / idle_wall:.2f} cores, during sleep-tasks {wait_cpu / wait_wall:.2f} cores")
    write_csv("cpu_burn.csv", rows)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["idle_cores_burned"] for r in rows], 0.4, label="pool idle (main sleeps 1 s)")
    ax.bar(x + 0.2, [r["sleeptask_cores_burned"] for r in rows], 0.4, label="tasks blocked in sleep(20 ms)")
    ax.set_xticks(x, [r["system"] for r in rows], rotation=15)
    ax.set_ylabel("CPU seconds / wall second")
    ax.set_title("CPU burned while there is no work")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "cpu_burn.png", dpi=120)
    plt.close(fig)


def exp_async_overlap() -> None:
    print("[async] sync run() per launch vs runAsyncWithDeps")
    rows: list[dict] = []
    n_launch, n_tasks = 48, 9
    works = {"sleep_2ms": SleepRunnable(0.002), "numpy_ufunc": NumpyRunnable(n_tasks, elems=1 << 15, reps=4)}
    for wname, r in works.items():
        sys_ = make_system("sleeping", THREADS)
        try:
            sys_.run(r, n_tasks)
            modes = {}
            t0 = time.perf_counter()
            for _ in range(n_launch):
                sys_.run(r, n_tasks)
            modes["sync_run_each"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(n_launch):
                sys_.run_async_with_deps(r, n_tasks, [])
            sys_.sync()
            modes["async_independent"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            prev: list[int] = []
            for _ in range(n_launch):
                prev = [sys_.run_async_with_deps(r, n_tasks, prev)]
            sys_.sync()
            modes["async_chain"] = time.perf_counter() - t0
        finally:
            sys_.close()
        for mode, t in modes.items():
            rows.append({"work": wname, "launches": n_launch, "tasks_per_launch": n_tasks, "threads": THREADS,
                         "mode": mode, "wall_s": round(t, 4)})
        print(f"  {wname}: " + ", ".join(f"{m}={t * 1e3:.1f}ms" for m, t in modes.items()))
    write_csv("async_overlap.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, wname in zip(axes, works):
        rs = [r for r in rows if r["work"] == wname]
        ax.bar([r["mode"] for r in rs], [r["wall_s"] * 1e3 for r in rs])
        ax.set_title(f"{wname}: 48 launches x 9 tasks, 8 threads")
        ax.set_ylabel("ms")
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "async_overlap.png", dpi=120)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"switchinterval={sys.getswitchinterval()}")
    exp_launch_overhead()
    exp_throughput()
    exp_cpu_burn()
    exp_async_overlap()
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
