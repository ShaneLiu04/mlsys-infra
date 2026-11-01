"""Run all asst1 experiments; writes artifacts/raw/*.csv and artifacts/figures/*.png."""

from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from cpupar.mandelbrot import (  # noqa: E402
    VIEW1,
    VIEW2,
    MandelSpec,
    assignment_work,
    interleaved_rows,
    mandel_image,
    mandel_points,
    mandel_points_masked,
    mandel_scalar,
    row_coords,
    static_blocks,
)
from cpupar.parallel import make_pool, run_parallel  # noqa: E402
from cpupar.saxpy import (  # noqa: E402
    effective_gbps,
    saxpy_numpy,
    saxpy_numpy_threads,
    saxpy_torch,
    time_best,
)
from cpupar.simd import (  # noqa: E402
    gang_issued_slots,
    gang_utilization,
    sqrt_inputs,
    sqrt_iterations,
    sqrt_masked,
    sqrt_scalar,
)

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
MAX_WORKERS = 8
WIDTHS = [1, 2, 4, 8, 16]
CHUNK_ROWS = 16


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def exp_mandelbrot() -> None:
    print("[mandelbrot] kernel block size (cache effect, serial)")
    blk_rows: list[dict] = []
    spec1 = MandelSpec(1600, 1200, 256, VIEW1)
    for rows_per_call in (2, 10, 40, 160, 1200):
        t0 = time.perf_counter()
        mandel_image(spec1, block_points=rows_per_call * spec1.width)
        dt = time.perf_counter() - t0
        blk_rows.append({"view": "view1", "rows_per_call": rows_per_call,
                         "points_per_call": rows_per_call * spec1.width, "serial_s": round(dt, 4)})
        print(f"  rows/call={rows_per_call}: {dt:.3f}s")
    write_csv("mandel_block_size.csv", blk_rows)

    print("[mandelbrot] process-pool schedules")
    views = {"view1": MandelSpec(1600, 1200, 256, VIEW1), "view2": MandelSpec(1600, 1200, 256, VIEW2)}
    worker_sets = {"view1": list(range(1, MAX_WORKERS + 1)), "view2": [1, 2, 4, 8]}
    speed_rows: list[dict] = []
    task_rows: list[dict] = []
    imb_rows: list[dict] = []
    counts = {}
    serial = {}
    for name, spec in views.items():
        t0 = time.perf_counter()
        counts[name] = mandel_image(spec)
        serial[name] = time.perf_counter() - t0
        print(f"  {name} serial in-process {serial[name]:.3f}s")
        for p in range(1, MAX_WORKERS + 1):
            for sched, parts in (("static", static_blocks(spec.height, p)),
                                 ("interleaved", interleaved_rows(spec.height, p))):
                work = assignment_work(counts[name], parts)
                imb_rows.append({"view": name, "schedule": sched, "workers": p,
                                 "work_max_over_mean": round(float(work.max() / work.mean()), 4),
                                 "ideal_speedup": round(float(work.sum() / work.max()), 3)})
    all_workers = sorted(set(sum(worker_sets.values(), [])))
    for p in all_workers:
        wp = make_pool(p)
        try:
            for name, spec in views.items():
                if p not in worker_sets[name]:
                    continue
                for sched in ("static", "interleaved", "dynamic"):
                    best = None
                    for _ in range(3):
                        res = run_parallel(wp, spec, sched, CHUNK_ROWS)
                        assert np.array_equal(res.image, counts[name])
                        if best is None or res.wall_s < best.wall_s:
                            best = res
                    busy = np.array(list(best.worker_busy_s.values()))
                    speed_rows.append({
                        "view": name, "schedule": sched, "workers": p, "tasks": best.n_tasks,
                        "wall_s": round(best.wall_s, 4),
                        "speedup_vs_serial": round(serial[name] / best.wall_s, 3),
                        "busy_max_s": round(float(busy.max()), 4), "busy_min_s": round(float(busy.min()), 4),
                        "busy_max_over_mean": round(float(busy.max() / busy.mean()), 3),
                        "workers_used": len(busy),
                    })
                    if sched in ("static", "interleaved") and p in (4, 8):
                        for k, first, dt, _pid in best.task_times:
                            task_rows.append({"view": name, "schedule": sched, "workers": p,
                                              "task": k, "first_row": first, "compute_s": round(dt, 4)})
                    print(f"  p={p} {name} {sched:11s} wall={best.wall_s:.3f}s "
                          f"speedup={serial[name] / best.wall_s:.2f}")
        finally:
            wp.close()
    for r in speed_rows:
        r["serial_s"] = round(serial[r["view"]], 4)
    write_csv("mandel_speedup.csv", speed_rows)
    write_csv("mandel_task_times.csv", task_rows)
    write_csv("mandel_work_imbalance.csv", imb_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, views):
        for sched, mk in (("static", "o-"), ("interleaved", "s-"), ("dynamic", "^-")):
            rs = [r for r in speed_rows if r["view"] == name and r["schedule"] == sched]
            ax.plot([r["workers"] for r in rs], [r["speedup_vs_serial"] for r in rs], mk, label=sched)
        ws = worker_sets[name]
        ax.plot(ws, ws, "k:", label="linear")
        ax.axvline(4, color="gray", lw=0.6)
        ax.set_title(f"Mandelbrot {name} 1600x1200 (4 cores / 8 HW threads)")
        ax.set_xlabel("worker processes")
        ax.set_ylabel("speedup vs in-process serial")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "mandel_speedup.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, sched in zip(axes, ("static", "interleaved")):
        rs = [r for r in task_rows if r["view"] == "view1" and r["schedule"] == sched and r["workers"] == 8]
        ax.bar([r["task"] for r in rs], [r["compute_s"] for r in rs])
        ax.set_title(f"view1, 8 workers, {sched}: per-worker compute time")
        ax.set_xlabel("worker (static: block index, top→bottom)")
        ax.set_ylabel("seconds")
    fig.tight_layout()
    fig.savefig(FIG / "mandel_worker_time.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.imshow(counts["view1"], cmap="magma")
    ax.set_title("view1 iteration counts (bright = expensive)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "mandel_view1_counts.png", dpi=100)
    plt.close(fig)
    np.save(RAW / "_counts_view1.npy", counts["view1"])
    np.save(RAW / "_counts_view2.npy", counts["view2"])


def exp_simd() -> None:
    print("[simd] gang utilization model")
    n = 1 << 16
    rows: list[dict] = []
    sources: dict[str, dict[int, np.ndarray]] = {}
    for name in ("view1", "view2"):
        work = np.load(RAW / f"_counts_{name}.npy").astype(np.int64).ravel() + 1
        sources[f"mandel_{name}"] = {w: work for w in WIDTHS}
    for pat in ("uniform", "all_fast", "all_slow"):
        it = sqrt_iterations(sqrt_inputs(pat, n, seed=0))[1].astype(np.int64) + 1
        sources[f"sqrt_{pat}"] = {w: it for w in WIDTHS}
    sources["sqrt_one_slow_per_gang"] = {
        w: sqrt_iterations(sqrt_inputs("one_slow_per_gang", n, width=w))[1].astype(np.int64) + 1 for w in WIDTHS
    }
    for src, by_w in sources.items():
        for w in WIDTHS:
            work = by_w[w]
            u = gang_utilization(work, w)
            rows.append({"source": src, "W": w, "utilization": round(u, 4),
                         "simd_speedup_vs_scalar": round(u * w, 3),
                         "useful_lane_iters": int(work.sum()), "issued_lane_slots": gang_issued_slots(work, w)})
    write_csv("simd_utilization.csv", rows)
    for f in RAW.glob("_counts_*.npy"):
        f.unlink()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for src in sources:
        rs = [r for r in rows if r["source"] == src]
        axes[0].plot([r["W"] for r in rs], [r["utilization"] for r in rs], "o-", label=src)
        axes[1].plot([r["W"] for r in rs], [r["simd_speedup_vs_scalar"] for r in rs], "o-", label=src)
    axes[1].plot(WIDTHS, WIDTHS, "k:", label="ideal")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
        ax.set_xlabel("gang width W")
    axes[0].set_ylabel("lane utilization")
    axes[1].set_ylabel("modelled speedup = W x utilization")
    axes[0].legend(fontsize=7)
    fig.suptitle("SIMD divergence model (work = iterations + 1 per lane)")
    fig.tight_layout()
    fig.savefig(FIG / "simd_utilization.png", dpi=120)
    plt.close(fig)


def _best(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def exp_simd_timing() -> None:
    print("[simd] measured: Python scalar vs numpy compacted vs numpy masked")
    rows: list[dict] = []
    n_scalar, n_vec = 20_000, 1 << 20
    for pat in ("uniform", "all_fast", "all_slow", "one_slow_per_gang"):
        xs = sqrt_inputs(pat, n_scalar, width=16, seed=0)
        xv = sqrt_inputs(pat, n_vec, width=16, seed=0)
        xs_list = [float(v) for v in xs]
        t_sc = _best(lambda: [sqrt_scalar(v) for v in xs_list], repeats=1)
        t_cp = _best(lambda: sqrt_iterations(xv))
        t_mk = _best(lambda: sqrt_masked(xv))
        mean_it = float(sqrt_iterations(xv)[1].mean())
        rows.append({"kernel": "sqrt", "input": pat, "mean_iters": round(mean_it, 3),
                     "python_scalar_ns_per_elem": round(t_sc / n_scalar * 1e9, 2),
                     "numpy_compacted_ns_per_elem": round(t_cp / n_vec * 1e9, 3),
                     "numpy_masked_ns_per_elem": round(t_mk / n_vec * 1e9, 3)})
        print(f"  sqrt {pat}: {rows[-1]}")
    spec_s = MandelSpec(160, 120, 256, VIEW1)
    spec_v = MandelSpec(800, 600, 256, VIEW1)
    cr_s, ci_s = row_coords(spec_s, np.arange(spec_s.height))
    cr_v, ci_v = row_coords(spec_v, np.arange(spec_v.height))
    pts = [(float(a), float(b)) for a, b in zip(cr_s, ci_s)]
    t_sc = _best(lambda: [mandel_scalar(a, b, 256) for a, b in pts], repeats=1)
    blk = 16384

    def blocked(kernel) -> np.ndarray:
        return np.concatenate([kernel(cr_v[s:s + blk], ci_v[s:s + blk], 256) for s in range(0, cr_v.size, blk)])

    t_cp = _best(lambda: blocked(mandel_points), repeats=2)
    t_mk = _best(lambda: blocked(mandel_points_masked), repeats=1)
    mean_it = float(mandel_points(cr_v, ci_v, 256).mean())
    rows.append({"kernel": "mandel_view1", "input": "800x600, 16384-pt blocks (scalar: 160x120)",
                 "mean_iters": round(mean_it, 3),
                 "python_scalar_ns_per_elem": round(t_sc / cr_s.size * 1e9, 2),
                 "numpy_compacted_ns_per_elem": round(t_cp / cr_v.size * 1e9, 3),
                 "numpy_masked_ns_per_elem": round(t_mk / cr_v.size * 1e9, 3)})
    print(f"  mandel: {rows[-1]}")
    write_csv("simd_timing.csv", rows)


def exp_saxpy() -> None:
    print("[saxpy] bandwidth vs size and vs workers")
    a = 2.0
    rows: list[dict] = []
    for logn in range(10, 27, 2):
        n = 1 << logn
        x = np.random.default_rng(0).random(n, dtype=np.float32)
        y = np.random.default_rng(1).random(n, dtype=np.float32)
        out = np.empty_like(x)
        tx, ty, to = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(out)
        results = {}
        results["numpy_2pass_1t"] = time_best(lambda: saxpy_numpy(a, x, y, out))
        torch.set_num_threads(1)
        results["torch_fused_1t"] = time_best(lambda: saxpy_torch(a, tx, ty, to))
        torch.set_num_threads(MAX_WORKERS)
        results["torch_fused_8t"] = time_best(lambda: saxpy_torch(a, tx, ty, to))
        for method, t in results.items():
            rows.append({"n": n, "bytes_3arrays": 12 * n, "method": method, "time_s": f"{t:.3e}",
                         "effective_GBps": round(effective_gbps(n, t), 3),
                         "with_write_allocate_GBps": round(effective_gbps(n, t) * 4 / 3, 3)})
        print(f"  n=2^{logn}: " + ", ".join(f"{k}={effective_gbps(n, v):.1f}GB/s" for k, v in results.items()))
        del x, y, out, tx, ty, to
    write_csv("saxpy_bandwidth.csv", rows)

    scal_rows: list[dict] = []
    for logn in (16, 20, 25):
        n = 1 << logn
        x = np.random.default_rng(0).random(n, dtype=np.float32)
        y = np.random.default_rng(1).random(n, dtype=np.float32)
        out = np.empty_like(x)
        tx, ty, to = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(out)
        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            for p in range(1, MAX_WORKERS + 1):
                torch.set_num_threads(p)
                t_t = time_best(lambda: saxpy_torch(a, tx, ty, to))
                t_n = time_best(lambda: saxpy_numpy_threads(pool, p, a, x, y, out))
                for method, t in (("torch_fused_threads", t_t), ("numpy_2pass_threadchunks", t_n)):
                    scal_rows.append({"n": n, "MB_3arrays": round(12 * n / 2**20, 2), "method": method,
                                      "workers": p, "time_s": f"{t:.3e}",
                                      "effective_GBps": round(effective_gbps(n, t), 3)})
        print(f"  n=2^{logn} scaling done")
    torch.set_num_threads(MAX_WORKERS)
    write_csv("saxpy_scaling.csv", scal_rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in ("numpy_2pass_1t", "torch_fused_1t", "torch_fused_8t"):
        rs = [r for r in rows if r["method"] == method]
        ax.plot([r["bytes_3arrays"] for r in rs], [r["effective_GBps"] for r in rs], "o-", label=method)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("working set = 3 arrays x 4 B x N (bytes)")
    ax.set_ylabel("effective GB/s (12N bytes / time)")
    ax.set_title("SAXPY: cache vs DRAM regimes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "saxpy_bandwidth.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, method in zip(axes, ("torch_fused_threads", "numpy_2pass_threadchunks")):
        for n in sorted({r["n"] for r in scal_rows}):
            rs = [r for r in scal_rows if r["n"] == n and r["method"] == method]
            ax.plot([r["workers"] for r in rs], [r["effective_GBps"] for r in rs], "o-",
                    label=f"{rs[0]['MB_3arrays']} MB")
        ax.set_title(method)
        ax.set_xlabel("threads")
        ax.set_ylabel("effective GB/s")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "saxpy_scaling.png", dpi=120)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"cpu_count={os.cpu_count()} numpy={np.__version__} torch={torch.__version__}")
    t0 = time.perf_counter()
    exp_mandelbrot()
    exp_simd()
    exp_simd_timing()
    exp_saxpy()
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
