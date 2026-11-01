"""Run all asst3 experiments; writes artifacts/raw/*.csv and artifacts/figures/*.png."""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from scanrender.primitives import find_repeats, segmented_exclusive_scan, stream_compact  # noqa: E402
from scanrender.renderer import (  # noqa: E402
    brute_force_tests,
    make_scene,
    render_reference,
    render_tiled,
    render_unordered,
)
from scanrender.scan import (  # noqa: E402
    analytic_stats,
    blelloch_exclusive,
    exclusive_scan_sequential,
    hillis_steele_exclusive,
)

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
SCANS = {"sequential": exclusive_scan_sequential, "hillis_steele": hillis_steele_exclusive,
         "blelloch": blelloch_exclusive}


def write_csv(name: str, rows: list[dict]) -> None:
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {name} ({len(rows)} rows)")


def best_time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def exp_scan() -> None:
    print("[scan] work / span / wall time")
    rows: list[dict] = []
    for n in (1000, 1 << 10, 1 << 14, (1 << 17) + 1, 1 << 20):
        x = np.random.default_rng(n).integers(0, 100, n)
        ref = np.concatenate([[0], np.cumsum(x)[:-1]])
        t_np = best_time(lambda: np.cumsum(x))
        for algo, fn in SCANS.items():
            out, st = fn(x)
            assert np.array_equal(out, ref)
            an = analytic_stats(algo, n)
            assert (st.adds, st.steps) == (an.adds, an.steps)
            t = best_time(lambda: fn(x), repeats=1 if algo == "sequential" else 3)
            rows.append({
                "n": n, "algo": algo, "padded_n": st.padded_n, "adds": st.adds, "steps": st.steps,
                "adds_over_n": round(st.adds / n, 3),
                "brent_T_P8": math.ceil(st.adds / 8) + st.steps if algo != "sequential" else st.adds,
                "brent_T_P2048": math.ceil(st.adds / 2048) + st.steps if algo != "sequential" else st.adds,
                "wall_ms": round(t * 1e3, 3), "np_cumsum_ms": round(t_np * 1e3, 4),
            })
            print(f"  n={n:8d} {algo:13s} adds={st.adds:9d} steps={st.steps:7d} {t * 1e3:9.2f} ms")
    write_csv("scan_work_span.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for algo in SCANS:
        rs = [r for r in rows if r["algo"] == algo and r["n"] != 1000]
        ns = [r["n"] for r in rs]
        axes[0].plot(ns, [r["adds"] for r in rs], "o-", label=algo)
        axes[1].plot(ns, [r["steps"] for r in rs], "o-", label=algo)
        axes[2].plot(ns, [r["wall_ms"] for r in rs], "o-", label=algo)
    rs = [r for r in rows if r["algo"] == "blelloch" and r["n"] != 1000]
    axes[2].plot([r["n"] for r in rs], [r["np_cumsum_ms"] for r in rs], "k:", label="np.cumsum")
    for ax, t in zip(axes, ("work (adds)", "span (dependent steps)", "wall ms (numpy, 1 core)")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_title(t)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "scan_work_span.png", dpi=120)
    plt.close(fig)


def exp_primitives() -> None:
    print("[primitives] find_repeats / compaction / segmented scan")
    rows: list[dict] = []
    n = 1 << 20
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, n)
    fr = find_repeats(a)
    ok = np.array_equal(fr, np.nonzero(a[:-1] == a[1:])[0])
    rows.append({"primitive": "find_repeats", "n": n, "output_len": fr.size, "correct": ok,
                 "scan_based_ms": round(best_time(lambda: find_repeats(a)) * 1e3, 3),
                 "numpy_builtin_ms": round(best_time(lambda: np.nonzero(a[:-1] == a[1:])[0]) * 1e3, 3)})
    v = rng.normal(size=n)
    sc = stream_compact(v, v > 1.0)
    rows.append({"primitive": "stream_compact(v>1)", "n": n, "output_len": sc.size,
                 "correct": bool(np.array_equal(sc, v[v > 1.0])),
                 "scan_based_ms": round(best_time(lambda: stream_compact(v, v > 1.0)) * 1e3, 3),
                 "numpy_builtin_ms": round(best_time(lambda: v[v > 1.0]) * 1e3, 3)})
    heads = rng.random(n) < 0.01
    vals = rng.integers(0, 10, n)
    seg = segmented_exclusive_scan(vals, heads)
    heads0 = heads.copy()
    heads0[0] = True
    seg_id = np.cumsum(heads0) - 1
    incl = np.cumsum(vals)
    starts = np.nonzero(heads0)[0]
    base = (incl - vals)[starts][seg_id]
    expect = incl - vals - base
    rows.append({"primitive": "segmented_exclusive_scan", "n": n, "output_len": n,
                 "correct": bool(np.array_equal(seg, expect)),
                 "scan_based_ms": round(best_time(lambda: segmented_exclusive_scan(vals, heads)) * 1e3, 3),
                 "numpy_builtin_ms": round(best_time(lambda: np.cumsum(vals)) * 1e3, 3)})
    for r in rows:
        print(f"  {r}")
    write_csv("primitives.csv", rows)


def exp_render() -> None:
    print("[render] reference vs unordered vs tile-binned")
    size = 512
    rows: list[dict] = []
    err_rows: list[dict] = []
    images = {}
    diffs = {}
    for kind in ("few_big", "mixed", "many_small"):
        sc = make_scene(kind, size, seed=0)
        t0 = time.perf_counter()
        ref, rst = render_reference(sc, size, size)
        t_ref = time.perf_counter() - t0
        images[kind] = ref
        for seed in range(3):
            bad = render_unordered(sc, size, size, seed)
            d = np.abs(bad - ref).max(axis=2)
            if seed == 0:
                diffs[kind] = d
            err_rows.append({"scene": kind, "n_circles": sc.n, "shuffle_seed": seed,
                             "max_abs_err": round(float(d.max()), 4),
                             "frac_pixels_err_gt_1e-6": round(float((d > 1e-6).mean()), 4),
                             "frac_pixels_err_gt_0.01": round(float((d > 0.01).mean()), 4)})
        brute = brute_force_tests(sc, size, size)
        for tile in (8, 16, 32, 64):
            t0 = time.perf_counter()
            out, st = render_tiled(sc, size, size, tile)
            t_tile = time.perf_counter() - t0
            n_tiles = (size // tile) ** 2
            total = st.bbox_tests + st.circle_pixel_tests
            rows.append({
                "scene": kind, "n_circles": sc.n, "tile": tile, "tiles": n_tiles,
                "brute_force_tests": brute, "ref_bbox_pixel_tests": rst.circle_pixel_tests,
                "tiled_bbox_tests": st.bbox_tests, "tiled_pixel_tests": st.circle_pixel_tests,
                "tiled_total_tests": total, "brute_over_tiled": round(brute / total, 2),
                "avg_circles_per_tile": round(st.pairs / n_tiles, 3),
                "max_abs_err_vs_ref": float(np.abs(out - ref).max()),
                "ref_s": round(t_ref, 3), "tiled_s": round(t_tile, 3),
            })
            print(f"  {kind:10s} tile={tile:2d} brute/tiled={brute / total:8.1f} "
                  f"err={np.abs(out - ref).max():.1e} ref={t_ref:.2f}s tiled={t_tile:.2f}s")
    write_csv("render_tests.csv", rows)
    write_csv("render_unordered_error.csv", err_rows)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    for j, kind in enumerate(images):
        axes[0, j].imshow(np.clip(images[kind], 0, 1))
        axes[0, j].set_title(f"{kind}: ordered reference")
        im = axes[1, j].imshow(diffs[kind], cmap="inferno", vmin=0)
        axes[1, j].set_title("|unordered - reference| (max over RGB)")
        fig.colorbar(im, ax=axes[1, j], fraction=0.046)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "render_order_error.png", dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    kinds = list(images)
    tiles = (8, 16, 32, 64)
    width = 0.18
    for i, tile in enumerate(tiles):
        vals = [next(r["brute_over_tiled"] for r in rows if r["scene"] == k and r["tile"] == tile) for k in kinds]
        ax.bar(np.arange(len(kinds)) + (i - 1.5) * width, vals, width, label=f"tile {tile}")
    ax.set_xticks(np.arange(len(kinds)), [f"{k}\n(N={next(r['n_circles'] for r in rows if r['scene'] == k)})"
                                          for k in kinds])
    ax.set_yscale("log")
    ax.set_ylabel("brute-force tests / tiled tests")
    ax.set_title("Work saved by tile binning (512x512)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "render_binning_savings.png", dpi=120)
    plt.close(fig)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    exp_scan()
    exp_primitives()
    exp_render()
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
