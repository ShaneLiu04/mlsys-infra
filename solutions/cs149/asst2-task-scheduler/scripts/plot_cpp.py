"""Figures for the C++ version of asst2 (README PNGs + report PDFs).

Reads artifacts/raw/cpp/<platform>/*.jsonl and writes artifacts/figures/cpp_*.png|pdf.
Each raw row stores all repetitions (lists such as wall_s / us_per_launch); plots show the
median with min-max error bars.

Usage: python scripts/plot_cpp.py [--platform server12c]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parents[1]
RAW = LAB / "artifacts" / "raw" / "cpp"
FIG = LAB / "artifacts" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
SYS = ["serial", "spawn", "spinning", "sleeping", "async_spinning", "async_sleeping"]
C = {"serial": "#7f7f7f", "spawn": "#d62728", "spinning": "#ff7f0e", "sleeping": "#1f77b4",
     "async_spinning": "#9467bd", "async_sleeping": "#2ca02c"}
M = {"serial": "s", "spawn": "x", "spinning": "^", "sleeping": "o", "async_spinning": "v", "async_sleeping": "D"}
PLAT = "server12c"


MEAS = {"wall_min_s", "wall_med_s", "cpu_med_s", "us_med", "us_min", "ns_per_task_med", "cpu_per_wall", "ns_med",
        "cores_burned", "wall_s", "cpu_s", "correct"}


def jl(name: str, reps: list[str] | None = None) -> pd.DataFrame:
    """Rows of one experiment pooled over campaigns: list columns (all repetitions) are
    concatenated, scalar measurements take the median over campaigns."""
    frames = []
    for p in reps or [PLAT, f"{PLAT}_r2", f"{PLAT}_r3"]:
        f = RAW / p / f"{name}.jsonl"
        if f.exists():
            frames.append(pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()]).assign(rep=p))
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    lists = [c for c in d if isinstance(d[c].iloc[0], list)]
    meas = [c for c in d if c in MEAS and c not in lists]
    keys = [c for c in d if c not in lists + meas + ["rep"]]
    agg = {c: (lambda s: sum(s, [])) for c in lists}
    agg.update({c: ("min" if c == "correct" else "median") for c in meas})
    agg["rep"] = "nunique"
    out = d.groupby(keys, dropna=False).agg(agg).reset_index()
    if "wall_s" in lists:
        out["wall_med_s"] = out.wall_s.apply(np.median)
    if "cpu_s" in lists:
        out["cpu_med_s"] = out.cpu_s.apply(np.median)
    return out


def stats(d: pd.DataFrame, col: str, scale: float = 1.0) -> pd.DataFrame:
    d = d.copy()
    d["med"] = d[col].apply(lambda v: float(np.median(v)) * scale)
    d["lo"] = d[col].apply(lambda v: float(np.min(v)) * scale)
    d["hi"] = d[col].apply(lambda v: float(np.max(v)) * scale)
    return d


def ebar(ax, x, d, **kw):
    ax.errorbar(x, d["med"], yerr=[d["med"] - d["lo"], d["hi"] - d["med"]], capsize=2, **kw)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def short(t: str) -> str:
    return (t.replace("math_operations_in_tight_for_loop", "math_ops").replace("recursive_fibonacci", "fib")
            .replace("spin_between_run_calls", "spin_between").replace("mandelbrot_chunked", "mandelbrot"))


def fig_overhead():
    d = jl("overhead")
    if d.empty:
        return
    d = stats(d, "us_per_launch")
    P = d.threads.max()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    for s in SYS:
        x = d[(d.system == s) & (d.threads == P)].sort_values("tasks")
        ebar(ax, x.tasks, x, label=s, color=C[s], marker=M[s], ms=4)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("tasks per launch (empty runTask)")
    ax.set_ylabel("µs per run() call")
    ax.set_title(f"(a) launch overhead, {P} threads")
    ax.legend(fontsize=8)
    ax = axs[1]
    for s in SYS:
        if s == "serial":
            continue
        x = d[(d.system == s) & (d.tasks == 64)].sort_values("threads")
        ebar(ax, x.threads, x, label=s, color=C[s], marker=M[s], ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("worker threads")
    ax.set_ylabel("µs per run() call")
    ax.set_title("(b) 64 empty tasks: cost vs thread count")
    save(fig, "cpp_launch_overhead")


def fig_heatmap():
    d = jl("tests")
    if d.empty:
        return
    d = stats(d, "wall_s")
    piv = d.pivot(index="test", columns="system", values="med")[SYS]
    norm = piv.div(piv.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(np.log2(norm.values), cmap="RdYlGn_r", vmin=0, vmax=6, aspect="auto")
    ax.set_xticks(range(len(SYS)), SYS, rotation=30, ha="right")
    ax.set_yticks(range(len(norm)), [short(t) for t in norm.index], fontsize=8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm.values[i, j]
            ax.text(j, i, f"{v:.2f}" if v < 100 else f"{v:.0f}", ha="center", va="center", fontsize=7)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("log2(time / best system)")
    ax.set_title(f"official tests: wall time normalised to the best system ({d.threads.iloc[0]} threads)")
    ax.grid(False)
    save(fig, "cpp_test_heatmap")

    # CPU cores burned per test
    d["cores"] = d.cpu_med_s / d.wall_med_s
    piv = d.pivot(index="test", columns="system", values="cores")[SYS]
    fig, ax = plt.subplots(figsize=(8.5, 8))
    im = ax.imshow(piv.values, cmap="magma_r", vmin=0, vmax=d.threads.iloc[0] + 1, aspect="auto")
    ax.set_xticks(range(len(SYS)), SYS, rotation=30, ha="right")
    ax.set_yticks(range(len(piv)), [short(t) for t in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if piv.values[i, j] > 7 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7).set_label("process CPU time / wall time (cores busy)")
    ax.set_title("CPU cost of each test: spinning burns every core, sleeping only what it uses")
    ax.grid(False)
    save(fig, "cpp_test_cpu_cost")


def fig_scaling():
    d = jl("scaling")
    if d.empty:
        return
    d = stats(d, "wall_s")
    ser = d[d.system == "serial"].set_index("test")["med"]
    tests = list(ser.index)
    fig, axs = plt.subplots(1, len(tests), figsize=(3.2 * len(tests), 3.6), sharey=True)
    P = d.threads.max()
    for ax, t in zip(np.atleast_1d(axs), tests):
        for s in ["spawn", "spinning", "sleeping", "async_sleeping"]:
            x = d[(d.system == s) & (d.test == t)].sort_values("threads")
            ax.plot(x.threads, ser[t] / x.med, marker=M[s], color=C[s], label=s, ms=4)
        ax.plot([1, P], [1, P], "k--", lw=0.8, label="linear")
        ax.set_title(short(t))
        ax.set_xlabel("threads")
    np.atleast_1d(axs)[0].set_ylabel("speedup vs serial")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    save(fig, "cpp_thread_scaling")


def fig_idle_and_spin():
    d = jl("idle")
    t = jl("tests")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    if not d.empty:
        ax = axs[0]
        for s in SYS:
            x = d[d.system == s].sort_values("threads")
            ax.plot(x.threads, x.cores_burned, marker=M[s], color=C[s], label=s, ms=4)
        ax.set_xlabel("worker threads (pool idle for 1 s)")
        ax.set_ylabel("cores burned = CPU s / wall s")
        ax.set_title("(a) idle pool CPU burn")
        ax.legend(fontsize=8)
    if not t.empty:
        ax = axs[1]
        t = stats(t, "wall_s")
        x = t[t.test == "spin_between_run_calls"].set_index("system").loc[SYS]
        xx = np.arange(len(SYS))
        ax.bar(xx - 0.2, x.med, 0.4, color=[C[s] for s in SYS], label="wall s")
        ax.bar(xx + 0.2, x.cpu_med_s, 0.4, color=[C[s] for s in SYS], alpha=0.4, hatch="//", label="CPU s")
        ax.set_xticks(xx, SYS, rotation=25, ha="right")
        ax.set_ylabel("seconds")
        ax.set_title("(b) spin_between_run_calls: wall (solid) vs CPU (hatched)")
        ax.legend(fontsize=8)
    save(fig, "cpp_idle_cpu_burn")


def fig_oversub():
    d = jl("oversub")
    if d.empty:
        return
    d = stats(d, "wall_s")
    tests = sorted(d.test.unique())
    fig, axs = plt.subplots(1, len(tests), figsize=(3.2 * len(tests), 3.6))
    for ax, t in zip(np.atleast_1d(axs), tests):
        for s in ["spinning", "sleeping", "async_spinning", "async_sleeping"]:
            x = d[(d.system == s) & (d.test == t)].sort_values("mult")
            ebar(ax, x["mult"], x, marker=M[s], color=C[s], label=s, ms=4)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("threads / cores")
        ax.set_title(short(t))
    np.atleast_1d(axs)[0].set_ylabel("wall s (scale 0.25)")
    np.atleast_1d(axs)[0].legend(fontsize=7)
    save(fig, "cpp_oversubscription")


def fig_granularity():
    d = jl("granularity")
    if d.empty:
        return
    d = stats(d, "wall_s", 1e3)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s in ["spawn", "spinning", "sleeping", "async_sleeping"]:
        x = d[d.system == s].sort_values("tasks")
        ebar(ax, x.tasks, x, marker=M[s], color=C[s], label=s, ms=4)
    g11 = jl("granularity_11workers", [PLAT])
    if not g11.empty:
        g11 = stats(g11, "wall_s", 1e3)
        for s in ["spinning", "sleeping"]:
            x = g11[g11.system == s].sort_values("tasks")
            ax.plot(x.tasks, x.med, ls="--", color=C[s], marker=M[s], ms=3, mfc="none",
                    label=f"{s}, 11 workers (one core left to caller)")
    P = d.threads.iloc[0]
    ax.axvline(P, color="k", lw=0.7, ls=":")
    ax.text(P * 1.1, ax.get_ylim()[1] * 0.6, f"tasks = {P} threads", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("number of tasks (fixed total work, ±50% per-task cost jitter)")
    ax.set_ylabel("ms per run()")
    ax.set_title("task granularity U-curve: imbalance (left) vs dispatch overhead (right)")
    ax.legend(fontsize=8)
    save(fig, "cpp_granularity")


def fig_dag():
    d = jl("dag")
    if d.empty:
        return
    d = stats(d, "wall_s", 1e3)
    ser = d[d.system == "serial"]
    T1 = float(ser.med.iloc[0])
    K, n = int(d.launches.iloc[0]), int(d.tasks.iloc[0])
    t_task = T1 / (K * n)
    P = int(d[d.system != "serial"].threads.iloc[0])
    shapes = ["indep_sync", "indep", "chain", "fanin", "tree", "random"]
    fig, ax = plt.subplots(figsize=(8, 4))
    xx = np.arange(len(shapes))
    for k, s in enumerate(["sleeping", "async_spinning", "async_sleeping"]):
        x = d[d.system == s].set_index("shape").reindex(shapes)
        ax.bar(xx + (k - 1) * 0.25, x.med, 0.25, color=C[s], label=s,
               yerr=[x.med - x.lo, x.hi - x.med], capsize=2)
    sp = d[d.system == "async_sleeping"].set_index("shape").reindex(shapes).span_launches
    lb = [max(T1 / P, sp[s] * np.ceil(n / P) * t_task) if s != "indep_sync" else K * np.ceil(n / P) * t_task
          for s in shapes]
    ax.plot(xx, lb, "k_", ms=25, mew=2, label="bound max(T1/P, span·⌈n/P⌉·t)")
    ax.set_xticks(xx, ["indep\n(sync each)", "indep\n(async)", "chain", "fan-in", "tree", "random"])
    ax.set_ylabel("ms")
    ax.set_title(f"{K} launches × {n} tasks on {P} threads (T1 = {T1:.0f} ms serial)")
    ax.legend(fontsize=8)
    save(fig, "cpp_dag_shapes")

    g = jl("dag_tasks")
    if g.empty:
        return
    g = stats(g, "wall_s")
    sync = g[g["shape"] == "indep_sync"].set_index("tasks").med
    asy = g[g["shape"] == "indep"].set_index("tasks").med
    ns = np.array(sorted(sync.index))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(ns, sync.loc[ns] / asy.loc[ns], "o-", color=C["async_sleeping"], label="measured sync/async")
    K = int(g.launches.iloc[0])
    ideal = [K * np.ceil(m / P) / np.ceil(K * m / P) for m in ns]
    ax.plot(ns, ideal, "k--", label="ideal K⌈n/P⌉ / ⌈Kn/P⌉")
    ax.set_xlabel(f"tasks per launch n (P = {P})")
    ax.set_ylabel("speedup of runAsyncWithDeps")
    ax.set_title("filling the tail: independent launches overlap")
    ax.legend(fontsize=8)
    save(fig, "cpp_async_tail")


def fig_contention():
    f = jl("false_sharing")
    dsp = jl("dispatch")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    if not f.empty:
        f = stats(f, "ns_per_inc")
        ax = axs[0]
        for op, ls in (("atomic", "-"), ("store", "--")):
            for lay, col in (("packed", "#d62728"), ("padded", "#2ca02c")):
                x = f[(f.layout == lay) & (f.placement == "cores") & (f.op == op)].sort_values("threads")
                ebar(ax, x.threads, x, marker="o", color=col, ls=ls, label=f"{op} {lay}", ms=4)
        smt = f[(f.placement == "smt_siblings") & (f.op == "atomic")]
        for lay, col in (("packed", "#d62728"), ("padded", "#2ca02c")):
            x = smt[smt.layout == lay]
            if not x.empty:
                ax.scatter([2.3], x.med, marker="*", s=90, color=col, label=f"atomic {lay}, SMT pair")
        ax.set_yscale("log")
        ax.set_xlabel("threads")
        ax.set_ylabel("ns per increment (per thread)")
        ax.set_title("(a) false sharing: counters in one 64 B line")
        ax.legend(fontsize=7)
    if not dsp.empty:
        dsp = stats(dsp, "ns_per_item")
        for ax, w in zip(axs[1:], (0, 50)):
            for sch, c in (("mutex", 1), ("atomic", 1), ("atomic_chunk", 8), ("atomic_chunk", 64), ("static", 0)):
                x = dsp[(dsp.scheme == sch) & (dsp.chunk == c) & (dsp.work == w)].sort_values("threads")
                lab = sch if sch != "atomic_chunk" else f"atomic chunk={c}"
                ebar(ax, x.threads, x, marker="o", label=lab, ms=4)
            ax.set_yscale("log")
            ax.set_xlabel("threads")
            ax.set_ylabel("ns per item (wall / n)")
            ax.set_title(f"({'b' if w == 0 else 'c'}) task dispatch, {w} LCG steps per item")
        axs[1].legend(fontsize=7)
    save(fig, "cpp_contention")


def fig_caller_and_smt():
    cc = jl("caller_core", [PLAT])
    smt = jl("smt", [PLAT])
    if cc.empty and smt.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    if not cc.empty:
        cc = stats(cc, "wall_s", 1e3)
        tests = sorted(cc.test.unique())
        ax = axs[0]
        syss = ["spinning", "sleeping", "async_spinning", "async_sleeping"]
        w = 0.8 / len(syss)
        for k, s in enumerate(syss):
            r = [float(cc[(cc.system == s) & (cc.test == t) & (cc.threads == 12)].med.iloc[0]) /
                 float(cc[(cc.system == s) & (cc.test == t) & (cc.threads == 11)].med.iloc[0]) for t in tests]
            ax.bar(np.arange(len(tests)) + (k - 1.5) * w, r, w, color=C[s], label=s)
        ax.axhline(1, color="k", lw=0.7)
        ax.set_xticks(range(len(tests)), [short(t) for t in tests], rotation=20, ha="right")
        ax.set_ylabel("time(12 workers) / time(11 workers)")
        ax.set_title("(a) 12 cores: leave one core for the caller thread?")
        ax.legend(fontsize=7)
    if not smt.empty:
        smt = stats(smt, "wall_s", 1e3)
        tests = sorted(smt.test.unique())
        ax = axs[1]
        for k, s in enumerate(["spinning", "sleeping"]):
            r = [float(smt[(smt.system == s) & (smt.test == t) & (smt.threads == 12)].med.iloc[0]) /
                 float(smt[(smt.system == s) & (smt.test == t) & (smt.threads == 24)].med.iloc[0]) for t in tests]
            ax.bar(np.arange(len(tests)) + (k - 0.5) * 0.4, r, 0.4, color=C[s], label=s)
        ax.axhline(1, color="k", lw=0.7)
        ax.set_xticks(range(len(tests)), [short(t) for t in tests], rotation=20, ha="right")
        ax.set_ylabel("speedup of 24 threads (12 cores + SMT) over 12")
        ax.set_title("(b) SMT siblings as extra workers")
        ax.legend(fontsize=7)
    save(fig, "cpp_caller_smt")


def main() -> None:
    global PLAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=PLAT)
    PLAT = ap.parse_args().platform
    fig_overhead()
    fig_heatmap()
    fig_scaling()
    fig_idle_and_spin()
    fig_oversub()
    fig_granularity()
    fig_dag()
    fig_contention()
    fig_caller_and_smt()


if __name__ == "__main__":
    main()
