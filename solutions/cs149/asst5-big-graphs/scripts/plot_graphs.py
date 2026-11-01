"""Figures for the CS149 A5 campaign. Reads artifacts/raw/campaign.jsonl and extra.jsonl."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw" / "campaign.jsonl"
EXTRA = ROOT / "artifacts" / "raw" / "extra.jsonl"
FIG = ROOT / "artifacts" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

C = {
    "serial": "#7f7f7f",
    "local": "#1f77b4",
    "atomic": "#17becf",
    "critical": "#d62728",
    "bottom": "#ff7f0e",
    "bottom_static": "#c49c94",
    "hybrid": "#2ca02c",
    "pull": "#2ca02c",
    "push": "#d62728",
}


def load():
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def main():
    df = load()
    bfs = df[df.kind == "bfs"].copy()
    lvl = df[df.kind == "level"].copy()
    pr = df[df.kind == "pagerank"].copy()

    # 1. frontier and step time on rmat and grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, graph, title in ((0, "rmat_1m_8m", "RMAT, 1M vertices, 8M edges"), (1, "grid_256", "256x256 grid")):
        sub = lvl[(lvl.graph == graph) & (lvl.threads == 4) & (lvl.algo.isin(["local", "bottom", "hybrid"]))]
        for algo, g in sub.groupby("algo"):
            axes[0, col].plot(g.level, g.frontier, "o-", ms=3, color=C[algo], label=algo)
            axes[1, col].plot(g.level, g.ms, "o-", ms=3, color=C[algo], label=algo)
        axes[0, col].set_title(title + ": frontier")
        axes[0, col].set_ylabel("frontier vertices")
        axes[1, col].set_title(title + ": step time")
        axes[1, col].set_xlabel("BFS level")
        axes[1, col].set_ylabel("ms")
        axes[0, col].legend(fontsize=8)
        axes[1, col].legend(fontsize=8)
        if graph.startswith("rmat"):
            axes[0, col].set_yscale("log")
    save(fig, "a5_levels")

    # 2. speedup vs threads
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, graph in zip(axes, ("rmat_1m_8m", "grid_256")):
        sub = bfs[bfs.graph == graph]
        base = sub[sub.algo == "serial"].ms.iloc[0]
        for algo in ("local", "atomic", "bottom", "hybrid"):
            a = sub[sub.algo == algo].sort_values("threads")
            if a.empty:
                continue
            ax.plot(a.threads, base / a.ms, "o-", color=C.get(algo, "k"), label=algo)
        ax.axhline(1, color="gray", lw=0.6)
        ax.set_xlabel("threads")
        ax.set_ylabel("speedup vs serial")
        ax.set_title(graph + ("  (parallel hurts)" if graph.startswith("grid") else ""))
        ax.legend(fontsize=8)
        ax.set_yscale("log")
    save(fig, "a5_speedup")

    # 3. edges examined at 4 threads
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = bfs[(bfs.threads == 4) & (bfs.algo.isin(["local", "bottom", "hybrid", "serial"]))]
    graphs = ["rmat_1m_8m", "er_1m_4m", "grid_256", "chain_20k", "star_2m"]
    import numpy as np

    x = np.arange(len(graphs))
    width = 0.18
    for i, algo in enumerate(("serial", "local", "bottom", "hybrid")):
        vals = []
        for gname in graphs:
            row = sub[(sub.graph == gname) & (sub.algo == algo)]
            vals.append(float(row.examined.iloc[0]) if len(row) else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=algo, color=C[algo])
    ax.set_xticks(x, graphs, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("edges examined")
    ax.set_title("Work, not just time: bottom-up scans unvisited vertices")
    ax.legend(fontsize=8)
    save(fig, "a5_examined")

    # 4. synchronization variants
    fig, ax = plt.subplots(figsize=(7, 4))
    rows = []
    labels = []
    for graph, algo, th, label in (
        ("rmat_1m_8m", "local", 4, "RMAT local CAS"),
        ("rmat_1m_8m", "atomic", 4, "RMAT CAS + shared index"),
        ("rmat_1m_8m", "serial", 1, "RMAT serial"),
        ("chain_20k", "local", 4, "chain local CAS"),
        ("chain_20k", "critical", 4, "chain critical"),
        ("chain_20k", "serial", 1, "chain serial"),
    ):
        row = bfs[(bfs.graph == graph) & (bfs.algo == algo) & (bfs.threads == th)]
        rows.append(float(row.ms.iloc[0]))
        labels.append(label)
    ax.barh(labels[::-1], rows[::-1], color="#1f77b4")
    ax.set_xlabel("ms (best of 3)")
    ax.set_title("Where the synchronization goes")
    save(fig, "a5_sync")

    # 5. hybrid alpha
    fig, ax = plt.subplots(figsize=(6.5, 4))
    h = bfs[(bfs.graph == "rmat_1m_8m") & (bfs.algo.str.startswith("hybrid_a"))].copy()
    h["alpha"] = h.algo.str.replace("hybrid_a", "", regex=False).astype(float)
    h = h.sort_values("alpha")
    ax.plot(h.alpha, h.ms, "o-", color=C["hybrid"])
    ref = bfs[(bfs.graph == "rmat_1m_8m") & (bfs.algo == "local") & (bfs.threads == 4)].ms.iloc[0]
    bu = bfs[(bfs.graph == "rmat_1m_8m") & (bfs.algo == "bottom") & (bfs.threads == 4)].ms.iloc[0]
    ax.axhline(ref, color=C["local"], ls="--", label=f"top-down {ref:.0f} ms")
    ax.axhline(bu, color=C["bottom"], ls="--", label=f"bottom-up {bu:.0f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (larger = switch to bottom-up sooner)")
    ax.set_ylabel("ms, 4 threads")
    ax.set_title("Hybrid threshold on RMAT")
    ax.legend(fontsize=8)
    save(fig, "a5_alpha")

    # 6. pagerank
    if not pr.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for algo, g in pr.groupby("algo"):
            g = g.sort_values("threads")
            ax.plot(g.threads, g.ms, "o-", color=C[algo], label=algo)
        ax.set_xlabel("threads")
        ax.set_ylabel("ms for 10 iterations")
        ax.set_title("PageRank on the same RMAT graph")
        ax.legend()
        save(fig, "a5_pagerank")

    # 7. which algorithm wins
    fig, ax = plt.subplots(figsize=(8, 4))
    algos = ["local", "bottom", "hybrid"]
    x = np.arange(len(graphs))
    for i, algo in enumerate(algos):
        vals = []
        for gname in graphs:
            base = bfs[(bfs.graph == gname) & (bfs.algo == "serial")].ms.iloc[0]
            row = bfs[(bfs.graph == gname) & (bfs.algo == algo) & (bfs.threads == 4)]
            vals.append(base / float(row.ms.iloc[0]))
        ax.bar(x + (i - 1) * 0.25, vals, 0.25, label=algo, color=C[algo])
    ax.axhline(1, color="gray", lw=0.6)
    ax.set_xticks(x, graphs, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("speedup vs serial at 4 threads")
    ax.set_title("No single BFS wins on every graph")
    ax.legend()
    save(fig, "a5_which")

    # 8. static vs dynamic bottom-up
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for algo in ("bottom", "bottom_static"):
        a = bfs[(bfs.graph == "rmat_1m_8m") & (bfs.algo == algo)].sort_values("threads")
        ax.plot(a.threads, a.ms, "o-", color=C[algo], label=algo)
    ax.set_xlabel("threads")
    ax.set_ylabel("ms")
    ax.set_title("Bottom-up schedule: dynamic chunk 1024 vs static")
    ax.legend()
    save(fig, "a5_schedule")

    # 9. hybrid direction along the RMAT levels
    fig, ax = plt.subplots(figsize=(7, 3.6))
    h = lvl[(lvl.graph == "rmat_1m_8m") & (lvl.algo == "hybrid") & (lvl.threads == 4)]
    colors = ["#1f77b4" if w == "t" else "#ff7f0e" for w in h.which]
    ax.bar(h.level, h.ms, color=colors)
    ax.set_xlabel("level (blue = top-down, orange = bottom-up)")
    ax.set_ylabel("ms")
    ax.set_title("Hybrid switches direction as the frontier grows, then switches back")
    save(fig, "a5_switch")

    if EXTRA.exists():
        extra_figures()


def extra_figures():
    import numpy as np

    rows = [json.loads(line) for line in EXTRA.read_text(encoding="utf-8").splitlines() if line.strip()]
    ex = pd.DataFrame(rows)
    bfs = ex[ex.kind == "bfs"].copy()
    bfs["lo"] = bfs.times.apply(min)
    bfs["hi"] = bfs.times.apply(max)
    bfs["med"] = bfs.times.apply(lambda t: float(np.median(t)))

    # 10. RMAT scale sweep
    sc = bfs[bfs.suite == "scale"].copy()
    sc["scale"] = sc.graph.str.replace("rmat_s", "", regex=False).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algo in ("serial", "local", "bottom", "hybrid", "hybrid_edges"):
        a = sc[sc.algo == algo].sort_values("scale")
        axes[0].plot(a.scale, a.ms, "o-", color=C.get(algo, "#9467bd"), label=algo if algo == "serial" else f"{algo} (8 thr)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("RMAT scale (n = 2^scale, m = 8n)")
    axes[0].set_ylabel("ms (best of 5)")
    axes[0].set_title("BFS time vs graph size")
    axes[0].legend(fontsize=8)
    serial = sc[sc.algo == "serial"].set_index("scale")
    for algo in ("local", "bottom", "hybrid"):
        a = sc[sc.algo == algo].set_index("scale").sort_index()
        axes[1].plot(a.index, serial.ms.loc[a.index] / a.ms, "o-", color=C[algo], label=f"{algo} speedup")
    axes[1].axhline(8, color="gray", lw=0.6, ls=":")
    axes[1].set_xlabel("RMAT scale")
    axes[1].set_ylabel("speedup vs serial (8 threads)")
    ax2 = axes[1].twinx()
    ax2.plot(serial.index, serial.ms * 1e6 / serial.m, "s--", color="#7f7f7f", label="serial ns / edge")
    ax2.set_ylabel("serial ns per edge")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    axes[1].set_title("Speedup grows once the graph leaves the cache")
    save(fig, "a5_scale")

    # 11. level shape across scales
    lv = ex[(ex.kind == "level") & (ex.suite == "scale")].copy()
    graphs = ex[ex.kind == "graph"].set_index("graph")
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    names = sorted(lv.graph.unique(), key=lambda s: int(s.replace("rmat_s", "")))
    for i, gname in enumerate(names):
        g = lv[lv.graph == gname].sort_values("level")
        n = graphs.loc[gname, "n"]
        ax.plot(g.level, g.frontier / n, "o-", ms=3, color=cmap(i / max(1, len(names) - 1)), label=gname.replace("rmat_s", "scale "))
    ax.set_yscale("log")
    ax.set_xlabel("BFS level")
    ax.set_ylabel("frontier / n")
    ax.set_title("RMAT level shape is almost scale-free")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "a5_scale_levels")

    # 12. edge-range splitting
    ed = bfs[bfs.suite == "edges"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, gname in zip(axes, ("star_2m", "rmat_1m_8m", "er_1m_4m")):
        sub = ed[ed.graph == gname]
        base = sub[sub.algo == "serial"].ms.iloc[0]
        for algo, color in (("local", C["local"]), ("edges", "#9467bd"), ("hybrid", C["hybrid"]), ("hybrid_edges", "#8c564b")):
            a = sub[sub.algo == algo].sort_values("threads")
            ax.errorbar(a.threads, a.med, yerr=[a.med - a.lo, a.hi - a.med], fmt="o-", capsize=3, color=color, label=algo)
        ax.axhline(base, color=C["serial"], ls="--", label=f"serial {base:.1f} ms")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("threads")
        ax.set_ylabel("ms (median, min-max)")
        ax.set_title(gname)
        ax.legend(fontsize=7)
    fig.suptitle("Splitting the frontier by edges balances the hub, but per-vertex claim cost stays")
    save(fig, "a5_edges")

    # 13. independent graph seeds
    sd = bfs[bfs.suite == "seed"]
    fig, ax = plt.subplots(figsize=(8, 4))
    configs = [("serial", 1), ("local", 4), ("local", 8), ("bottom", 4), ("bottom", 8), ("hybrid", 4), ("hybrid", 8)]
    x = np.arange(len(configs))
    for j, seed in enumerate(sorted(sd.seed.unique())):
        meds, los, his = [], [], []
        for algo, th in configs:
            r = sd[(sd.seed == seed) & (sd.algo == algo) & (sd.threads == th)].iloc[0]
            meds.append(r.med)
            los.append(r.med - r.lo)
            his.append(r.hi - r.med)
        ax.errorbar(x + (j - 1) * 0.2, meds, yerr=[los, his], fmt="o", capsize=3, label=f"seed {int(seed)}")
    ax.set_xticks(x, [f"{a}\n{t} thr" for a, t in configs])
    ax.set_yscale("log")
    ax.set_ylabel("ms (median of 7, min-max)")
    ax.set_title("Three RMAT graphs (2^20, 8M edges) with different seeds")
    ax.legend(fontsize=8)
    save(fig, "a5_seeds")


if __name__ == "__main__":
    main()
