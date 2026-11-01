"""The mandelbrot_chunked scaling gap: server (3 rounds) against a quiet local 4-core control.

Writes artifacts/figures/cpp_mandel_anomaly.{png,pdf} and prints the numbers quoted in the report.
CPU efficiency = process CPU seconds / serial seconds for the same work. For the spinning pools the
caller spins for the whole wall time, so its share (one core x wall) is subtracted first.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw" / "cpp"
FIG = ROOT / "artifacts" / "figures"
SYSTEMS = ["spawn", "spinning", "sleeping", "async_sleeping"]
COLORS = {"spawn": "#ff7f0e", "spinning": "#d62728", "sleeping": "#1f77b4", "async_sleeping": "#2ca02c", "async_spinning": "#9467bd"}


def load(path):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("test") == "mandelbrot_chunked"]


def efficiency(r, serial_s):
    cpu = r["cpu_med_s"]
    if "spinning" in r["system"]:
        cpu -= r["wall_med_s"]
    return cpu / serial_s


def main():
    rounds = [load(RAW / d / "scaling.jsonl") for d in ("server12c", "server12c_r2", "server12c_r3")]
    local = load(RAW / "local4c" / "scaling_mandel.jsonl")
    cpu0 = load(RAW / "local4c" / "scaling_mandel_cpu0.jsonl")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for s in SYSTEMS:
        th = sorted({r["threads"] for r in rounds[0] if r["system"] == s})
        sp, ef = [], []
        for t in th:
            sp_r, ef_r = [], []
            for rows in rounds:
                serial = next(r for r in rows if r["system"] == "serial")["wall_med_s"]
                r = next(r for r in rows if r["system"] == s and r["threads"] == t)
                sp_r.append(serial / r["wall_med_s"])
                ef_r.append(efficiency(r, serial))
            sp.append(np.median(sp_r))
            ef.append(np.median(ef_r))
            print(f"server {s:15s} th={t:2d} speedup {np.median(sp_r):5.2f} cpu/serial {np.median(ef_r):5.2f}")
        axes[0].plot(th, sp, "o-", color=COLORS[s], label=s)
        axes[1].plot(th, ef, "o-", color=COLORS[s], label=s)
    axes[0].plot([1, 12], [1, 12], ":", color="gray", lw=0.8)
    axes[0].set_title("Server (12 pinned cores, median of 3 rounds)")
    axes[0].set_xlabel("worker threads")
    axes[0].set_ylabel("speedup vs serial")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color="gray", lw=0.8)
    axes[1].set_title("Server: CPU seconds per unit of work")
    axes[1].set_xlabel("worker threads")
    axes[1].set_ylabel("worker CPU time / serial time")
    axes[1].legend(fontsize=8)

    serial_local = next(r for r in local if r["system"] == "serial" and r["threads"] == 1)["wall_med_s"]
    for s in SYSTEMS + ["async_spinning"]:
        rs = sorted((r for r in local if r["system"] == s), key=lambda r: r["threads"])
        axes[2].plot([r["threads"] for r in rs], [serial_local / r["wall_med_s"] for r in rs], "o-", color=COLORS[s], label=s)
        for r in rs:
            print(f"local  {s:15s} th={r['threads']} speedup {serial_local / r['wall_med_s']:5.2f} cpu/serial {efficiency(r, serial_local):5.2f}")
    for r in cpu0:
        print(f"local cpu0 {r['system']:15s} {r['wall_med_s'] * 1e3:7.1f} ms ({r['wall_med_s'] / serial_local:.3f}x serial)")
    axes[2].plot([1, 4], [1, 4], ":", color="gray", lw=0.8)
    worst = max(r["wall_med_s"] for r in cpu0)
    axes[2].scatter([1] * len(cpu0), [serial_local / r["wall_med_s"] for r in cpu0], marker="x", color="k",
                    label=f"1 worker pinned to CPU 0 (all systems, >= {serial_local / worst:.2f})")
    axes[2].set_title("Local 4-core control (CPUs 2,4,6,0)")
    axes[2].set_xlabel("worker threads")
    axes[2].set_ylabel("speedup vs serial")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "cpp_mandel_anomaly.png", dpi=140)
    fig.savefig(FIG / "cpp_mandel_anomaly.pdf")
    print("wrote cpp_mandel_anomaly")


if __name__ == "__main__":
    main()
