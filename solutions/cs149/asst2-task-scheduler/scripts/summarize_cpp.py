"""Print the tables cited in README / report from the pooled raw jsonl files.

Usage: python scripts/summarize_cpp.py [--platform server12c] > artifacts/raw/cpp/summary_server12c.md
All values are medians over every repetition of every campaign (server12c, _r2, _r3).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import plot_cpp as P  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)


def med(d: pd.DataFrame, col: str) -> pd.Series:
    return d[col].apply(lambda v: float(np.median(v)))


def md(df: pd.DataFrame, fmt: str = "{:.3g}") -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for i, row in df.iterrows():
        lines.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                                    for v in row]) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="server12c")
    P.PLAT = ap.parse_args().platform
    print(f"# Summary of {P.PLAT} (+_r2, _r3), medians over all repetitions\n")

    o = P.jl("overhead")
    if not o.empty:
        o["us"] = med(o, "us_per_launch")
        Pm = o.threads.max()
        print(f"## Launch overhead, µs per run(), {Pm} threads, empty tasks (overhead.jsonl)\n")
        print(md(o[o.threads == Pm].pivot(index="tasks", columns="system", values="us")[P.SYS]), "\n")
        print("## Launch overhead vs threads, 64 empty tasks\n")
        print(md(o[o.tasks == 64].pivot(index="threads", columns="system", values="us")[P.SYS]), "\n")
        print("## cpu_per_wall during overhead loop, 64 tasks\n")
        print(md(o[o.tasks == 64].pivot(index="threads", columns="system", values="cpu_per_wall")[P.SYS]), "\n")

    t = P.jl("tests")
    if not t.empty:
        t["ms"] = med(t, "wall_s") * 1e3
        t["cores"] = t.cpu_s.apply(np.median) / med(t, "wall_s")
        print(f"## Official tests, ms, {t.threads.iloc[0]} threads (tests.jsonl); all correct = {bool(t.correct.min())}\n")
        print(md(t.pivot(index="test", columns="system", values="ms")[P.SYS]), "\n")
        print("## Official tests, CPU s / wall s\n")
        print(md(t.pivot(index="test", columns="system", values="cores")[P.SYS], "{:.2f}"), "\n")

    s = P.jl("scaling")
    if not s.empty:
        s["ms"] = med(s, "wall_s") * 1e3
        ser = s[s.system == "serial"].set_index("test").ms
        s = s[s.system != "serial"].copy()
        s["speedup"] = s.ms.values / 1.0
        s["speedup"] = [ser[a] / b for a, b in zip(s.test, s.ms)]
        print("## Serial ms (scaling.jsonl)\n")
        print(ser.round(2).to_string(), "\n")
        print("## Speedup vs serial\n")
        print(md(s.pivot_table(index=["test", "threads"], columns="system", values="speedup"), "{:.2f}"), "\n")

    i = P.jl("idle")
    if not i.empty:
        print("## Idle pool: cores burned (idle.jsonl)\n")
        print(md(i.pivot(index="threads", columns="system", values="cores_burned")[P.SYS], "{:.2f}"), "\n")

    v = P.jl("oversub")
    if not v.empty:
        v["ms"] = med(v, "wall_s") * 1e3
        v["cpu"] = v.cpu_s.apply(np.median)
        print("## Oversubscription, ms (scale 0.25) (oversub.jsonl)\n")
        print(md(v.pivot_table(index=["test", "mult"], columns="system", values="ms")), "\n")
        print("## Oversubscription, CPU s\n")
        print(md(v.pivot_table(index=["test", "mult"], columns="system", values="cpu")), "\n")

    g = P.jl("granularity")
    if not g.empty:
        g["ms"] = med(g, "wall_s") * 1e3
        print("## Granularity, ms per run(), total 2e8 LCG steps (granularity.jsonl)\n")
        print(md(g.pivot(index="tasks", columns="system", values="ms")), "\n")
    g11 = P.jl("granularity_11workers", [P.PLAT])
    if not g11.empty:
        g11["ms"] = med(g11, "wall_s") * 1e3
        print("## Granularity with 11 workers (granularity_11workers.jsonl)\n")
        print(md(g11.pivot(index="tasks", columns="system", values="ms")), "\n")

    d = P.jl("dag")
    if not d.empty:
        d["ms"] = med(d, "wall_s") * 1e3
        print("## DAG shapes, ms (dag.jsonl)\n")
        print(md(d.pivot_table(index=["shape", "span_launches"], columns="system", values="ms")), "\n")
    dt = P.jl("dag_tasks")
    if not dt.empty:
        dt["ms"] = med(dt, "wall_s") * 1e3
        pv = dt.pivot(index="tasks", columns="shape", values="ms")
        pv["speedup"] = pv.indep_sync / pv.indep
        print("## Sync-per-launch vs async, ms (dag_tasks.jsonl)\n")
        print(md(pv), "\n")

    f = P.jl("false_sharing")
    if not f.empty and "op" in f:
        f["ns"] = med(f, "ns_per_inc")
        print("## False sharing, ns per increment (false_sharing.jsonl)\n")
        print(md(f.pivot_table(index=["placement", "threads"], columns=["op", "layout"], values="ns")), "\n")

    p = P.jl("dispatch")
    if not p.empty:
        p["ns"] = med(p, "ns_per_item")
        print("## Dispatch, ns per item (dispatch.jsonl)\n")
        print(md(p.pivot_table(index=["work", "threads"], columns=["scheme", "chunk"], values="ns")), "\n")

    for name in ("caller_core", "smt"):
        c = P.jl(name, [P.PLAT])
        if not c.empty:
            c["ms"] = med(c, "wall_s") * 1e3
            print(f"## {name}, ms ({name}.jsonl)\n")
            print(md(c.pivot_table(index="test", columns=["system", "threads"], values="ms")), "\n")


if __name__ == "__main__":
    main()
