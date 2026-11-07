"""Figures (PNG + PDF) and summary JSON from artifacts/raw/gpu/*.jsonl; also re-runs the lab's
roofline scheduler model with measured RTX 5090 numbers for a model-vs-measurement comparison.

python scripts/plot_gpu.py      # local, no GPU needed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LAB.parents[2]), str(LAB / "code"), str(LAB.parents[2] / "tools")]
import mlsys_calc  # noqa: E402
from mlsys_harness.devices import get  # noqa: E402
from servesim.scheduler import CostModel, simulate_continuous, simulate_static, summarize  # noqa: E402
from servesim.workload import poisson_arrivals, sample_lengths  # noqa: E402

RAW = LAB / "artifacts" / "raw" / "gpu"
FIG = LAB / "artifacts" / "figures"
GB = 1e9
POL = ["static", "continuous", "chunked"]
POL_LABEL = {"static": "static batching", "continuous": "continuous (Orca/vLLM)", "chunked": "chunked prefill (Sarathi)"}
COL = {"static": "#c44e52", "continuous": "#4c72b0", "chunked": "#55a868"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
QWEN = mlsys_calc.MODELS["qwen2.5-0.5b"]


def load(name: str) -> list[dict]:
    p = RAW / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=140)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def measured_gpu(summary: dict) -> mlsys_calc.GPU:
    spec = get("rtx5090")
    return mlsys_calc.GPU("RTX 5090 (measured)",
                          summary.get("gemm_tflops", spec.spec_bf16_flops / 1e12) * 1e12,
                          summary.get("copy_gbps", spec.reference_copy_bw / 1e9) * 1e9, 32e9, 64e9)


# --------------------------------------------------------------------------- roofline
def fig_roofline(summary: dict) -> None:
    rows = load("roofline")
    if not rows:
        return
    for r in rows:
        if r["kind"] == "probe" and r["what"].startswith("d2d"):
            summary["copy_gbps"] = r["gbps"]
        if r["kind"] == "probe" and r["what"].startswith("gemm"):
            summary["gemm_tflops"] = r["tflops"]
        if r["kind"] == "model":
            summary["model"] = r
    gpu = measured_gpu(summary)
    W = summary["model"]["weight_bytes"]
    kvt = summary["model"]["kv_bytes_per_token"]
    N = summary["model"]["params"]
    dec = [r for r in rows if r["kind"] == "decode"]
    pre = [r for r in rows if r["kind"] == "prefill"]

    # decode tokens/s vs batch
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ctxs = sorted({r["ctx"] for r in dec})
    knees = {}
    for ctx, col in zip(ctxs, plt.cm.viridis(np.linspace(0, 0.85, len(ctxs)))):
        rr = sorted([r for r in dec if r["ctx"] == ctx], key=lambda r: r["batch"])
        B = np.array([r["batch"] for r in rr])
        ax[0].loglog(B, [r["tok_s_graph"] for r in rr], "o-", color=col, ms=4, label=f"ctx={ctx} CUDA graph")
        ax[0].loglog(B, [r["tok_s_wall"] for r in rr], "x:", color=col, ms=4, lw=0.8)
        model = [B_ / mlsys_calc.roofline_time(2 * N * B_ + 4 * QWEN.layers * QWEN.d_model * B_ * (ctx + 1),
                                               W + B_ * (ctx + 1) * kvt, gpu)[0] for B_ in B]
        ax[0].loglog(B, model, "--", color=col, lw=0.8)
        t1 = rr[0]["t_graph_s"]
        knee = next((r["batch"] for r in rr if r["t_graph_s"] > 2 * t1), None)
        knees[ctx] = knee
    ax[0].set_xlabel("decode batch B")
    ax[0].set_ylabel("tokens/s")
    ax[0].set_title("Decode throughput: graph (o), eager wall (x), roofline w/ measured 5090 (--)")
    ax[0].legend(fontsize=7)
    summary["decode_knee_batch_2x_step"] = knees
    summary["decode"] = [{k: r[k] for k in ("batch", "ctx", "t_graph_s", "t_wall_s", "t_eager_s", "tok_s_graph",
                                            "tok_s_wall", "achieved_gbps_graph", "intensity")} for r in dec]
    # roofline plot
    I = np.logspace(-0.5, 3.5, 100)
    roof = np.minimum(gpu.bf16_flops, I * gpu.hbm_bw) / 1e12
    ax[1].loglog(I, roof, "k-", lw=1.2, label=f"5090 measured roof ({gpu.hbm_bw/1e12:.2f} TB/s, {gpu.bf16_flops/1e12:.0f} TF)")
    h = mlsys_calc.GPUS["h100"]
    ax[1].loglog(I, np.minimum(h.bf16_flops, I * h.hbm_bw) / 1e12, "k:", lw=1, label="H100 roof (lab model)")
    for ctx, col in zip(ctxs, plt.cm.viridis(np.linspace(0, 0.85, len(ctxs)))):
        rr = [r for r in dec if r["ctx"] == ctx]
        ax[1].loglog([r["intensity"] for r in rr], [r["flops"] / r["t_graph_s"] / 1e12 for r in rr], "o", color=col,
                     ms=4, label=f"decode ctx={ctx}")
    for r in pre:
        tok = r["batch"] * r["seq"]
        inten = (2 * N * tok) / (W + tok * kvt)
        ax[1].loglog(inten, r["tflops"], "s", color="#c44e52", ms=4, label="prefill" if r is pre[0] else None)
    ax[1].axvline(gpu.ridge, color="gray", ls="--", lw=0.8)
    ax[1].text(gpu.ridge * 1.1, 0.5, f"ridge {gpu.ridge:.0f} FLOP/B", fontsize=7, color="gray")
    ax[1].set_xlabel("arithmetic intensity (FLOP / HBM byte)")
    ax[1].set_ylabel("achieved TFLOP/s")
    ax[1].set_title("Qwen2.5-0.5B bf16 on RTX 5090: roofline")
    ax[1].legend(fontsize=6, loc="lower right")
    save(fig, "srv_decode_roofline")

    # step time heatmap + graph speedup
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    Bs = sorted({r["batch"] for r in dec})
    grid = np.full((len(ctxs), len(Bs)), np.nan)
    for r in dec:
        grid[ctxs.index(r["ctx"]), Bs.index(r["batch"])] = r["t_graph_s"] * 1e3
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="magma")
    for i in range(len(ctxs)):
        for j in range(len(Bs)):
            if not np.isnan(grid[i, j]):
                ax[0].text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7,
                           color="w" if grid[i, j] < np.nanmax(grid) * 0.6 else "k")
    ax[0].set_xticks(range(len(Bs)), Bs)
    ax[0].set_yticks(range(len(ctxs)), ctxs)
    ax[0].set_xlabel("batch")
    ax[0].set_ylabel("context length")
    ax[0].set_title("Decode step time (ms, CUDA graph); blank = KV > 12 GB")
    fig.colorbar(im, ax=ax[0])
    for ctx, col in zip(ctxs, plt.cm.viridis(np.linspace(0, 0.85, len(ctxs)))):
        rr = sorted([r for r in dec if r["ctx"] == ctx], key=lambda r: r["batch"])
        ax[1].semilogx([r["batch"] for r in rr], [r["t_wall_s"] / r["t_graph_s"] for r in rr], "o-", color=col,
                       ms=4, label=f"ctx={ctx}")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_xlabel("batch")
    ax[1].set_ylabel("eager wall time / CUDA-graph time")
    ax[1].set_title("CPU launch overhead dominates small-batch decode")
    ax[1].legend(fontsize=7)
    save(fig, "srv_decode_heatmap_graph")

    # prefill
    if pre:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        p1 = sorted([r for r in pre if r["batch"] == 1], key=lambda r: r["seq"])
        ax.semilogx([r["seq"] for r in p1], [r["tflops"] for r in p1], "o-", label="batch 1: TFLOP/s")
        ax.axhline(gpu.bf16_flops / 1e12, color="k", ls=":", lw=1, label="best GEMM")
        ax2 = ax.twinx()
        ax2.semilogx([r["seq"] for r in p1], [r["tokens_per_s"] / 1e3 for r in p1], "s--", color="#dd8452",
                     label="batch 1: k tokens/s")
        ax2.grid(False)
        ax2.set_ylabel("k tokens/s")
        for r in pre:
            if r["batch"] > 1:
                ax.plot(r["seq"], r["tflops"], "^", color="#55a868", ms=6)
                ax.annotate(f"B={r['batch']}", (r["seq"], r["tflops"]), fontsize=7, xytext=(3, -8), textcoords="offset points")
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("achieved TFLOP/s (2N·tok + attention)")
        ax.set_title("Prefill is compute-bound once tokens ≳ ridge")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
        save(fig, "srv_prefill")
        summary["prefill"] = [{k: r[k] for k in ("batch", "seq", "time_s", "tokens_per_s", "tflops")} for r in pre]

    # model assumptions: H100 model vs 5090 model vs measured, ctx=512
    sel = sorted([r for r in dec if r["ctx"] == 512], key=lambda r: r["batch"])
    if sel:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        B = [r["batch"] for r in sel]
        for gname, g, ls in (("H100 (lab assumption)", mlsys_calc.GPUS["h100"], ":"),
                             ("RTX 5090 spec 1.79 TB/s / 209.5 TF",
                              mlsys_calc.GPU("5090", get("rtx5090").spec_bf16_flops, get("rtx5090").spec_hbm_bw, 32e9, 64e9), "--"),
                             ("RTX 5090 measured", gpu, "-")):
            ax.loglog(B, [mlsys_calc.decode_step_time(QWEN, g, 513, b)["time_s"] * 1e3 for b in B], ls, label=f"model: {gname}")
        ax.loglog(B, [r["t_graph_s"] * 1e3 for r in sel], "o", color="k", label="measured (CUDA graph)")
        ax.loglog(B, [r["t_wall_s"] * 1e3 for r in sel], "x", color="r", label="measured (eager, wall)")
        ax.set_xlabel("batch")
        ax.set_ylabel("decode step (ms), ctx=512")
        ax.set_title("decode_step_time model vs measurement")
        ax.legend(fontsize=7)
        save(fig, "srv_decode_model_vs_measured")
        summary["decode_model_ratio_ctx512"] = {r["batch"]: r["t_graph_s"] / mlsys_calc.decode_step_time(QWEN, gpu, 513, r["batch"])["time_s"]
                                                for r in sel}


# --------------------------------------------------------------------------- paged
def fig_paged(summary: dict) -> None:
    rows = load("paged")
    pt = [r for r in rows if r["kind"] == "paged_time"]
    if not pt:
        return
    shapes = sorted({(r["H"], r["Hk"], r["D"]) for r in pt})
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.5 * len(shapes), 4))
    axes = np.atleast_1d(axes)
    impls = [("triton_paged", "Triton paged (shuffled blocks)", "o-"), ("triton_contig_table", "Triton, contiguous blocks", "s--"),
             ("torch_gather", "torch gather + SDPA", "^-")]
    for ax, sh in zip(axes, shapes):
        rr = [r for r in pt if (r["H"], r["Hk"], r["D"]) == sh and r["batch"] == 64]
        for impl, lab, st in impls:
            q = sorted([r for r in rr if r["impl"] == impl], key=lambda r: r["block"])
            ax.plot([r["block"] for r in q], [r["time_s"] * 1e6 for r in q], st, label=lab)
        for impl, lab, ls in (("sdpa_padded_4096", "SDPA, contiguous padded to 4096", ":"),
                              ("sdpa_padded_batchmax", "SDPA, contiguous padded to batch max", "-.")):
            q = [r for r in rr if r["impl"] == impl]
            if q:
                ax.axhline(q[0]["time_s"] * 1e6, ls=ls, color="gray", label=lab)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([8, 16, 32, 64], [8, 16, 32, 64])
        ax.set_xlabel("block size (tokens)")
        ax.set_ylabel("one decode-attention call (µs)")
        ax.set_title(f"H={sh[0]}, Hk={sh[1]}, D={sh[2]}, batch 64, lab length mix")
        ax.legend(fontsize=6)
    save(fig, "srv_paged_blocksize")
    summary["paged"] = pt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for sh, mk in zip(shapes, "os"):
        for impl, col in (("triton_paged", "#4c72b0"), ("triton_contig_table", "#55a868"), ("torch_gather", "#c44e52"),
                          ("sdpa_padded_4096", "gray")):
            q = sorted([r for r in pt if (r["H"], r["Hk"], r["D"]) == sh and r["impl"] == impl and r["block"] in (0, 16)],
                       key=lambda r: r["batch"])
            ax.semilogx([r["batch"] for r in q], [r["eff_gbps"] for r in q], mk + "-", color=col, ms=4,
                        label=f"{impl} (H={sh[0]},Hk={sh[1]},D={sh[2]})")
    ax.axhline(summary.get("copy_gbps", 1500), color="k", ls=":", lw=1)
    ax.set_xlabel("batch (sequences)")
    ax.set_ylabel("useful KV bytes / time (GB/s)")
    ax.set_title("Effective KV bandwidth, block=16 (only live tokens counted)")
    ax.legend(fontsize=6)
    save(fig, "srv_paged_bandwidth")


# --------------------------------------------------------------------------- waste
def fig_waste(summary: dict) -> None:
    rows = load("waste")
    wm = [r for r in rows if r["kind"] == "waste_model"]
    ws = [r for r in rows if r["kind"] == "waste_measured"]
    churn = next((r for r in rows if r["kind"] == "churn"), None)
    if not ws:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lab = [("contiguous_max4096", 0, "contig\nmax 4096"), ("contiguous_exact_final_len", 0, "contig\nexact final"),
           ("paged", 8, "paged\n8"), ("paged", 16, "paged\n16"), ("paged", 32, "paged\n32"), ("paged", 64, "paged\n64")]
    x = np.arange(len(lab))
    meas = [next(r for r in ws if r["scheme"] == s and r["block"] == b) for s, b, _ in lab]
    mod = [next((r for r in wm if r["scheme"] == s and (s != "paged" or r["block"] == b)), None) for s, b, _ in lab]
    ax[0].bar(x - 0.2, [r["waste_frac"] * 100 for r in meas], 0.4, label="measured on GPU (snapshot)")
    ax[0].bar(x + 0.2, [r["waste_frac"] * 100 if r else 0 for r in mod], 0.4, label="waste.py (lifetime avg)")
    for xi, r in zip(x, meas):
        ax[0].text(xi - 0.2, r["waste_frac"] * 100 + 1, f"{r['waste_frac']*100:.1f}", ha="center", fontsize=7)
    ax[0].set_xticks(x, [l for *_, l in lab], fontsize=7)
    ax[0].set_ylabel("KV memory wasted [%]")
    ax[0].set_title("KV waste: real allocations vs lab model (4 GiB budget)")
    ax[0].legend(fontsize=7)
    ax[1].bar(x - 0.2, [r["concurrent_seqs"] for r in meas], 0.4, label="measured: sequences that fit")
    ax[1].bar(x + 0.2, [r["concurrent_seqs"] if r else 0 for r in mod], 0.4, label="waste.py prediction")
    for xi, r in zip(x, meas):
        ax[1].text(xi - 0.2, r["concurrent_seqs"] + 5, str(r["concurrent_seqs"]), ha="center", fontsize=7)
    ax[1].set_xticks(x, [l for *_, l in lab], fontsize=7)
    ax[1].set_ylabel("concurrent sequences in 4 GiB of KV")
    ax[1].set_title("Concurrency = budget / allocated tokens per sequence")
    ax[1].legend(fontsize=7)
    save(fig, "srv_waste_measured")
    summary["waste_measured"] = ws
    summary["waste_model"] = wm
    if churn:
        tr = np.array(churn["trace"], float)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(tr[:, 0], tr[:, 1] / GB, label="allocated (live tensors)")
        ax.plot(tr[:, 0], tr[:, 2] / GB, label="reserved by caching allocator")
        ax.set_xlabel("requests admitted")
        ax.set_ylabel("GB")
        ax.set_title(f"Exact-length per-request KV tensors, {churn['slots']} slots: external fragmentation")
        ax.legend(fontsize=7)
        save(fig, "srv_alloc_churn")
        summary["churn"] = {"reserved_over_alloc_mean": float(np.mean(tr[len(tr) // 4:, 2] / tr[len(tr) // 4:, 1])),
                            "reserved_over_alloc_max": float(np.max(tr[len(tr) // 4:, 2] / tr[len(tr) // 4:, 1])),
                            "segments": churn["segments"], "num_alloc_retries": churn["num_alloc_retries"],
                            "inactive_split_bytes": churn["inactive_split_bytes"]}


# --------------------------------------------------------------------------- engine
def engine_rows():
    out = {}
    for pol in POL:
        for r in load(f"engine_{pol}"):
            if r["kind"] in ("engine", "engine_requests", "engine_steps"):
                out.setdefault(pol, {}).setdefault(r["rate"], {})[r["kind"]] = r
    return out


def sim_with_5090(summary: dict, eng: dict) -> dict:
    """Lab roofline scheduler with measured 5090 numbers, same workload / KV pool as the engine."""
    gpu = measured_gpu(summary)
    cost = CostModel(QWEN, gpu)
    res = {}
    for pol, by_rate in eng.items():
        for rate, d in by_rate.items():
            e = d.get("engine")
            if not e:
                continue
            p, o = sample_lengths(e["n"], seed=0)
            arr = poisson_arrivals(e["n"], rate, seed=1)
            if pol == "static":
                sim = simulate_static(cost, arr, p, o, max_batch=e["static_batch"])
            else:
                sim = simulate_continuous(cost, arr, p, o, chunked=pol == "chunked", max_num_seqs=128,
                                          kv_tokens=e["num_blocks"] * e["block"])
            res[(pol, rate)] = summarize(sim, 1.0, 0.05)
    return res


def fig_engine(summary: dict) -> None:
    eng = engine_rows()
    if not eng:
        return
    sim = sim_with_5090(summary, eng)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for pol in POL:
        if pol not in eng:
            continue
        rates = sorted(r for r in eng[pol] if "engine" in eng[pol][r])
        E = [eng[pol][r]["engine"] for r in rates]
        S = [sim.get((pol, r)) for r in rates]
        ax[0].plot(rates, [e["throughput_tok_s"] for e in E], "o-", color=COL[pol], label=f"{POL_LABEL[pol]} measured")
        ax[0].plot(rates, [s["throughput_tok_s"] for s in S], "--", color=COL[pol], lw=0.8, label=f"{pol}: lab sim (5090 roofline)")
        ax[1].semilogy(rates, [e["ttft_p50_s"] for e in E], "o-", color=COL[pol], label=f"{pol} P50")
        ax[1].semilogy(rates, [e["ttft_p99_s"] for e in E], "^:", color=COL[pol], label=f"{pol} P99")
        ax[2].plot(rates, [e["tpot_mean_ms"] for e in E], "o-", color=COL[pol], label=f"{pol} mean TPOT")
        ax[2].plot(rates, [e["itl_p99_ms"] for e in E], "^:", color=COL[pol], label=f"{pol} P99 inter-token")
    ax[0].set_xlabel("arrival rate λ (req/s)")
    ax[0].set_ylabel("output tokens/s")
    ax[0].set_title("Throughput vs load")
    ax[0].legend(fontsize=6)
    ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("λ (req/s)")
    ax[1].set_ylabel("TTFT (s)")
    ax[1].set_title("Time to first token")
    ax[1].legend(fontsize=6)
    ax[2].axhline(50, color="k", ls=":", lw=0.8)
    ax[2].set_xlabel("λ (req/s)")
    ax[2].set_ylabel("ms")
    ax[2].set_title("TPOT and P99 inter-token latency")
    ax[2].legend(fontsize=6)
    save(fig, "srv_engine_load")
    summary["engine"] = {pol: {r: d["engine"] for r, d in by.items() if "engine" in d} for pol, by in eng.items()}
    summary["engine_sim_5090"] = {f"{p}@{r}": s for (p, r), s in sim.items()}

    # latency CDFs at the highest common rate
    common = sorted(set.intersection(*[set(eng[p]) for p in eng]))
    hi = common[-1] if common else None
    mid = common[len(common) // 2] if common else None
    if hi is not None:
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
        for j, (col, lab) in enumerate(((3, "TTFT (s)"), (4, "end-to-end latency (s)"), (6, "max inter-token gap (s)"))):
            for pol in POL:
                for rate, ls in ((mid, ":"), (hi, "-")):
                    d = eng.get(pol, {}).get(rate, {}).get("engine_requests")
                    if not d:
                        continue
                    v = np.sort([row[col] for row in d["rows"]])
                    ax[j].plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=COL[pol], label=f"{pol} λ={rate}")
            ax[j].set_xscale("log")
            ax[j].set_xlabel(lab)
            ax[j].set_ylabel("CDF")
            ax[j].legend(fontsize=6)
        fig.suptitle("Per-request latency CDFs (measured engine)", fontsize=10)
        save(fig, "srv_engine_cdf")

        # step timeline: decode stalls
        fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        for a, pol in zip(ax, ("continuous", "chunked")):
            d = eng.get(pol, {}).get(mid, {}).get("engine_steps")
            if not d:
                continue
            st = np.array(d["rows"], float)
            sl = slice(0, min(len(st), 600))
            a.bar(st[sl, 0] - st[sl, 1], st[sl, 1] * 1e3, width=st[sl, 1], align="edge",
                  color=np.where(st[sl, 3] > 0, "#c44e52", "#4c72b0"), lw=0)
            a.set_ylabel("iteration time (ms)")
            a.set_title(f"{POL_LABEL[pol]}, λ={mid}: red = iteration contains prefill tokens; "
                        f"max {st[:, 1].max()*1e3:.0f} ms, P99 {np.percentile(st[:, 1], 99)*1e3:.0f} ms")
        ax[-1].set_xlabel("engine time (s)")
        save(fig, "srv_engine_timeline")

    # measured step time vs cost model
    gpu = measured_gpu(summary)
    cost = CostModel(QWEN, gpu)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fits = {}
    for pol in ("continuous", "chunked", "static"):
        for rate, d in eng.get(pol, {}).items():
            if "engine_steps" not in d or rate != mid:
                continue
            cols = d["engine_steps"]["cols"]
            st = np.array(d["engine_steps"]["rows"], float)
            ci = {c: i for i, c in enumerate(cols)}
            pred = []
            for row in st:
                n_tok = row[ci["n_dec"]] + row[ci["n_prefill"]]
                pairs = row[ci["ctx_sum"]] + row[ci["prefill_pairs"]]
                flops = 2 * QWEN.params() * n_tok + 4 * QWEN.layers * QWEN.d_model * pairs
                nbytes = cost.weight_bytes + cost.kv_per_token * (row[ci["ctx_sum"]] + row[ci["prefill_kv"]])
                pred.append(mlsys_calc.roofline_time(flops, nbytes, gpu)[0])
            pred = np.array(pred)
            ax[0].loglog(pred * 1e3, st[:, 1] * 1e3, ".", ms=2, alpha=0.4, color=COL[pol], label=f"{pol} λ={rate}")
            A = np.vstack([np.ones_like(pred), pred]).T
            coef, *_ = np.linalg.lstsq(A, st[:, 1], rcond=None)
            fits[pol] = {"overhead_ms": coef[0] * 1e3, "slope": coef[1], "median_ratio": float(np.median(st[:, 1] / pred)),
                         "n_steps": len(st)}
            ntok = st[:, ci["n_dec"]] + st[:, ci["n_prefill"]]
            ax[1].loglog(ntok, st[:, 1] * 1e3, ".", ms=2, alpha=0.4, color=COL[pol], label=f"{pol}")
    abl = [r for r in load("engine_chunked_maskedsdpa") if r["kind"] == "engine_steps" and r["rate"] == mid]
    if abl:
        cols = abl[0]["cols"]
        st = np.array(abl[0]["rows"], float)
        ci = {c: i for i, c in enumerate(cols)}
        ntok = st[:, ci["n_dec"]] + st[:, ci["n_prefill"]]
        ax[1].loglog(ntok, st[:, 1] * 1e3, "x", ms=2, alpha=0.3, color="0.5", label="chunked, boolean-mask SDPA (ablation)")
        summary["engine_chunked_maskedsdpa"] = {r["rate"]: r for r in load("engine_chunked_maskedsdpa") if r["kind"] == "engine"}
    lim = [0.3, 3000]
    ax[0].plot(lim, lim, "k:", lw=1, label="y = x")
    ax[0].set_xlabel("roofline prediction with measured 5090 (ms)")
    ax[0].set_ylabel("measured iteration (ms)")
    ax[0].set_title("Engine iterations vs lab cost model")
    ax[0].legend(fontsize=6, markerscale=4)
    ax[1].set_xlabel("tokens in iteration (decode + prefill)")
    ax[1].set_ylabel("measured iteration (ms)")
    ax[1].set_title("Iteration time vs batched tokens")
    ax[1].legend(fontsize=6, markerscale=4)
    save(fig, "srv_engine_costmodel")
    summary["engine_costmodel_fit"] = fits


def fig_stall(summary: dict) -> None:
    """10% of prompts replaced by 3072-token prompts: generation stalls with and without chunked prefill."""
    eng = {}
    for pol in ("continuous", "chunked"):
        for r in load(f"engine_{pol}_long"):
            if r["kind"] in ("engine", "engine_requests", "engine_steps"):
                eng.setdefault(pol, {}).setdefault(r["rate"], {})[r["kind"]] = r
    if len(eng) < 2:
        return
    rates = sorted(set(eng["continuous"]) & set(eng["chunked"]))
    hi = rates[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    for a, pol in zip(ax, ("continuous", "chunked")):
        st = np.array(eng[pol][hi]["engine_steps"]["rows"], float)
        sl = slice(0, min(len(st), 700))
        a.bar(st[sl, 0] - st[sl, 1], st[sl, 1] * 1e3, width=st[sl, 1], align="edge",
              color=np.where(st[sl, 3] > 0, "#c44e52", "#4c72b0"), lw=0)
        a.set_ylabel("iteration time (ms)")
        a.set_title(f"{POL_LABEL[pol]}, 10% prompts = 3072 tokens, λ={hi}: max {st[:, 1].max()*1e3:.0f} ms, "
                    f"P99 {np.percentile(st[:, 1], 99)*1e3:.0f} ms (red = has prefill)")
    ax[-1].set_xlabel("engine time (s)")
    save(fig, "srv_stall_timeline")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for pol in ("continuous", "chunked"):
        for rate, ls in zip(rates, (":", "-")):
            d = eng[pol][rate]["engine_requests"]
            rows = np.array(d["rows"], float)
            for j, col in ((0, 6), (1, 5)):
                v = np.sort(rows[:, col] * 1e3)
                ax[j].plot(v, np.arange(1, len(v) + 1) / len(v), ls, color=COL[pol], label=f"{pol} λ={rate}")
            if rate == hi:
                for long_, mk in ((False, "-"), (True, "--")):
                    sel = (rows[:, 1] >= 3072) == long_
                    v = np.sort(rows[sel, 3])
                    ax[2].plot(v, np.arange(1, len(v) + 1) / len(v), mk, color=COL[pol],
                               label=f"{pol}, {'3072-token' if long_ else 'other'} prompts")
    for a, lab in zip(ax, ("per-request max inter-token gap (ms)", "per-request TPOT (ms)", f"TTFT at λ={hi} (s)")):
        a.set_xlabel(lab)
        a.set_ylabel("CDF")
        a.legend(fontsize=6)
    ax[2].set_xscale("log")
    save(fig, "srv_stall_cdf")
    summary["engine_long"] = {pol: {r: d["engine"] for r, d in by.items()} for pol, by in eng.items()}
    for pol in eng:
        st = np.array(eng[pol][hi]["engine_steps"]["rows"], float)
        rows = np.array(eng[pol][hi]["engine_requests"]["rows"], float)
        lg = rows[:, 1] >= 3072
        summary["engine_long"][pol]["detail"] = {
            "rate": hi, "step_max_ms": st[:, 1].max() * 1e3, "step_p99_ms": float(np.percentile(st[:, 1], 99) * 1e3),
            "ttft_long_p50_s": float(np.median(rows[lg, 3])), "ttft_short_p50_s": float(np.median(rows[~lg, 3])),
            "max_itl_p50_ms": float(np.median(rows[:, 6]) * 1e3), "max_itl_p99_ms": float(np.percentile(rows[:, 6], 99) * 1e3),
            "n_long": int(lg.sum())}


def main() -> int:
    summary: dict = {}
    for f in (fig_roofline, fig_paged, fig_waste, fig_engine, fig_stall):
        f(summary)
    v = load("verify")
    summary["verify"] = [r for r in v if r["kind"] != "env"]
    (RAW / "summary_gpu.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
