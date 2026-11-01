"""Produce every CSV/figure cited in the README. CPU only, ~1-2 minutes."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "code"))

from tfkernels.attention import IOCounter, flash_backward, flash_forward, naive_attention  # noqa: E402
from tfkernels.conv1d import conv_direct, conv_fft, conv_loop, conv_windows  # noqa: E402
from tfkernels.io_model import (attention_io_row, decode_attention_intensity, flash_blocks,  # noqa: E402
                                flash_elements)
from tfkernels.recall import RecallTask, build_hand_coded, recall_accuracy  # noqa: E402

RAW = LAB / "artifacts" / "raw"
FIG = LAB / "artifacts" / "figures"
SRAM = 100 * 1024  # bytes of on-chip SRAM per SM assumed by the IO model (bf16)


def write_csv(name: str, rows: list[dict]) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {name} ({len(rows)} rows)")


def best_time(fn, repeats: int = 3, min_time: float = 0.05) -> float:
    best = float("inf")
    for _ in range(repeats):
        n, t0 = 0, time.perf_counter()
        while True:
            fn()
            n += 1
            dt = time.perf_counter() - t0
            if dt >= min_time:
                break
        best = min(best, dt / n)
    return best


def exp_attention_correctness() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for (b, h, t, d, br, bc) in [(1, 1, 128, 64, 64, 64), (2, 4, 255, 64, 64, 48),
                                 (1, 2, 512, 64, 128, 100), (1, 1, 1024, 128, 64, 64)]:
        for causal in (False, True):
            q, k, v = (rng.standard_normal((b, h, t, d)) for _ in range(3))
            do = rng.standard_normal(q.shape)
            qt, kt, vt = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
            ot = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
            ot.backward(torch.tensor(do))
            o, lse = flash_forward(q, k, v, causal, br, bc)
            dq, dk, dv = flash_backward(q, k, v, o, do, lse, causal, br, bc)
            t_naive = best_time(lambda: naive_attention(q, k, v, causal), repeats=3)
            t_flash = best_time(lambda: flash_forward(q, k, v, causal, br, bc), repeats=3)
            rows.append({"B": b, "H": h, "T": t, "d": d, "Br": br, "Bc": bc, "causal": causal,
                         "err_o": float(np.abs(o - ot.detach().numpy()).max()),
                         "err_dq": float(np.abs(dq - qt.grad.numpy()).max()),
                         "err_dk": float(np.abs(dk - kt.grad.numpy()).max()),
                         "err_dv": float(np.abs(dv - vt.grad.numpy()).max()),
                         "naive_S_mb_fp64": b * h * t * t * 8 / 1e6,
                         "flash_S_block_mb_fp64": b * h * br * bc * 8 / 1e6,
                         "numpy_naive_ms": t_naive * 1e3, "numpy_flash_ms": t_flash * 1e3})
    write_csv("attention_correctness.csv", rows)


def exp_io() -> None:
    rows = []
    ts = [2 ** i for i in range(9, 17)]
    for d in (64, 128):
        for causal in (False, True):
            rows += [attention_io_row(t, d, SRAM, 2, causal) for t in ts]
    write_csv("attention_io.csv", rows)

    br, bc = flash_blocks(SRAM, 64)
    rng = np.random.default_rng(1)
    ver = []
    for t in (256, 512, 1024, 2048):
        for causal in (False, True):
            q, k, v = (rng.standard_normal((t, 64)) for _ in range(3))
            io = IOCounter()
            flash_forward(q, k, v, causal, br, bc, io=io)
            ver.append({"T": t, "d": 64, "Br": br, "Bc": bc, "causal": causal, "counted_elements": io.total,
                        "model_elements": flash_elements(t, 64, br, bc, causal),
                        "kv_blocks_visited": io.blocks_visited, "kv_blocks_skipped": io.blocks_skipped})
    write_csv("attention_io_counter_check.csv", ver)

    sram_rows = [attention_io_row(8192, 64, kib * 1024, 2, False) for kib in (16, 32, 64, 100, 192, 256)]
    write_csv("attention_io_vs_sram.csv", sram_rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d, c in ((64, "C0"), (128, "C1")):
        nc = [r for r in rows if r["d"] == d and not r["causal"]]
        ca = [r for r in rows if r["d"] == d and r["causal"]]
        ax[0].loglog(ts, [r["naive_mb"] for r in nc], "o-", color=c, base=2, label=f"naive d={d}")
        ax[0].loglog(ts, [r["flash_mb"] for r in nc], "s--", color=c, base=2, label=f"flash d={d}")
        ax[0].loglog(ts, [r["flash_mb"] for r in ca], "^:", color=c, base=2, label=f"flash causal d={d}")
        ax[1].semilogx(ts, [r["ratio"] for r in nc], "o-", color=c, base=2, label=f"d={d}")
        ax[1].semilogx(ts, [r["ratio"] for r in ca], "^:", color=c, base=2, label=f"d={d} (flash causal)")
    ax[0].set(xlabel="sequence length T", ylabel="HBM MB (bf16, 1 head, fwd)", title="Attention HBM traffic")
    ax[1].set(xlabel="sequence length T", ylabel="naive / flash bytes", title=f"Traffic ratio, SRAM={SRAM // 1024} KiB")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "attention_io.png", dpi=130)
    plt.close(fig)


def exp_decode() -> None:
    write_csv("decode_attention_intensity.csv", decode_attention_intensity("llama3-8b", "h100", 4096))


def exp_conv() -> None:
    rng = np.random.default_rng(2)
    c = 16
    rows = []
    cases = [(mode, n, k) for n in [2 ** i for i in range(6, 17)] for mode, k in (("K=N", n), ("K=4", 4))]
    cases += [("N=16384", 16384, k) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)]
    for mode, n, k in cases:
        x, h = rng.standard_normal((c, n)), rng.standard_normal((c, k))
        ref = conv_fft(x, h)
        entry = {"mode": mode, "C": c, "N": n, "K": k}
        for name, fn, ok in (("loop", conv_loop, n * min(k, 64) <= 2 ** 16),
                             ("direct", conv_direct, n * k <= 2 ** 26),
                             ("windows", conv_windows, n * k <= 2 ** 24),
                             ("fft", conv_fft, True)):
            if ok:
                entry[f"{name}_ms"] = best_time(lambda: fn(x, h), repeats=3) * 1e3
                if name != "fft":
                    assert np.allclose(fn(x, h), ref, atol=1e-8 * max(1, k))
            else:
                entry[f"{name}_ms"] = ""
        rows.append(entry)
        print(f"  conv {mode:<7} N={n:<6} K={k:<6} " + " ".join(
            f"{kk}={vv:.3g}" for kk, vv in entry.items() if kk.endswith("_ms") and vv != ""))
    write_csv("conv1d_timing.csv", rows)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, mode in enumerate(("K=N", "K=4", "N=16384")):
        rr = [r for r in rows if r["mode"] == mode]
        xkey = "K" if mode == "N=16384" else "N"
        for name, st in (("loop", "x-"), ("direct", "o-"), ("windows", "s-"), ("fft", "^-")):
            pts = [(r[xkey], r[f"{name}_ms"]) for r in rr if r[f"{name}_ms"] != ""]
            if pts:
                ax[i].loglog(*zip(*pts), st, base=2, label=name)
        ax[i].set(xlabel="kernel length K" if xkey == "K" else "sequence length N",
                  ylabel="ms (C=16, float64)", title=f"Depthwise causal conv, {mode}")
        ax[i].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "conv1d_timing.png", dpi=130)
    plt.close(fig)


def exp_recall() -> None:
    rows = []
    for pairs in (4, 8, 16, 32):
        task = RecallTask(n_keys=max(16, pairs), n_values=16, n_pairs=pairs)
        for beta in (1.0, 2.0, 4.0, 8.0, 16.0, 30.0):
            acc = recall_accuracy(build_hand_coded(task, beta, beta), task, 5000, seed=100 + pairs)
            rows.append({"n_pairs": pairs, "n_keys": task.n_keys, "n_values": 16, "seq_len": task.seq_len,
                         "beta": beta, "accuracy": acc, "n_test": 5000})
    write_csv("recall_accuracy.csv", rows)

    task = RecallTask(n_keys=16, n_values=16, n_pairs=6)
    seq, tgt = task.sample(1, np.random.default_rng(7))
    _, (a1, a2) = build_hand_coded(task)(seq, return_attn=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    labels = [("BOS" if t == 0 else (f"k{t}" if t <= 16 else f"v{t - 17}")) for t in seq[0].tolist()]
    for a, att, title in ((ax[0], a1, "layer 1: previous-token head"), (ax[1], a2, "layer 2: match head")):
        a.imshow(att[0].numpy(), cmap="viridis", vmin=0, vmax=1)
        a.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
        a.set_yticks(range(len(labels)), labels, fontsize=6)
        a.set(title=title, xlabel="key position", ylabel="query position")
    for pairs in (4, 8, 16, 32):
        rr = [r for r in rows if r["n_pairs"] == pairs]
        ax[2].semilogx([r["beta"] for r in rr], [r["accuracy"] for r in rr], "o-", base=2, label=f"{pairs} pairs")
    ax[2].set(xlabel="attention sharpness beta", ylabel="recall accuracy", title="Hand-set weights", ylim=(0, 1.05))
    ax[2].legend(fontsize=8)
    fig.suptitle(f"target = v{int(tgt[0]) - 17}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "recall_attention.png", dpi=130)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    exp_attention_correctness()
    exp_io()
    exp_decode()
    exp_conv()
    exp_recall()
    meta = {"machine": platform.processor(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "sram_bytes_model": SRAM,
            "elapsed_s": round(time.perf_counter() - t0, 1)}
    (RAW / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"done in {meta['elapsed_s']} s")


if __name__ == "__main__":
    main()
