"""Back-of-envelope calculator for ML systems interviews.

Every function is a closed-form formula that should also be derivable on a
whiteboard; see interview/FORMULAS.md for the derivations.

Examples
--------
    python tools/mlsys_calc.py hw
    python tools/mlsys_calc.py train-mem --params 7e9 --dp 8 --zero 3
    python tools/mlsys_calc.py kv --model llama3-8b --seq 8192 --batch 16
    python tools/mlsys_calc.py decode --model llama3-8b --gpu h100 --seq 4096 --batch 1
    python tools/mlsys_calc.py allreduce --bytes 14e9 --gpus 8 --bw 450e9
    python tools/mlsys_calc.py bubble --stages 8 --micro 32
    python tools/mlsys_calc.py spec --alpha 0.8 --gamma 4
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GB = 1e9
TB = 1e12


@dataclass(frozen=True)
class GPU:
    name: str
    bf16_flops: float  # dense tensor-core FLOP/s
    hbm_bw: float  # bytes/s
    hbm_bytes: float
    link_bw: float  # NVLink (or PCIe) per-GPU bandwidth, one direction, bytes/s

    @property
    def ridge(self) -> float:
        """Arithmetic intensity (FLOP/byte) where compute time equals memory time."""
        return self.bf16_flops / self.hbm_bw


GPUS = {
    "a100": GPU("A100 SXM 80GB", 312e12, 2.039 * TB, 80 * GB, 300 * GB),
    "h100": GPU("H100 SXM 80GB", 989e12, 3.35 * TB, 80 * GB, 450 * GB),
    "h200": GPU("H200 SXM 141GB", 989e12, 4.8 * TB, 141 * GB, 450 * GB),
    "4090": GPU("RTX 4090 24GB", 165e12, 1.008 * TB, 24 * GB, 32 * GB),
}


@dataclass(frozen=True)
class Model:
    name: str
    layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab: int
    gated_mlp: bool = True
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def params(self) -> int:
        return transformer_params(self, self.tied_embeddings)


MODELS = {
    "gpt2-124m": Model("GPT-2 124M", 12, 768, 12, 12, 3072, 50257, gated_mlp=False,
                       tied_embeddings=True),
    "llama2-7b": Model("Llama-2-7B", 32, 4096, 32, 32, 11008, 32000),
    "llama3-8b": Model("Llama-3-8B", 32, 4096, 32, 8, 14336, 128256),
    "llama3-70b": Model("Llama-3-70B", 80, 8192, 64, 8, 28672, 128256),
    "qwen2.5-0.5b": Model("Qwen2.5-0.5B", 24, 896, 14, 2, 4864, 151936, tied_embeddings=True),
}


# --------------------------------------------------------------------------- parallel laws

def amdahl_speedup(parallel_fraction: float, workers: int) -> float:
    """S(n) = 1 / ((1-p) + p/n)."""
    if not 0.0 <= parallel_fraction <= 1.0 or workers < 1:
        raise ValueError("need 0<=p<=1 and n>=1")
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / workers)


def gustafson_speedup(parallel_fraction: float, workers: int) -> float:
    """Scaled speedup with problem size growing with n: S(n) = (1-p) + p*n."""
    return (1.0 - parallel_fraction) + parallel_fraction * workers


# --------------------------------------------------------------------------- roofline

def roofline_time(flops: float, bytes_moved: float, gpu: GPU) -> tuple[float, str]:
    """Lower-bound runtime max(F/peak, B/bw) and which roof binds."""
    t_compute = flops / gpu.bf16_flops
    t_memory = bytes_moved / gpu.hbm_bw
    return max(t_compute, t_memory), ("compute" if t_compute >= t_memory else "memory")


def gemm_intensity(m: int, n: int, k: int, bytes_per_el: int = 2) -> float:
    """FLOP/byte of C[m,n] = A[m,k] @ B[k,n] when each matrix crosses HBM once."""
    flops = 2 * m * n * k
    traffic = bytes_per_el * (m * k + k * n + m * n)
    return flops / traffic


# --------------------------------------------------------------------------- transformer

def transformer_params(model: Model, tied_embeddings: bool = False) -> int:
    d, L = model.d_model, model.layers
    kv_dim = model.n_kv_heads * model.head_dim
    attn = d * d + 2 * d * kv_dim + d * d  # Wq, Wk, Wv, Wo
    mlp = (3 if model.gated_mlp else 2) * d * model.d_ff
    norms = 2 * d
    embed = model.vocab * d * (1 if tied_embeddings else 2)
    return L * (attn + mlp + norms) + embed + d


def train_flops(params: float, tokens: float) -> float:
    """C ~= 6 N D (forward 2ND, backward 4ND); ignores attention's T^2 term."""
    return 6.0 * params * tokens


def mfu(tokens_per_s: float, params: float, peak_flops: float, n_gpus: int = 1) -> float:
    return 6.0 * params * tokens_per_s / (peak_flops * n_gpus)


def kv_cache_bytes(model: Model, seq: int, batch: int = 1, bytes_per_el: int = 2) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x tokens x bytes."""
    return 2 * model.layers * model.n_kv_heads * model.head_dim * seq * batch * bytes_per_el


def decode_step_time(model: Model, gpu: GPU, seq: int, batch: int,
                     bytes_per_el: int = 2) -> dict[str, float]:
    """Roofline bound for one decode step: read all weights once plus every KV cache."""
    n = model.params()
    weight_bytes = n * bytes_per_el
    kv = kv_cache_bytes(model, seq, batch, bytes_per_el)
    flops = 2 * n * batch + 4 * model.layers * model.d_model * seq * batch
    t, bound = roofline_time(flops, weight_bytes + kv, gpu)
    return {
        "time_s": t,
        "tokens_per_s": batch / t,
        "weight_gb": weight_bytes / GB,
        "kv_gb": kv / GB,
        "intensity": flops / (weight_bytes + kv),
        "bound": bound,
    }


def critical_batch(model: Model, gpu: GPU, bytes_per_el: int = 2) -> float:
    """Batch at which decode GEMMs reach the ridge (ignoring KV): 2NB / (bytes*N) = ridge."""
    return gpu.ridge * bytes_per_el / 2.0


# --------------------------------------------------------------------------- training memory

def training_memory_bytes(params: float, dp: int = 1, zero: int = 0) -> dict[str, float]:
    """Mixed-precision Adam: bf16 weights 2N, bf16 grads 2N, fp32 master+m+v 12N.

    ZeRO-1 shards the 12N optimizer states, ZeRO-2 also the gradients,
    ZeRO-3 also the weights. Activations are not included.
    """
    if zero not in (0, 1, 2, 3):
        raise ValueError("zero stage must be 0..3")
    w, g, o = 2 * params, 2 * params, 12 * params
    if zero >= 1:
        o /= dp
    if zero >= 2:
        g /= dp
    if zero >= 3:
        w /= dp
    return {"weights": w, "grads": g, "optimizer": o, "total": w + g + o}


# --------------------------------------------------------------------------- communication

def ring_allreduce_time(nbytes: float, gpus: int, bw: float, latency: float = 0.0) -> float:
    """Ring all-reduce: each GPU sends and receives 2(p-1)/p * S, in 2(p-1) steps."""
    if gpus < 2:
        return 0.0
    return 2 * (gpus - 1) / gpus * nbytes / bw + 2 * (gpus - 1) * latency


def allreduce_bus_bytes(nbytes: float, gpus: int) -> float:
    return 2 * (gpus - 1) / gpus * nbytes


def pipeline_bubble(stages: int, micro_batches: int, interleave: int = 1) -> float:
    """Idle fraction of GPipe / 1F1B: (p-1)/(m+p-1); interleaving divides the bubble by v."""
    bubble = (stages - 1) / interleave
    return bubble / (micro_batches + bubble)


# --------------------------------------------------------------------------- inference tricks

def speculative_expected_tokens(alpha: float, gamma: int) -> float:
    """Expected tokens per target forward with gamma drafts and i.i.d. acceptance alpha."""
    if alpha >= 1.0:
        return gamma + 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speculative_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """Walltime speedup when one draft forward costs c target forwards (Leviathan et al.)."""
    return speculative_expected_tokens(alpha, gamma) / (gamma * cost_ratio + 1)


def littles_law_concurrency(throughput: float, latency_s: float) -> float:
    return throughput * latency_s


# --------------------------------------------------------------------------- CLI

def _fmt_bytes(x: float) -> str:
    return f"{x / GB:,.2f} GB"


def _cmd_hw(_: argparse.Namespace) -> None:
    print(f"{'gpu':<16}{'bf16 TF/s':>10}{'HBM TB/s':>10}{'ridge F/B':>11}{'HBM GB':>8}")
    for g in GPUS.values():
        print(f"{g.name:<16}{g.bf16_flops / 1e12:>10.0f}{g.hbm_bw / TB:>10.2f}"
              f"{g.ridge:>11.0f}{g.hbm_bytes / GB:>8.0f}")
    print()
    print(f"{'model':<14}{'params':>9}{'KV/token':>11}")
    for m in MODELS.values():
        print(f"{m.name:<14}{m.params() / 1e9:>8.2f}B{kv_cache_bytes(m, 1) / 1024:>9.0f}KB")


def _cmd_train_mem(a: argparse.Namespace) -> None:
    mem = training_memory_bytes(a.params, a.dp, a.zero)
    for k, v in mem.items():
        print(f"{k:<10}{_fmt_bytes(v):>14}")


def _cmd_kv(a: argparse.Namespace) -> None:
    m = MODELS[a.model]
    print(f"{m.name}: {_fmt_bytes(kv_cache_bytes(m, a.seq, a.batch, a.bytes))} "
          f"for seq={a.seq}, batch={a.batch}")


def _cmd_decode(a: argparse.Namespace) -> None:
    r = decode_step_time(MODELS[a.model], GPUS[a.gpu], a.seq, a.batch)
    for k, v in r.items():
        print(f"{k:<14}{v if isinstance(v, str) else f'{v:,.4g}':>14}")
    print(f"{'crit. batch':<14}{critical_batch(MODELS[a.model], GPUS[a.gpu]):>14,.0f}")


def _cmd_allreduce(a: argparse.Namespace) -> None:
    t = ring_allreduce_time(a.bytes, a.gpus, a.bw, a.latency)
    print(f"bus bytes {_fmt_bytes(allreduce_bus_bytes(a.bytes, a.gpus))}, time {t * 1e3:.2f} ms")


def _cmd_bubble(a: argparse.Namespace) -> None:
    print(f"bubble = {pipeline_bubble(a.stages, a.micro, a.interleave):.3f}")


def _cmd_spec(a: argparse.Namespace) -> None:
    print(f"E[tokens/step] = {speculative_expected_tokens(a.alpha, a.gamma):.3f}, "
          f"speedup(c={a.c}) = {speculative_speedup(a.alpha, a.gamma, a.c):.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hw").set_defaults(fn=_cmd_hw)

    s = sub.add_parser("train-mem")
    s.add_argument("--params", type=float, required=True)
    s.add_argument("--dp", type=int, default=1)
    s.add_argument("--zero", type=int, default=0)
    s.set_defaults(fn=_cmd_train_mem)

    s = sub.add_parser("kv")
    s.add_argument("--model", choices=MODELS, default="llama3-8b")
    s.add_argument("--seq", type=int, default=4096)
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--bytes", type=int, default=2)
    s.set_defaults(fn=_cmd_kv)

    s = sub.add_parser("decode")
    s.add_argument("--model", choices=MODELS, default="llama3-8b")
    s.add_argument("--gpu", choices=GPUS, default="h100")
    s.add_argument("--seq", type=int, default=4096)
    s.add_argument("--batch", type=int, default=1)
    s.set_defaults(fn=_cmd_decode)

    s = sub.add_parser("allreduce")
    s.add_argument("--bytes", type=float, required=True)
    s.add_argument("--gpus", type=int, default=8)
    s.add_argument("--bw", type=float, default=450e9)
    s.add_argument("--latency", type=float, default=0.0)
    s.set_defaults(fn=_cmd_allreduce)

    s = sub.add_parser("bubble")
    s.add_argument("--stages", type=int, required=True)
    s.add_argument("--micro", type=int, required=True)
    s.add_argument("--interleave", type=int, default=1)
    s.set_defaults(fn=_cmd_bubble)

    s = sub.add_parser("spec")
    s.add_argument("--alpha", type=float, required=True)
    s.add_argument("--gamma", type=int, required=True)
    s.add_argument("--c", type=float, default=0.05)
    s.set_defaults(fn=_cmd_spec)

    args = p.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
