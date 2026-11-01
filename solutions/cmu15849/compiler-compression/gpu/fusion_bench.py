"""Q1: operator fusion on GPU -- eager vs torch.compile (inductor) vs handwritten Triton.

Parts (select with --part):
  chains   memory-bound chains x sizes x {eager, default, max-autotune}: time, kernels/call,
           eager HBM bytes (dispatch-mode count) vs ideal bytes, achieved bandwidth, compile time
  mlp      Transformer MLP (the lab's graph: x + W2 GELU(W1 LN(x)+b1)+b2) and a pre-LN block
  triton   handwritten fused RMSNorm / bias+GELU vs eager vs inductor
  dynamic  recompilation under changing sequence length: dynamic=False / automatic / mark_dynamic
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from common import RTX5090, Writer, bench, env_info, set_threads
import triton_kernels as tk

DT = torch.bfloat16
DEV = "cuda"

_VIEW_OPS = {"view", "_unsafe_view", "t", "transpose", "expand", "reshape", "permute", "unsqueeze",
             "squeeze", "slice", "select", "as_strided", "detach", "alias", "split", "unbind",
             "_reshape_alias", "split_with_sizes", "empty", "empty_like", "lift_fresh"}


class ByteCounter(TorchDispatchMode):
    """Eager traffic model: every non-view aten op reads its tensor inputs once and writes its outputs."""

    def __init__(self):
        super().__init__()
        self.bytes = 0
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.overloadpacket.__name__
        if name not in _VIEW_OPS:
            flat = list(args) + list((kwargs or {}).values())
            flat += out if isinstance(out, (tuple, list)) else [out]
            seen = set()
            for t in flat:
                if isinstance(t, torch.Tensor) and t.device.type == "cuda" and id(t) not in seen:
                    seen.add(id(t))
                    self.bytes += t.numel() * t.element_size()
            self.ops += 1
        return out


def eager_bytes(fn, *args) -> tuple[int, int]:
    with ByteCounter() as bc:
        fn(*args)
    return bc.bytes, bc.ops


def count_kernels(fn, calls: int = 5) -> float:
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    n = 0
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.lower().startswith(("memcpy", "memset")):
            n += 1
    return n / calls


def compile_fn(fn, mode: str):
    torch._dynamo.reset()
    if mode == "eager":
        return fn
    return torch.compile(fn, mode=None if mode == "default" else mode, dynamic=False)


def first_call_s(fn, *args) -> float:
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn(*args)
    torch.cuda.synchronize()
    return time.perf_counter() - t


# --------------------------------------------------------------------------- chains

def bias_gelu_dropout(x, b):
    return F.dropout(F.gelu(x + b), p=0.1, training=True)


def layernorm(x, w, b):
    return F.layer_norm(x, (x.shape[-1],), w, b)


def rmsnorm(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)


def softmax(x):
    return torch.softmax(x.float(), -1).to(x.dtype)


def residual_chain(x, r1, r2, r3):
    y = x + r1
    y = y * 0.5 + r2
    return F.relu(y) + r3


def chain_inputs(name: str, m: int, n: int):
    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(m, n, device=DEV, dtype=DT, generator=g)
    vec = torch.randn(n, device=DEV, dtype=DT, generator=g)
    if name == "bias_gelu_dropout":
        return bias_gelu_dropout, (x, vec)
    if name == "layernorm":
        return layernorm, (x, vec, vec.clone())
    if name == "rmsnorm":
        return rmsnorm, (x, vec)
    if name == "softmax":
        return softmax, (x,)
    if name == "residual_chain":
        return residual_chain, (x, torch.randn_like(x), torch.randn_like(x), torch.randn_like(x))
    raise KeyError(name)


def ideal_bytes(args, out) -> int:
    return sum(a.numel() * a.element_size() for a in args) + out.numel() * out.element_size()


def check(name, ref_fn, fn, args) -> float:
    if name == "bias_gelu_dropout":
        y = fn(*args).float()
        ref = F.gelu(args[0] + args[1]).float() / 0.9
        keep = y != 0
        return float((y[keep] - ref[keep]).abs().max())
    return float((fn(*args).float() - ref_fn(*args).float()).abs().max())


def part_chains(w: Writer, sizes, modes):
    for name in ["bias_gelu_dropout", "layernorm", "rmsnorm", "softmax", "residual_chain"]:
        for m in sizes:
            n = 4096
            fn, args = chain_inputs(name, m, n)
            out = fn(*args)
            ib = ideal_bytes(args, out)
            eb, eops = eager_bytes(fn, *args)
            for mode in modes:
                cf = compile_fn(fn, mode)
                c_s = first_call_s(cf, *args)
                t = bench(lambda: cf(*args))
                k = count_kernels(lambda: cf(*args))
                err = check(name, fn, cf, args)
                w(part="chains", chain=name, M=m, N=n, mode=mode, time_s=t, kernels=k,
                  ideal_bytes=ib, eager_bytes=eb, eager_ops=eops, achieved_bw=ib / t,
                  bw_frac=ib / t / RTX5090["hbm_bw"], compile_s=c_s, max_err=err,
                  fits_l2=bool(eb < RTX5090["l2_bytes"]))
                del cf
            torch._dynamo.reset()


# --------------------------------------------------------------------------- MLP / block

class MLP(torch.nn.Module):
    def __init__(self, d: int, dff: int):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, dff)
        self.fc2 = torch.nn.Linear(dff, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.ln(x))))


class Block(torch.nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1 = torch.nn.LayerNorm(d)
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)
        self.mlp = MLP(d, 4 * d)

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, s, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, s, d))
        return self.mlp(x)


def part_mlp(w: Writer, tokens_list, modes, ma_tokens):
    d = 1024
    for kind in ["mlp", "block"]:
        for t_ in tokens_list:
            torch.manual_seed(0)
            mod = (MLP(d, 4 * d) if kind == "mlp" else Block(d, 16)).to(DEV, DT).eval()
            x = torch.randn(t_, d, device=DEV, dtype=DT) if kind == "mlp" else \
                torch.randn(t_ // 1024, 1024, d, device=DEV, dtype=DT)
            flops = 2 * t_ * d * 4 * d * 2
            if kind == "block":
                flops += 2 * t_ * d * 4 * d + 2 * 2 * t_ * 1024 * d / 2  # qkv+proj, causal attention
            with torch.no_grad():
                ref = mod(x)
                eb, eops = eager_bytes(mod, x)
                wbytes = sum(p.numel() * p.element_size() for p in mod.parameters())
                ib = 2 * x.numel() * x.element_size() + wbytes
                for mode in modes:
                    if mode.startswith("max-autotune") and t_ not in ma_tokens:
                        continue
                    cf = compile_fn(mod, mode)
                    c_s = first_call_s(cf, x)
                    t = bench(lambda: cf(x))
                    k = count_kernels(lambda: cf(x))
                    err = float((cf(x).float() - ref.float()).abs().max())
                    w(part="mlp", chain=kind, M=t_, N=d, mode=mode, time_s=t, kernels=k,
                      ideal_bytes=ib, eager_bytes=eb, eager_ops=eops, flops=flops,
                      tflops=flops / t / 1e12, achieved_bw=ib / t, compile_s=c_s, max_err=err)
                    del cf
            torch._dynamo.reset()


# --------------------------------------------------------------------------- handwritten Triton

def part_triton(w: Writer, sizes):
    for name in ["rmsnorm", "bias_gelu"]:
        for m in sizes:
            for n in [1024, 4096, 8192]:
                torch.manual_seed(0)
                x = torch.randn(m, n, device=DEV, dtype=DT)
                v = torch.randn(n, device=DEV, dtype=DT)
                if name == "rmsnorm":
                    eager, trit = rmsnorm, tk.rmsnorm
                else:
                    def eager(a, b):
                        return F.gelu(a + b)
                    trit = tk.bias_gelu
                ref = eager(x, v)
                ib = 2 * x.numel() * x.element_size() + v.numel() * v.element_size()
                ind = compile_fn(eager, "default")
                for impl, f in [("eager", eager), ("inductor", ind), ("triton", trit)]:
                    t = bench(lambda: f(x, v))
                    err = float((f(x, v).float() - ref.float()).abs().max())
                    w(part="triton", chain=name, M=m, N=n, impl=impl, time_s=t,
                      achieved_bw=ib / t, bw_frac=ib / t / RTX5090["hbm_bw"], max_err=err,
                      kernels=count_kernels(lambda: f(x, v)))
                torch._dynamo.reset()


# --------------------------------------------------------------------------- dynamic shapes

def part_dynamic(w: Writer):
    from torch._dynamo.utils import counters
    d = 1024
    seqs = [512, 768, 1024, 1536, 2048, 3072, 4096, 640, 1024, 2048]
    torch.manual_seed(0)
    mod = MLP(d, 4 * d).to(DEV, DT).eval()
    for strategy in ["static", "automatic", "mark_dynamic"]:
        torch._dynamo.reset()
        counters.clear()
        if strategy == "static":
            cf = torch.compile(mod, dynamic=False)
        else:
            cf = torch.compile(mod)
        total = 0.0
        with torch.no_grad():
            for i, s in enumerate(seqs):
                x = torch.randn(s, d, device=DEV, dtype=DT)
                if strategy == "mark_dynamic":
                    torch._dynamo.mark_dynamic(x, 0)
                dt = first_call_s(cf, x)
                total += dt
                w(part="dynamic", strategy=strategy, step=i, seq=s, first_call_s=dt,
                  cum_s=total, graphs=int(counters["stats"]["unique_graphs"]))
            for s in [512, 1024, 2048, 4096]:
                x = torch.randn(s, d, device=DEV, dtype=DT)
                if strategy == "mark_dynamic":
                    torch._dynamo.mark_dynamic(x, 0)
                t = bench(lambda: cf(x))
                w(part="dynamic_steady", strategy=strategy, seq=s, time_s=t,
                  eager_s=bench(lambda: mod(x)))
        w(part="dynamic_total", strategy=strategy, total_s=total,
          graphs=int(counters["stats"]["unique_graphs"]),
          recompiles=int(counters["recompiles"].get("total", 0)) if "recompiles" in counters else -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["chains", "mlp", "triton", "dynamic"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    set_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    import torch._inductor.config as ic
    ic.compile_threads = 2
    w = Writer(f"fusion_{a.part}")
    w(part="env", **env_info())
    modes = ["eager", "default", "max-autotune", "max-autotune-no-cudagraphs"]
    if a.part == "chains":
        part_chains(w, [256, 1024] if a.quick else [256, 1024, 4096, 16384], modes)
    elif a.part == "mlp":
        part_mlp(w, [1024] if a.quick else [1024, 4096, 16384], modes, {4096} if not a.quick else {1024})
    elif a.part == "triton":
        part_triton(w, [256] if a.quick else [256, 2048, 16384])
    else:
        part_dynamic(w)


if __name__ == "__main__":
    main()
