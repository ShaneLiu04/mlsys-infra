"""ctypes bindings for libsgemm.so / libmicro.so / libconv.so (built from the .cu files next to this file).

All functions take torch CUDA tensors and launch on the default stream (the stream torch uses unless told
otherwise), so torch.cuda.Event timing and torch reference ops interleave correctly.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
NVCC = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
LIBS = ("sgemm", "micro", "conv")

SGEMM_VARIANTS = {0: "naive", 1: "coalesced", 2: "smem8", 3: "smem16", 4: "smem32", 5: "reg1d",
                  6: "reg2d", 7: "vec4", 8: "vec4_db", 9: "vec4_db_lb2", 10: "reg2d_nc"}
# (BM, BN) block tile each variant keeps in shared memory / registers; drives the traffic model.
SGEMM_TILES = {"naive": (1, 1), "coalesced": (1, 1), "smem8": (8, 8), "smem16": (16, 16), "smem32": (32, 32),
               "reg1d": (64, 64), "reg2d": (128, 128), "vec4": (128, 128), "vec4_db": (128, 128),
               "vec4_db_lb2": (128, 128), "reg2d_nc": (128, 128),
               "wmma_fp16": (128, 128), "wmma_bf16": (128, 128)}


def build(force: bool = False, arch: str | None = None) -> None:
    # CUDA_ARCH=native (or sm_89, sm_120, ...) so the same sources build on Ada and Blackwell.
    arch = arch or os.environ.get("CUDA_ARCH", "sm_120")
    BUILD.mkdir(exist_ok=True)
    for name in LIBS:
        src, out = HERE / f"{name}.cu", BUILD / f"lib{name}.so"
        if out.exists() and not force and out.stat().st_mtime > src.stat().st_mtime:
            continue
        cmd = [NVCC, "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-lineinfo", str(src), "-o", str(out)]
        subprocess.run(cmd, check=True)


_cache: dict[str, ctypes.CDLL] = {}


def lib(name: str) -> ctypes.CDLL:
    if name not in _cache:
        _cache[name] = ctypes.CDLL(str(BUILD / f"lib{name}.so"))
    return _cache[name]


def _p(t) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed with code {rc}")


# ------------------------------------------------------------------ GEMM
def sgemm(variant: int, a, b, c) -> None:
    m, k = a.shape
    n = b.shape[1]
    _check(lib("sgemm").sgemm(variant, m, n, k, _p(a), _p(b), _p(c)), SGEMM_VARIANTS.get(variant, variant))


def sgemm_tile(bm: int, bn: int, a, b, c) -> None:
    m, k = a.shape
    n = b.shape[1]
    _check(lib("sgemm").sgemm_reg2d_tile(bm, bn, m, n, k, _p(a), _p(b), _p(c)), f"reg2d {bm}x{bn}")


def sgemm_skinny(a, b, c, ksplit: int) -> None:
    """c must be zero-filled (split-K atomics)."""
    m, k = a.shape
    n = b.shape[1]
    _check(lib("sgemm").sgemm_skinny(m, n, k, ksplit, _p(a), _p(b), _p(c)), "skinny")


def hgemm_wmma(a, b, c) -> None:
    import torch
    m, k = a.shape
    n = b.shape[1]
    dt = 0 if a.dtype == torch.float16 else 1
    _check(lib("sgemm").hgemm_wmma(dt, m, n, k, _p(a), _p(b), _p(c)), "wmma")


KERNEL_INFO_NAMES = [SGEMM_VARIANTS[i] for i in range(9)] + ["wmma_fp16", "skinny8", "vec4_db_lb2", "reg2d_nc"]


def kernel_info(idx: int) -> dict:
    out = (ctypes.c_int * 5)()
    _check(lib("sgemm").kernel_info(idx, out), "kernel_info")
    return {"kernel": KERNEL_INFO_NAMES[idx], "regs": out[0], "smem_bytes": out[1], "threads": out[2],
            "blocks_per_sm": out[3], "warps_per_sm": out[4], "theoretical_occupancy": out[4] / 48}


# ------------------------------------------------------------------ microbenchmarks
def bw_copy(a, b, blocks: int) -> None:
    f = lib("micro").bw_copy
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _check(f(_p(a), _p(b), a.numel(), blocks), "copy")


def bw_triad(a, b, c, blocks: int) -> None:
    f = lib("micro").bw_triad
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _check(f(_p(a), _p(b), _p(c), a.numel(), blocks), "triad")


def fma_chain(out, blocks: int, threads: int, iters: int, chains: int) -> None:
    _check(lib("micro").fma_chain(_p(out), blocks, threads, iters, chains), "fma")


# ------------------------------------------------------------------ conv
def out_hw(h: int, k: int, s: int, p: int) -> int:
    return (h + 2 * p - k) // s + 1


def conv_direct(x, w, y, s: int, p: int) -> None:
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    _check(lib("conv").conv_direct(n, c, h, wd, f, kh, kw, s, p, _p(x), _p(w), _p(y)), "conv_direct")


def conv_im2col(x, cols, kh: int, kw: int, s: int, p: int) -> None:
    n, c, h, wd = x.shape
    _check(lib("conv").conv_im2col(n, c, h, wd, 0, kh, kw, s, p, _p(x), _p(cols)), "im2col")


def conv_igemm(x, w, y, s: int, p: int, relu: bool = False) -> None:
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    _check(lib("conv").conv_igemm(n, c, h, wd, f, kh, kw, s, p, _p(x), _p(w), _p(y), int(relu)), "igemm")


def conv2x2(fuse: bool, x, w, y, s: int, p: int) -> None:
    n, c, h, wd = x.shape
    f, _, kh, kw = w.shape
    _check(lib("conv").conv2x2(int(fuse), n, c, h, wd, f, kh, kw, s, p, _p(x), _p(w), _p(y)), "conv2x2")


def relu(a, b) -> None:
    f = lib("conv").relu
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    _check(f(_p(a), _p(b), a.numel()), "relu")


def maxpool2x2(a, b, with_relu: bool = False) -> None:
    n, f, oh, ow = a.shape
    _check(lib("conv").maxpool2x2(int(with_relu), _p(a), _p(b), n * f, oh, ow), "maxpool")
