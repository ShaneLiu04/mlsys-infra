"""Depthwise causal 1D convolution on GPU: three hand-written CUDA kernels (load_inline) + torch baselines.

y[b, c, t] = sum_{j < K} h[c, j] * x[b, c, t - j],  x[., ., t < 0] = 0, fp32, x is (B, C, N), h is (C, K).

naive    : one thread per output, K global loads of x and h per output (L1/L2 serve the reuse).
smem     : a block owns TILE=256 outputs of one (b, c) row, stages x[t0-K+1, t0+TILE) (halo K-1) and
           h[c, :] in shared memory; still 2 shared loads per FMA.
regblock : each thread owns R=8 consecutive outputs; taps processed in chunks of R with a 2R-1
           register window, so R*R FMAs cost 3R-1 shared loads; padded shared indexing avoids the
           R-way bank conflicts of the stride-R access; outputs staged through shared memory so the
           global store is coalesced.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor dw_naive(torch::Tensor x, torch::Tensor h);
torch::Tensor dw_smem(torch::Tensor x, torch::Tensor h);
torch::Tensor dw_regblock(torch::Tensor x, torch::Tensor h);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define PADI(i) ((i) + ((i) >> 5))

__global__ void k_naive(const float* __restrict__ x, const float* __restrict__ h, float* __restrict__ y,
                        int C, int N, int K) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (t >= N) return;
  const float* xr = x + (size_t)row * N;
  const float* hr = h + (size_t)(row % C) * K;
  int jmax = min(K, t + 1);
  float acc = 0.f;
  for (int j = 0; j < jmax; ++j) acc += hr[j] * xr[t - j];
  y[(size_t)row * N + t] = acc;
}

__global__ void k_smem(const float* __restrict__ x, const float* __restrict__ h, float* __restrict__ y,
                       int C, int N, int K) {
  extern __shared__ float sm[];
  const int TILE = blockDim.x;
  float* sh = sm;
  float* sx = sm + K;
  int row = blockIdx.y;
  int t0 = blockIdx.x * TILE;
  const float* xr = x + (size_t)row * N;
  const float* hr = h + (size_t)(row % C) * K;
  for (int i = threadIdx.x; i < K; i += blockDim.x) sh[i] = hr[i];
  for (int i = threadIdx.x; i < TILE + K - 1; i += blockDim.x) {
    int t = t0 - (K - 1) + i;
    sx[i] = (t >= 0 && t < N) ? xr[t] : 0.f;
  }
  __syncthreads();
  int o = threadIdx.x;
  int t = t0 + o;
  if (t >= N) return;
  float acc = 0.f;
  for (int j = 0; j < K; ++j) acc += sh[j] * sx[o + K - 1 - j];
  y[(size_t)row * N + t] = acc;
}

template <int R>
__global__ void k_regblock(const float* __restrict__ x, const float* __restrict__ h, float* __restrict__ y,
                           int C, int N, int K, int Kp) {
  extern __shared__ float sm[];
  const int TILE = blockDim.x * R;
  float* sh = sm;            // Kp taps, zero padded
  float* sx = sm + Kp;       // x[t0 - (Kp-1) + i], padded index
  int row = blockIdx.y;
  int t0 = blockIdx.x * TILE;
  const float* xr = x + (size_t)row * N;
  const float* hr = h + (size_t)(row % C) * K;
  for (int i = threadIdx.x; i < Kp; i += blockDim.x) sh[i] = i < K ? hr[i] : 0.f;
  for (int i = threadIdx.x; i < TILE + Kp - 1; i += blockDim.x) {
    int t = t0 - (Kp - 1) + i;
    sx[PADI(i)] = (t >= 0 && t < N) ? xr[t] : 0.f;
  }
  __syncthreads();
  const int o0 = threadIdx.x * R;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
  for (int jb = 0; jb < Kp; jb += R) {
    const int base = o0 + Kp - jb - R;
    float w[2 * R - 1];
#pragma unroll
    for (int i = 0; i < 2 * R - 1; ++i) w[i] = sx[PADI(base + i)];
#pragma unroll
    for (int jj = 0; jj < R; ++jj) {
      const float hv = sh[jb + jj];
#pragma unroll
      for (int r = 0; r < R; ++r) acc[r] = fmaf(hv, w[r - jj + R - 1], acc[r]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < R; ++r) sx[PADI(o0 + r)] = acc[r];
  __syncthreads();
  for (int i = threadIdx.x; i < TILE; i += blockDim.x) {
    int t = t0 + i;
    if (t < N) y[(size_t)row * N + t] = sx[PADI(i)];
  }
}

static void check(const torch::Tensor& x, const torch::Tensor& h) {
  TORCH_CHECK(x.is_cuda() && h.is_cuda(), "cuda tensors");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32 && h.scalar_type() == torch::kFloat32, "fp32");
  TORCH_CHECK(x.dim() == 3 && h.dim() == 2 && x.size(1) == h.size(0), "x (B,C,N), h (C,K)");
  TORCH_CHECK(x.size(0) * x.size(1) <= 65535, "B*C rows must fit gridDim.y");
}

static void set_smem(const void* fn, size_t bytes) {
  TORCH_CHECK(bytes <= 99 * 1024, "kernel length too large for shared memory");
  if (bytes > 48 * 1024)
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

torch::Tensor dw_naive(torch::Tensor x, torch::Tensor h) {
  check(x, h);
  x = x.contiguous(); h = h.contiguous();
  int B = x.size(0), C = x.size(1), N = x.size(2), K = h.size(1);
  auto y = torch::empty_like(x);
  dim3 grid((N + 255) / 256, B * C);
  k_naive<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), h.data_ptr<float>(),
                                                              y.data_ptr<float>(), C, N, K);
  return y;
}

torch::Tensor dw_smem(torch::Tensor x, torch::Tensor h) {
  check(x, h);
  x = x.contiguous(); h = h.contiguous();
  int B = x.size(0), C = x.size(1), N = x.size(2), K = h.size(1);
  const int TILE = 256;
  auto y = torch::empty_like(x);
  size_t bytes = (size_t)(K + TILE + K - 1) * sizeof(float);
  set_smem((const void*)k_smem, bytes);
  dim3 grid((N + TILE - 1) / TILE, B * C);
  k_smem<<<grid, TILE, bytes, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), h.data_ptr<float>(),
                                                                   y.data_ptr<float>(), C, N, K);
  return y;
}

torch::Tensor dw_regblock(torch::Tensor x, torch::Tensor h) {
  check(x, h);
  x = x.contiguous(); h = h.contiguous();
  int B = x.size(0), C = x.size(1), N = x.size(2), K = h.size(1);
  constexpr int R = 8;
  const int THREADS = 128, TILE = THREADS * R;
  int Kp = (K + R - 1) / R * R;
  auto y = torch::empty_like(x);
  int sx_len = TILE + Kp - 1;
  size_t bytes = (size_t)(Kp + sx_len + (sx_len >> 5) + 1) * sizeof(float);
  set_smem((const void*)k_regblock<R>, bytes);
  dim3 grid((N + TILE - 1) / TILE, B * C);
  k_regblock<R><<<grid, THREADS, bytes, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), h.data_ptr<float>(), y.data_ptr<float>(), C, N, K, Kp);
  return y;
}
"""

_EXT = None


def load_ext(verbose: bool = False):
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load_inline
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        build = Path(os.environ.get("P1_BUILD_DIR", Path(__file__).resolve().parent / "_build"))
        build.mkdir(parents=True, exist_ok=True)
        _EXT = load_inline(name="p1_dwconv", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=["dw_naive", "dw_smem", "dw_regblock"],
                           extra_cuda_cflags=["-O3", "-lineinfo"], build_directory=str(build),
                           verbose=verbose)
    return _EXT


def run_kernel(ext, name: str, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return getattr(ext, f"dw_{name}")(x, h)


def conv_torch(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """cuDNN depthwise conv1d (groups=C); conv1d is cross-correlation, so flip taps and left-pad K-1."""
    k = h.shape[-1]
    return F.conv1d(F.pad(x, (k - 1, 0)), h.flip(-1)[:, None, :], groups=x.shape[1])


def conv_fft(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """O(N log N): zero-pad to a power of two >= N + K - 1 so circular == linear convolution."""
    n, k = x.shape[-1], h.shape[-1]
    size = 1 << (n + k - 2).bit_length()
    return torch.fft.irfft(torch.fft.rfft(x, n=size) * torch.fft.rfft(h, n=size), n=size)[..., :n]
