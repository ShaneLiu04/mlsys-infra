// Roofline microbenchmarks: streaming copy / triad (memory roof) and an FMA chain (fp32 compute roof).
//   nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC micro.cu -o libmicro.so
#include <cuda_runtime.h>

__global__ void k_copy(const float4* __restrict__ a, float4* __restrict__ b, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x)
    b[i] = a[i];
}

__global__ void k_triad(float4* __restrict__ a, const float4* __restrict__ b, const float4* __restrict__ c,
                        float s, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
    float4 x = b[i], y = c[i];
    a[i] = make_float4(x.x + s * y.x, x.y + s * y.y, x.z + s * y.z, x.w + s * y.w);
  }
}

// CH independent FMA chains per thread hide the 4-cycle FMA latency; 2*CH*iters FLOP per thread.
template <int CH>
__global__ void k_fma(float* out, int iters, float a, float b) {
  float x[CH];
#pragma unroll
  for (int c = 0; c < CH; ++c) x[c] = threadIdx.x * 1e-7f + c;
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < CH; ++c) x[c] = fmaf(x[c], a, b);
  }
  float s = 0.f;
#pragma unroll
  for (int c = 0; c < CH; ++c) s += x[c];
  if (s == 12345.678f) out[0] = s;  // never true; keeps the chains alive
}

extern "C" {
int bw_copy(const float* a, float* b, size_t n, int blocks) {
  k_copy<<<blocks, 256>>>((const float4*)a, (float4*)b, n / 4);
  return (int)cudaGetLastError();
}
int bw_triad(float* a, const float* b, const float* c, size_t n, int blocks) {
  k_triad<<<blocks, 256>>>((float4*)a, (const float4*)b, (const float4*)c, 1.5f, n / 4);
  return (int)cudaGetLastError();
}
// chains in {1, 2, 4, 8, 16}
int fma_chain(float* out, int blocks, int threads, int iters, int chains) {
  switch (chains) {
    case 1: k_fma<1><<<blocks, threads>>>(out, iters, 0.999f, 1e-3f); break;
    case 2: k_fma<2><<<blocks, threads>>>(out, iters, 0.999f, 1e-3f); break;
    case 4: k_fma<4><<<blocks, threads>>>(out, iters, 0.999f, 1e-3f); break;
    case 8: k_fma<8><<<blocks, threads>>>(out, iters, 0.999f, 1e-3f); break;
    case 16: k_fma<16><<<blocks, threads>>>(out, iters, 0.999f, 1e-3f); break;
    default: return -1;
  }
  return (int)cudaGetLastError();
}
}
