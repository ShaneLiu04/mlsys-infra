// SGEMM optimisation ladder, row-major C[M,N] = A[M,K] @ B[K,N] (fp32).
// Built as a shared library and driven from Python via ctypes (see run_gpu.py):
//   nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC sgemm.cu -o libsgemm.so
// Every launcher returns 0 on success, otherwise a cudaError_t / -1 for an unsupported config.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

#define CDIV(a, b) (((a) + (b) - 1) / (b))

// ---------------------------------------------------------------- 0. naive, uncoalesced
// threadIdx.x walks rows: a warp reads 32 different rows of A (stride K) and writes 32 rows of C.
__global__ void k_naive(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                        float* __restrict__ C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------- 1. naive, coalesced
// threadIdx.x walks columns: B loads and C stores are contiguous across the warp, A is a broadcast.
__global__ void k_coalesced(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// ---------------------------------------------------------------- 2. shared-memory tiling (T x T)
template <int T>
__global__ void k_smem(int M, int N, int K, const float* __restrict__ A, const float* __restrict__ B,
                       float* __restrict__ C) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------- 3. 1D register tiling
// Block BM x BN, each thread computes TM consecutive rows of one column.
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM) k_reg1d(int M, int N, int K, const float* __restrict__ A,
                                                          const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM * BN) / TM;
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  const int tid = threadIdx.x;
  const int tcol = tid % BN, trow = tid / BN;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  float acc[TM] = {0.f};
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += NT) {
      int r = i / BK, c = i % BK;
      As[i] = (bm + r < M && k0 + c < K) ? A[(bm + r) * K + k0 + c] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += NT) {
      int r = i / BN, c = i % BN;
      Bs[i] = (k0 + r < K && bn + c < N) ? B[(k0 + r) * N + bn + c] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float b = Bs[k * BN + tcol];
#pragma unroll
      for (int t = 0; t < TM; ++t) acc[t] += As[(trow * TM + t) * BK + k] * b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int t = 0; t < TM; ++t) {
    int r = bm + trow * TM + t, c = bn + tcol;
    if (r < M && c < N) C[r * N + c] = acc[t];
  }
}

// ---------------------------------------------------------------- 4. 2D register tiling (scalar loads)
// Block BM x BN, thread tile TM x TN; generic strided loads so any (BM, BN) instantiation works.
// NC=true ("no conflict"): As rows padded by 4 floats so the transposing store hits 32 distinct banks, and a
// thread owns columns tcol + j*(BN/TN) instead of tcol*TN + j so the B reads of a warp are consecutive words.
template <int BM, int BN, int BK, int TM, int TN, bool NC = false>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) k_reg2d(int M, int N, int K, const float* __restrict__ A,
                                                                const float* __restrict__ B, float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);
  constexpr int LDA = NC ? BM + 4 : BM;
  constexpr int CS = BN / TN;  // column stride between a thread's outputs when NC
  __shared__ float As[BK * LDA];  // transposed: As[k][m]
  __shared__ float Bs[BK * BN];
  const int tid = threadIdx.x;
  const int tcol = tid % (BN / TN), trow = tid / (BN / TN);
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  float acc[TM][TN] = {{0.f}};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += NT) {
      int r = i / BK, c = i % BK;
      As[c * LDA + r] = (bm + r < M && k0 + c < K) ? A[(bm + r) * K + k0 + c] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += NT) {
      int r = i / BN, c = i % BN;
      Bs[i] = (k0 + r < K && bn + c < N) ? B[(k0 + r) * N + bn + c] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[k * LDA + trow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = NC ? Bs[k * BN + tcol + j * CS] : Bs[k * BN + tcol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = bm + trow * TM + i, c = bn + (NC ? tcol + j * CS : tcol * TN + j);
      if (r < M && c < N) C[r * N + c] = acc[i][j];
    }
}

// ---------------------------------------------------------------- 5/6. 2D tiling + float4 (+ double buffering)
// BM=BN=128, BK=8, TM=TN=8, 256 threads. Requires M%128==N%128==K%8==0.
// DB=true: two smem stages; the next tile is prefetched into registers while the current one is consumed.
// MINB=2 asks ptxas to fit 2 blocks/SM (<= 128 regs/thread), trading a few spills for occupancy.
template <bool DB, int MINB = 1>
__global__ void __launch_bounds__(256, MINB) k_vec(int M, int N, int K, const float* __restrict__ A,
                                             const float* __restrict__ B, float* __restrict__ C) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  constexpr int S = DB ? 2 : 1;
  __shared__ __align__(16) float As[S][BK][BM];
  __shared__ __align__(16) float Bs[S][BK][BN];
  const int tid = threadIdx.x;
  const int tcol = tid % 16, trow = tid / 16;
  const float* Ab = A + (size_t)blockIdx.y * BM * K;
  const float* Bb = B + (size_t)blockIdx.x * BN;
  // A tile 128x8: each thread one float4 (row = tid/2, k = (tid%2)*4); B tile 8x128: row = tid/32, col=(tid%32)*4
  const int ar = tid >> 1, ak = (tid & 1) * 4;
  const int br = tid >> 5, bc = (tid & 31) * 4;
  float acc[TM][TN] = {{0.f}};
  float ra[TM], rb[TN];
  float4 pa = *reinterpret_cast<const float4*>(Ab + (size_t)ar * K + ak);
  float4 pb = *reinterpret_cast<const float4*>(Bb + (size_t)br * N + bc);
  As[0][ak + 0][ar] = pa.x; As[0][ak + 1][ar] = pa.y; As[0][ak + 2][ar] = pa.z; As[0][ak + 3][ar] = pa.w;
  *reinterpret_cast<float4*>(&Bs[0][br][bc]) = pb;
  __syncthreads();
  int st = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool more = k0 + BK < K;
    if (DB && more) {
      pa = *reinterpret_cast<const float4*>(Ab + (size_t)ar * K + k0 + BK + ak);
      pb = *reinterpret_cast<const float4*>(Bb + (size_t)(k0 + BK + br) * N + bc);
    }
#pragma unroll
    for (int k = 0; k < BK; ++k) {
      float4 a0 = *reinterpret_cast<const float4*>(&As[st][k][trow * TM]);
      float4 a1 = *reinterpret_cast<const float4*>(&As[st][k][trow * TM + 4]);
      float4 b0 = *reinterpret_cast<const float4*>(&Bs[st][k][tcol * TN]);
      float4 b1 = *reinterpret_cast<const float4*>(&Bs[st][k][tcol * TN + 4]);
      ra[0] = a0.x; ra[1] = a0.y; ra[2] = a0.z; ra[3] = a0.w; ra[4] = a1.x; ra[5] = a1.y; ra[6] = a1.z; ra[7] = a1.w;
      rb[0] = b0.x; rb[1] = b0.y; rb[2] = b0.z; rb[3] = b0.w; rb[4] = b1.x; rb[5] = b1.y; rb[6] = b1.z; rb[7] = b1.w;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    if (!more) break;
    if (DB) {
      const int nx = st ^ 1;
      As[nx][ak + 0][ar] = pa.x; As[nx][ak + 1][ar] = pa.y; As[nx][ak + 2][ar] = pa.z; As[nx][ak + 3][ar] = pa.w;
      *reinterpret_cast<float4*>(&Bs[nx][br][bc]) = pb;
      __syncthreads();
      st = nx;
    } else {
      __syncthreads();
      pa = *reinterpret_cast<const float4*>(Ab + (size_t)ar * K + k0 + BK + ak);
      pb = *reinterpret_cast<const float4*>(Bb + (size_t)(k0 + BK + br) * N + bc);
      As[0][ak + 0][ar] = pa.x; As[0][ak + 1][ar] = pa.y; As[0][ak + 2][ar] = pa.z; As[0][ak + 3][ar] = pa.w;
      *reinterpret_cast<float4*>(&Bs[0][br][bc]) = pb;
      __syncthreads();
    }
  }
  float* Cb = C + (size_t)(blockIdx.y * BM + trow * TM) * N + blockIdx.x * BN + tcol * TN;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    *reinterpret_cast<float4*>(Cb + (size_t)i * N) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    *reinterpret_cast<float4*>(Cb + (size_t)i * N + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
}

// ---------------------------------------------------------------- 7. skinny / decode GEMM (M small)
// C[M,N] = A[M,K] B[K,N] for M <= 64: memory-bound on B. Each thread owns one column and MT rows,
// K is split over gridDim.y and partial sums are atomically added (C must be zeroed by the caller).
template <int MT>
__global__ void __launch_bounds__(256) k_skinny(int M, int N, int K, int kchunk, const float* __restrict__ A,
                                                const float* __restrict__ B, float* __restrict__ C) {
  __shared__ float As[MT][256];
  const int col = blockIdx.x * 256 + threadIdx.x;
  const int kb = blockIdx.y * kchunk, ke = min(K, kb + kchunk);
  for (int m0 = 0; m0 < M; m0 += MT) {
    float acc[MT] = {0.f};
    for (int k1 = kb; k1 < ke; k1 += 256) {
      __syncthreads();
      for (int t = 0; t < MT; ++t) {
        int k = k1 + threadIdx.x;
        As[t][threadIdx.x] = (m0 + t < M && k < ke) ? A[(size_t)(m0 + t) * K + k] : 0.f;
      }
      __syncthreads();
      const int kn = min(256, ke - k1);
      if (col < N) {
#pragma unroll 4
        for (int k = 0; k < kn; ++k) {
          float b = __ldg(B + (size_t)(k1 + k) * N + col);
#pragma unroll
          for (int t = 0; t < MT; ++t) acc[t] += As[t][k] * b;
        }
      }
    }
    if (col < N)
      for (int t = 0; t < MT && m0 + t < M; ++t) atomicAdd(C + (size_t)(m0 + t) * N + col, acc[t]);
  }
}

// ---------------------------------------------------------------- 8. tensor cores (WMMA 16x16x16)
// fp16/bf16 inputs, fp32 accumulate. Block 128x128x32, 8 warps as 2x4, warp tile 64x32 = 4x2 fragments.
// Two smem stages filled with cp.async so global loads overlap the MMAs. Requires M%128==N%128==K%32==0.
namespace wm = nvcuda::wmma;
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }

template <typename T>
__global__ void __launch_bounds__(256) k_wmma(int M, int N, int K, const T* __restrict__ A, const T* __restrict__ B,
                                              float* __restrict__ C) {
  constexpr int BM = 128, BN = 128, BK = 32, PADA = 8, PADB = 8;
  __shared__ __align__(128) T As[2][BM][BK + PADA];
  __shared__ __align__(128) T Bs[2][BK][BN + PADB];
  const int tid = threadIdx.x, warp = tid / 32;
  const int wm_ = warp / 4, wn_ = warp % 4;  // 2 x 4 warps, warp tile 64 x 32
  const T* Ab = A + (size_t)blockIdx.y * BM * K;
  const T* Bb = B + (size_t)blockIdx.x * BN;
  wm::fragment<wm::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wm::fill_fragment(acc[i][j], 0.f);
  auto load = [&](int stage, int k0) {
    // A: 128x32 halves = 512 x 16B chunks; B: 32x128 = 512 chunks; 2 chunks each per thread.
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      int id = tid + c * 256;
      int r = id / 4, q = (id % 4) * 8;
      cp_async16(&As[stage][r][q], Ab + (size_t)r * K + k0 + q);
      int rb = id / 16, qb = (id % 16) * 8;
      cp_async16(&Bs[stage][rb][qb], Bb + (size_t)(k0 + rb) * N + qb);
    }
    cp_async_commit();
  };
  load(0, 0);
  int st = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    cp_async_wait0();
    __syncthreads();
    if (k0 + BK < K) load(st ^ 1, k0 + BK);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wm::fragment<wm::matrix_a, 16, 16, 16, T, wm::row_major> fa[4];
      wm::fragment<wm::matrix_b, 16, 16, 16, T, wm::row_major> fb[2];
#pragma unroll
      for (int i = 0; i < 4; ++i) wm::load_matrix_sync(fa[i], &As[st][wm_ * 64 + i * 16][kk], BK + PADA);
#pragma unroll
      for (int j = 0; j < 2; ++j) wm::load_matrix_sync(fb[j], &Bs[st][kk][wn_ * 32 + j * 16], BN + PADB);
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wm::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
    __syncthreads();
    st ^= 1;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      float* p = C + (size_t)(blockIdx.y * BM + wm_ * 64 + i * 16) * N + blockIdx.x * BN + wn_ * 32 + j * 16;
      wm::store_matrix_sync(p, acc[i][j], N, wm::mem_row_major);
    }
}

// ---------------------------------------------------------------- launchers
static int ret() { return (int)cudaGetLastError(); }

template <int BM, int BN>
static void launch_reg2d(int M, int N, int K, const float* A, const float* B, float* C) {
  dim3 g(CDIV(N, BN), CDIV(M, BM));
  k_reg2d<BM, BN, 8, 8, 8><<<g, (BM / 8) * (BN / 8)>>>(M, N, K, A, B, C);
}

extern "C" {

// variant: 0 naive, 1 coalesced, 2 smem8, 3 smem16, 4 smem32, 5 reg1d(64x64x8,TM=8),
//          6 reg2d(128x128x8, 8x8), 7 vec float4, 8 vec float4 + double buffer,
//          9 vec float4 + double buffer + __launch_bounds__(256, 2), 10 reg2d without bank conflicts
int sgemm(int variant, int M, int N, int K, const float* A, const float* B, float* C) {
  switch (variant) {
    case 0: { dim3 b(32, 32), g(CDIV(M, 32), CDIV(N, 32)); k_naive<<<g, b>>>(M, N, K, A, B, C); break; }
    case 1: { dim3 b(32, 32), g(CDIV(N, 32), CDIV(M, 32)); k_coalesced<<<g, b>>>(M, N, K, A, B, C); break; }
    case 2: { dim3 b(8, 8), g(CDIV(N, 8), CDIV(M, 8)); k_smem<8><<<g, b>>>(M, N, K, A, B, C); break; }
    case 3: { dim3 b(16, 16), g(CDIV(N, 16), CDIV(M, 16)); k_smem<16><<<g, b>>>(M, N, K, A, B, C); break; }
    case 4: { dim3 b(32, 32), g(CDIV(N, 32), CDIV(M, 32)); k_smem<32><<<g, b>>>(M, N, K, A, B, C); break; }
    case 5: { dim3 g(CDIV(N, 64), CDIV(M, 64)); k_reg1d<64, 64, 8, 8><<<g, 512>>>(M, N, K, A, B, C); break; }
    case 6: launch_reg2d<128, 128>(M, N, K, A, B, C); break;
    case 7:
    case 8:
    case 9: {
      if (M % 128 || N % 128 || K % 8) return -1;
      dim3 g(N / 128, M / 128);
      if (variant == 7) k_vec<false><<<g, 256>>>(M, N, K, A, B, C);
      else if (variant == 8) k_vec<true><<<g, 256>>>(M, N, K, A, B, C);
      else k_vec<true, 2><<<g, 256>>>(M, N, K, A, B, C);
      break;
    }
    case 10: {
      dim3 g(CDIV(N, 128), CDIV(M, 128));
      k_reg2d<128, 128, 8, 8, 8, true><<<g, 256>>>(M, N, K, A, B, C);
      break;
    }
    default: return -1;
  }
  return ret();
}

// 2D register tiling with block (BM, BN) in {32,64,128,256}^2, BK=8, TM=TN=8 (for the tile heatmap).
int sgemm_reg2d_tile(int BM, int BN, int M, int N, int K, const float* A, const float* B, float* C) {
#define CASE(bm, bn) if (BM == bm && BN == bn) { launch_reg2d<bm, bn>(M, N, K, A, B, C); return ret(); }
  CASE(32, 32) CASE(32, 64) CASE(32, 128) CASE(32, 256)
  CASE(64, 32) CASE(64, 64) CASE(64, 128) CASE(64, 256)
  CASE(128, 32) CASE(128, 64) CASE(128, 128) CASE(128, 256)
  CASE(256, 32) CASE(256, 64) CASE(256, 128) CASE(256, 256)
#undef CASE
  return -1;
}

// C must be zero on entry.
int sgemm_skinny(int M, int N, int K, int ksplit, const float* A, const float* B, float* C) {
  int kchunk = CDIV(CDIV(K, ksplit), 256) * 256;
  dim3 g(CDIV(N, 256), CDIV(K, kchunk));
  if (M <= 1) k_skinny<1><<<g, 256>>>(M, N, K, kchunk, A, B, C);
  else if (M <= 4) k_skinny<4><<<g, 256>>>(M, N, K, kchunk, A, B, C);
  else k_skinny<8><<<g, 256>>>(M, N, K, kchunk, A, B, C);
  return ret();
}

// Static resource usage + theoretical occupancy (no perf counters needed).
// id: 0..8 sgemm variants, 9 wmma fp16, 10 skinny<8>. out = {regs, static smem, block threads, blocks/SM, warps/SM}
int kernel_info(int id, int* out) {
  const void* f = nullptr;
  int threads = 0;
  switch (id) {
    case 0: f = (const void*)k_naive; threads = 1024; break;
    case 1: f = (const void*)k_coalesced; threads = 1024; break;
    case 2: f = (const void*)k_smem<8>; threads = 64; break;
    case 3: f = (const void*)k_smem<16>; threads = 256; break;
    case 4: f = (const void*)k_smem<32>; threads = 1024; break;
    case 5: f = (const void*)k_reg1d<64, 64, 8, 8>; threads = 512; break;
    case 6: f = (const void*)k_reg2d<128, 128, 8, 8, 8>; threads = 256; break;
    case 7: f = (const void*)k_vec<false>; threads = 256; break;
    case 8: f = (const void*)k_vec<true>; threads = 256; break;
    case 9: f = (const void*)k_wmma<__half>; threads = 256; break;
    case 10: f = (const void*)k_skinny<8>; threads = 256; break;
    case 11: f = (const void*)k_vec<true, 2>; threads = 256; break;
    case 12: f = (const void*)k_reg2d<128, 128, 8, 8, 8, true>; threads = 256; break;
    default: return -1;
  }
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f);
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, f, threads, 0);
  out[0] = a.numRegs; out[1] = (int)a.sharedSizeBytes; out[2] = threads; out[3] = blocks; out[4] = blocks * threads / 32;
  return ret();
}

// dtype 0 = fp16, 1 = bf16
int hgemm_wmma(int dtype, int M, int N, int K, const void* A, const void* B, float* C) {
  if (M % 128 || N % 128 || K % 32) return -1;
  dim3 g(N / 128, M / 128);
  if (dtype == 0) k_wmma<__half><<<g, 256>>>(M, N, K, (const __half*)A, (const __half*)B, C);
  else k_wmma<__nv_bfloat16><<<g, 256>>>(M, N, K, (const __nv_bfloat16*)A, (const __nv_bfloat16*)B, C);
  return ret();
}

}  // extern "C"
