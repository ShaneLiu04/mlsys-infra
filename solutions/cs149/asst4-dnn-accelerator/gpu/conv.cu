// Conv2d (NCHW, fp32): direct, explicit im2col, implicit GEMM, and fused conv+ReLU+maxpool2x2.
//   nvcc -O3 -arch=sm_120 -shared -Xcompiler -fPIC conv.cu -o libconv.so
// x: (N,C,H,W), w: (F,C,KH,KW), y: (N,F,OH,OW).
#include <cuda_runtime.h>
#include <cfloat>

#define CDIV(a, b) (((a) + (b) - 1) / (b))

struct Shape {
  int N, C, H, W, F, KH, KW, S, P, OH, OW;
};

// ---------------------------------------------------------------- direct, one output per thread
__global__ void k_conv_direct(Shape s, const float* __restrict__ x, const float* __restrict__ w,
                              float* __restrict__ y) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y;
  int nf = blockIdx.z;
  int n = nf / s.F, f = nf % s.F;
  if (ow >= s.OW) return;
  float acc = 0.f;
  for (int c = 0; c < s.C; ++c) {
    const float* xc = x + ((size_t)n * s.C + c) * s.H * s.W;
    const float* wc = w + ((size_t)f * s.C + c) * s.KH * s.KW;
    for (int i = 0; i < s.KH; ++i) {
      int ih = oh * s.S - s.P + i;
      if (ih < 0 || ih >= s.H) continue;
      for (int j = 0; j < s.KW; ++j) {
        int iw = ow * s.S - s.P + j;
        if (iw >= 0 && iw < s.W) acc += xc[ih * s.W + iw] * wc[i * s.KW + j];
      }
    }
  }
  y[(((size_t)n * s.F + f) * s.OH + oh) * s.OW + ow] = acc;
}

// ---------------------------------------------------------------- explicit im2col: cols[n][c*KH*KW][oh*OW+ow]
__global__ void k_im2col(Shape s, const float* __restrict__ x, float* __restrict__ cols) {
  size_t P = (size_t)s.OH * s.OW, R = (size_t)s.C * s.KH * s.KW;
  size_t total = (size_t)s.N * R * P;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    size_t p = idx % P, r = (idx / P) % R, n = idx / (P * R);
    int ow = p % s.OW, oh = p / s.OW;
    int j = r % s.KW, i = (r / s.KW) % s.KH, c = r / (s.KW * s.KH);
    int ih = oh * s.S - s.P + i, iw = ow * s.S - s.P + j;
    cols[idx] = (ih >= 0 && ih < s.H && iw >= 0 && iw < s.W) ? x[((n * s.C + c) * s.H + ih) * s.W + iw] : 0.f;
  }
}

// ---------------------------------------------------------------- implicit GEMM
// GEMM view: M = F (A = w as [F, C*KH*KW]), Ncol = N*OH*OW pixels, K = C*KH*KW.
// The B tile is gathered from x on the fly (im2col never materialised). BM=64, BN=128, BK=8, TM=TN=8.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) k_conv_igemm(Shape s, const float* __restrict__ x,
                                                                     const float* __restrict__ w,
                                                                     float* __restrict__ y, int relu) {
  constexpr int NT = (BM / TM) * (BN / TN);
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  const int M = s.F, K = s.C * s.KH * s.KW;
  const int P = s.OH * s.OW, NP = s.N * P;
  const int tid = threadIdx.x;
  const int tcol = tid % (BN / TN), trow = tid / (BN / TN);
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;
  float acc[TM][TN] = {{0.f}};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = tid; i < BM * BK; i += NT) {
      int r = i / BK, c = i % BK;
      As[c][r] = (bm + r < M && k0 + c < K) ? w[(size_t)(bm + r) * K + k0 + c] : 0.f;
    }
    for (int i = tid; i < BK * BN; i += NT) {
      int r = i / BN, cc = i % BN;
      int k = k0 + r, p = bn + cc;
      float v = 0.f;
      if (k < K && p < NP) {
        int n = p / P, pp = p % P, oh = pp / s.OW, ow = pp % s.OW;
        int kj = k % s.KW, ki = (k / s.KW) % s.KH, c = k / (s.KW * s.KH);
        int ih = oh * s.S - s.P + ki, iw = ow * s.S - s.P + kj;
        if (ih >= 0 && ih < s.H && iw >= 0 && iw < s.W) v = x[(((size_t)n * s.C + c) * s.H + ih) * s.W + iw];
      }
      Bs[r][cc] = v;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) ra[i] = As[k][trow * TM + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tcol * TN + j];
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
      int f = bm + trow * TM + i, p = bn + tcol * TN + j;
      if (f < M && p < NP) {
        int n = p / P, pp = p % P;
        float v = acc[i][j];
        y[((size_t)n * s.F + f) * P + pp] = relu ? fmaxf(v, 0.f) : v;
      }
    }
}

// ---------------------------------------------------------------- 2x2-output direct conv (fused / unfused)
// Each thread computes the 2x2 conv outputs that feed one maxpool window, for FT filters at once
// (input pixels are reused across the FT filters from registers).
// FUSE=true: ReLU + max in registers, write 1 value per filter. FUSE=false: write the 4 raw conv outputs.
template <bool FUSE, int FT>
__global__ void k_conv2x2(Shape s, const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y) {
  const int PH = s.OH / 2, PW = s.OW / 2;
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y;
  int nf = blockIdx.z;
  int nfb = CDIV(s.F, FT);
  int n = nf / nfb, f0 = (nf % nfb) * FT;
  if (px >= PW) return;
  float acc[FT][4];
#pragma unroll
  for (int t = 0; t < FT; ++t) acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.f;
  const int oh0 = 2 * py, ow0 = 2 * px;
  for (int c = 0; c < s.C; ++c) {
    const float* xc = x + ((size_t)n * s.C + c) * s.H * s.W;
    for (int i = 0; i < s.KH; ++i) {
      for (int j = 0; j < s.KW; ++j) {
        float xv[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int ih = (oh0 + q / 2) * s.S - s.P + i, iw = (ow0 + q % 2) * s.S - s.P + j;
          xv[q] = (ih >= 0 && ih < s.H && iw >= 0 && iw < s.W) ? __ldg(xc + ih * s.W + iw) : 0.f;
        }
#pragma unroll
        for (int t = 0; t < FT; ++t) {
          if (f0 + t < s.F) {
            float wv = __ldg(w + (((size_t)(f0 + t) * s.C + c) * s.KH + i) * s.KW + j);
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[t][q] += xv[q] * wv;
          }
        }
      }
    }
  }
#pragma unroll
  for (int t = 0; t < FT; ++t) {
    int f = f0 + t;
    if (f >= s.F) continue;
    if (FUSE) {
      float m = fmaxf(fmaxf(acc[t][0], acc[t][1]), fmaxf(acc[t][2], acc[t][3]));
      y[(((size_t)n * s.F + f) * PH + py) * PW + px] = fmaxf(m, 0.f);
    } else {
      float* yo = y + (((size_t)n * s.F + f) * s.OH + oh0) * s.OW + ow0;
      yo[0] = acc[t][0]; yo[1] = acc[t][1]; yo[s.OW] = acc[t][2]; yo[s.OW + 1] = acc[t][3];
    }
  }
}

__global__ void k_relu(const float* __restrict__ a, float* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    b[i] = fmaxf(a[i], 0.f);
}

// maxpool 2x2 stride 2 on (N*F, OH, OW); RELU=true also fuses the ReLU (epilogue-only fusion).
template <bool RELU>
__global__ void k_pool(const float* __restrict__ a, float* __restrict__ b, int NF, int OH, int OW) {
  int PH = OH / 2, PW = OW / 2;
  size_t total = (size_t)NF * PH * PW;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
    int px = idx % PW, py = (idx / PW) % PH;
    size_t nf = idx / ((size_t)PW * PH);
    const float* p = a + (nf * OH + 2 * py) * OW + 2 * px;
    float m = fmaxf(fmaxf(p[0], p[1]), fmaxf(p[OW], p[OW + 1]));
    b[idx] = RELU ? fmaxf(m, 0.f) : m;
  }
}

static Shape mk(int N, int C, int H, int W, int F, int KH, int KW, int S, int P) {
  Shape s{N, C, H, W, F, KH, KW, S, P, (H + 2 * P - KH) / S + 1, (W + 2 * P - KW) / S + 1};
  return s;
}

extern "C" {
int conv_direct(int N, int C, int H, int W, int F, int KH, int KW, int S, int P, const float* x, const float* w,
                float* y) {
  Shape s = mk(N, C, H, W, F, KH, KW, S, P);
  dim3 b(128), g(CDIV(s.OW, 128), s.OH, N * F);
  k_conv_direct<<<g, b>>>(s, x, w, y);
  return (int)cudaGetLastError();
}

int conv_im2col(int N, int C, int H, int W, int F, int KH, int KW, int S, int P, const float* x, float* cols) {
  Shape s = mk(N, C, H, W, F, KH, KW, S, P);
  k_im2col<<<170 * 16, 256>>>(s, x, cols);
  return (int)cudaGetLastError();
}

int conv_igemm(int N, int C, int H, int W, int F, int KH, int KW, int S, int P, const float* x, const float* w,
               float* y, int relu) {
  Shape s = mk(N, C, H, W, F, KH, KW, S, P);
  int NP = N * s.OH * s.OW;
  dim3 g(CDIV(NP, 128), CDIV(F, 64));
  k_conv_igemm<64, 128, 8, 8, 8><<<g, 128>>>(s, x, w, y, relu);
  return (int)cudaGetLastError();
}

// fuse=1: y is (N,F,OH/2,OW/2) = maxpool(relu(conv)). fuse=0: y is the raw conv (N,F,OH,OW).
int conv2x2(int fuse, int N, int C, int H, int W, int F, int KH, int KW, int S, int P, const float* x,
            const float* w, float* y) {
  Shape s = mk(N, C, H, W, F, KH, KW, S, P);
  if (s.OH % 2 || s.OW % 2) return -1;
  constexpr int FT = 8;
  int PW = s.OW / 2, PH = s.OH / 2;
  int bx = PW >= 64 ? 64 : 32;
  dim3 b(bx), g(CDIV(PW, bx), PH, N * CDIV(F, FT));
  if (fuse) k_conv2x2<true, FT><<<g, b>>>(s, x, w, y);
  else k_conv2x2<false, FT><<<g, b>>>(s, x, w, y);
  return (int)cudaGetLastError();
}

int relu(const float* a, float* b, size_t n) {
  k_relu<<<170 * 8, 256>>>(a, b, n);
  return (int)cudaGetLastError();
}

int maxpool2x2(int with_relu, const float* a, float* b, int NF, int OH, int OW) {
  if (with_relu) k_pool<true><<<170 * 8, 256>>>(a, b, NF, OH, OW);
  else k_pool<false><<<170 * 8, 256>>>(a, b, NF, OH, OW);
  return (int)cudaGetLastError();
}
}
