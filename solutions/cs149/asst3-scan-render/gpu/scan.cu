// CS149 asst3 Part 2: exclusive scan and find_repeats on the GPU.
//   naive    : starter-style work-efficient Blelloch on global memory, padded to 2^k,
//              one kernel launch per up-sweep / down-sweep level
//   smem     : Blelloch block scan in shared memory (1024 elems / block, bank-conflict padding)
//              + recursive scan of block sums + uniform add
//   smem_nopad: same without the padding (shows the bank-conflict cost)
//   raking   : 16 elems / thread in registers, warp-shuffle scan, recursive block sums
//   thrust   : thrust::exclusive_scan     cub: cub::DeviceScan::ExclusiveSum
// find_repeats = flag kernel -> exclusive scan -> scatter, compared with thrust::copy_if.
// Usage: scan <scan.jsonl> <find_repeats.jsonl> [maxlog=28]
#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <cmath>
#include <cstdint>
#include <functional>

#include "common.h"

// ---------------------------------------------------------------- naive (starter structure)
__global__ void upsweep_k(int* a, long long m, long long two_d) {
  long long i = (long long)(blockIdx.x * (long long)blockDim.x + threadIdx.x) * 2 * two_d;
  if (i < m) a[i + 2 * two_d - 1] += a[i + two_d - 1];
}
__global__ void downsweep_k(int* a, long long m, long long two_d) {
  long long i = (long long)(blockIdx.x * (long long)blockDim.x + threadIdx.x) * 2 * two_d;
  if (i < m) {
    int t = a[i + two_d - 1];
    a[i + two_d - 1] = a[i + 2 * two_d - 1];
    a[i + 2 * two_d - 1] += t;
  }
}
static long long next_pow2(long long n) {
  long long m = 1;
  while (m < n) m <<= 1;
  return m;
}
static int g_launches = 0;
// in -> out, out must hold next_pow2(n) ints.
static void scan_naive(const int* in, int* out, long long n) {
  long long m = next_pow2(n);
  CK(cudaMemcpyAsync(out, in, n * 4, cudaMemcpyDeviceToDevice));
  if (m > n) CK(cudaMemsetAsync(out + n, 0, (m - n) * 4));
  for (long long td = 1; td < m; td *= 2) {
    long long threads = m / (2 * td);
    upsweep_k<<<(unsigned)((threads + 255) / 256), 256>>>(out, m, td);
    ++g_launches;
  }
  CK(cudaMemsetAsync(out + m - 1, 0, 4));
  for (long long td = m / 2; td >= 1; td /= 2) {
    long long threads = m / (2 * td);
    downsweep_k<<<(unsigned)((threads + 255) / 256), 256>>>(out, m, td);
    ++g_launches;
  }
}

// ---------------------------------------------------------------- shared-memory Blelloch
constexpr int SB = 512;          // threads per block
constexpr int SE = 2 * SB;       // elements per block
#define CF(i) ((i) >> 5)
template <bool PAD>
__global__ void blelloch_block(const int* __restrict__ in, int* __restrict__ out,
                               int* __restrict__ sums, long long n) {
  __shared__ int s[SE + SE / 32];
  int t = threadIdx.x;
  long long base = (long long)blockIdx.x * SE;
  int ai = t, bi = t + SB;
  int sa = ai + (PAD ? CF(ai) : 0), sb = bi + (PAD ? CF(bi) : 0);
  s[sa] = base + ai < n ? in[base + ai] : 0;
  s[sb] = base + bi < n ? in[base + bi] : 0;
  int off = 1;
  for (int d = SE >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (t < d) {
      int a = off * (2 * t + 1) - 1, b = off * (2 * t + 2) - 1;
      if (PAD) a += CF(a), b += CF(b);
      s[b] += s[a];
    }
    off <<= 1;
  }
  if (t == 0) {
    int last = SE - 1 + (PAD ? CF(SE - 1) : 0);
    if (sums) sums[blockIdx.x] = s[last];
    s[last] = 0;
  }
  for (int d = 1; d < SE; d <<= 1) {
    off >>= 1;
    __syncthreads();
    if (t < d) {
      int a = off * (2 * t + 1) - 1, b = off * (2 * t + 2) - 1;
      if (PAD) a += CF(a), b += CF(b);
      int x = s[a];
      s[a] = s[b];
      s[b] += x;
    }
  }
  __syncthreads();
  if (base + ai < n) out[base + ai] = s[sa];
  if (base + bi < n) out[base + bi] = s[sb];
}

// ---------------------------------------------------------------- raking + warp shuffle
constexpr int RB = 256, RI = 16, RE = RB * RI;  // 4096 elems / block
__device__ __forceinline__ int rpad(int i) { return i + (i >> 5); }
__global__ void raking_block(const int* __restrict__ in, int* __restrict__ out,
                             int* __restrict__ sums, long long n) {
  __shared__ int s[RE + RE / 32];
  __shared__ int wsum[RB / 32];
  int t = threadIdx.x, lane = t & 31, w = t >> 5;
  long long base = (long long)blockIdx.x * RE;
#pragma unroll
  for (int j = 0; j < RI; ++j) {
    int i = j * RB + t;
    s[rpad(i)] = base + i < n ? in[base + i] : 0;
  }
  __syncthreads();
  int v[RI], tot = 0;
#pragma unroll
  for (int j = 0; j < RI; ++j) {
    v[j] = tot;
    tot += s[rpad(t * RI + j)];
  }
  int inc = tot;  // inclusive warp scan of per-thread totals
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += y;
  }
  if (lane == 31) wsum[w] = inc;
  __syncthreads();
  if (w == 0) {
    int x = lane < RB / 32 ? wsum[lane] : 0, xi = x;
#pragma unroll
    for (int o = 1; o < RB / 32; o <<= 1) {
      int y = __shfl_up_sync(0xffffffffu, xi, o);
      if (lane >= o) xi += y;
    }
    if (lane < RB / 32) wsum[lane] = xi - x;  // exclusive warp offsets
    if (lane == RB / 32 - 1 && sums) sums[blockIdx.x] = xi;
  }
  __syncthreads();
  int pre = wsum[w] + inc - tot;
#pragma unroll
  for (int j = 0; j < RI; ++j) s[rpad(t * RI + j)] = v[j] + pre;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < RI; ++j) {
    int i = j * RB + t;
    if (base + i < n) out[base + i] = s[rpad(i)];
  }
}

__global__ void uniform_add(int* out, const int* incr, long long n, int per_block) {
  long long base = (long long)blockIdx.x * per_block;
  int add = incr[blockIdx.x];
  for (int i = threadIdx.x; i < per_block; i += blockDim.x)
    if (base + i < n) out[base + i] += add;
}

// Recursive block-scan driver; level buffers are preallocated.
struct Rec {
  int per_block;
  std::function<void(const int*, int*, int*, long long, unsigned)> block;
  std::vector<int*> lvl_sums, lvl_scan;
  void alloc(long long n) {
    while (n > per_block) {
      long long nb = (n + per_block - 1) / per_block;
      int *a, *b;
      CK(cudaMalloc(&a, nb * 4));
      CK(cudaMalloc(&b, nb * 4));
      lvl_sums.push_back(a);
      lvl_scan.push_back(b);
      n = nb;
    }
  }
  void run(const int* in, int* out, long long n, size_t lvl = 0) {
    unsigned nb = (unsigned)((n + per_block - 1) / per_block);
    if (nb == 1) {
      block(in, out, nullptr, n, 1);
      ++g_launches;
      return;
    }
    block(in, out, lvl_sums[lvl], n, nb);
    run(lvl_sums[lvl], lvl_scan[lvl], nb, lvl + 1);
    uniform_add<<<nb, 256>>>(out, lvl_scan[lvl], n, per_block);
    g_launches += 2;
  }
};

// ---------------------------------------------------------------- find_repeats
__global__ void repeat_flags(const int* a, int* f, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) f[i] = (i + 1 < n && a[i] == a[i + 1]) ? 1 : 0;
}
__global__ void repeat_scatter(const int* a, const int* pos, int* out, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i + 1 < n && a[i] == a[i + 1]) out[pos[i]] = (int)i;
}
struct IsRepeat {
  const int* a;
  __device__ bool operator()(int i) const { return a[i] == a[i + 1]; }
};

static uint32_t hash32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

int main(int argc, char** argv) {
  Jsonl out(argc > 1 ? argv[1] : "scan.jsonl");
  Jsonl outf(argc > 2 ? argv[2] : "find_repeats.jsonl");
  if (argc > 3 && std::string(argv[3]) == "attrs") {  // kernel resources + theoretical occupancy
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    struct K {
      const char* name;
      const void* f;
      int block;
    };
    for (K k : {K{"upsweep_k", (const void*)upsweep_k, 256},
                K{"blelloch_block_pad", (const void*)blelloch_block<true>, SB},
                K{"blelloch_block_nopad", (const void*)blelloch_block<false>, SB},
                K{"raking_block", (const void*)raking_block, RB},
                K{"uniform_add", (const void*)uniform_add, 256}}) {
      cudaFuncAttributes a;
      CK(cudaFuncGetAttributes(&a, k.f));
      int blocks = 0;
      CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.f, k.block, 0));
      out.line(obj(kvs("kind", "attrs"), kvs("kernel", k.name), kvi("block", k.block),
                   kvi("regs", a.numRegs), kvi("static_smem", (long long)a.sharedSizeBytes),
                   kvi("blocks_per_sm", blocks),
                   kv("occupancy", blocks * k.block / (double)p.maxThreadsPerMultiProcessor)));
    }
    return 0;
  }
  int maxlog = argc > 3 ? std::atoi(argv[3]) : 28;
  const long long NMAX = 1LL << maxlog;
  int *din, *dout, *dpad;
  CK(cudaMalloc(&din, NMAX * 4));
  CK(cudaMalloc(&dout, NMAX * 4));
  CK(cudaMalloc(&dpad, NMAX * 4));
  size_t cub_bytes = 0;
  CK(cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, din, dout, (int)NMAX));
  void* dcub;
  CK(cudaMalloc(&dcub, cub_bytes));
  std::vector<int> h(NMAX), ref(NMAX), got(NMAX);
  for (long long i = 0; i < NMAX; ++i) h[i] = hash32((uint32_t)i) & 7;  // sum < 2^31 at 2^28
  CK(cudaMemcpy(din, h.data(), NMAX * 4, cudaMemcpyHostToDevice));

  Rec smem{SE, [](const int* i, int* o, int* s, long long n, unsigned nb) {
             blelloch_block<true><<<nb, SB>>>(i, o, s, n);
           }};
  Rec nopad{SE, [](const int* i, int* o, int* s, long long n, unsigned nb) {
              blelloch_block<false><<<nb, SB>>>(i, o, s, n);
            }};
  Rec rak{RE, [](const int* i, int* o, int* s, long long n, unsigned nb) {
            raking_block<<<nb, RB>>>(i, o, s, n);
          }};
  smem.alloc(NMAX);
  nopad.alloc(NMAX);
  rak.alloc(NMAX);

  std::vector<long long> ns = {1000, 1024, 4096, 10000, 65536, 100000, 1 << 18, (1 << 20),
                               (1 << 20) + 1, 1000000, 3000007, 1 << 22, 10000000,
                               (1 << 24) - 1, 1 << 24, 1 << 26, 100000000, 1 << 27, 1 << 28};
  const bool single = argc > 4;  // profiling mode: one size, one timed rep, no find_repeats
  if (single) ns = {std::atoll(argv[4])};
  for (long long n : ns) {
    if (n > NMAX) continue;
    long long acc = 0;
    for (long long i = 0; i < n; ++i) ref[i] = (int)acc, acc += h[i];
    struct Algo {
      const char* name;
      std::function<void()> fn;
    };
    std::vector<Algo> algos = {
        {"naive", [&] { scan_naive(din, dpad, n); }},
        {"smem_nopad", [&] { nopad.run(din, dout, n); }},
        {"smem", [&] { smem.run(din, dout, n); }},
        {"raking", [&] { rak.run(din, dout, n); }},
        {"thrust", [&] { thrust::exclusive_scan(thrust::device, din, din + n, dout); }},
        {"cub", [&] { size_t b = cub_bytes;
                      CK(cub::DeviceScan::ExclusiveSum(dcub, b, din, dout, (int)n)); }},
    };
    for (auto& al : algos) {
      CK(cudaMemset(dout, 0xff, n * 4));
      CK(cudaMemset(dpad, 0xff, n * 4));
      g_launches = 0;
      al.fn();
      CK(cudaDeviceSynchronize());
      int launches = g_launches;
      const int* res = std::string(al.name) == "naive" ? dpad : dout;
      CK(cudaMemcpy(got.data(), res, n * 4, cudaMemcpyDeviceToHost));
      long long bad = 0;
      for (long long i = 0; i < n; ++i) bad += got[i] != ref[i];
      int reps = single ? 1 : n >= (1 << 26) ? 10 : 30;
      double ms = time_gpu(al.fn, reps, single ? 0 : 3);
      out.line(obj(kvs("kind", "scan"), kvs("algo", al.name), kvi("n", n),
                   kvi("padded_n", next_pow2(n)), kvi("pow2", (n & (n - 1)) == 0),
                   kv("ms", ms), kv("gelem_s", n / ms / 1e6),
                   kv("eff_gbs", 8.0 * n / ms / 1e6), kvi("kernel_launches", launches),
                   kvi("mismatch", bad)));
    }
  }

  // ------------------------------------------------------------ find_repeats
  if (single) return 0;
  std::vector<long long> fns = {10000, 100000, 1000000, 10000000, 1 << 24, 100000000};
  int *dflag, *dpos, *dres;
  CK(cudaMalloc(&dflag, NMAX * 4));
  CK(cudaMalloc(&dpos, NMAX * 4));
  CK(cudaMalloc(&dres, NMAX * 4));
  for (int range : {100, 2}) {
    for (long long n : fns) {
      if (n > NMAX) continue;
      std::vector<int> a(n);
      for (long long i = 0; i < n; ++i) a[i] = hash32((uint32_t)(i * 2654435761u + range)) % range;
      CK(cudaMemcpy(din, a.data(), n * 4, cudaMemcpyHostToDevice));
      std::vector<int> expect;
      for (long long i = 0; i + 1 < n; ++i)
        if (a[i] == a[i + 1]) expect.push_back((int)i);
      unsigned g = (unsigned)((n + 255) / 256);
      int count = 0;
      auto ours = [&](Rec& r) {
        repeat_flags<<<g, 256>>>(din, dflag, n);
        r.run(dflag, dpos, n);
        repeat_scatter<<<g, 256>>>(din, dpos, dres, n);
        CK(cudaMemcpy(&count, dpos + n - 1, 4, cudaMemcpyDeviceToHost));  // flag[n-1] == 0
      };
      auto thr = [&] {
        int* e = thrust::copy_if(thrust::device, thrust::counting_iterator<int>(0),
                                 thrust::counting_iterator<int>((int)n - 1), dres, IsRepeat{din});
        count = (int)(e - dres);
      };
      struct FA {
        const char* name;
        std::function<void()> fn;
      };
      std::vector<FA> fa = {{"scan_smem", [&] { ours(smem); }},
                            {"scan_raking", [&] { ours(rak); }},
                            {"thrust_copy_if", thr}};
      for (auto& f : fa) {
        CK(cudaMemset(dres, 0xff, n * 4));
        f.fn();
        CK(cudaDeviceSynchronize());
        std::vector<int> r(count);
        CK(cudaMemcpy(r.data(), dres, size_t(count) * 4, cudaMemcpyDeviceToHost));
        bool ok = r == expect;
        double ms = time_gpu(f.fn, n >= 10000000 ? 10 : 30, 2);
        outf.line(obj(kvs("kind", "find_repeats"), kvs("algo", f.name), kvi("n", n),
                      kvi("value_range", range), kvi("repeats", count),
                      kvi("expected", (long long)expect.size()), kvi("correct", ok), kv("ms", ms),
                      kv("gelem_s", n / ms / 1e6)));
      }
    }
  }
  return 0;
}
