// CS149 asst3 Part 3: ordered, atomic circle renderer.
// Semantics (as in the official starter): image starts white; for every pixel, circles are
// composited in input order with c <- a*rgb + (1-a)*c; plain scenes use a = 0.5, snow scenes use
// a radial falloff. Pixel centers are ((x+0.5)/W, (y+0.5)/H) in normalized coordinates.
//   naive      : one thread per circle, read-modify-write of its bounding box (starter; WRONG)
//   allcircles : one thread per pixel, loops over every circle in order (correct, O(N*P))
//   binned     : global binning = per-circle tile counts -> exclusive scan -> emit (tile,circle)
//                pairs -> stable radix sort by tile (+ scan of per-tile counts for offsets);
//                one block per tile, each pixel walks its tile list from global memory
//   binned_smem: same lists, circles staged into shared memory in batches of blockDim
//   blockscan  : no global lists; each block tests blockDim circles at a time against its tile,
//                block-wide exclusive scan of the hit flags compacts them in order into shared
//                memory, then every pixel shades the compacted batch
// Usage: render <out.jsonl> <imgdir> [scene_filter|all] [ncu]
#include <cub/cub.cuh>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "common.h"

#define HD __host__ __device__ __forceinline__

struct SceneH {
  std::string name;
  int snow = 0;
  std::vector<float> x, y, z, r, cr, cg, cb;
  int n() const { return (int)x.size(); }
  void add(float px, float py, float pz, float rad, float R, float G, float B) {
    x.push_back(px), y.push_back(py), z.push_back(pz), r.push_back(rad);
    cr.push_back(R), cg.push_back(G), cb.push_back(B);
  }
};
struct SceneD {
  int n, snow;
  const float *x, *y, *z, *r, *cr, *cg, *cb;
};

HD void shade(float cx, float cy, float cz, float rad, float R, float G, float B, int snow,
              float fx, float fy, float4& c) {
  float dx = cx - fx, dy = cy - fy;
  float d2 = dx * dx + dy * dy;
  if (d2 > rad * rad) return;
  float a;
  if (snow) {
    float nd = sqrtf(d2) / rad;
    R = 1.f - 0.6f * nd;
    G = 1.f - 0.35f * nd;
    B = 1.f;
    float ma = .6f + .4f * (1.f - cz);
    ma = 0.5f * fmaxf(fminf(ma, 1.f), 0.f);
    a = ma * expf(-4.f * nd * nd);
  } else {
    a = 0.5f;
  }
  float om = 1.f - a;
  c.x = a * R + om * c.x;
  c.y = a * G + om * c.y;
  c.z = a * B + om * c.z;
}

// Conservative inclusive pixel bounding box; empty if x0 > x1 or y0 > y1.
HD void circ_bbox(float cx, float cy, float rad, int W, int H, int& x0, int& x1, int& y0, int& y1) {
  x0 = max(0, (int)floorf((cx - rad) * W));
  x1 = min(W - 1, (int)ceilf((cx + rad) * W));
  y0 = max(0, (int)floorf((cy - rad) * H));
  y1 = min(H - 1, (int)ceilf((cy + rad) * H));
}

// ------------------------------------------------------------------ scenes (deterministic)
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ULL + 1) {}
  float u() {  // xorshift64*, uniform [0,1)
    s ^= s >> 12, s ^= s << 25, s ^= s >> 27;
    return (float)((s * 2685821657736338717ULL) >> 40) / 16777216.f;
  }
};
static SceneH make_scene(const std::string& name) {
  SceneH s;
  s.name = name;
  Rng g(std::hash<std::string>{}(name) & 0xffffff);
  auto rnd = [&](int n, float rmin, float rmax) {
    for (int i = 0; i < n; ++i) {
      float x = g.u(), y = g.u(), r = rmin + (rmax - rmin) * g.u();
      s.add(x, y, 0.f, r, g.u(), g.u(), g.u());
    }
  };
  if (name == "rgb") {
    s.add(0.40f, 0.45f, 0, 0.30f, 1, 0, 0);
    s.add(0.60f, 0.45f, 0, 0.30f, 0, 1, 0);
    s.add(0.50f, 0.65f, 0, 0.30f, 0, 0, 1);
  } else if (name == "rand10k") {
    rnd(10000, 0.01f, 0.06f);
  } else if (name == "rand100k") {
    rnd(100000, 0.005f, 0.03f);
  } else if (name == "rand1M") {
    rnd(1000000, 0.001f, 0.007f);
  } else if (name == "micro2M") {
    rnd(2000000, 0.0005f, 0.0015f);
  } else if (name == "biglittle" || name == "littlebig") {
    SceneH big, little;
    for (int i = 0; i < 20; ++i)
      big.add(g.u(), g.u(), 0, 0.15f + 0.2f * g.u(), g.u(), g.u(), g.u());
    for (int i = 0; i < 50000; ++i)
      little.add(g.u(), g.u(), 0, 0.002f + 0.008f * g.u(), g.u(), g.u(), g.u());
    const SceneH& first = name == "biglittle" ? big : little;  // biglittle: big ones at the back
    const SceneH& second = name == "biglittle" ? little : big;
    for (const SceneH* p : {&first, &second})
      for (int i = 0; i < p->n(); ++i)
        s.add(p->x[i], p->y[i], 0, p->r[i], p->cr[i], p->cg[i], p->cb[i]);
  } else if (name == "pattern") {
    for (int layer = 0; layer < 2; ++layer) {
      int k = layer ? 64 : 32;
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i) {
          float x = (i + 0.5f) / k, y = (j + 0.5f) / k;
          float r = (0.35f + 0.3f * sinf(6.2832f * (x + 2 * y))) / k * (layer ? 1.0f : 1.6f);
          s.add(x, y, 0, r, 0.5f + 0.5f * sinf(9 * x), 0.5f + 0.5f * cosf(7 * y), layer ? 0.2f : 0.9f);
        }
    }
  } else if (name.rfind("sweep_", 0) == 0) {  // rand10k radii scaled so coverage stays ~45
    int n = std::atoi(name.c_str() + 6);
    float k = std::sqrt(1e4f / n);
    rnd(n, 0.01f * k, 0.06f * k);
  } else if (name == "snowsingle") {
    s.snow = 1;
    std::vector<float> zs(50000);
    for (float& z : zs) z = g.u();
    std::sort(zs.begin(), zs.end(), std::greater<float>());  // far (z=1) first
    for (float z : zs) {
      float x = g.u(), y = g.u();
      s.add(x, y, z, 0.003f + 0.03f * (1.f - z) * (1.f - z), 1, 1, 1);
    }
  } else {
    std::fprintf(stderr, "unknown scene %s\n", name.c_str());
    std::exit(1);
  }
  return s;
}

// ------------------------------------------------------------------ CPU reference
static void render_cpu(const SceneH& s, int W, int H, std::vector<float4>& img) {
  img.assign((size_t)W * H, make_float4(1, 1, 1, 1));
  float iw = 1.f / W, ih = 1.f / H;
  for (int k = 0; k < s.n(); ++k) {
    int x0, x1, y0, y1;
    circ_bbox(s.x[k], s.y[k], s.r[k], W, H, x0, x1, y0, y1);
    for (int y = y0; y <= y1; ++y) {
      float fy = (y + 0.5f) * ih;
      for (int x = x0; x <= x1; ++x)
        shade(s.x[k], s.y[k], s.z[k], s.r[k], s.cr[k], s.cg[k], s.cb[k], s.snow,
              (x + 0.5f) * iw, fy, img[(size_t)y * W + x]);
    }
  }
}

// ------------------------------------------------------------------ GPU kernels
#define CIRCLE(s, k) s.x[k], s.y[k], s.z[k], s.r[k], s.cr[k], s.cg[k], s.cb[k]

__global__ void k_clear(float4* img, int np) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < np) img[i] = make_float4(1, 1, 1, 1);
}

__global__ void k_naive(SceneD s, float4* img, int W, int H) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s.n) return;
  int x0, x1, y0, y1;
  circ_bbox(s.x[k], s.y[k], s.r[k], W, H, x0, x1, y0, y1);
  float iw = 1.f / W, ih = 1.f / H;
  for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
      float4 c = img[y * W + x];  // non-atomic read-modify-write, no ordering
      shade(CIRCLE(s, k), s.snow, (x + 0.5f) * iw, (y + 0.5f) * ih, c);
      img[y * W + x] = c;
    }
}

__global__ void k_allcircles(SceneD s, float4* img, int W, int H) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  float fx = (x + 0.5f) * (1.f / W), fy = (y + 0.5f) * (1.f / H);
  float4 c = img[y * W + x];
  for (int k = 0; k < s.n; ++k) shade(CIRCLE(s, k), s.snow, fx, fy, c);
  img[y * W + x] = c;
}

__global__ void k_bin_count(SceneD s, int W, int H, int T, int TX, int* cnt, int* tile_cnt) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s.n) return;
  int x0, x1, y0, y1;
  circ_bbox(s.x[k], s.y[k], s.r[k], W, H, x0, x1, y0, y1);
  if (x0 > x1 || y0 > y1) {
    cnt[k] = 0;
    return;
  }
  int tx0 = x0 / T, tx1 = x1 / T, ty0 = y0 / T, ty1 = y1 / T;
  cnt[k] = (tx1 - tx0 + 1) * (ty1 - ty0 + 1);
  for (int ty = ty0; ty <= ty1; ++ty)
    for (int tx = tx0; tx <= tx1; ++tx) atomicAdd(&tile_cnt[ty * TX + tx], 1);
}

__global__ void k_bin_emit(SceneD s, int W, int H, int T, int TX, const int* off, int* keys,
                           int* vals) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= s.n) return;
  int x0, x1, y0, y1;
  circ_bbox(s.x[k], s.y[k], s.r[k], W, H, x0, x1, y0, y1);
  if (x0 > x1 || y0 > y1) return;
  int j = off[k];
  for (int ty = y0 / T; ty <= y1 / T; ++ty)
    for (int tx = x0 / T; tx <= x1 / T; ++tx) keys[j] = ty * TX + tx, vals[j] = k, ++j;
}

__global__ void __launch_bounds__(1024) k_render_binned(SceneD s, float4* img, int W, int H, const int* tile_start,
                                const int* list) {
  int tile = blockIdx.y * gridDim.x + blockIdx.x;
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;
  float fx = (x + 0.5f) * (1.f / W), fy = (y + 0.5f) * (1.f / H);
  float4 c = img[y * W + x];
  for (int j = tile_start[tile], e = tile_start[tile + 1]; j < e; ++j) {
    int k = list[j];
    shade(CIRCLE(s, k), s.snow, fx, fy, c);
  }
  img[y * W + x] = c;
}

// shared-memory circle batch, SoA, 7 floats per circle
struct Batch {
  float *x, *y, *z, *r, *R, *G, *B;
  __device__ Batch(float* sh, int bs)
      : x(sh), y(sh + bs), z(sh + 2 * bs), r(sh + 3 * bs), R(sh + 4 * bs), G(sh + 5 * bs),
        B(sh + 6 * bs) {}
  __device__ void put(int i, const SceneD& s, int k) {
    x[i] = s.x[k], y[i] = s.y[k], z[i] = s.z[k], r[i] = s.r[k];
    R[i] = s.cr[k], G[i] = s.cg[k], B[i] = s.cb[k];
  }
};

__global__ void __launch_bounds__(1024) k_render_binned_smem(SceneD s, float4* img, int W, int H, const int* tile_start,
                                     const int* list) {
  extern __shared__ float sh[];
  int bs = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
  Batch b(sh, bs);
  int tile = blockIdx.y * gridDim.x + blockIdx.x;
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  bool in = x < W && y < H;
  float fx = (x + 0.5f) * (1.f / W), fy = (y + 0.5f) * (1.f / H);
  float4 c = in ? img[y * W + x] : make_float4(0, 0, 0, 0);
  int beg = tile_start[tile], end = tile_start[tile + 1];
  for (int base = beg; base < end; base += bs) {
    if (base + tid < end) b.put(tid, s, list[base + tid]);
    __syncthreads();
    int m = min(bs, end - base);
    for (int i = 0; i < m; ++i)
      shade(b.x[i], b.y[i], b.z[i], b.r[i], b.R[i], b.G[i], b.B[i], s.snow, fx, fy, c);
    __syncthreads();
  }
  if (in) img[y * W + x] = c;
}

// block-wide exclusive scan of v (blockDim multiple of 32); ws needs 33 ints
__device__ __forceinline__ int block_excl_scan(int v, int* ws, int& total) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x, lane = tid & 31, w = tid >> 5;
  int nw = (blockDim.x * blockDim.y) >> 5;
  int inc = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int t = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) inc += t;
  }
  if (lane == 31) ws[w] = inc;
  __syncthreads();
  if (w == 0) {
    int x = lane < nw ? ws[lane] : 0, xi = x;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int t = __shfl_up_sync(0xffffffffu, xi, o);
      if (lane >= o) xi += t;
    }
    if (lane < nw) ws[lane] = xi - x;
    if (lane == 31) ws[32] = xi;
  }
  __syncthreads();
  total = ws[32];
  return ws[w] + inc - v;
}

__global__ void __launch_bounds__(1024) k_render_blockscan(SceneD s, float4* img, int W, int H) {
  extern __shared__ float sh[];
  __shared__ int ws[33];
  int bs = blockDim.x * blockDim.y, tid = threadIdx.y * blockDim.x + threadIdx.x;
  Batch b(sh, bs);
  int tx0 = blockIdx.x * blockDim.x, ty0 = blockIdx.y * blockDim.y;
  int tx1 = min(W - 1, tx0 + (int)blockDim.x - 1), ty1 = min(H - 1, ty0 + (int)blockDim.y - 1);
  int x = tx0 + threadIdx.x, y = ty0 + threadIdx.y;
  bool in = x < W && y < H;
  float fx = (x + 0.5f) * (1.f / W), fy = (y + 0.5f) * (1.f / H);
  float4 c = in ? img[y * W + x] : make_float4(0, 0, 0, 0);
  for (int cb = 0; cb < s.n; cb += bs) {
    int k = cb + tid, hit = 0;
    if (k < s.n) {
      int x0, x1, y0, y1;
      circ_bbox(s.x[k], s.y[k], s.r[k], W, H, x0, x1, y0, y1);
      hit = x0 <= tx1 && x1 >= tx0 && y0 <= ty1 && y1 >= ty0;
    }
    int total;
    int pos = block_excl_scan(hit, ws, total);
    if (hit) b.put(pos, s, k);
    __syncthreads();
    for (int i = 0; i < total; ++i)
      shade(b.x[i], b.y[i], b.z[i], b.r[i], b.R[i], b.G[i], b.B[i], s.snow, fx, fy, c);
    __syncthreads();
  }
  if (in) img[y * W + x] = c;
}

// ------------------------------------------------------------------ host drivers
struct Binner {
  int *cnt = nullptr, *off = nullptr, *tcnt = nullptr, *tstart = nullptr;
  int *k1 = nullptr, *v1 = nullptr, *k2 = nullptr, *v2 = nullptr;
  void* tmp = nullptr;
  size_t tmp_bytes = 0, cap = 0;
  long long pairs = 0;
  int ntiles = 0, nmax = 0;
  void grow_tmp(size_t b) {
    if (b > tmp_bytes) {
      if (tmp) CK(cudaFree(tmp));
      tmp_bytes = b * 2;
      CK(cudaMalloc(&tmp, tmp_bytes));
    }
  }
  void build(const SceneD& s, int W, int H, int T) {
    int TX = (W + T - 1) / T, TY = (H + T - 1) / T;
    if (s.n > nmax || TX * TY > ntiles) {  // (re)allocate per-scene buffers (warm-up only)
      nmax = std::max(nmax, s.n);
      ntiles = std::max(ntiles, TX * TY);
      for (int* p : {cnt, off, tcnt, tstart})
        if (p) CK(cudaFree(p));
      CK(cudaMalloc(&cnt, (size_t)nmax * 4 + 4));
      CK(cudaMalloc(&off, (size_t)nmax * 4 + 4));
      CK(cudaMalloc(&tcnt, (size_t)ntiles * 4 + 4));
      CK(cudaMalloc(&tstart, (size_t)ntiles * 4 + 4));
    }
    int nt = TX * TY;
    CK(cudaMemsetAsync(tcnt, 0, (size_t)(nt + 1) * 4));
    k_bin_count<<<(s.n + 255) / 256, 256>>>(s, W, H, T, TX, cnt, tcnt);
    size_t b = 0;
    CK(cub::DeviceScan::ExclusiveSum(nullptr, b, cnt, off, s.n));
    grow_tmp(b);
    b = tmp_bytes;
    CK(cub::DeviceScan::ExclusiveSum(tmp, b, cnt, off, s.n));
    CK(cub::DeviceScan::ExclusiveSum(nullptr, b, tcnt, tstart, nt + 1));
    grow_tmp(b);
    b = tmp_bytes;
    CK(cub::DeviceScan::ExclusiveSum(tmp, b, tcnt, tstart, nt + 1));
    int last[2];
    CK(cudaMemcpy(&last[0], off + s.n - 1, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&last[1], cnt + s.n - 1, 4, cudaMemcpyDeviceToHost));
    pairs = (long long)last[0] + last[1];
    if ((size_t)pairs > cap) {
      for (int* p : {k1, v1, k2, v2})
        if (p) CK(cudaFree(p));
      cap = (size_t)pairs * 5 / 4 + 1024;
      CK(cudaMalloc(&k1, cap * 4));
      CK(cudaMalloc(&v1, cap * 4));
      CK(cudaMalloc(&k2, cap * 4));
      CK(cudaMalloc(&v2, cap * 4));
    }
    k_bin_emit<<<(s.n + 255) / 256, 256>>>(s, W, H, T, TX, off, k1, v1);
    int bits = 1;
    while ((1 << bits) < nt) ++bits;
    CK(cub::DeviceRadixSort::SortPairs(nullptr, b, k1, k2, v1, v2, (int)pairs, 0, bits));
    grow_tmp(b);
    b = tmp_bytes;
    CK(cub::DeviceRadixSort::SortPairs(tmp, b, k1, k2, v1, v2, (int)pairs, 0, bits));
  }
};

int main(int argc, char** argv) {
  Jsonl out(argc > 1 ? argv[1] : "render.jsonl");
  std::string imgdir = argc > 2 ? argv[2] : ".";
  std::string filt = argc > 3 ? argv[3] : "all";
  bool ncu = argc > 4 && std::string(argv[4]) == "ncu";
  const int W = 1024, H = 1024, NP = W * H;
  std::vector<std::string> names = {"rgb",     "rand10k",    "rand100k", "biglittle", "littlebig",
                                    "pattern", "snowsingle", "rand1M",   "micro2M"};
  const bool sweep = filt == "sweep";
  if (sweep) {
    names.clear();
    for (int n : {1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000})
      names.push_back("sweep_" + std::to_string(n));
  }
  if (filt == "attrs") {  // static kernel resources + theoretical occupancy (ncu is not permitted here)
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    struct K {
      const char* name;
      const void* f;
    };
    std::vector<K> ks = {{"k_naive", (const void*)k_naive},
                         {"k_allcircles", (const void*)k_allcircles},
                         {"k_bin_count", (const void*)k_bin_count},
                         {"k_bin_emit", (const void*)k_bin_emit},
                         {"k_render_binned", (const void*)k_render_binned},
                         {"k_render_binned_smem", (const void*)k_render_binned_smem},
                         {"k_render_blockscan", (const void*)k_render_blockscan}};
    for (auto& k : ks) {
      cudaFuncAttributes a;
      CK(cudaFuncGetAttributes(&a, k.f));
      for (int bs : {64, 128, 256, 1024}) {
        int blocks = 0;
        size_t dyn = std::string(k.name).find("smem") != std::string::npos ||
                             std::string(k.name).find("blockscan") != std::string::npos
                         ? 7 * bs * 4
                         : 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k.f, bs, dyn));
        out.line(obj(kvs("kind", "attrs"), kvs("kernel", k.name), kvi("block", bs),
                     kvi("regs", a.numRegs), kvi("static_smem", (long long)a.sharedSizeBytes),
                     kvi("dyn_smem", (long long)dyn), kvi("blocks_per_sm", blocks),
                     kv("occupancy", blocks * bs / (double)p.maxThreadsPerMultiProcessor),
                     kvi("sms", p.multiProcessorCount), kvi("l2_bytes", p.l2CacheSize),
                     kvi("smem_per_sm", (long long)p.sharedMemPerMultiprocessor)));
      }
    }
    return 0;
  }
  float4* dimg;
  CK(cudaMalloc(&dimg, (size_t)NP * sizeof(float4)));
  Binner bin;
  std::vector<float4> ref, got, got2;
  auto save_ppm = [&](const std::string& path, const std::vector<float4>& im) {
    FILE* f = std::fopen(path.c_str(), "wb");
    std::fprintf(f, "P6\n%d %d\n255\n", W, H);
    std::vector<unsigned char> row((size_t)W * 3);
    for (int y = H - 1; y >= 0; --y) {  // y up in scene space
      for (int x = 0; x < W; ++x) {
        const float4& c = im[(size_t)y * W + x];
        float v[3] = {c.x, c.y, c.z};
        for (int ch = 0; ch < 3; ++ch)
          row[x * 3 + ch] = (unsigned char)std::lround(255.f * fminf(fmaxf(v[ch], 0.f), 1.f));
      }
      std::fwrite(row.data(), 1, row.size(), f);
    }
    std::fclose(f);
  };
  auto compare = [&](const std::vector<float4>& a, double& maxd, long long& bad_any,
                     long long& bad_1e3) {
    maxd = 0, bad_any = 0, bad_1e3 = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      double d = std::max({std::fabs(a[i].x - ref[i].x), std::fabs(a[i].y - ref[i].y),
                           std::fabs(a[i].z - ref[i].z)});
      maxd = std::max(maxd, d);
      bad_any += d > 1e-6;
      bad_1e3 += d > 1e-3;
    }
  };

  for (const std::string& name : names) {
    if (filt != "all" && filt != name && !sweep) continue;
    SceneH sh = make_scene(name);
    const int n = sh.n();
    std::vector<float*> dbuf;
    auto up = [&](const std::vector<float>& v) {
      float* p;
      CK(cudaMalloc(&p, v.size() * 4));
      CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
      dbuf.push_back(p);
      return (const float*)p;
    };
    SceneD sd{n, sh.snow, up(sh.x), up(sh.y), up(sh.z), up(sh.r), up(sh.cr), up(sh.cg), up(sh.cb)};
    double cpu_ms = 0;
    if (!ncu) {
      double t0 = wall_ms();
      render_cpu(sh, W, H, ref);
      cpu_ms = wall_ms() - t0;
    }
    // per-scene coverage statistics: total and mean circles covering a pixel
    double cover = 0;
    for (int k = 0; k < n; ++k) cover += M_PI * sh.r[k] * sh.r[k];
    out.line(obj(kvs("kind", "scene"), kvs("scene", name), kvi("n", n), kvi("W", W), kvi("H", H),
                 kv("mean_coverage_per_pixel", cover), kv("cpu_ref_ms", cpu_ms)));
    auto clear = [&] { k_clear<<<(NP + 255) / 256, 256>>>(dimg, NP); };
    int reps = ncu ? 1 : 5, warm = ncu ? 0 : 2;
    auto record = [&](const std::string& stage, int T, double ms, double render_ms, bool check,
                      const std::string& extra) {
      double maxd = -1;
      long long bad = -1, bad3 = -1;
      if (check && !ncu) {
        got.resize(NP);
        CK(cudaMemcpy(got.data(), dimg, (size_t)NP * sizeof(float4), cudaMemcpyDeviceToHost));
        compare(got, maxd, bad, bad3);
      }
      std::string line =
          obj(kvs("kind", "render"), kvs("scene", name), kvi("n", n), kvs("stage", stage),
              kvi("tile", T), kv("ms", ms), kv("render_ms", render_ms), kv("max_abs_diff", maxd),
              kvi("pixels_diff_gt_1e-6", bad), kvi("pixels_diff_gt_1e-3", bad3),
              kv("cpu_ref_ms", cpu_ms), kv("speedup_vs_cpu", ncu ? 0 : cpu_ms / ms));
      line.pop_back();
      out.line(line + extra + "}");
    };

    // (a) naive one thread per circle: incorrect; also measure run-to-run nondeterminism
    {
      auto fn = [&] { clear(); k_naive<<<(n + 127) / 128, 128>>>(sd, dimg, W, H); };
      double ms = time_gpu(fn, reps, warm);
      long long rr = -1;
      if (!ncu) {
        fn();
        CK(cudaDeviceSynchronize());
        got2.resize(NP);
        CK(cudaMemcpy(got2.data(), dimg, (size_t)NP * sizeof(float4), cudaMemcpyDeviceToHost));
        fn();
        CK(cudaDeviceSynchronize());
        got.resize(NP);
        CK(cudaMemcpy(got.data(), dimg, (size_t)NP * sizeof(float4), cudaMemcpyDeviceToHost));
        rr = 0;
        for (int i = 0; i < NP; ++i)
          rr += got[i].x != got2[i].x || got[i].y != got2[i].y || got[i].z != got2[i].z;
        if (!sweep && (name == "rgb" || name == "rand10k" || name == "pattern" || name == "biglittle"))
          save_ppm(imgdir + "/" + name + "_naive.ppm", got);
      }
      record("naive_per_circle", 0, ms, ms, true,
             ",\"pixels_differ_between_runs\":" + std::to_string(rr));
    }
    // (b) correct: per-pixel loop over all circles (skip when it would take seconds per run)
    if ((double)n * NP <= (sweep ? 3.5e11 : 2.2e12)) {
      auto fn = [&] {
        clear();
        k_allcircles<<<dim3(W / 16, H / 16), dim3(16, 16)>>>(sd, dimg, W, H);
      };
      int r = n >= 1000000 ? 1 : reps, w = n >= 1000000 ? 0 : warm;
      double ms = time_gpu(fn, r, w);
      record("allcircles", 16, ms, ms, true, "");
    }
    if (!ncu && !sweep) save_ppm(imgdir + "/" + name + "_ref.ppm", ref);
    // (c) binned (global lists), (d) binned + shared batching, (e) block-scan in-kernel binning
    for (int T : sweep ? std::vector<int>{16} : std::vector<int>{8, 16, 32}) {
      dim3 grid((W + T - 1) / T, (H + T - 1) / T), blk(T, T);
      size_t shb = 7 * T * T * sizeof(float);
      double bin_ms = time_gpu([&] { bin.build(sd, W, H, T); }, reps, warm);
      int nt = grid.x * grid.y;
      std::vector<int> tc(nt);
      CK(cudaMemcpy(tc.data(), bin.tcnt, nt * 4, cudaMemcpyDeviceToHost));
      long long mx = 0, nonempty = 0;
      for (int v : tc) mx = std::max<long long>(mx, v), nonempty += v > 0;
      char ex[256];
      std::snprintf(ex, sizeof ex, ",\"bin_ms\":%.6g,\"pairs\":%lld,\"mean_list\":%.6g,\"max_list\":%lld,\"nonempty_tiles\":%lld",
                    bin_ms, bin.pairs, (double)bin.pairs / nt, mx, nonempty);
      std::string extra = ex;
      for (int smem = 0; smem < 2; ++smem) {
        auto rk = [&] {
          if (smem)
            k_render_binned_smem<<<grid, blk, shb>>>(sd, dimg, W, H, bin.tstart, bin.v2);
          else
            k_render_binned<<<grid, blk>>>(sd, dimg, W, H, bin.tstart, bin.v2);
        };
        double r_ms = time_gpu([&] { clear(); rk(); }, reps, warm);
        double ms = time_gpu([&] { clear(); bin.build(sd, W, H, T); rk(); }, reps, warm);
        record(smem ? "binned_smem" : "binned", T, ms, r_ms, true, extra);
      }
      {
        auto fn = [&] { clear(); k_render_blockscan<<<grid, blk, shb>>>(sd, dimg, W, H); };
        double ms = time_gpu(fn, reps, warm);
        record("blockscan_smem", T, ms, ms, true, ",\"bin_ms\":0");
      }
    }
    for (float* p : dbuf) CK(cudaFree(p));
  }
  return 0;
}
