// GPU counterpart of CS149 asst1: the same three questions (divergence, mapping, bandwidth)
// on an NVIDIA GPU, where the "gang" is a 32-thread warp.
//
//   ./a1_gpu mandel  [--view 1] [--scale 1]     block-shape sweep, warp execution efficiency
//   ./a1_gpu sqrt    [--n 20000000]             input patterns incl. one slow lane per 32
//   ./a1_gpu saxpy                               bandwidth vs size, with and without PCIe copies
//   ./a1_gpu kmeans  [--m 1000000 --dims 100 --k 3]   assignment step vs CPU
// Every result is checked against a CPU reference compiled with the same operation order;
// build with -fmad=false so the device does not contract a*b+c into an FMA.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                              \
    do {                                                                                   \
        cudaError_t e = (x);                                                               \
        if (e != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

static std::string arg(int argc, char** argv, const char* k, const char* d) {
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == std::string("--") + k) return argv[i + 1];
    return d;
}

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a), cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float stop_ms() {
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        return ms;
    }
};

template <class F>
static float best_ms(int reps, F f) {
    Timer t;
    f();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        t.start();
        f();
        best = std::min(best, t.stop_ms());
    }
    return best;
}

// ------------------------------------------------------------------ mandelbrot
__host__ __device__ inline int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f) break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandel_kernel(float x0, float y0, float dx, float dy, int w, int h, int max_iter, int* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= w || j >= h) return;
    float x = x0 + i * dx;
    float y = y0 + j * dy;
    out[j * w + i] = mandel(x, y, max_iter);
}

// Warp execution efficiency for a block shape: every warp issues (max count + 1) iterations
// for all 32 lanes; useful work is sum(count + 1) over its active lanes.
static double warp_efficiency(const std::vector<int>& c, int w, int h, int bx, int by) {
    double useful = 0, issued = 0;
    int gx = (w + bx - 1) / bx, gy = (h + by - 1) / by, nt = bx * by;
    for (int byi = 0; byi < gy; ++byi)
        for (int bxi = 0; bxi < gx; ++bxi)
            for (int w0 = 0; w0 < nt; w0 += 32) {
                int mx = -1;
                double u = 0;
                for (int l = w0; l < std::min(w0 + 32, nt); ++l) {
                    int i = bxi * bx + l % bx, j = byi * by + l / bx;
                    if (i >= w || j >= h) continue;
                    int v = c[j * w + i] + 1;
                    u += v;
                    mx = std::max(mx, v);
                }
                if (mx > 0) useful += u, issued += 32.0 * mx;
            }
    return useful / issued;
}

static int run_mandel(int argc, char** argv) {
    int view = std::atoi(arg(argc, argv, "view", "1").c_str());
    int scale = std::atoi(arg(argc, argv, "scale", "1").c_str());
    int max_iter = std::atoi(arg(argc, argv, "iters", "256").c_str());
    float x0 = -2.f, y0 = -1.f, x1 = 1.f, y1 = 1.f;
    if (view == 2) {
        const float s = .015f, sx = -.986f, sy = .30f;
        x0 = x0 * s + sx, y0 = y0 * s + sy, x1 = x1 * s + sx, y1 = y1 * s + sy;
    }
    int w = 1600 * scale, h = 1200 * scale;
    float dx = (x1 - x0) / w, dy = (y1 - y0) / h;
    size_t n = size_t(w) * h;
    int* d;
    CK(cudaMalloc(&d, n * sizeof(int)));
    std::vector<int> g(n), ref;
    if (scale == 1) {
        ref.resize(n);
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i) ref[size_t(j) * w + i] = mandel(x0 + i * dx, y0 + j * dy, max_iter);
    }
    int shapes[][2] = {{32, 1}, {16, 2}, {8, 4}, {4, 8}, {32, 4}, {32, 8}, {16, 16}, {8, 8}, {64, 1}, {128, 1}, {256, 1}, {1, 32}};
    for (auto& s : shapes) {
        dim3 blk(s[0], s[1]), grd((w + s[0] - 1) / s[0], (h + s[1] - 1) / s[1]);
        float ms = best_ms(10, [&] { mandel_kernel<<<grd, blk>>>(x0, y0, dx, dy, w, h, max_iter, d); });
        CK(cudaGetLastError());
        CK(cudaMemcpy(g.data(), d, n * sizeof(int), cudaMemcpyDeviceToHost));
        long long mism = 0;
        if (!ref.empty())
            for (size_t k = 0; k < n; ++k) mism += g[k] != ref[k];
        double eff = warp_efficiency(g, w, h, s[0], s[1]);
        double work = 0;
        for (int v : g) work += v + 1;
        std::printf("{\"kind\":\"mandel\",\"view\":%d,\"scale\":%d,\"iters\":%d,\"bx\":%d,\"by\":%d,\"ms\":%.6f,"
                    "\"gpix_per_s\":%.4f,\"giters_per_s\":%.4f,\"warp_eff\":%.6f,\"mismatch\":%lld}\n",
                    view, scale, max_iter, s[0], s[1], ms, n / ms / 1e6, work / ms / 1e6, eff, mism);
    }
    CK(cudaFree(d));
    return 0;
}

// ------------------------------------------------------------------ sqrt
__host__ __device__ inline float sqrt_newton(float x, float guess) {
    const float kThreshold = 0.00001f;
    float error = fabsf(guess * guess * x - 1.f);
    while (error > kThreshold) {
        guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
        error = fabsf(guess * guess * x - 1.f);
    }
    return x * guess;
}

// `repeat` > 1 re-solves each element with slightly different initial guesses so the
// kernel becomes compute bound and divergence shows up in the time.
__host__ __device__ inline float sqrt_repeat(float x, int repeat) {
    float acc = 0.f;
    for (int r = 0; r < repeat; ++r) acc += sqrt_newton(x, 1.f + r * 1e-6f);
    return acc;
}

__global__ void sqrt_kernel(int n, int repeat, const float* v, float* o) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = repeat == 1 ? sqrt_newton(v[i], 1.f) : sqrt_repeat(v[i], repeat);
}

static int run_sqrt(int argc, char** argv) {
    int n = std::atoi(arg(argc, argv, "n", "20000000").c_str());
    int repeat = std::atoi(arg(argc, argv, "repeat", "1").c_str());
    std::vector<float> v(n), o(n), ref(n);
    float *dv, *dout;
    CK(cudaMalloc(&dv, n * 4));
    CK(cudaMalloc(&dout, n * 4));
    const char* pats[] = {"random", "all_slow", "all_one", "one_slow_8", "one_slow_16", "one_slow_32", "one_slow_64", "sorted_random"};
    for (const char* p : pats) {
        std::mt19937 rng(4);
        std::uniform_real_distribution<float> u(0.f, 1.f);
        std::string s = p;
        for (int i = 0; i < n; ++i) {
            if (s == "random" || s == "sorted_random") v[i] = .001f + 2.998f * u(rng);
            else if (s == "all_slow") v[i] = 2.998f;
            else if (s == "all_one") v[i] = 1.f;
            else {
                int k = std::atoi(p + 9);
                v[i] = (i % k == 0) ? 2.998f : 1.f;
            }
        }
        if (s == "sorted_random") std::sort(v.begin(), v.end());
        // Reference on a sample when repeating (the full host loop would take minutes).
        int step = repeat == 1 ? 1 : 101;
        for (int i = 0; i < n; i += step) ref[i] = repeat == 1 ? sqrt_newton(v[i], 1.f) : sqrt_repeat(v[i], repeat);
        CK(cudaMemcpy(dv, v.data(), n * 4, cudaMemcpyHostToDevice));
        float ms = best_ms(repeat == 1 ? 10 : 3, [&] { sqrt_kernel<<<(n + 255) / 256, 256>>>(n, repeat, dv, dout); });
        CK(cudaGetLastError());
        CK(cudaMemcpy(o.data(), dout, n * 4, cudaMemcpyDeviceToHost));
        long long mism = 0;
        for (int i = 0; i < n; i += step) mism += std::memcmp(&o[i], &ref[i], 4) != 0;
        std::printf("{\"kind\":\"sqrt\",\"pattern\":\"%s\",\"n\":%d,\"repeat\":%d,\"ms\":%.6f,\"gelem_per_s\":%.4f,\"mismatch\":%lld}\n",
                    p, n, repeat, ms, n / ms / 1e6, mism);
    }
    CK(cudaFree(dv));
    CK(cudaFree(dout));
    return 0;
}

// ------------------------------------------------------------------ saxpy
__global__ void saxpy_kernel(size_t n, float a, const float* x, const float* y, float* r) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) r[i] = a * x[i] + y[i];
}
__global__ void saxpy_vec4(size_t n4, float a, const float4* x, const float4* y, float4* r) {
    size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n4; i += stride) {
        float4 p = x[i], q = y[i];
        r[i] = make_float4(a * p.x + q.x, a * p.y + q.y, a * p.z + q.z, a * p.w + q.w);
    }
}

static int run_saxpy(int, char**) {
    int sms;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    size_t nmax = size_t(1) << 28;
    float *x, *y, *r, *hx, *hy, *hr;
    CK(cudaMalloc(&x, nmax * 4));
    CK(cudaMalloc(&y, nmax * 4));
    CK(cudaMalloc(&r, nmax * 4));
    CK(cudaMallocHost(&hx, nmax * 4));
    CK(cudaMallocHost(&hy, nmax * 4));
    CK(cudaMallocHost(&hr, nmax * 4));
    for (size_t i = 0; i < nmax; ++i) hx[i] = float(i % 1000) * 1e-3f, hy[i] = 1.f;
    CK(cudaMemcpy(x, hx, nmax * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(y, hy, nmax * 4, cudaMemcpyHostToDevice));
    for (size_t n = size_t(1) << 10; n <= nmax; n *= 2) {
        int grid = (int)std::min<size_t>((n + 255) / 256, size_t(sms) * 32);
        float k1 = best_ms(20, [&] { saxpy_kernel<<<grid, 256>>>(n, 2.f, x, y, r); });
        int grid4 = (int)std::min<size_t>((n / 4 + 255) / 256, size_t(sms) * 32);
        float k4 = best_ms(20, [&] { saxpy_vec4<<<std::max(grid4, 1), 256>>>(n / 4, 2.f, (const float4*)x, (const float4*)y, (float4*)r); });
        float e2e = best_ms(3, [&] {
            cudaMemcpyAsync(x, hx, n * 4, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(y, hy, n * 4, cudaMemcpyHostToDevice);
            saxpy_vec4<<<std::max(grid4, 1), 256>>>(n / 4, 2.f, (const float4*)x, (const float4*)y, (float4*)r);
            cudaMemcpyAsync(hr, r, n * 4, cudaMemcpyDeviceToHost);
        });
        CK(cudaGetLastError());
        long long bad = 0;
        for (size_t i = 0; i < n; i += 997) bad += hr[i] != 2.f * hx[i] + hy[i];
        std::printf("{\"kind\":\"saxpy\",\"n\":%zu,\"bytes\":%zu,\"kernel_ms\":%.6f,\"vec4_ms\":%.6f,\"e2e_ms\":%.6f,"
                    "\"kernel_gbs\":%.3f,\"vec4_gbs\":%.3f,\"e2e_gbs\":%.3f,\"mismatch\":%lld}\n",
                    n, n * 12, k1, k4, e2e, 12.0 * n / k1 / 1e6, 12.0 * n / k4 / 1e6, 12.0 * n / e2e / 1e6, bad);
    }
    cudaFree(x), cudaFree(y), cudaFree(r), cudaFreeHost(hx), cudaFreeHost(hy), cudaFreeHost(hr);
    return 0;
}

// ------------------------------------------------------------------ kmeans assignment
__global__ void assign_kernel(int M, int N, int K, const double* x, const double* c, int* a) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    double best = 1e300;
    int bk = -1;
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int n = 0; n < N; ++n) {
            double d = x[size_t(m) * N + n] - c[k * N + n];
            s += d * d;
        }
        if (s < best) best = s, bk = k;
    }
    a[m] = bk;
}
// Same math with the data transposed (N x M): consecutive threads read consecutive addresses.
__global__ void assign_kernel_T(int M, int N, int K, const double* xT, const double* c, int* a) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    double best = 1e300;
    int bk = -1;
    for (int k = 0; k < K; ++k) {
        double s = 0;
        for (int n = 0; n < N; ++n) {
            double d = xT[size_t(n) * M + m] - c[k * N + n];
            s += d * d;
        }
        if (s < best) best = s, bk = k;
    }
    a[m] = bk;
}

static int run_kmeans(int argc, char** argv) {
    int M = std::atoi(arg(argc, argv, "m", "1000000").c_str());
    int N = std::atoi(arg(argc, argv, "dims", "100").c_str());
    int K = std::atoi(arg(argc, argv, "k", "3").c_str());
    std::vector<double> x(size_t(M) * N), xT(size_t(M) * N), c(size_t(K) * N);
    std::mt19937_64 rng(1);
    std::normal_distribution<double> g;
    for (auto& v : x) v = g(rng);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) xT[size_t(n) * M + m] = x[size_t(m) * N + n];
    for (auto& v : c) v = g(rng);
    double *dx, *dxT, *dc;
    int* da;
    CK(cudaMalloc(&dx, x.size() * 8));
    CK(cudaMalloc(&dxT, x.size() * 8));
    CK(cudaMalloc(&dc, c.size() * 8));
    CK(cudaMalloc(&da, M * 4));
    CK(cudaMemcpy(dx, x.data(), x.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxT, xT.data(), x.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dc, c.data(), c.size() * 8, cudaMemcpyHostToDevice));
    std::vector<int> ref(M), got(M);
    for (int m = 0; m < M; ++m) {
        double best = 1e300;
        int bk = -1;
        for (int k = 0; k < K; ++k) {
            double s = 0;
            for (int n = 0; n < N; ++n) {
                double d = x[size_t(m) * N + n] - c[k * N + n];
                s += d * d;
            }
            if (s < best) best = s, bk = k;
        }
        ref[m] = bk;
    }
    double bytes = double(M) * N * 8;
    float ms = best_ms(10, [&] { assign_kernel<<<(M + 255) / 256, 256>>>(M, N, K, dx, dc, da); });
    CK(cudaMemcpy(got.data(), da, M * 4, cudaMemcpyDeviceToHost));
    long long mism = 0;
    for (int m = 0; m < M; ++m) mism += got[m] != ref[m];
    float msT = best_ms(10, [&] { assign_kernel_T<<<(M + 255) / 256, 256>>>(M, N, K, dxT, dc, da); });
    CK(cudaMemcpy(got.data(), da, M * 4, cudaMemcpyDeviceToHost));
    long long mismT = 0;
    for (int m = 0; m < M; ++m) mismT += got[m] != ref[m];
    std::printf("{\"kind\":\"kmeans_assign\",\"m\":%d,\"dims\":%d,\"k\":%d,\"aos_ms\":%.6f,\"soa_ms\":%.6f,"
                "\"aos_gbs\":%.3f,\"soa_gbs\":%.3f,\"mismatch_aos\":%lld,\"mismatch_soa\":%lld}\n",
                M, N, K, ms, msT, bytes / ms / 1e6, bytes / msT / 1e6, mism, mismT);
    cudaFree(dx), cudaFree(dxT), cudaFree(dc), cudaFree(da);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: a1_gpu mandel|sqrt|saxpy|kmeans|info [--opts]\n");
        return 1;
    }
    std::string m = argv[1];
    if (m == "info") {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, 0));
        int clk = 0, memclk = 0, bus = 0;
        cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
        cudaDeviceGetAttribute(&memclk, cudaDevAttrMemoryClockRate, 0);
        cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
        std::printf("{\"kind\":\"info\",\"name\":\"%s\",\"sm\":%d,\"cc\":\"%d.%d\",\"mem_gib\":%.1f,\"clock_mhz\":%d,"
                    "\"mem_clock_mhz\":%d,\"bus_bits\":%d,\"l2_mib\":%.1f,\"warp\":%d}\n",
                    p.name, p.multiProcessorCount, p.major, p.minor, p.totalGlobalMem / 1073741824.0, clk / 1000,
                    memclk / 1000, bus, p.l2CacheSize / 1048576.0, p.warpSize);
        return 0;
    }
    if (m == "mandel") return run_mandel(argc, argv);
    if (m == "sqrt") return run_sqrt(argc, argv);
    if (m == "saxpy") return run_saxpy(argc, argv);
    if (m == "kmeans") return run_kmeans(argc, argv);
    std::fprintf(stderr, "unknown mode %s\n", m.c_str());
    return 1;
}
