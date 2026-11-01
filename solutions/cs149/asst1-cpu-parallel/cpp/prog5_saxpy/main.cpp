// CS149 asst1 prog5: saxpy is bandwidth bound. Kernels:
//   scalar   plain loop (gcc -O3 auto-vectorizes it; kept as the starter's serial version)
//   simd     explicit AVX2/AVX-512 loads and stores (stores trigger write-allocate: 16 B/elem real traffic)
//   stream   non-temporal stores (_mm*_stream_ps): no read-for-ownership, 12 B/elem real traffic
//   read     sum of x and y only: 8 B/elem, the read-bandwidth ceiling
// A persistent spinning thread pool keeps per-call overhead to a few microseconds so that
// cache-resident sizes can be measured with many threads.
#include <immintrin.h>

#include "../common/common.h"

struct Pool {
    int n;
    std::vector<std::thread> th;
    std::atomic<long long> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> quit{false};
    std::function<void(int)> job;

    Pool(int n_, const std::vector<int>& cpus) : n(n_) {
        for (int t = 1; t < n; ++t)
            th.emplace_back([this, t, cpus] {
                pin_thread(cpu_for(cpus, t));
                long long seen = 0;
                for (;;) {
                    long long g;
                    while ((g = gen.load(std::memory_order_acquire)) == seen)
                        if (quit.load(std::memory_order_relaxed)) return;
                        else _mm_pause();
                    seen = g;
                    job(t);
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
            });
        pin_thread(cpu_for(cpus, 0));
    }
    void run(const std::function<void(int)>& f) {
        job = f;
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_acq_rel);
        job(0);
        while (done.load(std::memory_order_acquire) != n - 1) _mm_pause();
    }
    ~Pool() {
        quit = true;
        for (auto& x : th) x.join();
    }
};

static void saxpy_scalar(size_t lo, size_t hi, float a, const float* x, const float* y, float* r) {
    for (size_t i = lo; i < hi; ++i) r[i] = a * x[i] + y[i];
}

__attribute__((target("avx512f"))) static void saxpy_simd(size_t lo, size_t hi, float a, const float* x, const float* y, float* r) {
    __m512 va = _mm512_set1_ps(a);
    size_t i = lo;
    for (; i + 16 <= hi; i += 16)
        _mm512_store_ps(r + i, _mm512_add_ps(_mm512_mul_ps(va, _mm512_load_ps(x + i)), _mm512_load_ps(y + i)));
    saxpy_scalar(i, hi, a, x, y, r);
}

__attribute__((target("avx512f"))) static void saxpy_stream(size_t lo, size_t hi, float a, const float* x, const float* y, float* r) {
    __m512 va = _mm512_set1_ps(a);
    size_t i = lo;
    for (; i + 16 <= hi; i += 16)
        _mm512_stream_ps(r + i, _mm512_add_ps(_mm512_mul_ps(va, _mm512_load_ps(x + i)), _mm512_load_ps(y + i)));
    _mm_sfence();
    saxpy_scalar(i, hi, a, x, y, r);
}

__attribute__((target("avx2"))) static void saxpy_simd_avx2(size_t lo, size_t hi, float a, const float* x, const float* y, float* r) {
    __m256 va = _mm256_set1_ps(a);
    size_t i = lo;
    for (; i + 8 <= hi; i += 8)
        _mm256_store_ps(r + i, _mm256_add_ps(_mm256_mul_ps(va, _mm256_load_ps(x + i)), _mm256_load_ps(y + i)));
    saxpy_scalar(i, hi, a, x, y, r);
}

__attribute__((target("avx2"))) static void saxpy_stream_avx2(size_t lo, size_t hi, float a, const float* x, const float* y, float* r) {
    __m256 va = _mm256_set1_ps(a);
    size_t i = lo;
    for (; i + 8 <= hi; i += 8)
        _mm256_stream_ps(r + i, _mm256_add_ps(_mm256_mul_ps(va, _mm256_load_ps(x + i)), _mm256_load_ps(y + i)));
    _mm_sfence();
    saxpy_scalar(i, hi, a, x, y, r);
}

static float g_sink[64 * 16];

__attribute__((target("avx2"))) static void read_sum(size_t lo, size_t hi, float, const float* x, const float* y, float* r) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    size_t i = lo;
    for (; i + 8 <= hi; i += 8) {
        s0 = _mm256_add_ps(s0, _mm256_load_ps(x + i));
        s1 = _mm256_add_ps(s1, _mm256_load_ps(y + i));
    }
    float t[8];
    _mm256_storeu_ps(t, _mm256_add_ps(s0, s1));
    (void)r;
    g_sink[(lo / 4096) % 64 * 16] += t[0] + t[7];
}

using Fn = void (*)(size_t, size_t, float, const float*, const float*, float*);

int main(int argc, char** argv) {
    Args a(argc, argv);
    auto cpus = a.ints("cpus");
    std::string tag = a.str("tag", "");
    std::string mode = a.str("mode", "sizes");
    bool has512 = __builtin_cpu_supports("avx512f");
    std::vector<int> threads = a.has("threads") ? a.ints("threads") : std::vector<int>{1, 8};
    double min_s = a.real("min_s", 0.2);

    struct K {
        const char* name;
        Fn fn;
        double real_bytes;  // per element, including write-allocate where it happens
    };
    std::vector<K> kernels = {
        {"scalar", saxpy_scalar, 16},
        {"simd", has512 ? saxpy_simd : saxpy_simd_avx2, 16},
        {"stream", has512 ? saxpy_stream : saxpy_stream_avx2, 12},
        {"read", read_sum, 8},
    };

    std::vector<size_t> sizes;
    if (mode == "sizes")
        for (size_t n = 1 << 10; n <= (size_t(1) << (int)a.num("max_log2", 27)); n *= 2) sizes.push_back(n);
    else
        sizes = {size_t(a.num("n", 1 << 26))};

    size_t nmax = sizes.back();
    float* x = aligned_alloc_n<float>(nmax);
    float* y = aligned_alloc_n<float>(nmax);
    float* r = aligned_alloc_n<float>(nmax);
    for (size_t i = 0; i < nmax; ++i) x[i] = float(i % 1000) * 1e-3f, y[i] = 1.f, r[i] = 0.f;

    int bad = 0;
    for (int T : threads) {
        Pool pool(T, cpus);
        for (size_t n : sizes)
            for (const auto& k : kernels) {
                auto call = [&] {
                    pool.run([&](int t) {
                        size_t lo = (n * t / T) & ~size_t(15), hi = t + 1 == T ? n : (n * (t + 1) / T) & ~size_t(15);
                        k.fn(lo, hi, 2.f, x, y, r);
                    });
                };
                call();
                double t0 = now_s();
                call();
                double one = std::max(now_s() - t0, 1e-7);
                long long inner = std::max(1LL, (long long)(min_s / 5 / one));
                double best = 1e30;
                for (int rep = 0; rep < 5; ++rep) {
                    double s = now_s();
                    for (long long q = 0; q < inner; ++q) call();
                    best = std::min(best, (now_s() - s) / inner);
                }
                if (std::string(k.name) != "read")
                    for (size_t i = 0; i < n; i += 4099)
                        if (r[i] != 2.f * x[i] + y[i]) { ++bad; break; }
                double visible = (std::string(k.name) == "read" ? 8.0 : 12.0) * n;
                Json().kv("kind", "bw").kv("tag", tag).kv("kernel", k.name).kv("threads", T).kv("n", (long long)n)
                    .kv("bytes_ws", (double)n * 12).kv("time_s", best).kv("visible_gbs", visible / best / 1e9)
                    .kv("real_gbs", k.real_bytes * n / best / 1e9).kv("correct", (int)(bad == 0)).emit();
            }
    }
    aligned_free(x);
    aligned_free(y);
    aligned_free(r);
    return bad ? 2 : 0;
}
