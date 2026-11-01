// CS149 asst1 prog4: Newton's method for sqrt, where the input decides the iteration count.
// Versions: scalar (starter code), AVX2 8-wide, AVX-512 16-wide, and SIMD + threads (ISPC tasks).
// Input patterns probe the best and worst cases for SIMD:
//   random     the starter's .001 + 2.998 * rand
//   all_slow   every element 2.998: maximal, uniform work -> best SIMD speedup
//   all_one    every element 1.0: zero iterations -> pure load/store
//   one_slow_W one 2.998 per W consecutive elements, the rest 1.0 -> worst divergence for width W
#include <immintrin.h>

#include <cmath>
#include <random>

#include "../common/common.h"

static const float kThreshold = 0.00001f;

static void sqrt_scalar(int lo, int hi, float initialGuess, const float* values, float* output) {
    for (int i = lo; i < hi; i++) {
        float x = values[i];
        float guess = initialGuess;
        float error = std::fabs(guess * guess * x - 1.f);
        while (error > kThreshold) {
            guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
            error = std::fabs(guess * guess * x - 1.f);
        }
        output[i] = x * guess;
    }
}

__attribute__((target("avx2"))) static void sqrt_avx2(int lo, int hi, float initialGuess, const float* values, float* output) {
    const __m256 one = _mm256_set1_ps(1.f), three = _mm256_set1_ps(3.f), half = _mm256_set1_ps(0.5f);
    const __m256 thr = _mm256_set1_ps(kThreshold);
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    int i = lo;
    for (; i + 8 <= hi; i += 8) {
        __m256 x = _mm256_loadu_ps(values + i);
        __m256 g = _mm256_set1_ps(initialGuess);
        __m256 err = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(g, g), x), one), absmask);
        __m256 active = _mm256_cmp_ps(err, thr, _CMP_GT_OQ);
        while (_mm256_movemask_ps(active)) {
            __m256 xg3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(x, g), g), g);
            __m256 ng = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(three, g), xg3), half);
            g = _mm256_blendv_ps(g, ng, active);
            err = _mm256_and_ps(_mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(g, g), x), one), absmask);
            active = _mm256_and_ps(active, _mm256_cmp_ps(err, thr, _CMP_GT_OQ));
        }
        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, g));
    }
    sqrt_scalar(i, hi, initialGuess, values, output);
}

__attribute__((target("avx512f"))) static void sqrt_avx512(int lo, int hi, float initialGuess, const float* values, float* output) {
    const __m512 one = _mm512_set1_ps(1.f), three = _mm512_set1_ps(3.f), half = _mm512_set1_ps(0.5f);
    const __m512 thr = _mm512_set1_ps(kThreshold);
    int i = lo;
    for (; i + 16 <= hi; i += 16) {
        __m512 x = _mm512_loadu_ps(values + i);
        __m512 g = _mm512_set1_ps(initialGuess);
        __m512 err = _mm512_abs_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(g, g), x), one));
        __mmask16 active = _mm512_cmp_ps_mask(err, thr, _CMP_GT_OQ);
        while (active) {
            __m512 xg3 = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(x, g), g), g);
            __m512 ng = _mm512_mul_ps(_mm512_sub_ps(_mm512_mul_ps(three, g), xg3), half);
            g = _mm512_mask_mov_ps(g, active, ng);
            err = _mm512_abs_ps(_mm512_sub_ps(_mm512_mul_ps(_mm512_mul_ps(g, g), x), one));
            active = _mm512_mask_cmp_ps_mask(active, err, thr, _CMP_GT_OQ);
        }
        _mm512_storeu_ps(output + i, _mm512_mul_ps(x, g));
    }
    sqrt_scalar(i, hi, initialGuess, values, output);
}

using Fn = void (*)(int, int, float, const float*, float*);

static void run_threads_fn(Fn fn, int N, int threads, const std::vector<int>& cpus, const float* v, float* o) {
    // 64 tasks per thread keeps the tail short when per-element work varies.
    int tasks = threads * 64;
    std::atomic<int> next{0};
    run_threads(threads, cpus, [&](int) {
        for (;;) {
            int t = next.fetch_add(1);
            if (t >= tasks) break;
            int lo = (int)((long long)N * t / tasks) & ~15, hi = t + 1 == tasks ? N : (int)((long long)N * (t + 1) / tasks) & ~15;
            fn(lo, hi, 1.f, v, o);
        }
    });
}

static int iterations(float x) {
    float guess = 1.f, error = std::fabs(guess * guess * x - 1.f);
    int it = 0;
    while (error > kThreshold) guess = (3.f * guess - x * guess * guess * guess) * 0.5f, error = std::fabs(guess * guess * x - 1.f), ++it;
    return it;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    int N = (int)a.num("n", 20 * 1000 * 1000);
    int reps = (int)a.num("reps", 3);
    auto cpus = a.ints("cpus");
    std::vector<int> threads = a.has("threads") ? a.ints("threads") : std::vector<int>{8};
    std::string tag = a.str("tag", "");
    bool has512 = __builtin_cpu_supports("avx512f");
    std::vector<std::string> patterns = {"random", "all_slow", "all_one", "one_slow_8", "one_slow_16"};
    if (a.has("pattern")) patterns = {a.str("pattern", "random")};
    if (a.has("xsweep")) patterns = {"xsweep"};

    float* v = aligned_alloc_n<float>(N);
    float* ref = aligned_alloc_n<float>(N);
    float* out = aligned_alloc_n<float>(N);
    int bad = 0;

    if (patterns[0] == "xsweep") {  // iteration count as a function of x (figure for the report)
        for (int k = 0; k <= 300; ++k) {
            float x = 0.001f + 2.998f * k / 300.f;
            Json().kv("kind", "iters").kv("x", (double)x).kv("iterations", iterations(x)).emit();
        }
        return 0;
    }

    for (const auto& pat : patterns) {
        std::mt19937 rng(4);
        std::uniform_real_distribution<float> u(0.f, 1.f);
        for (int i = 0; i < N; ++i) {
            if (pat == "random") v[i] = .001f + 2.998f * u(rng);
            else if (pat == "all_slow") v[i] = 2.998f;
            else if (pat == "all_one") v[i] = 1.f;
            else if (pat == "one_slow_8") v[i] = (i % 8 == 0) ? 2.998f : 1.f;
            else if (pat == "one_slow_16") v[i] = (i % 16 == 0) ? 2.998f : 1.f;
        }
        double mean_it = 0;
        for (int i = 0; i < N; i += 97) mean_it += iterations(v[i]);
        mean_it /= (N + 96) / 97;

        double ts = min_time(reps, [&] { sqrt_scalar(0, N, 1.f, v, ref); });
        auto check = [&] {
            if (std::memcmp(out, ref, sizeof(float) * N)) ++bad;
            std::memset(out, 0, sizeof(float) * N);
        };
        double t2 = min_time(reps, [&] { sqrt_avx2(0, N, 1.f, v, out); });
        check();
        double t5 = 0;
        if (has512) {
            t5 = min_time(reps, [&] { sqrt_avx512(0, N, 1.f, v, out); });
            check();
        }
        Json().kv("kind", "simd").kv("tag", tag).kv("pattern", pat).kv("n", N).kv("mean_iters", mean_it)
            .kv("scalar_s", ts).kv("avx2_s", t2).kv("avx512_s", t5).kv("avx2_speedup", ts / t2)
            .kv("avx512_speedup", has512 ? ts / t5 : 0.0).kv("correct", (int)(bad == 0)).emit();
        for (int T : threads) {
            Fn fn = has512 ? sqrt_avx512 : sqrt_avx2;
            double tt = min_time(reps, [&] { run_threads_fn(fn, N, T, cpus, v, out); });
            check();
            Json().kv("kind", "tasks").kv("tag", tag).kv("pattern", pat).kv("isa", has512 ? "avx512" : "avx2")
                .kv("threads", T).kv("wall_s", tt).kv("speedup_vs_scalar", ts / tt)
                .kv("speedup_vs_simd1", (has512 ? t5 : t2) / tt).kv("correct", (int)(bad == 0)).emit();
        }
    }
    aligned_free(v);
    aligned_free(ref);
    aligned_free(out);
    return bad ? 2 : 0;
}
