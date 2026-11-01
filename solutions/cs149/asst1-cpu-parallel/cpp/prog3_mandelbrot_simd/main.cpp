// CS149 asst1 prog3 without an ISPC compiler: the same "gang of program instances"
// written by hand with AVX2 (8 x float) and AVX-512 (16 x float) intrinsics.
// An ISPC `launch[n]` task system is modelled by a pool of T threads pulling n row
// bands from an atomic counter.
//
// Lanes follow the scalar code operation by operation (mul, sub, add; `2*zr` before
// `*zi`; "not greater than 4" so NaN keeps iterating), so with -ffp-contract=off every
// pixel matches the scalar reference exactly.
#include <immintrin.h>

#include "../common/common.h"

struct View {
    float x0, y0, x1, y1;
};
static View make_view(int idx) {
    View v{-2.f, -1.f, 1.f, 1.f};
    if (idx == 2) {
        const float s = .015f, sx = -.986f, sy = .30f;
        v = {v.x0 * s + sx, v.y0 * s + sy, v.x1 * s + sx, v.y1 * s + sy};
    }
    return v;
}
struct Problem {
    View v;
    int width = 1600, height = 1200, max_iter = 256;
    float dx() const { return (v.x1 - v.x0) / width; }
    float dy() const { return (v.y1 - v.y0) / height; }
};

static inline int mandel(float c_re, float c_im, int count) {
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

static void rows_scalar(const Problem& p, int r0, int r1, int* out) {
    float dx = p.dx(), dy = p.dy();
    for (int j = r0; j < r1; ++j)
        for (int i = 0; i < p.width; ++i) out[j * p.width + i] = mandel(p.v.x0 + i * dx, p.v.y0 + j * dy, p.max_iter);
}

__attribute__((target("avx2"))) static void rows_avx2(const Problem& p, int r0, int r1, int* out) {
    const float dx = p.dx(), dy = p.dy();
    const __m256 four = _mm256_set1_ps(4.f), two = _mm256_set1_ps(2.f);
    const __m256 x0 = _mm256_set1_ps(p.v.x0), vdx = _mm256_set1_ps(dx);
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (int j = r0; j < r1; ++j) {
        const __m256 ci = _mm256_set1_ps(p.v.y0 + j * dy);
        for (int i = 0; i < p.width; i += 8) {
            __m256 idx = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(i), lane));
            __m256 cr = _mm256_add_ps(x0, _mm256_mul_ps(idx, vdx));
            __m256 zr = cr, zi = ci;
            __m256 active = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            __m256i cnt = _mm256_setzero_si256();
            for (int k = 0; k < p.max_iter; ++k) {
                __m256 zr2 = _mm256_mul_ps(zr, zr), zi2 = _mm256_mul_ps(zi, zi);
                active = _mm256_and_ps(active, _mm256_cmp_ps(_mm256_add_ps(zr2, zi2), four, _CMP_NGT_UQ));
                if (!_mm256_movemask_ps(active)) break;
                cnt = _mm256_sub_epi32(cnt, _mm256_castps_si256(active));
                __m256 nr = _mm256_sub_ps(zr2, zi2);
                __m256 ni = _mm256_mul_ps(_mm256_mul_ps(two, zr), zi);
                zr = _mm256_add_ps(cr, nr);
                zi = _mm256_add_ps(ci, ni);
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + j * p.width + i), cnt);
        }
    }
}

__attribute__((target("avx512f"))) static void rows_avx512(const Problem& p, int r0, int r1, int* out) {
    const float dx = p.dx(), dy = p.dy();
    const __m512 four = _mm512_set1_ps(4.f), two = _mm512_set1_ps(2.f);
    const __m512 x0 = _mm512_set1_ps(p.v.x0), vdx = _mm512_set1_ps(dx);
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i one = _mm512_set1_epi32(1);
    for (int j = r0; j < r1; ++j) {
        const __m512 ci = _mm512_set1_ps(p.v.y0 + j * dy);
        for (int i = 0; i < p.width; i += 16) {
            __m512 idx = _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(i), lane));
            __m512 cr = _mm512_add_ps(x0, _mm512_mul_ps(idx, vdx));
            __m512 zr = cr, zi = ci;
            __mmask16 active = 0xFFFF;
            __m512i cnt = _mm512_setzero_si512();
            for (int k = 0; k < p.max_iter; ++k) {
                __m512 zr2 = _mm512_mul_ps(zr, zr), zi2 = _mm512_mul_ps(zi, zi);
                active = _mm512_mask_cmp_ps_mask(active, _mm512_add_ps(zr2, zi2), four, _CMP_NGT_UQ);
                if (!active) break;
                cnt = _mm512_mask_add_epi32(cnt, active, cnt, one);
                __m512 nr = _mm512_sub_ps(zr2, zi2);
                __m512 ni = _mm512_mul_ps(_mm512_mul_ps(two, zr), zi);
                zr = _mm512_add_ps(cr, nr);
                zi = _mm512_add_ps(ci, ni);
            }
            _mm512_storeu_si512(out + j * p.width + i, cnt);
        }
    }
}

// Two independent 16-wide gangs per loop: their dependency chains interleave, so the core's
// FP pipelines see twice the independent work (the single-thread analogue of what SMT does).
__attribute__((target("avx512f"))) static void rows_avx512_x2(const Problem& p, int r0, int r1, int* out) {
    const float dx = p.dx(), dy = p.dy();
    const __m512 four = _mm512_set1_ps(4.f), two = _mm512_set1_ps(2.f);
    const __m512 x0 = _mm512_set1_ps(p.v.x0), vdx = _mm512_set1_ps(dx);
    const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m512i one = _mm512_set1_epi32(1);
    for (int j = r0; j < r1; ++j) {
        const __m512 ci = _mm512_set1_ps(p.v.y0 + j * dy);
        for (int i = 0; i < p.width; i += 32) {
            __m512 cra = _mm512_add_ps(x0, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(i), lane)), vdx));
            __m512 crb = _mm512_add_ps(x0, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(i + 16), lane)), vdx));
            __m512 zra = cra, zia = ci, zrb = crb, zib = ci;
            __mmask16 ma = 0xFFFF, mb = 0xFFFF;
            __m512i ca = _mm512_setzero_si512(), cb = _mm512_setzero_si512();
            for (int k = 0; k < p.max_iter; ++k) {
                __m512 ra2 = _mm512_mul_ps(zra, zra), ia2 = _mm512_mul_ps(zia, zia);
                __m512 rb2 = _mm512_mul_ps(zrb, zrb), ib2 = _mm512_mul_ps(zib, zib);
                ma = _mm512_mask_cmp_ps_mask(ma, _mm512_add_ps(ra2, ia2), four, _CMP_NGT_UQ);
                mb = _mm512_mask_cmp_ps_mask(mb, _mm512_add_ps(rb2, ib2), four, _CMP_NGT_UQ);
                if (!(ma | mb)) break;
                ca = _mm512_mask_add_epi32(ca, ma, ca, one);
                cb = _mm512_mask_add_epi32(cb, mb, cb, one);
                __m512 nia = _mm512_mul_ps(_mm512_mul_ps(two, zra), zia);
                __m512 nib = _mm512_mul_ps(_mm512_mul_ps(two, zrb), zib);
                zra = _mm512_add_ps(cra, _mm512_sub_ps(ra2, ia2));
                zrb = _mm512_add_ps(crb, _mm512_sub_ps(rb2, ib2));
                zia = _mm512_add_ps(ci, nia);
                zib = _mm512_add_ps(ci, nib);
            }
            _mm512_storeu_si512(out + j * p.width + i, ca);
            _mm512_storeu_si512(out + j * p.width + i + 16, cb);
        }
    }
}

using RowFn = void (*)(const Problem&, int, int, int*);

// ISPC-style task launch: `tasks` equal row bands, `threads` workers pulling them dynamically.
static void run_tasks(const Problem& p, RowFn fn, int tasks, int threads, const std::vector<int>& cpus, int* out) {
    std::atomic<int> next{0};
    run_threads(threads, cpus, [&](int) {
        for (;;) {
            int t = next.fetch_add(1);
            if (t >= tasks) break;
            int r0 = (int)((long long)p.height * t / tasks), r1 = (int)((long long)p.height * (t + 1) / tasks);
            fn(p, r0, r1, out);
        }
    });
}

// SIMD efficiency from the reference counts: useful lane-iterations / issued lane-iterations,
// where a W-wide gang of adjacent pixels issues W * (max count + 1) lane-iterations.
static double gang_util(const std::vector<int>& c, int W) {
    double useful = 0, issued = 0;
    for (size_t g = 0; g < c.size(); g += W) {
        int mx = 0;
        for (int k = 0; k < W; ++k) mx = std::max(mx, c[g + k]), useful += c[g + k] + 1;
        issued += double(W) * (mx + 1);
    }
    return useful / issued;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    Problem p;
    int view = (int)a.num("view", 1);
    p.v = make_view(view);
    int reps = (int)a.num("reps", 5);
    auto cpus = a.ints("cpus");
    std::string mode = a.str("mode", "simd");
    std::string tag = a.str("tag", "");
    bool has512 = __builtin_cpu_supports("avx512f");

    const size_t n = size_t(p.width) * p.height;
    std::vector<int> ref(n), out(n);
    double t_scalar = min_time(reps, [&] { rows_scalar(p, 0, p.height, ref.data()); });
    int bad = 0;
    auto check = [&] {
        if (out != ref) ++bad;
        std::fill(out.begin(), out.end(), -1);
    };

    if (mode == "simd") {
        double t2 = min_time(reps, [&] { rows_avx2(p, 0, p.height, out.data()); });
        check();
        double t5 = 0, t5x2 = 0;
        if (has512) {
            t5 = min_time(reps, [&] { rows_avx512(p, 0, p.height, out.data()); });
            check();
            t5x2 = min_time(reps, [&] { rows_avx512_x2(p, 0, p.height, out.data()); });
            check();
        }
        Json().kv("kind", "simd").kv("tag", tag).kv("view", view).kv("scalar_s", t_scalar).kv("avx2_s", t2)
            .kv("avx512_s", t5).kv("avx512x2_s", t5x2).kv("avx2_speedup", t_scalar / t2)
            .kv("avx512_speedup", has512 ? t_scalar / t5 : 0.0).kv("avx512x2_speedup", has512 ? t_scalar / t5x2 : 0.0)
            .kv("util_w8", gang_util(ref, 8)).kv("util_w16", gang_util(ref, 16))
            .kv("correct", (int)(bad == 0)).emit();
    } else if (mode == "tasks") {
        int threads = (int)a.num("threads", 8);
        std::vector<int> tasks_list = a.has("tasks") ? a.ints("tasks") : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 300, 1200};
        std::string isa = a.str("isa", has512 ? "avx512" : "avx2");
        RowFn fn = isa == "avx512x2" ? rows_avx512_x2 : isa == "avx512" ? rows_avx512 : isa == "avx2" ? rows_avx2 : rows_scalar;
        double t1 = min_time(reps, [&] { fn(p, 0, p.height, out.data()); });
        check();
        for (int tk : tasks_list) {
            double tt = min_time(reps, [&] { run_tasks(p, fn, tk, std::min(threads, tk), cpus, out.data()); });
            check();
            Json().kv("kind", "tasks").kv("tag", tag).kv("view", view).kv("isa", isa).kv("threads", std::min(threads, tk))
                .kv("tasks", tk).kv("scalar_s", t_scalar).kv("simd1_s", t1).kv("wall_s", tt)
                .kv("speedup_vs_scalar", t_scalar / tt).kv("speedup_vs_simd1", t1 / tt)
                .kv("correct", (int)(bad == 0)).emit();
        }
    }
    return bad ? 2 : 0;
}
