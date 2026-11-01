// CS149 asst1 prog2: clampedExp and arraySum with the fake vector intrinsics.
// Prints vector-lane utilization for several exponent distributions.
#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "../common/common.h"
#include "cs149intrin.h"

static void clampedExpSerial(const float* values, const int* exponents, float* output, int N) {
    for (int i = 0; i < N; i++) {
        float x = values[i];
        int y = exponents[i];
        if (y == 0) {
            output[i] = 1.f;
        } else {
            float result = x;
            int count = y - 1;
            while (count > 0) {
                result *= x;
                count--;
            }
            if (result > 9.999999f) result = 9.999999f;
            output[i] = result;
        }
    }
}

static void clampedExpVector(const float* values, const int* exponents, float* output, int N) {
    __cs149_vec_float x, result, clampV;
    __cs149_vec_int y, count, zero, one;
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        __cs149_mask maskAll = _cs149_init_ones(std::min(VECTOR_WIDTH, N - i));
        __cs149_mask maskActive = _cs149_init_ones(0), maskClamp = _cs149_init_ones(0);
        _cs149_vset_int(zero, 0, maskAll);
        _cs149_vset_int(one, 1, maskAll);
        _cs149_vset_float(clampV, 9.999999f, maskAll);
        _cs149_vload_float(x, values + i, maskAll);
        _cs149_vload_int(y, exponents + i, maskAll);
        _cs149_vset_float(result, 1.f, maskAll);
        _cs149_vmove_int(count, y, maskAll);
        _cs149_vgt_int(maskActive, count, zero, maskAll);
        while (_cs149_cntbits(maskActive) > 0) {
            _cs149_vmult_float(result, result, x, maskActive);
            _cs149_vsub_int(count, count, one, maskActive);
            _cs149_vgt_int(maskActive, count, zero, maskActive);
        }
        _cs149_vgt_float(maskClamp, result, clampV, maskAll);
        _cs149_vmove_float(result, clampV, maskClamp);
        _cs149_vstore_float(output + i, result, maskAll);
    }
}

static float arraySumSerial(const float* values, int N) {
    float s = 0.f;
    for (int i = 0; i < N; i++) s += values[i];
    return s;
}

// O(N/W + log2 W) vector instructions; assumes N % VECTOR_WIDTH == 0 and W a power of two.
static float arraySumVector(const float* values, int N) {
    __cs149_mask all = _cs149_init_ones();
    __cs149_vec_float acc, v;
    _cs149_vset_float(acc, 0.f, all);
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        _cs149_vload_float(v, values + i, all);
        _cs149_vadd_float(acc, acc, v, all);
    }
    for (int w = VECTOR_WIDTH; w > 1; w /= 2) {
        _cs149_hadd_float(acc, acc);
        _cs149_interleave_float(acc, acc);
    }
    return acc.value[0];
}

static std::vector<int> make_exponents(const std::string& dist, int N, std::mt19937& rng) {
    std::vector<int> e(N);
    std::uniform_int_distribution<int> u(0, 9);
    if (dist == "uniform") {
        for (auto& v : e) v = u(rng);
    } else if (dist == "sorted") {
        for (auto& v : e) v = u(rng);
        std::sort(e.begin(), e.end());
    } else if (dist == "constant") {
        std::fill(e.begin(), e.end(), 5);
    } else if (dist == "skewed") {  // mostly cheap, rare expensive lanes
        std::uniform_real_distribution<float> f(0.f, 1.f);
        for (auto& v : e) v = f(rng) < 0.1f ? 9 : 1;
    } else if (dist == "one_slow_per_vector") {
        for (int i = 0; i < N; ++i) e[i] = (i % VECTOR_WIDTH == 0) ? 9 : 1;
    } else {
        std::fprintf(stderr, "unknown dist %s\n", dist.c_str());
        std::exit(1);
    }
    return e;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    int N = (int)a.num("n", 10000);
    std::vector<std::string> dists = {"uniform", "sorted", "constant", "skewed", "one_slow_per_vector"};
    if (a.has("dist")) dists = {a.str("dist", "uniform")};
    int fails = 0;

    for (const auto& d : dists) {
        std::mt19937 rng(149);
        std::uniform_real_distribution<float> uv(0.5f, 1.5f);
        std::vector<float> values(N), gold(N), out(N);
        for (auto& v : values) v = uv(rng);
        auto ex = make_exponents(d, N, rng);
        clampedExpSerial(values.data(), ex.data(), gold.data(), N);
        CS149_LOG.reset();
        clampedExpVector(values.data(), ex.data(), out.data(), N);
        int ok = (gold == out);
        fails += !ok;
        Json().kv("kind", "clamped_exp").kv("width", VECTOR_WIDTH).kv("n", N).kv("dist", d)
            .kv("instructions", CS149_LOG.instructions).kv("enabled_lanes", CS149_LOG.enabled_lanes)
            .kv("total_lanes", CS149_LOG.total_lanes).kv("utilization", CS149_LOG.utilization())
            .kv("correct", ok).emit();
    }

    int Ns = (N / VECTOR_WIDTH) * VECTOR_WIDTH;
    std::vector<float> vals(Ns);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uv(0.f, 1.f);
    for (auto& v : vals) v = uv(rng);
    CS149_LOG.reset();
    float sv = arraySumVector(vals.data(), Ns);
    float ss = arraySumSerial(vals.data(), Ns);
    double rel = std::fabs(sv - ss) / std::fabs(ss);
    int ok = rel < 1e-4;
    fails += !ok;
    Json().kv("kind", "array_sum").kv("width", VECTOR_WIDTH).kv("n", Ns).kv("instructions", CS149_LOG.instructions)
        .kv("expected_instructions", (long long)(1 + 2 * (Ns / VECTOR_WIDTH) + 2 * (int)std::log2(VECTOR_WIDTH)))
        .kv("utilization", CS149_LOG.utilization()).kv("rel_err", rel).kv("correct", ok).emit();

    // Tail handling: N not a multiple of the width must still be exact.
    for (int n : {1, 3, VECTOR_WIDTH + 1, 2 * VECTOR_WIDTH - 1, 1001}) {
        std::mt19937 r2(n);
        std::vector<float> v(n), g(n), o(n);
        std::vector<int> e(n);
        for (int i = 0; i < n; ++i) v[i] = uv(r2) + 0.5f, e[i] = (int)(r2() % 10);
        clampedExpSerial(v.data(), e.data(), g.data(), n);
        clampedExpVector(v.data(), e.data(), o.data(), n);
        if (g != o) {
            ++fails;
            Json().kv("kind", "tail_fail").kv("width", VECTOR_WIDTH).kv("n", n).emit();
        }
    }
    return fails ? 2 : 0;
}
