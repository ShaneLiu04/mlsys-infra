// CS149 asst1 prog6: profile and speed up K-means.
//
// `baseline` mirrors the starter's kmeansThread.cpp: computeAssignments loops k outside and
// m inside with sqrt distances, then computeCentroids and computeCost run serially.
// Optimizations, each checked against the baseline assignments and cost:
//   v1  computeAssignments split over points m across T threads (the intended fix)
//   v2  v1 + computeCentroids / computeCost with per-thread partial sums
//   v3  v2 + loop interchange (m outer, k inner: each point is read once), squared distance
//       (no sqrt), AVX-512 distance kernel
// Data: M points in N dims drawn from K Gaussian blobs (fixed seed); K=3, epsilon=0.1 as in the starter.
#include <immintrin.h>

#include <cmath>
#include <random>

#include "../common/common.h"

struct Data {
    int M, N, K;
    std::vector<double> x;     // M x N
    std::vector<double> init;  // K x N initial centroids
};

static Data make_data(int M, int N, int K, unsigned seed) {
    Data d{M, N, K, std::vector<double>(size_t(M) * N), std::vector<double>(size_t(K) * N)};
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> g(0.0, 1.0);
    std::uniform_real_distribution<double> u(-10.0, 10.0);
    int C = 8;  // true blobs; K < C so the clustering has real work to do
    std::vector<double> centers(size_t(C) * N);
    for (auto& c : centers) c = u(rng);
    std::uniform_int_distribution<int> pick(0, C - 1);
    for (int m = 0; m < M; ++m) {
        int c = pick(rng);
        for (int n = 0; n < N; ++n) d.x[size_t(m) * N + n] = centers[size_t(c) * N + n] + 2.0 * g(rng);
    }
    std::uniform_int_distribution<int> pm(0, M - 1);
    for (int k = 0; k < K; ++k) {
        int m = pm(rng);
        for (int n = 0; n < N; ++n) d.init[size_t(k) * N + n] = d.x[size_t(m) * N + n];
    }
    return d;
}

static double dist(const double* x, const double* y, int n) {
    double accum = 0.0;
    for (int i = 0; i < n; i++) accum += (x[i] - y[i]) * (x[i] - y[i]);
    return std::sqrt(accum);
}

__attribute__((target("avx512f"))) static double dist2_avx512(const double* x, const double* y, int n) {
    __m512d acc = _mm512_setzero_pd();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d d = _mm512_sub_pd(_mm512_loadu_pd(x + i), _mm512_loadu_pd(y + i));
        acc = _mm512_add_pd(acc, _mm512_mul_pd(d, d));
    }
    double s = _mm512_reduce_add_pd(acc);
    for (; i < n; ++i) s += (x[i] - y[i]) * (x[i] - y[i]);
    return s;
}

__attribute__((target("avx2"))) static double dist2_avx2(const double* x, const double* y, int n) {
    __m256d acc = _mm256_setzero_pd();
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d d = _mm256_sub_pd(_mm256_loadu_pd(x + i), _mm256_loadu_pd(y + i));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(d, d));
    }
    double t[4];
    _mm256_storeu_pd(t, acc);
    double s = (t[0] + t[1]) + (t[2] + t[3]);
    for (; i < n; ++i) s += (x[i] - y[i]) * (x[i] - y[i]);
    return s;
}

struct State {
    const Data* d;
    std::vector<double> cent;
    std::vector<int> assign;
    std::vector<double> cost;  // per cluster
};

struct Timers {
    double assign = 0, centroids = 0, cost = 0;
    int iters = 0;
};

static bool converged(const std::vector<double>& prev, const std::vector<double>& cur, double eps) {
    for (size_t k = 0; k < prev.size(); ++k)
        if (std::fabs(prev[k] - cur[k]) > eps) return false;
    return true;
}

// ---- baseline (starter structure) ------------------------------------------------------------
static void assign_baseline(State& s, int m0, int m1) {
    const Data& d = *s.d;
    std::vector<double> minDist(m1 - m0, 1e30);
    for (int m = m0; m < m1; ++m) s.assign[m] = -1;
    for (int k = 0; k < d.K; k++)
        for (int m = m0; m < m1; m++) {
            double dd = dist(&d.x[size_t(m) * d.N], &s.cent[size_t(k) * d.N], d.N);
            if (dd < minDist[m - m0]) minDist[m - m0] = dd, s.assign[m] = k;
        }
}

static void centroids_serial(State& s) {
    const Data& d = *s.d;
    std::vector<int> counts(d.K, 0);
    std::fill(s.cent.begin(), s.cent.end(), 0.0);
    for (int m = 0; m < d.M; ++m) {
        int k = s.assign[m];
        counts[k]++;
        for (int n = 0; n < d.N; ++n) s.cent[size_t(k) * d.N + n] += d.x[size_t(m) * d.N + n];
    }
    for (int k = 0; k < d.K; ++k)
        for (int n = 0; n < d.N; ++n) s.cent[size_t(k) * d.N + n] /= std::max(counts[k], 1);
}

static void cost_serial(State& s) {
    const Data& d = *s.d;
    std::fill(s.cost.begin(), s.cost.end(), 0.0);
    for (int m = 0; m < d.M; ++m) {
        int k = s.assign[m];
        s.cost[k] += dist(&d.x[size_t(m) * d.N], &s.cent[size_t(k) * d.N], d.N);
    }
}

// ---- parallel pieces ---------------------------------------------------------------------------
static void assign_v3(State& s, int m0, int m1, bool avx512) {
    const Data& d = *s.d;
    for (int m = m0; m < m1; ++m) {
        const double* xm = &d.x[size_t(m) * d.N];
        double best = 1e300;
        int bk = -1;
        for (int k = 0; k < d.K; ++k) {
            double dd = avx512 ? dist2_avx512(xm, &s.cent[size_t(k) * d.N], d.N) : dist2_avx2(xm, &s.cent[size_t(k) * d.N], d.N);
            if (dd < best) best = dd, bk = k;
        }
        s.assign[m] = bk;
    }
}

static void centroids_parallel(State& s, int T, const std::vector<int>& cpus) {
    const Data& d = *s.d;
    std::vector<std::vector<double>> part(T, std::vector<double>(size_t(d.K) * d.N, 0.0));
    std::vector<std::vector<int>> cnt(T, std::vector<int>(d.K, 0));
    run_threads(T, cpus, [&](int t) {
        int m0 = (int)((long long)d.M * t / T), m1 = (int)((long long)d.M * (t + 1) / T);
        auto& p = part[t];
        for (int m = m0; m < m1; ++m) {
            int k = s.assign[m];
            cnt[t][k]++;
            const double* xm = &d.x[size_t(m) * d.N];
            double* c = &p[size_t(k) * d.N];
            for (int n = 0; n < d.N; ++n) c[n] += xm[n];
        }
    });
    std::fill(s.cent.begin(), s.cent.end(), 0.0);
    std::vector<int> counts(d.K, 0);
    for (int t = 0; t < T; ++t) {
        for (size_t i = 0; i < s.cent.size(); ++i) s.cent[i] += part[t][i];
        for (int k = 0; k < d.K; ++k) counts[k] += cnt[t][k];
    }
    for (int k = 0; k < d.K; ++k)
        for (int n = 0; n < d.N; ++n) s.cent[size_t(k) * d.N + n] /= std::max(counts[k], 1);
}

static void cost_parallel(State& s, int T, const std::vector<int>& cpus) {
    const Data& d = *s.d;
    std::vector<std::vector<double>> part(T, std::vector<double>(d.K, 0.0));
    run_threads(T, cpus, [&](int t) {
        int m0 = (int)((long long)d.M * t / T), m1 = (int)((long long)d.M * (t + 1) / T);
        for (int m = m0; m < m1; ++m) {
            int k = s.assign[m];
            part[t][k] += dist(&d.x[size_t(m) * d.N], &s.cent[size_t(k) * d.N], d.N);
        }
    });
    std::fill(s.cost.begin(), s.cost.end(), 0.0);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < d.K; ++k) s.cost[k] += part[t][k];
}

static Timers kmeans(State& s, const std::string& ver, int T, const std::vector<int>& cpus, double eps, int max_iter) {
    const Data& d = *s.d;
    Timers tm;
    bool avx512 = __builtin_cpu_supports("avx512f");
    std::vector<double> prev(d.K, 1e30);
    std::fill(s.cost.begin(), s.cost.end(), 0.0);
    while (!converged(prev, s.cost, eps) && tm.iters < max_iter) {
        prev = s.cost;
        double t0 = now_s();
        if (ver == "baseline") {
            assign_baseline(s, 0, d.M);
        } else if (ver == "v1" || ver == "v2") {
            run_threads(T, cpus, [&](int t) {
                assign_baseline(s, (int)((long long)d.M * t / T), (int)((long long)d.M * (t + 1) / T));
            });
        } else {
            run_threads(T, cpus, [&](int t) {
                assign_v3(s, (int)((long long)d.M * t / T), (int)((long long)d.M * (t + 1) / T), avx512);
            });
        }
        double t1 = now_s();
        if (ver == "baseline" || ver == "v1") centroids_serial(s);
        else centroids_parallel(s, T, cpus);
        double t2 = now_s();
        if (ver == "baseline" || ver == "v1") cost_serial(s);
        else cost_parallel(s, T, cpus);
        double t3 = now_s();
        tm.assign += t1 - t0, tm.centroids += t2 - t1, tm.cost += t3 - t2;
        ++tm.iters;
    }
    return tm;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    int M = (int)a.num("m", 1000000), N = (int)a.num("dims", 100), K = (int)a.num("k", 3);
    double eps = a.real("eps", 0.1);
    int max_iter = (int)a.num("max_iter", 100);
    auto cpus = a.ints("cpus");
    std::string tag = a.str("tag", "");
    std::vector<int> threads = a.has("threads") ? a.ints("threads") : std::vector<int>{1, 2, 4, 8};
    std::vector<std::string> vers = {"baseline", "v1", "v2", "v3"};
    if (a.has("ver")) vers = {a.str("ver", "baseline")};

    Data d = make_data(M, N, K, 2024);
    State ref{&d, d.init, std::vector<int>(M), std::vector<double>(K)};
    Timers tb = kmeans(ref, "baseline", 1, cpus, eps, max_iter);
    double cost_ref = 0;
    for (double c : ref.cost) cost_ref += c;
    double tot_b = tb.assign + tb.centroids + tb.cost;
    Json().kv("kind", "kmeans").kv("tag", tag).kv("ver", "baseline").kv("threads", 1).kv("m", M).kv("dims", N).kv("k", K)
        .kv("iters", tb.iters).kv("assign_s", tb.assign).kv("centroids_s", tb.centroids).kv("cost_s", tb.cost)
        .kv("total_s", tot_b).kv("speedup", 1.0).kv("cost", cost_ref).kv("mismatch", 0LL).emit();

    int bad = 0;
    for (const auto& v : vers) {
        if (v == "baseline") continue;
        for (int T : threads) {
            State s{&d, d.init, std::vector<int>(M), std::vector<double>(K)};
            Timers t = kmeans(s, v, T, cpus, eps, max_iter);
            long long mism = 0;
            for (int m = 0; m < M; ++m) mism += s.assign[m] != ref.assign[m];
            double c = 0;
            for (double x : s.cost) c += x;
            double tot = t.assign + t.centroids + t.cost;
            bool ok = t.iters == tb.iters && mism <= M / 100000 && std::fabs(c - cost_ref) <= 1e-9 * std::fabs(cost_ref) * 1e3;
            bad += !ok;
            Json().kv("kind", "kmeans").kv("tag", tag).kv("ver", v).kv("threads", T).kv("m", M).kv("dims", N).kv("k", K)
                .kv("iters", t.iters).kv("assign_s", t.assign).kv("centroids_s", t.centroids).kv("cost_s", t.cost)
                .kv("total_s", tot).kv("speedup", tot_b / tot).kv("cost", c).kv("mismatch", mism)
                .kv("correct", (int)ok).emit();
        }
    }
    return bad ? 2 : 0;
}
