// Micro-benchmarks behind the task-dispatch design choices.
//   --mode false_sharing --threads 1,2,4,.. --iters 50000000
//       per-thread counters packed in one cache line vs padded to 64 B (alignas(64)).
//   --mode dispatch --threads 1,2,4,.. --n 20000000 --work 0 --chunks 1,8,64
//       hand out indices 0..n-1 with: mutex counter, atomic fetch_add, atomic fetch_add of a
//       chunk, static per-thread partition.
#include <mutex>

#include "../common/common.h"

static std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) v.push_back(std::atoi(s.substr(p, q - p).c_str()));
        p = q + 1;
    }
    return v;
}

struct alignas(64) Padded {
    volatile long long v;
};

static inline uint32_t work_fn(uint32_t x, int w) {
    for (int k = 0; k < w; ++k) x = x * 1103515245u + 12345u;
    return x;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string mode = a.str("mode", "false_sharing");
    std::vector<int> cpus = a.ints("cpus");
    auto threads = ints(a.str("threads", "1,2,4,8"));
    int reps = (int)a.num("reps", 3);
    int bad = 0;

    if (mode == "false_sharing") {
        // Both layouts run the same code on a pointer into one 64 B-aligned buffer; only the
        // stride differs (8 B packed: all counters share a line; 64 B padded: one line each).
        long long iters = a.num("iters", 50000000);
        for (int T : threads)
            for (std::string op : {"store", "atomic"})
                for (std::string layout : {"packed", "padded"}) {
                    int stride = layout == "packed" ? 1 : 8;
                    std::vector<double> ns;
                    bool ok = true;
                    for (int r = 0; r < reps; ++r) {
                        long long* buf = aligned_alloc_n<long long>(8 * 64 + 8);
                        std::memset(buf, 0, sizeof(long long) * (8 * 64 + 8));
                        double t0 = now_s();
                        run_threads(T, cpus, [&](int t) {
                            if (op == "store") {
                                volatile long long* p = buf + t * stride;
                                for (long long i = 0; i < iters; ++i) *p = *p + 1;
                            } else {
                                auto* p = reinterpret_cast<std::atomic<long long>*>(buf + t * stride);
                                for (long long i = 0; i < iters; ++i) p->fetch_add(1, std::memory_order_relaxed);
                            }
                        });
                        double dt = now_s() - t0;
                        ns.push_back(dt / iters * 1e9);
                        for (int t = 0; t < T; ++t) ok &= buf[t * stride] == iters;
                        aligned_free(buf);
                    }
                    bad += !ok;
                    std::sort(ns.begin(), ns.end());
                    Json().kv("mode", "false_sharing").kv("op", op).kv("layout", layout).kv("threads", T)
                        .kv("iters", iters).kv("ns_per_inc", ns).kv("ns_med", ns[ns.size() / 2]).kv("correct", (int)ok)
                        .emit();
                }
    } else if (mode == "dispatch") {
        long long n = a.num("n", 20000000);
        int w = (int)a.num("work", 0);
        auto chunks = ints(a.str("chunks", "8,64"));
        std::vector<std::pair<std::string, int>> schemes = {{"mutex", 1}, {"atomic", 1}, {"static", 0}};
        for (int c : chunks) schemes.push_back({"atomic_chunk", c});
        for (int T : threads)
            for (auto& [sch, c] : schemes) {
                std::vector<double> ns;
                bool ok = true;
                for (int r = 0; r < reps; ++r) {
                    std::mutex m;
                    long long shared = 0;
                    std::atomic<long long> next{0};
                    std::vector<Padded> sum(T);
                    for (auto& p : sum) p.v = 0;
                    double t0 = now_s();
                    run_threads(T, cpus, [&](int t) {
                        long long acc = 0;
                        if (sch == "mutex") {
                            for (;;) {
                                long long i;
                                {
                                    std::lock_guard<std::mutex> lk(m);
                                    i = shared++;
                                }
                                if (i >= n) break;
                                acc += work_fn((uint32_t)i, w) & 1;
                                acc += 1;
                            }
                        } else if (sch == "atomic") {
                            for (long long i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;)
                                acc += (work_fn((uint32_t)i, w) & 1) + 1;
                        } else if (sch == "atomic_chunk") {
                            for (long long i; (i = next.fetch_add(c, std::memory_order_relaxed)) < n;)
                                for (long long j = i; j < std::min(n, i + c); ++j)
                                    acc += (work_fn((uint32_t)j, w) & 1) + 1;
                        } else {
                            long long lo = n * t / T, hi = n * (t + 1) / T;
                            for (long long j = lo; j < hi; ++j) acc += (work_fn((uint32_t)j, w) & 1) + 1;
                        }
                        sum[t].v = acc;
                    });
                    double dt = now_s() - t0;
                    ns.push_back(dt / n * 1e9);
                    long long tot = 0, want = 0;
                    for (auto& p : sum) tot += p.v;
                    for (long long j = 0; j < n; ++j) want += (work_fn((uint32_t)j, w) & 1) + 1;
                    ok &= tot == want;
                }
                bad += !ok;
                std::sort(ns.begin(), ns.end());
                Json().kv("mode", "dispatch").kv("scheme", sch).kv("chunk", c).kv("threads", T).kv("n", n)
                    .kv("work", w).kv("ns_per_item", ns).kv("ns_med", ns[ns.size() / 2]).kv("correct", (int)ok).emit();
            }
    }
    return bad ? 1 : 0;
}
