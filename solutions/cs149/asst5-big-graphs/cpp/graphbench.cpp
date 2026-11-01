// CS149 A5 (EC): direction-optimizing BFS (Beamer et al.) plus pull/push PageRank.
// Graph is a directed CSR with both outgoing and incoming edges. Source is vertex 0,
// distances match a serial queue BFS exactly (unreached vertices stay -1).
//
// Build: g++ -O3 -fopenmp -std=c++17 -o graphbench graphbench.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

struct Graph {
    int n = 0;
    std::vector<int> out_off, out_dst, in_off, in_dst, out_deg;
    long long m = 0;
};

static double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void build_csr(Graph& g, int n, std::vector<std::pair<int, int>>& edges) {
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    edges.erase(std::remove_if(edges.begin(), edges.end(), [](auto e) { return e.first == e.second; }), edges.end());
    g.n = n;
    g.m = (long long)edges.size();
    g.out_off.assign(n + 1, 0);
    g.in_off.assign(n + 1, 0);
    g.out_deg.assign(n, 0);
    for (auto [u, v] : edges) {
        g.out_off[u + 1]++;
        g.in_off[v + 1]++;
    }
    for (int i = 0; i < n; i++) {
        g.out_off[i + 1] += g.out_off[i];
        g.in_off[i + 1] += g.in_off[i];
        g.out_deg[i] = g.out_off[i + 1] - g.out_off[i];
    }
    g.out_dst.resize(edges.size());
    g.in_dst.resize(edges.size());
    std::vector<int> ow = g.out_off, iw = g.in_off;
    for (auto [u, v] : edges) {
        g.out_dst[ow[u]++] = v;
        g.in_dst[iw[v]++] = u;
    }
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint64_t next() {
        s ^= s << 7;
        s ^= s >> 9;
        s ^= s << 8;
        return s;
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    int below(int n) { return (int)(next() % (uint64_t)n); }
};

static Graph make_graph(const std::string& kind, int n, long long m, uint64_t seed) {
    std::vector<std::pair<int, int>> edges;
    Rng rng(seed);
    if (kind == "chain") {
        n = std::max(n, 2);
        edges.reserve(n - 1);
        for (int i = 0; i < n - 1; i++) edges.emplace_back(i, i + 1);
    } else if (kind == "star") {
        edges.reserve(n - 1);
        for (int i = 1; i < n; i++) edges.emplace_back(0, i);
    } else if (kind == "grid") {
        int s = std::max(2, (int)std::lround(std::sqrt((double)n)));
        n = s * s;
        auto id = [s](int r, int c) { return r * s + c; };
        for (int r = 0; r < s; r++)
            for (int c = 0; c < s; c++) {
                if (c + 1 < s) edges.emplace_back(id(r, c), id(r, c + 1));
                if (r + 1 < s) edges.emplace_back(id(r, c), id(r + 1, c));
                if (c > 0) edges.emplace_back(id(r, c), id(r, c - 1));
                if (r > 0) edges.emplace_back(id(r, c), id(r - 1, c));
            }
    } else if (kind == "er") {
        edges.reserve(m);
        for (long long i = 0; i < m; i++) edges.emplace_back(rng.below(n), rng.below(n));
    } else {  // rmat, Graph500 parameters
        int scale = 1;
        while ((1 << scale) < n) scale++;
        n = 1 << scale;
        const double A = 0.57, B = 0.19, C = 0.19;
        edges.reserve(m);
        for (long long i = 0; i < m; i++) {
            int u = 0, v = 0;
            for (int b = scale - 1; b >= 0; b--) {
                double r = rng.uniform();
                int bit = 1 << b;
                if (r < A) {
                } else if (r < A + B) u |= bit;
                else if (r < A + B + C) v |= bit;
                else {
                    u |= bit;
                    v |= bit;
                }
            }
            edges.emplace_back(u, v);
        }
    }
    Graph g;
    build_csr(g, n, edges);
    return g;
}

struct LevelStat {
    int frontier = 0;
    long long frontier_edges = 0;
    long long examined = 0;
    double seconds = 0;
    char which = 't';  // t top-down, b bottom-up
};

struct BfsResult {
    std::vector<int> dist;
    std::vector<LevelStat> levels;
    double seconds = 0;
    long long examined = 0;
};

static BfsResult bfs_serial(const Graph& g) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<int> frontier, next;
    frontier.push_back(0);
    r.dist[0] = 0;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        double ts = now_s();
        for (int u : frontier) {
            st.frontier_edges += g.out_deg[u];
            for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                int v = g.out_dst[k];
                st.examined++;
                if (r.dist[v] < 0) {
                    r.dist[v] = level + 1;
                    next.push_back(v);
                }
            }
        }
        st.seconds = now_s() - ts;
        st.which = 't';
        r.examined += st.examined;
        r.levels.push_back(st);
        frontier.swap(next);
        next.clear();
    }
    r.seconds = now_s() - t0;
    return r;
}

// Shared frontier with a critical section around every insertion. This is the
// version the assignment warns against: correct, and slow once the frontier is large.
static BfsResult bfs_topdown_critical(const Graph& g, int threads) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<int> frontier{0}, next;
    r.dist[0] = 0;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        long long examined = 0;
        double ts = now_s();
        int nf = (int)frontier.size();
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads) reduction(+ : examined)
        for (int i = 0; i < nf; i++) {
            int u = frontier[i];
            for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                int v = g.out_dst[k];
                examined++;
                if (r.dist[v] >= 0) continue;
#pragma omp critical
                {
                    if (r.dist[v] < 0) {
                        r.dist[v] = level + 1;
                        next.push_back(v);
                    }
                }
            }
        }
        st.examined = examined;
        st.seconds = now_s() - ts;
        r.examined += examined;
        r.levels.push_back(st);
        frontier.swap(next);
        next.clear();
    }
    r.seconds = now_s() - t0;
    return r;
}

// CAS claims the vertex; the winner appends it with an atomic index.
static BfsResult bfs_topdown_atomic(const Graph& g, int threads) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<int> frontier{0}, next;
    r.dist[0] = 0;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        next.assign(g.n, 0);
        int tail = 0;
        long long examined = 0;
        double ts = now_s();
        int nf = (int)frontier.size();
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads) reduction(+ : examined)
        for (int i = 0; i < nf; i++) {
            int u = frontier[i];
            for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                int v = g.out_dst[k];
                examined++;
                if (r.dist[v] >= 0) continue;
                if (__sync_bool_compare_and_swap(&r.dist[v], -1, level + 1)) {
                    int slot = __sync_fetch_and_add(&tail, 1);
                    next[slot] = v;
                }
            }
        }
        next.resize(tail);
        st.examined = examined;
        st.seconds = now_s() - ts;
        r.examined += examined;
        r.levels.push_back(st);
        frontier.swap(next);
    }
    r.seconds = now_s() - t0;
    return r;
}

// Each thread keeps its own next-frontier. The only shared write is the CAS.
static BfsResult bfs_topdown_local(const Graph& g, int threads) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<int> frontier{0};
    r.dist[0] = 0;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        for (int u : frontier) st.frontier_edges += g.out_deg[u];
        std::vector<std::vector<int>> local(threads);
        long long examined = 0;
        double ts = now_s();
        int nf = (int)frontier.size();
#pragma omp parallel num_threads(threads) reduction(+ : examined)
        {
            int tid = omp_get_thread_num();
            auto& mine = local[tid];
#pragma omp for schedule(dynamic, 64)
            for (int i = 0; i < nf; i++) {
                int u = frontier[i];
                for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                    int v = g.out_dst[k];
                    examined++;
                    if (r.dist[v] >= 0) continue;
                    if (__sync_bool_compare_and_swap(&r.dist[v], -1, level + 1)) mine.push_back(v);
                }
            }
        }
        size_t total = 0;
        for (auto& v : local) total += v.size();
        std::vector<int> next;
        next.reserve(total);
        for (auto& v : local) next.insert(next.end(), v.begin(), v.end());
        st.examined = examined;
        st.seconds = now_s() - ts;
        st.which = 't';
        r.examined += examined;
        r.levels.push_back(st);
        frontier.swap(next);
    }
    r.seconds = now_s() - t0;
    return r;
}

// Top-down step that splits the frontier's out-edges into equal ranges, so a single
// hub vertex is shared by all threads instead of landing on one of them.
static long long step_topdown_edges(const Graph& g, const std::vector<int>& frontier, std::vector<int>& dist, int level,
                                    int threads, std::vector<std::vector<int>>& local) {
    int nf = (int)frontier.size();
    std::vector<long long> pre(nf + 1, 0);
    for (int i = 0; i < nf; i++) pre[i + 1] = pre[i] + g.out_deg[frontier[i]];
    const long long total = pre[nf], chunk = 2048;
    const long long nchunks = (total + chunk - 1) / chunk;
    long long examined = 0;
#pragma omp parallel num_threads(threads) reduction(+ : examined)
    {
        auto& mine = local[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 4)
        for (long long c = 0; c < nchunks; c++) {
            long long lo = c * chunk, hi = std::min(total, lo + chunk);
            int i = (int)(std::upper_bound(pre.begin(), pre.end(), lo) - pre.begin()) - 1;
            for (long long e = lo; e < hi;) {
                while (pre[i + 1] <= e) i++;
                int u = frontier[i];
                long long end = std::min(hi, pre[i + 1]);
                const int* dst = &g.out_dst[g.out_off[u] + (e - pre[i])];
                for (long long k = 0; k < end - e; k++) {
                    int v = dst[k];
                    examined++;
                    if (dist[v] >= 0) continue;
                    if (__sync_bool_compare_and_swap(&dist[v], -1, level + 1)) mine.push_back(v);
                }
                e = end;
            }
        }
    }
    return examined;
}

static BfsResult bfs_topdown_edges(const Graph& g, int threads) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<int> frontier{0};
    r.dist[0] = 0;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        std::vector<std::vector<int>> local(threads);
        double ts = now_s();
        st.examined = step_topdown_edges(g, frontier, r.dist, level, threads, local);
        size_t total = 0;
        for (auto& v : local) total += v.size();
        std::vector<int> next;
        next.reserve(total);
        for (auto& v : local) next.insert(next.end(), v.begin(), v.end());
        st.seconds = now_s() - ts;
        r.examined += st.examined;
        r.levels.push_back(st);
        frontier.swap(next);
    }
    r.seconds = now_s() - t0;
    return r;
}

// Every unvisited vertex scans incoming edges until it finds one on the frontier.
// Each vertex is owned by one thread, so the distance write needs no atomic.
static BfsResult bfs_bottomup(const Graph& g, int threads, bool dynamic_sched) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<char> on_frontier(g.n, 0);
    std::vector<int> frontier{0};
    r.dist[0] = 0;
    on_frontier[0] = 1;
    double t0 = now_s();
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        LevelStat st;
        st.frontier = (int)frontier.size();
        st.which = 'b';
        std::vector<std::vector<int>> local(threads);
        long long examined = 0;
        double ts = now_s();
#pragma omp parallel num_threads(threads) reduction(+ : examined)
        {
            int tid = omp_get_thread_num();
            auto& mine = local[tid];
            if (dynamic_sched) {
#pragma omp for schedule(dynamic, 1024)
                for (int v = 0; v < g.n; v++) {
                    if (r.dist[v] >= 0) continue;
                    for (int k = g.in_off[v]; k < g.in_off[v + 1]; k++) {
                        examined++;
                        if (on_frontier[g.in_dst[k]]) {
                            r.dist[v] = level + 1;
                            mine.push_back(v);
                            break;
                        }
                    }
                }
            } else {
#pragma omp for schedule(static)
                for (int v = 0; v < g.n; v++) {
                    if (r.dist[v] >= 0) continue;
                    for (int k = g.in_off[v]; k < g.in_off[v + 1]; k++) {
                        examined++;
                        if (on_frontier[g.in_dst[k]]) {
                            r.dist[v] = level + 1;
                            mine.push_back(v);
                            break;
                        }
                    }
                }
            }
        }
        std::fill(on_frontier.begin(), on_frontier.end(), 0);
        frontier.clear();
        for (auto& v : local) {
            for (int x : v) {
                frontier.push_back(x);
                on_frontier[x] = 1;
            }
        }
        st.examined = examined;
        st.seconds = now_s() - ts;
        r.examined += examined;
        r.levels.push_back(st);
    }
    r.seconds = now_s() - t0;
    return r;
}

// Beamer, Asanovic, Patterson, SC 2012: top-down while the frontier is small,
// bottom-up while it would touch more edges than scanning the unvisited vertices.
// alpha discounts bottom-up work because a vertex stops at the first parent.
static BfsResult bfs_hybrid(const Graph& g, int threads, double alpha, bool edge_split = false) {
    BfsResult r;
    r.dist.assign(g.n, -1);
    std::vector<char> on_frontier(g.n, 0);
    std::vector<int> frontier{0};
    r.dist[0] = 0;
    on_frontier[0] = 1;
    long long visited = 1;
    double t0 = now_s();
    bool growing = true;
    int prev = 1;
    for (int level = 0; !frontier.empty() && level < g.n; level++) {
        long long frontier_edges = 0;
        for (int u : frontier) frontier_edges += g.out_deg[u];
        long long unvisited = (long long)g.n - visited;
        // Expected bottom-up edge inspections ~ (edges into unvisited) / alpha.
        long long bu_edges = 0;
        // Approximate with mean in-degree, which equals m/n, times unvisited / alpha.
        bu_edges = (long long)((double)unvisited * ((double)g.m / std::max(g.n, 1)) / alpha);
        bool use_bu = growing && frontier_edges > bu_edges && frontier.size() > 1;
        LevelStat st;
        st.frontier = (int)frontier.size();
        st.frontier_edges = frontier_edges;
        st.which = use_bu ? 'b' : 't';
        std::vector<std::vector<int>> local(threads);
        long long examined = 0;
        double ts = now_s();
        if (!use_bu && edge_split) {
            examined = step_topdown_edges(g, frontier, r.dist, level, threads, local);
        } else if (!use_bu) {
            int nf = (int)frontier.size();
#pragma omp parallel num_threads(threads) reduction(+ : examined)
            {
                int tid = omp_get_thread_num();
                auto& mine = local[tid];
#pragma omp for schedule(dynamic, 64)
                for (int i = 0; i < nf; i++) {
                    int u = frontier[i];
                    for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                        int v = g.out_dst[k];
                        examined++;
                        if (r.dist[v] >= 0) continue;
                        if (__sync_bool_compare_and_swap(&r.dist[v], -1, level + 1)) mine.push_back(v);
                    }
                }
            }
        } else {
#pragma omp parallel num_threads(threads) reduction(+ : examined)
            {
                int tid = omp_get_thread_num();
                auto& mine = local[tid];
#pragma omp for schedule(dynamic, 1024)
                for (int v = 0; v < g.n; v++) {
                    if (r.dist[v] >= 0) continue;
                    for (int k = g.in_off[v]; k < g.in_off[v + 1]; k++) {
                        examined++;
                        if (on_frontier[g.in_dst[k]]) {
                            r.dist[v] = level + 1;
                            mine.push_back(v);
                            break;
                        }
                    }
                }
            }
        }
        std::fill(on_frontier.begin(), on_frontier.end(), 0);
        frontier.clear();
        for (auto& v : local) {
            for (int x : v) {
                frontier.push_back(x);
                on_frontier[x] = 1;
            }
        }
        visited += (long long)frontier.size();
        growing = (int)frontier.size() >= prev;
        prev = (int)frontier.size();
        st.examined = examined;
        st.seconds = now_s() - ts;
        r.examined += examined;
        r.levels.push_back(st);
    }
    r.seconds = now_s() - t0;
    return r;
}

struct PrResult {
    std::vector<double> rank;
    double seconds = 0;
    int iters = 0;
};

static PrResult pagerank(const Graph& g, const std::string& algo, int threads, int iters, double damp = 0.85) {
    PrResult r;
    int n = g.n;
    std::vector<double> rank(n, 1.0 / n), next(n);
    double base = (1.0 - damp) / n;
    double t0 = now_s();
    for (int it = 0; it < iters; it++) {
        double dangling = 0;
#pragma omp parallel for reduction(+ : dangling) num_threads(threads)
        for (int u = 0; u < n; u++)
            if (g.out_deg[u] == 0) dangling += rank[u];
        double dang = damp * dangling / n;
        if (algo == "pull") {
#pragma omp parallel for schedule(dynamic, 1024) num_threads(threads)
            for (int v = 0; v < n; v++) {
                double s = 0;
                for (int k = g.in_off[v]; k < g.in_off[v + 1]; k++) {
                    int u = g.in_dst[k];
                    s += rank[u] / g.out_deg[u];
                }
                next[v] = base + dang + damp * s;
            }
        } else {  // push: scatter with atomics
            std::fill(next.begin(), next.end(), base + dang);
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads)
            for (int u = 0; u < n; u++) {
                if (g.out_deg[u] == 0) continue;
                double c = damp * rank[u] / g.out_deg[u];
                for (int k = g.out_off[u]; k < g.out_off[u + 1]; k++) {
                    int v = g.out_dst[k];
#pragma omp atomic
                    next[v] += c;
                }
            }
        }
        rank.swap(next);
    }
    r.seconds = now_s() - t0;
    r.iters = iters;
    r.rank.swap(rank);
    return r;
}

static long long reached(const std::vector<int>& d) {
    long long c = 0;
    for (int x : d)
        if (x >= 0) c++;
    return c;
}

static bool same_dist(const std::vector<int>& a, const std::vector<int>& b) { return a == b; }

static int fail(const char* msg) {
    std::fprintf(stderr, "FAIL %s\n", msg);
    return 1;
}

static int selftest() {
    auto check_all = [](const Graph& g, const char* name) {
        auto ref = bfs_serial(g);
        struct Item {
            const char* n;
            BfsResult r;
        };
        Item items[] = {
            {"critical", bfs_topdown_critical(g, 4)},
            {"atomic", bfs_topdown_atomic(g, 4)},
            {"local", bfs_topdown_local(g, 4)},
            {"bottom", bfs_bottomup(g, 4, true)},
            {"bottom_static", bfs_bottomup(g, 4, false)},
            {"hybrid", bfs_hybrid(g, 4, 14)},
            {"edges", bfs_topdown_edges(g, 4)},
            {"hybrid_edges", bfs_hybrid(g, 4, 14, true)},
            {"edges1", bfs_topdown_edges(g, 1)},
            {"local1", bfs_topdown_local(g, 1)},
            {"hybrid1", bfs_hybrid(g, 1, 4)},
        };
        for (auto& it : items) {
            if (!same_dist(it.r.dist, ref.dist)) {
                std::fprintf(stderr, "FAIL %s %s reached %lld vs %lld\n", name, it.n, reached(it.r.dist), reached(ref.dist));
                return 1;
            }
        }
        auto pull = pagerank(g, "pull", 4, 20);
        auto push = pagerank(g, "push", 4, 20);
        double l1 = 0, sum = 0;
        for (int i = 0; i < g.n; i++) {
            l1 += std::fabs(pull.rank[i] - push.rank[i]);
            sum += pull.rank[i];
        }
        if (std::fabs(sum - 1.0) > 1e-8) {
            std::fprintf(stderr, "FAIL %s pagerank sum %.12f\n", name, sum);
            return 1;
        }
        if (l1 > 1e-8) {
            std::fprintf(stderr, "FAIL %s pagerank push/pull L1 %.3e\n", name, l1);
            return 1;
        }
        std::printf("ok %s n=%d m=%lld reached=%lld pr_l1=%.3e\n", name, g.n, g.m, reached(ref.dist), l1);
        return 0;
    };
    if (check_all(make_graph("chain", 64, 0, 1), "chain")) return 1;
    if (check_all(make_graph("star", 128, 0, 1), "star")) return 1;
    if (check_all(make_graph("grid", 100, 0, 1), "grid")) return 1;
    if (check_all(make_graph("er", 400, 2000, 3), "er")) return 1;
    if (check_all(make_graph("rmat", 256, 2000, 5), "rmat")) return 1;
    // chain distances are exactly the index
    auto chain = bfs_hybrid(make_graph("chain", 40, 0, 1), 2, 14);
    for (int i = 0; i < 40; i++)
        if (chain.dist[i] != i) return fail("chain distance");
    std::printf("ok all\n");
    return 0;
}

static BfsResult run_algo(const Graph& g, const std::string& algo, int threads) {
    if (algo == "serial") return bfs_serial(g);
    if (algo == "critical") return bfs_topdown_critical(g, threads);
    if (algo == "atomic") return bfs_topdown_atomic(g, threads);
    if (algo == "local") return bfs_topdown_local(g, threads);
    if (algo == "edges") return bfs_topdown_edges(g, threads);
    if (algo == "bottom") return bfs_bottomup(g, threads, true);
    if (algo == "bottom_static") return bfs_bottomup(g, threads, false);
    if (algo == "hybrid") return bfs_hybrid(g, threads, 14);
    if (algo == "hybrid_edges") return bfs_hybrid(g, threads, 14, true);
    if (algo.rfind("hybrid_a", 0) == 0) return bfs_hybrid(g, threads, std::atof(algo.c_str() + 8));
    std::fprintf(stderr, "unknown algo %s\n", algo.c_str());
    std::exit(1);
}

static FILE* open_out(const std::string& path) {
    FILE* out = std::fopen(path.c_str(), "w");
    if (!out) {
        std::perror(path.c_str());
        std::exit(1);
    }
    return out;
}

// Extra campaign: edge-range splitting, an RMAT scale sweep and independent graph seeds.
// Every repetition is written out so the spread can be plotted, not just the minimum.
static void extra(const std::string& out_path, int reps) {
    FILE* out = open_out(out_path);
    auto timed = [&](const char* suite, const Graph& g, const std::string& gname, uint64_t seed, const std::string& algo,
                     int threads, int nrep, bool trace) {
        run_algo(g, algo, threads);  // warmup
        std::vector<double> ms;
        BfsResult kept;
        double best = 1e300;
        for (int r = 0; r < nrep; r++) {
            auto res = run_algo(g, algo, threads);
            ms.push_back(res.seconds * 1e3);
            if (res.seconds < best) {
                best = res.seconds;
                kept = std::move(res);
            }
        }
        auto ref = bfs_serial(g);
        int ok = same_dist(kept.dist, ref.dist);
        std::string times;
        for (size_t i = 0; i < ms.size(); i++) times += (i ? "," : "") + std::to_string(ms[i]);
        std::fprintf(out,
                     "{\"kind\":\"bfs\",\"suite\":\"%s\",\"graph\":\"%s\",\"seed\":%llu,\"n\":%d,\"m\":%lld,\"algo\":\"%s\","
                     "\"threads\":%d,\"ms\":%.4f,\"times\":[%s],\"examined\":%lld,\"levels\":%zu,\"reached\":%lld,\"correct\":%d}\n",
                     suite, gname.c_str(), (unsigned long long)seed, g.n, g.m, algo.c_str(), threads, best * 1e3, times.c_str(),
                     kept.examined, kept.levels.size(), reached(kept.dist), ok);
        if (trace) {
            for (size_t i = 0; i < kept.levels.size(); i++) {
                auto& st = kept.levels[i];
                std::fprintf(out,
                             "{\"kind\":\"level\",\"suite\":\"%s\",\"graph\":\"%s\",\"algo\":\"%s\",\"threads\":%d,\"level\":%zu,"
                             "\"frontier\":%d,\"examined\":%lld,\"ms\":%.4f,\"which\":\"%c\"}\n",
                             suite, gname.c_str(), algo.c_str(), threads, i, st.frontier, st.examined, st.seconds * 1e3, st.which);
            }
        }
        std::fflush(out);
        std::printf("%s %s %s th=%d best %.2f ms ok=%d\n", suite, gname.c_str(), algo.c_str(), threads, best * 1e3, ok);
        std::fflush(stdout);
        if (!ok) {
            std::fprintf(stderr, "MISMATCH %s %s\n", gname.c_str(), algo.c_str());
            std::exit(2);
        }
    };

    struct Spec {
        const char* kind;
        int n;
        long long m;
        const char* name;
    };
    Spec edge_specs[] = {
        {"star", 1 << 21, 0, "star_2m"},
        {"rmat", 1 << 20, 8LL << 20, "rmat_1m_8m"},
        {"er", 1 << 20, 4LL << 20, "er_1m_4m"},
        {"grid", 256 * 256, 0, "grid_256"},
    };
    for (auto s : edge_specs) {
        Graph g = make_graph(s.kind, s.n, s.m, 42);
        timed("edges", g, s.name, 42, "serial", 1, reps, false);
        for (const char* algo : {"local", "edges", "hybrid", "hybrid_edges"})
            for (int th : {1, 2, 4, 8}) timed("edges", g, s.name, 42, algo, th, reps, th == 8 && s.kind == std::string("star"));
    }

    for (int scale = 16; scale <= 23; scale++) {
        int n = 1 << scale;
        double tb = now_s();
        Graph g = make_graph("rmat", n, 8LL * n, 42);
        double build = now_s() - tb;
        std::string name = "rmat_s" + std::to_string(scale);
        std::fprintf(out, "{\"kind\":\"graph\",\"suite\":\"scale\",\"graph\":\"%s\",\"scale\":%d,\"n\":%d,\"m\":%lld,\"build_s\":%.3f}\n",
                     name.c_str(), scale, g.n, g.m, build);
        timed("scale", g, name, 42, "serial", 1, reps, false);
        for (const char* algo : {"local", "bottom", "hybrid", "hybrid_edges"}) timed("scale", g, name, 42, algo, 8, reps, std::string(algo) == "hybrid");
    }

    for (uint64_t seed : {1ULL, 2ULL, 3ULL}) {
        Graph g = make_graph("rmat", 1 << 20, 8LL << 20, seed);
        timed("seed", g, "rmat_1m_8m", seed, "serial", 1, 7, false);
        for (const char* algo : {"local", "bottom", "hybrid"})
            for (int th : {4, 8}) timed("seed", g, "rmat_1m_8m", seed, algo, th, 7, false);
    }
    std::fclose(out);
}

static void campaign(const std::string& out_path, int reps) {
    FILE* out = open_out(out_path);
    auto run_bfs = [&](const Graph& g, const std::string& gname, const std::string& algo, int threads, bool trace) {
        auto once = [&]() { return run_algo(g, algo, threads); };
        once();  // warmup
        double best = 1e300;
        BfsResult kept;
        for (int r = 0; r < reps; r++) {
            auto res = once();
            if (res.seconds < best) {
                best = res.seconds;
                kept = std::move(res);
            }
        }
        auto ref = bfs_serial(g);
        int ok = same_dist(kept.dist, ref.dist);
        std::string line = "{\"kind\":\"bfs\",\"graph\":\"" + gname + "\",\"n\":" + std::to_string(g.n) + ",\"m\":" + std::to_string(g.m) +
                           ",\"algo\":\"" + algo + "\",\"threads\":" + std::to_string(threads) + ",\"ms\":" + std::to_string(best * 1e3) +
                           ",\"examined\":" + std::to_string(kept.examined) + ",\"levels\":" + std::to_string(kept.levels.size()) +
                           ",\"reached\":" + std::to_string(reached(kept.dist)) + ",\"correct\":" + std::to_string(ok) + "}\n";
        std::fputs(line.c_str(), out);
        std::fflush(out);
        if (trace) {
            for (size_t i = 0; i < kept.levels.size(); i++) {
                auto& st = kept.levels[i];
                std::fprintf(out,
                             "{\"kind\":\"level\",\"graph\":\"%s\",\"algo\":\"%s\",\"threads\":%d,\"level\":%zu,\"frontier\":%d,"
                             "\"frontier_edges\":%lld,\"examined\":%lld,\"ms\":%.4f,\"which\":\"%c\"}\n",
                             gname.c_str(), algo.c_str(), threads, i, st.frontier, st.frontier_edges, st.examined, st.seconds * 1e3, st.which);
            }
        }
        std::printf("%s", line.c_str());
        if (!ok) {
            std::fprintf(stderr, "MISMATCH %s %s\n", gname.c_str(), algo.c_str());
            std::exit(2);
        }
    };

    struct Spec {
        const char* kind;
        int n;
        long long m;
        const char* name;
        bool heavy;
    };
    Spec specs[] = {
        {"rmat", 1 << 20, 8LL << 20, "rmat_1m_8m", true},
        {"er", 1 << 20, 4LL << 20, "er_1m_4m", true},
        {"grid", 256 * 256, 0, "grid_256", true},
        {"chain", 20000, 0, "chain_20k", false},
        {"star", 1 << 21, 0, "star_2m", false},
    };
    for (auto spec : specs) {
        std::printf("building %s\n", spec.name);
        std::fflush(stdout);
        Graph g = make_graph(spec.kind, spec.n, spec.m, 42);
        std::fprintf(out, "{\"kind\":\"graph\",\"graph\":\"%s\",\"n\":%d,\"m\":%lld,\"family\":\"%s\"}\n", spec.name, g.n, g.m, spec.kind);
        std::vector<std::string> algos = {"serial", "local", "bottom", "bottom_static", "hybrid"};
        if (!spec.heavy) algos.push_back("critical");
        if (spec.heavy) algos.push_back("atomic");
        std::vector<int> threads = spec.heavy ? std::vector<int>{1, 2, 4, 8} : std::vector<int>{1, 4};
        for (auto& algo : algos) {
            for (int th : threads) {
                if (algo == "serial" && th != 1) continue;
                if (algo == "critical" && th != 4) continue;
                // Chain levels are one vertex each; 20k trace lines per run add nothing.
                bool trace = th == 4 && algo != "atomic" && algo != "bottom_static" && std::string(spec.kind) != "chain";
                run_bfs(g, spec.name, algo, algo == "serial" ? 1 : th, trace);
            }
        }
        if (std::string(spec.kind) == "rmat") {
            for (double a : {2.0, 4.0, 8.0, 14.0, 32.0, 64.0}) {
                run_bfs(g, spec.name, "hybrid_a" + std::to_string(a), 4, false);
            }
            for (auto& algo : {"pull", "push"}) {
                for (int th : {1, 2, 4, 8}) {
                    pagerank(g, algo, th, 2);  // warmup
                    double best = 1e300;
                    PrResult kept;
                    for (int r = 0; r < reps; r++) {
                        auto pr = pagerank(g, algo, th, 10);
                        if (pr.seconds < best) {
                            best = pr.seconds;
                            kept = std::move(pr);
                        }
                    }
                    double sum = std::accumulate(kept.rank.begin(), kept.rank.end(), 0.0);
                    std::fprintf(out, "{\"kind\":\"pagerank\",\"graph\":\"%s\",\"algo\":\"%s\",\"threads\":%d,\"iters\":10,\"ms\":%.4f,\"sum\":%.8f}\n",
                                 spec.name, algo, th, best * 1e3, sum);
                    std::printf("pagerank %s th %d %.2f ms\n", algo, th, best * 1e3);
                }
            }
        }
    }
    std::fclose(out);
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "selftest";
    if (mode == "selftest") return selftest();
    if (mode == "campaign") {
        std::string out = "campaign.jsonl";
        int reps = 3;
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--out" && i + 1 < argc) out = argv[++i];
            if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        }
        campaign(out, reps);
        return 0;
    }
    if (mode == "extra") {
        std::string out = "extra.jsonl";
        int reps = 5;
        for (int i = 2; i < argc; i++) {
            if (std::string(argv[i]) == "--out" && i + 1 < argc) out = argv[++i];
            if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        }
        extra(out, reps);
        return 0;
    }
    std::fprintf(stderr, "usage: graphbench selftest | campaign --out f --reps 3 | extra --out f --reps 5\n");
    return 1;
}
