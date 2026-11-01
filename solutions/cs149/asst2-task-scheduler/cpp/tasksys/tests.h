// Re-implementations of the official asst2 test workloads (tests/tests.h of the starter).
// Sizes follow the structure of the official tests (tasks per launch, number of launches,
// sync vs async dependency pattern); absolute sizes are scaled by `scale`.
// Every test checks its output and returns correct = false on any mismatch.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "tasksys.h"

struct TestResult {
    bool correct = true;
};

inline uint32_t lcg_iter(uint32_t v, int iters) {
    for (int k = 0; k < iters; ++k) v = v * 1103515245u + 12345u;
    return v;
}

inline int fib_rec(int n) { return n < 2 ? n : fib_rec(n - 1) + fib_rec(n - 2); }

// ---------------------------------------------------------------- ping-pong family
// Two arrays; launch k reads one and writes the other, so consecutive launches depend on each
// other. Equal work: every task runs base_iters per element; unequal: task i runs ~2*base*(i+1)/n.
struct PingPongRunnable : IRunnable {
    const uint32_t* in;
    uint32_t* out;
    int n_elems, base_iters;
    bool equal;
    static int iters_for(int task, int ntasks, int base, bool equal) {
        return equal ? base : std::max(1, 2 * base * (task + 1) / ntasks);
    }
    void runTask(int t, int nt) override {
        int chunk = (n_elems + nt - 1) / nt, lo = t * chunk, hi = std::min(n_elems, lo + chunk);
        int it = iters_for(t, nt, base_iters, equal);
        for (int j = lo; j < hi; ++j) out[j] = lcg_iter(in[j], it);
    }
};

inline TestResult ping_pong(ITaskSystem* ts, bool equal, bool async, int n_elems, int base_iters, int launches,
                            int ntasks = 64) {
    std::vector<uint32_t> a(n_elems), b(n_elems);
    for (int j = 0; j < n_elems; ++j) a[j] = j;
    std::vector<PingPongRunnable> rs(launches);
    TaskID prev = -1;
    for (int k = 0; k < launches; ++k) {
        auto& r = rs[k];
        r.in = (k % 2 == 0) ? a.data() : b.data();
        r.out = (k % 2 == 0) ? b.data() : a.data();
        r.n_elems = n_elems;
        r.base_iters = base_iters;
        r.equal = equal;
        if (async) {
            std::vector<TaskID> deps;
            if (prev >= 0) deps.push_back(prev);
            prev = ts->runAsyncWithDeps(&r, ntasks, deps);
        } else {
            ts->run(&r, ntasks);
        }
    }
    if (async) ts->sync();
    const auto& res = (launches % 2 == 0) ? a : b;
    TestResult tr;
    int chunk = (n_elems + ntasks - 1) / ntasks;
    for (int s = 0; s < 97; ++s) {
        int j = (int)((long long)s * 7919 % n_elems);
        int it = PingPongRunnable::iters_for(j / chunk, ntasks, base_iters, equal);
        uint32_t v = j;
        for (int k = 0; k < launches; ++k) v = lcg_iter(v, it);
        if (launches == 0) v = j;
        if (res[j] != v) tr.correct = false;
    }
    return tr;
}

// ---------------------------------------------------------------- recursive fibonacci
struct FibRunnable : IRunnable {
    int n;
    int* out;
    void runTask(int t, int) override { out[t] = fib_rec(n); }
};

inline TestResult recursive_fibonacci(ITaskSystem* ts, bool async, int fib_n, int ntasks, int launches) {
    std::vector<std::vector<int>> outs(launches, std::vector<int>(ntasks, -1));
    std::vector<FibRunnable> rs(launches);
    for (int k = 0; k < launches; ++k) {
        rs[k].n = fib_n;
        rs[k].out = outs[k].data();
        if (async) ts->runAsyncWithDeps(&rs[k], ntasks, {});
        else ts->run(&rs[k], ntasks);
    }
    if (async) ts->sync();
    int want = fib_rec(fib_n);
    TestResult tr;
    for (auto& o : outs)
        for (int v : o) tr.correct &= (v == want);
    return tr;
}

// ---------------------------------------------------------------- math ops in a tight loop
// Many launches of a small array kernel: launch-overhead bound.
struct MathOpsRunnable : IRunnable {
    const float* in;
    float* out;
    int n;
    void runTask(int t, int nt) override {
        int chunk = (n + nt - 1) / nt, lo = t * chunk, hi = std::min(n, lo + chunk);
        for (int j = lo; j < hi; ++j) {
            float x = in[j];
            float y = x * x + 2.0f * x + 1.0f;
            y = y / (1.0f + x * x) - 0.5f * x;
            out[j] = y + x;
        }
    }
};

inline float math_ref(float x) {
    float y = x * x + 2.0f * x + 1.0f;
    y = y / (1.0f + x * x) - 0.5f * x;
    return y + x;
}

inline bool close_enough(float a, float b) {
    return std::fabs(a - b) <= 1e-5f * std::max(1.0f, std::fabs(b));
}

inline TestResult math_ops(ITaskSystem* ts, bool async, int n, int ntasks, int launches) {
    std::vector<float> in(n), out(launches * (size_t)n);
    for (int j = 0; j < n; ++j) in[j] = 0.001f * (j % 1000);
    std::vector<MathOpsRunnable> rs(launches);
    TaskID prev = -1;
    for (int k = 0; k < launches; ++k) {
        rs[k] = MathOpsRunnable();
        rs[k].in = in.data();
        rs[k].out = out.data() + (size_t)k * n;
        rs[k].n = n;
        if (async) {
            std::vector<TaskID> deps;
            if (prev >= 0) deps.push_back(prev);
            prev = ts->runAsyncWithDeps(&rs[k], ntasks, deps);
        } else {
            ts->run(&rs[k], ntasks);
        }
    }
    if (async) ts->sync();
    TestResult tr;
    for (int k = 0; k < launches; ++k)
        for (int j = 0; j < n; j += 37) tr.correct &= close_enough(out[(size_t)k * n + j], math_ref(in[j]));
    return tr;
}

// Fan-in: `width` independent math launches, then one launch depending on all of them that
// sums their outputs. Repeated `rounds` times, each round depending on the previous fan-in.
struct SumRunnable : IRunnable {
    std::vector<const float*> srcs;
    float* out;
    int n;
    void runTask(int t, int nt) override {
        int chunk = (n + nt - 1) / nt, lo = t * chunk, hi = std::min(n, lo + chunk);
        for (int j = lo; j < hi; ++j) {
            float s = 0;
            for (auto* p : srcs) s += p[j];
            out[j] = s;
        }
    }
};

inline TestResult math_ops_fan_in(ITaskSystem* ts, int n, int ntasks, int width, int rounds) {
    std::vector<float> in(n);
    for (int j = 0; j < n; ++j) in[j] = 0.001f * (j % 1000);
    std::vector<std::vector<float>> leaf(rounds * width, std::vector<float>(n));
    std::vector<std::vector<float>> sums(rounds, std::vector<float>(n));
    std::vector<MathOpsRunnable> lr(rounds * width);
    std::vector<SumRunnable> sr(rounds);
    TaskID prev = -1;
    for (int r = 0; r < rounds; ++r) {
        std::vector<TaskID> ids;
        for (int w = 0; w < width; ++w) {
            auto& m = lr[r * width + w];
            m.in = in.data();
            m.out = leaf[r * width + w].data();
            m.n = n;
            std::vector<TaskID> deps;
            if (prev >= 0) deps.push_back(prev);
            ids.push_back(ts->runAsyncWithDeps(&m, ntasks, deps));
        }
        sr[r].out = sums[r].data();
        sr[r].n = n;
        for (int w = 0; w < width; ++w) sr[r].srcs.push_back(leaf[r * width + w].data());
        prev = ts->runAsyncWithDeps(&sr[r], ntasks, ids);
    }
    ts->sync();
    TestResult tr;
    for (int r = 0; r < rounds; ++r)
        for (int j = 0; j < n; j += 41) {
            float s = 0;
            for (int w = 0; w < width; ++w) s += math_ref(in[j]);
            tr.correct &= close_enough(sums[r][j], s);
        }
    return tr;
}

// Reduction tree: `leaves` leaf launches, then a binary tree of add launches up to the root.
struct LeafRunnable : IRunnable {
    uint32_t* out;
    int n, leaf, iters;
    void runTask(int t, int nt) override {
        int chunk = (n + nt - 1) / nt, lo = t * chunk, hi = std::min(n, lo + chunk);
        for (int j = lo; j < hi; ++j) out[j] = lcg_iter((uint32_t)(leaf * 131 + j), iters);
    }
};
struct AddRunnable : IRunnable {
    const uint32_t *a, *b;
    uint32_t* out;
    int n;
    void runTask(int t, int nt) override {
        int chunk = (n + nt - 1) / nt, lo = t * chunk, hi = std::min(n, lo + chunk);
        for (int j = lo; j < hi; ++j) out[j] = a[j] + b[j];
    }
};

inline TestResult math_ops_reduction_tree(ITaskSystem* ts, int n, int ntasks, int leaves, int iters) {
    int nodes = 2 * leaves - 1;  // heap layout: node i has children 2i+1, 2i+2; leaves are the last `leaves`
    std::vector<std::vector<uint32_t>> buf(nodes, std::vector<uint32_t>(n));
    std::vector<LeafRunnable> lr(leaves);
    std::vector<AddRunnable> ar(nodes);
    std::vector<TaskID> id(nodes);
    for (int i = nodes - 1; i >= 0; --i) {
        if (i >= leaves - 1) {
            auto& L = lr[i - (leaves - 1)];
            L.out = buf[i].data();
            L.n = n;
            L.leaf = i - (leaves - 1);
            L.iters = iters;
            id[i] = ts->runAsyncWithDeps(&L, ntasks, {});
        } else {
            auto& A = ar[i];
            A.a = buf[2 * i + 1].data();
            A.b = buf[2 * i + 2].data();
            A.out = buf[i].data();
            A.n = n;
            id[i] = ts->runAsyncWithDeps(&A, ntasks, {id[2 * i + 1], id[2 * i + 2]});
        }
    }
    ts->sync();
    TestResult tr;
    for (int j = 0; j < n; j += 53) {
        uint32_t s = 0;
        for (int l = 0; l < leaves; ++l) s += lcg_iter((uint32_t)(l * 131 + j), iters);
        tr.correct &= (buf[0][j] == s);
    }
    return tr;
}

// ---------------------------------------------------------------- spin between run calls
// The caller thread does serial work between small launches. A spinning pool keeps burning the
// cores the caller needs; a sleeping pool gives them back.
inline TestResult spin_between_run_calls(ITaskSystem* ts, bool async, int iters, int fib_task, int fib_main) {
    std::vector<int> out(2 * iters, -1);
    std::vector<FibRunnable> rs(iters);
    volatile int sink = 0;
    for (int k = 0; k < iters; ++k) {
        rs[k].n = fib_task;
        rs[k].out = out.data() + 2 * k;
        if (async) {
            ts->runAsyncWithDeps(&rs[k], 2, {});
            ts->sync();
        } else {
            ts->run(&rs[k], 2);
        }
        sink = sink + fib_rec(fib_main);
    }
    TestResult tr;
    int want = fib_rec(fib_task);
    for (int v : out) tr.correct &= (v == want);
    return tr;
}

// ---------------------------------------------------------------- mandelbrot chunked
inline int mandel_px(float x0, float y0, int max_it) {
    float x = x0, y = y0;
    int i = 0;
    for (; i < max_it; ++i) {
        if (x * x + y * y > 4.f) break;
        float nx = x * x - y * y + x0;
        y = 2.f * x * y + y0;
        x = nx;
    }
    return i;
}

struct MandelRunnable : IRunnable {
    int w, h, max_it;
    int* out;
    void runTask(int t, int nt) override {
        int rows = (h + nt - 1) / nt, lo = t * rows, hi = std::min(h, lo + rows);
        for (int r = lo; r < hi; ++r)
            for (int c = 0; c < w; ++c)
                out[r * w + c] = mandel_px(-2.f + 3.f * c / w, -1.f + 2.f * r / h, max_it);
    }
};

inline TestResult mandelbrot_chunked(ITaskSystem* ts, bool async, int w, int h, int max_it, int ntasks,
                                     int launches) {
    std::vector<std::vector<int>> img(launches, std::vector<int>((size_t)w * h, -1));
    std::vector<MandelRunnable> rs(launches);
    for (int k = 0; k < launches; ++k) {
        rs[k].w = w;
        rs[k].h = h;
        rs[k].max_it = max_it;
        rs[k].out = img[k].data();
        if (async) ts->runAsyncWithDeps(&rs[k], ntasks, {});
        else ts->run(&rs[k], ntasks);
    }
    if (async) ts->sync();
    TestResult tr;
    for (int k = 0; k < launches; ++k)
        for (int r = 0; r < h; r += 7)
            for (int c = 0; c < w; c += 5)
                tr.correct &= (img[k][r * w + c] == mandel_px(-2.f + 3.f * c / w, -1.f + 2.f * r / h, max_it));
    return tr;
}

// ---------------------------------------------------------------- registry
inline int sc(double base, double scale, int lo = 1) { return std::max(lo, (int)std::lround(base * scale)); }

using TestFn = std::function<TestResult(ITaskSystem*, double)>;

inline const std::vector<std::pair<std::string, TestFn>>& all_tests() {
    static const std::vector<std::pair<std::string, TestFn>> v = {
        {"super_super_light", [](ITaskSystem* t, double s) { return ping_pong(t, true, false, 32 * 1024, 0, sc(400, s, 2)); }},
        {"super_super_light_async", [](ITaskSystem* t, double s) { return ping_pong(t, true, true, 32 * 1024, 0, sc(400, s, 2)); }},
        {"super_light", [](ITaskSystem* t, double s) { return ping_pong(t, true, false, 32 * 1024, 1, sc(400, s, 2)); }},
        {"super_light_async", [](ITaskSystem* t, double s) { return ping_pong(t, true, true, 32 * 1024, 1, sc(400, s, 2)); }},
        {"ping_pong_equal", [](ITaskSystem* t, double s) { return ping_pong(t, true, false, 128 * 1024, 32, sc(64, s, 2)); }},
        {"ping_pong_equal_async", [](ITaskSystem* t, double s) { return ping_pong(t, true, true, 128 * 1024, 32, sc(64, s, 2)); }},
        {"ping_pong_unequal", [](ITaskSystem* t, double s) { return ping_pong(t, false, false, 128 * 1024, 32, sc(64, s, 2)); }},
        {"ping_pong_unequal_async", [](ITaskSystem* t, double s) { return ping_pong(t, false, true, 128 * 1024, 32, sc(64, s, 2)); }},
        {"recursive_fibonacci", [](ITaskSystem* t, double s) { return recursive_fibonacci(t, false, 23, 256, sc(8, s)); }},
        {"recursive_fibonacci_async", [](ITaskSystem* t, double s) { return recursive_fibonacci(t, true, 23, 256, sc(8, s)); }},
        {"math_operations_in_tight_for_loop", [](ITaskSystem* t, double s) { return math_ops(t, false, 2048, 16, sc(2000, s, 2)); }},
        {"math_operations_in_tight_for_loop_async", [](ITaskSystem* t, double s) { return math_ops(t, true, 2048, 16, sc(2000, s, 2)); }},
        {"math_operations_in_tight_for_loop_fewer_tasks", [](ITaskSystem* t, double s) { return math_ops(t, false, 2048, 4, sc(2000, s, 2)); }},
        {"math_operations_in_tight_for_loop_fewer_tasks_async", [](ITaskSystem* t, double s) { return math_ops(t, true, 2048, 4, sc(2000, s, 2)); }},
        {"math_operations_in_tight_for_loop_fan_in", [](ITaskSystem* t, double s) { return math_ops_fan_in(t, 8192, 16, 16, sc(100, s)); }},
        {"math_operations_in_tight_for_loop_reduction_tree", [](ITaskSystem* t, double s) { return math_ops_reduction_tree(t, 16384, 16, std::max(2, 1 << (int)std::lround(std::log2(std::max(2.0, 256 * s)))), 8); }},
        {"spin_between_run_calls", [](ITaskSystem* t, double s) { return spin_between_run_calls(t, false, sc(20, s), 22, 27); }},
        {"spin_between_run_calls_async", [](ITaskSystem* t, double s) { return spin_between_run_calls(t, true, sc(20, s), 22, 27); }},
        {"mandelbrot_chunked", [](ITaskSystem* t, double s) { return mandelbrot_chunked(t, false, 800, 600, 256, 128, sc(4, s)); }},
        {"mandelbrot_chunked_async", [](ITaskSystem* t, double s) { return mandelbrot_chunked(t, true, 800, 600, 256, 128, sc(4, s)); }},
    };
    return v;
}
