// Test driver for the asst2 task systems. Prints one JSON line per measurement.
//   --mode tests        --systems a,b --tests x,y|all --threads 8[,..] --scale 1 --reps 3
//   --mode overhead     --systems .. --threads .. --tasks 1,4,.. --launches 2000
//   --mode granularity  --systems .. --threads .. --total 20000000 --tasks 1,2,4,..
//   --mode dag          --systems .. --threads .. --shape indep|chain|fanin|tree|random --launches 64 --tasks 9 --work 20000
//   --mode idle         --systems .. --threads .. --idle_s 1
// --cpus c0,c1,.. pins worker t to cpus[t % len].
#include <cmath>
#include <random>

#include "../common/common.h"
#include "tasksys.h"
#include "tests.h"

static std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (p <= s.size()) {
        size_t q = s.find(',', p);
        if (q == std::string::npos) q = s.size();
        if (q > p) out.push_back(s.substr(p, q - p));
        p = q + 1;
    }
    return out;
}

static std::vector<int> ints(const std::string& s) {
    std::vector<int> v;
    for (auto& x : split(s)) v.push_back(std::atoi(x.c_str()));
    return v;
}

struct Timed {
    double wall, cpu;
};
template <class F>
Timed timed(F&& f) {
    double c0 = cpu_time_s(), t0 = now_s();
    f();
    return {now_s() - t0, cpu_time_s() - c0};
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct NoopRunnable : IRunnable {
    void runTask(int, int) override {}
};

// Each task runs `iters` dependent LCG steps; the result is folded into a per-task slot so
// the work cannot be optimised away.
struct WorkRunnable : IRunnable {
    std::vector<int> iters;  // per task
    std::vector<uint32_t> sink;
    void runTask(int t, int) override { sink[t] = lcg_iter((uint32_t)t, iters[t]); }
};

int main(int argc, char** argv) {
    Args a(argc, argv);
    std::string mode = a.str("mode", "tests");
    std::vector<int> cpus = a.ints("cpus");
    auto systems = split(a.str("systems", "serial,spawn,spinning,sleeping,async_spinning,async_sleeping"));
    auto threads = ints(a.str("threads", "8"));
    int reps = (int)a.num("reps", 3);
    int bad = 0;

    if (mode == "tests") {
        std::string tl = a.str("tests", "all");
        double scale = a.real("scale", 1.0);
        std::vector<std::pair<std::string, TestFn>> sel;
        for (auto& t : all_tests())
            if (tl == "all" || ("," + tl + ",").find("," + t.first + ",") != std::string::npos) sel.push_back(t);
        for (int nt : threads)
            for (auto& s : systems)
                for (auto& [tn, fn] : sel) {
                    auto ts = make_system(s, nt, cpus);
                    std::vector<double> walls, cpus_used;
                    bool ok = true;
                    for (int r = 0; r < reps; ++r) {
                        TestResult tr;
                        auto tm = timed([&] { tr = fn(ts.get(), scale); });
                        walls.push_back(tm.wall);
                        cpus_used.push_back(tm.cpu);
                        ok &= tr.correct;
                    }
                    bad += !ok;
                    Json()
                        .kv("mode", "tests").kv("system", s).kv("test", tn).kv("threads", nt).kv("scale", scale)
                        .kv("correct", (int)ok).kv("wall_s", walls).kv("cpu_s", cpus_used)
                        .kv("wall_min_s", *std::min_element(walls.begin(), walls.end()))
                        .kv("wall_med_s", median(walls)).kv("cpu_med_s", median(cpus_used))
                        .emit();
                }
    } else if (mode == "overhead") {
        auto tasks = ints(a.str("tasks", "1,4,16,64,256,1024,4096"));
        int launches = (int)a.num("launches", 2000);
        NoopRunnable noop;
        for (int nt : threads)
            for (auto& s : systems) {
                auto ts = make_system(s, nt, cpus);
                for (int n : tasks) {
                    int L = std::max(10, (int)(launches * std::min(1.0, 256.0 / std::max(n, 1))));
                    if (s == "spawn") L = std::max(10, L / 10);
                    for (int w = 0; w < 10; ++w) ts->run(&noop, n);
                    std::vector<double> us;
                    double cpu_ratio = 0;
                    for (int r = 0; r < reps; ++r) {
                        auto tm = timed([&] {
                            for (int k = 0; k < L; ++k) ts->run(&noop, n);
                        });
                        us.push_back(tm.wall / L * 1e6);
                        cpu_ratio = tm.cpu / tm.wall;
                    }
                    Json().kv("mode", "overhead").kv("system", s).kv("threads", nt).kv("tasks", n).kv("launches", L)
                        .kv("us_per_launch", us).kv("us_med", median(us))
                        .kv("us_min", *std::min_element(us.begin(), us.end()))
                        .kv("ns_per_task_med", median(us) * 1e3 / std::max(n, 1)).kv("cpu_per_wall", cpu_ratio).emit();
                }
            }
    } else if (mode == "granularity") {
        auto tasks = ints(a.str("tasks", "1,2,4,8,16,32,64,128,256,1024,4096,16384,65536"));
        long long total = a.num("total", 20000000);
        double imbalance = a.real("imbalance", 0.5);  // task cost multiplier in [1-imb, 1+imb]
        for (int nt : threads)
            for (auto& s : systems) {
                auto ts = make_system(s, nt, cpus);
                for (int n : tasks) {
                    WorkRunnable w;
                    w.iters.resize(n);
                    w.sink.resize(n);
                    std::mt19937 rng(n);
                    std::uniform_real_distribution<double> u(1 - imbalance, 1 + imbalance);
                    for (int i = 0; i < n; ++i) w.iters[i] = std::max(0, (int)std::lround(u(rng) * total / n));
                    ts->run(&w, n);
                    std::vector<double> ws;
                    for (int r = 0; r < reps; ++r) ws.push_back(timed([&] { ts->run(&w, n); }).wall);
                    Json().kv("mode", "granularity").kv("system", s).kv("threads", nt).kv("tasks", n)
                        .kv("total_iters", total).kv("imbalance", imbalance).kv("wall_s", ws)
                        .kv("wall_med_s", median(ws)).kv("wall_min_s", *std::min_element(ws.begin(), ws.end())).emit();
                }
            }
    } else if (mode == "dag") {
        auto shapes = split(a.str("shape", "indep_sync,indep,chain,fanin,tree,random"));
        int K = (int)a.num("launches", 64), n = (int)a.num("tasks", 9), work = (int)a.num("work", 20000);
        for (int nt : threads)
            for (auto& s : systems) {
                auto ts = make_system(s, nt, cpus);
                for (auto& shape : shapes) {
                    // deps[k] = launches k depends on (indices < k)
                    std::vector<std::vector<int>> deps(K);
                    std::mt19937 rng(7);
                    for (int k = 1; k < K; ++k) {
                        if (shape == "chain") deps[k] = {k - 1};
                        else if (shape == "fanin") { if (k == K - 1) for (int j = 0; j < k; ++j) deps[k].push_back(j); }
                        else if (shape == "tree") { if (k >= K / 2) deps[k] = {2 * (k - K / 2), 2 * (k - K / 2) + 1}; }
                        else if (shape == "random") {
                            int nd = rng() % 3;
                            for (int d = 0; d < nd; ++d) deps[k].push_back(std::max(0, k - 1 - (int)(rng() % 8)));
                        }
                    }
                    // span in units of one task: a launch of n tasks on P threads takes ceil(n/P) rounds
                    std::vector<int> depth(K, 0);
                    int span = 0;
                    for (int k = 0; k < K; ++k) {
                        for (int d : deps[k]) depth[k] = std::max(depth[k], depth[d]);
                        depth[k] += 1;
                        span = std::max(span, depth[k]);
                    }
                    std::vector<WorkRunnable> rs(K);
                    for (auto& r : rs) {
                        r.iters.assign(n, work);
                        r.sink.assign(n, 0);
                    }
                    auto once = [&] {
                        if (shape == "indep_sync") {
                            for (int k = 0; k < K; ++k) ts->run(&rs[k], n);
                            return;
                        }
                        std::vector<TaskID> ids(K);
                        for (int k = 0; k < K; ++k) {
                            std::vector<TaskID> dd;
                            for (int d : deps[k]) dd.push_back(ids[d]);
                            ids[k] = ts->runAsyncWithDeps(&rs[k], n, dd);
                        }
                        ts->sync();
                    };
                    once();
                    std::vector<double> ws;
                    for (int r = 0; r < reps; ++r) ws.push_back(timed(once).wall);
                    Json().kv("mode", "dag").kv("system", s).kv("threads", nt).kv("shape", shape).kv("launches", K)
                        .kv("tasks", n).kv("work_iters", work).kv("total_tasks", (long long)K * n)
                        .kv("span_launches", span).kv("wall_s", ws).kv("wall_med_s", median(ws))
                        .kv("wall_min_s", *std::min_element(ws.begin(), ws.end())).emit();
                }
            }
    } else if (mode == "idle") {
        double idle = a.real("idle_s", 1.0);
        for (int nt : threads)
            for (auto& s : systems) {
                auto ts = make_system(s, nt, cpus);
                NoopRunnable noop;
                ts->run(&noop, nt);
                auto tm = timed([&] { std::this_thread::sleep_for(std::chrono::duration<double>(idle)); });
                Json().kv("mode", "idle").kv("system", s).kv("threads", nt).kv("wall_s", tm.wall).kv("cpu_s", tm.cpu)
                    .kv("cores_burned", tm.cpu / tm.wall).emit();
            }
    } else {
        std::fprintf(stderr, "unknown mode\n");
        return 2;
    }
    return bad ? 1 : 0;
}
