// CS149 asst1 prog1: Mandelbrot with std::thread.
//
// The kernel and the two views follow the starter (1600x1200, 256 iterations,
// view 2 = scaleAndShift(0.015, -0.986, 0.30)). Three row schedules:
//   block       thread t owns a contiguous band of rows (the starter's naive split)
//   interleaved row j belongs to thread j % T
//   dynamic     threads grab `chunk` rows at a time from an atomic counter
// Output: one JSON object per configuration (min of --reps runs) with per-thread compute times.
#include <cmath>

#include "../common/common.h"

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

struct View {
    float x0, y0, x1, y1;
};

static View make_view(int idx) {
    View v{-2.f, -1.f, 1.f, 1.f};
    if (idx == 2) {
        const float scale = .015f, sx = -.986f, sy = .30f;
        v = {v.x0 * scale + sx, v.y0 * scale + sy, v.x1 * scale + sx, v.y1 * scale + sy};
    }
    return v;
}

struct Problem {
    View v;
    int width = 1600, height = 1200, max_iter = 256;
};

static void mandel_row(const Problem& p, int j, int* out) {
    float dx = (p.v.x1 - p.v.x0) / p.width;
    float dy = (p.v.y1 - p.v.y0) / p.height;
    for (int i = 0; i < p.width; ++i) {
        float x = p.v.x0 + i * dx;
        float y = p.v.y0 + j * dy;
        out[j * p.width + i] = mandel(x, y, p.max_iter);
    }
}

static void mandel_serial(const Problem& p, int* out) {
    for (int j = 0; j < p.height; ++j) mandel_row(p, j, out);
}

struct RunStats {
    double wall = 1e30;
    std::vector<double> thread_s;
    std::vector<double> thread_rows;
};

static RunStats run_once(const Problem& p, const std::string& sched, int T, int chunk,
                         const std::vector<int>& cpus, int* out) {
    RunStats st;
    st.thread_s.assign(T, 0.0);
    st.thread_rows.assign(T, 0.0);
    std::atomic<int> next{0};
    double t0 = now_s();
    run_threads(T, cpus, [&](int t) {
        double ts = now_s();
        int rows = 0;
        if (sched == "block") {
            int lo = (int)((long long)p.height * t / T), hi = (int)((long long)p.height * (t + 1) / T);
            for (int j = lo; j < hi; ++j) mandel_row(p, j, out), ++rows;
        } else if (sched == "interleaved") {
            for (int j = t; j < p.height; j += T) mandel_row(p, j, out), ++rows;
        } else {
            for (;;) {
                int s = next.fetch_add(chunk, std::memory_order_relaxed);
                if (s >= p.height) break;
                int e = std::min(s + chunk, p.height);
                for (int j = s; j < e; ++j) mandel_row(p, j, out), ++rows;
            }
        }
        st.thread_s[t] = now_s() - ts;
        st.thread_rows[t] = rows;
    });
    st.wall = now_s() - t0;
    return st;
}

int main(int argc, char** argv) {
    Args a(argc, argv);
    Problem p;
    int view = (int)a.num("view", 1);
    p.v = make_view(view);
    p.max_iter = (int)a.num("iters", 256);
    int reps = (int)a.num("reps", 5);
    std::string mode = a.str("mode", "sweep");
    auto cpus = a.ints("cpus");
    std::string tag = a.str("tag", "");

    std::vector<int> ref(p.width * p.height), out(p.width * p.height);
    double serial = min_time(reps, [&] { mandel_serial(p, ref.data()); });

    if (mode == "counts") {
        // Dump per-row iteration totals (cost profile) and an 8x-downsampled image for plotting.
        for (int j = 0; j < p.height; ++j) {
            long long s = 0;
            for (int i = 0; i < p.width; ++i) s += ref[j * p.width + i] + 1;
            Json().kv("kind", "row").kv("view", view).kv("row", j).kv("work", s).emit();
        }
        return 0;
    }

    std::vector<std::string> scheds;
    if (a.has("sched")) scheds = {a.str("sched", "block")};
    else scheds = {"block", "interleaved", "dynamic"};
    std::vector<int> threads;
    if (a.has("threads")) threads = a.ints("threads");
    else threads = {1, 2, 3, 4, 5, 6, 7, 8, 16};
    int chunk = (int)a.num("chunk", 4);

    int bad = 0;
    for (const auto& sc : scheds)
        for (int T : threads) {
            RunStats best;
            for (int r = 0; r < reps; ++r) {
                std::fill(out.begin(), out.end(), -1);
                RunStats st = run_once(p, sc, T, chunk, cpus, out.data());
                if (out != ref) ++bad;
                if (st.wall < best.wall) best = st;
            }
            Json().kv("kind", "run").kv("tag", tag).kv("view", view).kv("sched", sc).kv("threads", T)
                .kv("chunk", sc == "dynamic" ? chunk : 0).kv("serial_s", serial).kv("wall_s", best.wall)
                .kv("speedup", serial / best.wall).kv("thread_s", best.thread_s)
                .kv("thread_rows", best.thread_rows).kv("correct", (int)(bad == 0)).emit();
        }
    return bad ? 2 : 0;
}
