#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

inline double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// User + system CPU seconds consumed by the whole process (all threads).
inline double cpu_time_s() {
#ifdef _WIN32
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto f = [](FILETIME t) { return ((unsigned long long)t.dwHighDateTime << 32 | t.dwLowDateTime) * 1e-7; };
    return f(k) + f(u);
#else
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + 1e-6 * (r.ru_utime.tv_usec + r.ru_stime.tv_usec);
#endif
}

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

// Run f() `reps` times and return the fastest wall time (the starter reports min-of-N too).
template <class F>
double min_time(int reps, F&& f) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        f();
        best = std::min(best, now_s() - t0);
    }
    return best;
}

// Pin the calling thread to one logical CPU (no-op for cpu < 0).
inline void pin_thread(int cpu) {
    if (cpu < 0) return;
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

struct Args {
    std::vector<std::string> v;
    Args(int argc, char** argv) : v(argv + 1, argv + argc) {}
    bool has(const std::string& k) const { return std::find(v.begin(), v.end(), "--" + k) != v.end(); }
    std::string str(const std::string& k, const std::string& d) const {
        for (size_t i = 0; i + 1 < v.size(); ++i)
            if (v[i] == "--" + k) return v[i + 1];
        return d;
    }
    long long num(const std::string& k, long long d) const {
        auto s = str(k, "");
        return s.empty() ? d : std::atoll(s.c_str());
    }
    double real(const std::string& k, double d) const {
        auto s = str(k, "");
        return s.empty() ? d : std::atof(s.c_str());
    }
    // "--cpus 0,2,4,6" -> {0,2,4,6}; empty means unpinned.
    std::vector<int> ints(const std::string& k) const {
        std::vector<int> out;
        auto s = str(k, "");
        size_t p = 0;
        while (p < s.size()) {
            size_t q = s.find(',', p);
            if (q == std::string::npos) q = s.size();
            if (q > p) out.push_back(std::atoi(s.substr(p, q - p).c_str()));
            p = q + 1;
        }
        return out;
    }
};

inline int cpu_for(const std::vector<int>& cpus, int t) { return cpus.empty() ? -1 : cpus[t % cpus.size()]; }

// Launch n threads running body(t) and join them; thread t is pinned to cpus[t % size] if given.
inline void run_threads(int n, const std::vector<int>& cpus, const std::function<void(int)>& body) {
    std::vector<std::thread> th;
    th.reserve(n);
    for (int t = 0; t < n; ++t)
        th.emplace_back([&, t] {
            pin_thread(cpu_for(cpus, t));
            body(t);
        });
    for (auto& x : th) x.join();
}

// Minimal JSON-lines writer: every experiment prints one object per line on stdout.
struct Json {
    std::string s = "{";
    bool first = true;
    Json& key(const char* k) {
        if (!first) s += ",";
        first = false;
        s += "\"";
        s += k;
        s += "\":";
        return *this;
    }
    Json& kv(const char* k, const std::string& v) { key(k); s += "\"" + v + "\""; return *this; }
    Json& kv(const char* k, const char* v) { return kv(k, std::string(v)); }
    Json& kv(const char* k, double v) { key(k); char b[64]; std::snprintf(b, sizeof b, "%.9g", v); s += b; return *this; }
    Json& kv(const char* k, long long v) { key(k); s += std::to_string(v); return *this; }
    Json& kv(const char* k, int v) { return kv(k, (long long)v); }
    Json& kv(const char* k, const std::vector<double>& v) {
        key(k);
        s += "[";
        for (size_t i = 0; i < v.size(); ++i) {
            char b[64];
            std::snprintf(b, sizeof b, "%s%.9g", i ? "," : "", v[i]);
            s += b;
        }
        s += "]";
        return *this;
    }
    void emit() { std::printf("%s}\n", s.c_str()); std::fflush(stdout); }
};

template <class T>
T* aligned_alloc_n(size_t n, size_t align = 64) {
#ifdef _WIN32
    return static_cast<T*>(_aligned_malloc(n * sizeof(T), align));
#else
    return static_cast<T*>(std::aligned_alloc(align, ((n * sizeof(T) + align - 1) / align) * align));
#endif
}

template <class T>
void aligned_free(T* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
