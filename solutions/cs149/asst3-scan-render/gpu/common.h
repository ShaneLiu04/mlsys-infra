// Shared helpers for the asst3 CUDA programs: error checking, event timing, JSONL output.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e_ = (x);                                                                  \
    if (e_ != cudaSuccess) {                                                               \
      std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__,               \
                   cudaGetErrorString(e_));                                                \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

inline double wall_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

inline double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Median GPU time (ms) of fn() measured with CUDA events, after `warm` warm-up calls.
template <class F>
double time_gpu(F fn, int reps = 10, int warm = 2) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  for (int i = 0; i < warm; ++i) fn();
  CK(cudaDeviceSynchronize());
  std::vector<double> ts;
  for (int i = 0; i < reps; ++i) {
    CK(cudaEventRecord(a));
    fn();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, a, b));
    ts.push_back(ms);
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return median(ts);
}

struct Jsonl {
  FILE* f;
  explicit Jsonl(const char* path) : f(std::fopen(path, "w")) {
    if (!f) {
      std::perror(path);
      std::exit(1);
    }
  }
  ~Jsonl() { std::fclose(f); }
  void line(const std::string& s) {
    std::fprintf(f, "%s\n", s.c_str());
    std::fflush(f);
    std::printf("%s\n", s.c_str());
  }
};

inline std::string kv(const char* k, double v) {
  char b[96];
  std::snprintf(b, sizeof b, "\"%s\":%.6g", k, v);
  return b;
}
inline std::string kvi(const char* k, long long v) {
  char b[96];
  std::snprintf(b, sizeof b, "\"%s\":%lld", k, v);
  return b;
}
inline std::string kvs(const char* k, const std::string& v) {
  return "\"" + std::string(k) + "\":\"" + v + "\"";
}
template <class... T>
std::string obj(const T&... parts) {
  std::string s = "{";
  bool first = true;
  for (const std::string& p : {std::string(parts)...}) {
    if (!first) s += ",";
    s += p;
    first = false;
  }
  return s + "}";
}
