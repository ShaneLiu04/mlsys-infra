// CS149 asst3 Part 1: SAXPY y = a*x + y. Kernel-only bandwidth vs N, and end-to-end time
// including host<->device copies with pinned vs pageable host memory (plus a 4-stream
// chunked pipeline on pinned memory). Usage: saxpy <out.jsonl>
#include "common.h"

__global__ void saxpy_kernel(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

static void launch(int n, float a, const float* x, float* y, cudaStream_t s = 0) {
  saxpy_kernel<<<(n + 255) / 256, 256, 0, s>>>(n, a, x, y);
}

int main(int argc, char** argv) {
  Jsonl out(argc > 1 ? argv[1] : "saxpy.jsonl");
  const float a = 2.0f;
  const int maxlog = 28;
  const size_t nmax = size_t(1) << maxlog;
  float *dx, *dy;
  CK(cudaMalloc(&dx, nmax * 4));
  CK(cudaMalloc(&dy, nmax * 4));
  float *px, *py, *pr;  // pinned
  CK(cudaMallocHost(&px, nmax * 4));
  CK(cudaMallocHost(&py, nmax * 4));
  CK(cudaMallocHost(&pr, nmax * 4));
  std::vector<float> gx(nmax), gy(nmax), gr(nmax);  // pageable
  for (size_t i = 0; i < nmax; ++i) {
    px[i] = gx[i] = float(i % 1000) * 0.001f;
    py[i] = gy[i] = float(i % 7);
  }
  cudaStream_t st[4];
  for (auto& s : st) CK(cudaStreamCreate(&s));

  for (int lg = 10; lg <= maxlog; lg += 2) {
    const int n = 1 << lg;
    const double bytes = 12.0 * n;  // read x, read y, write y
    CK(cudaMemcpy(dx, px, size_t(n) * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, py, size_t(n) * 4, cudaMemcpyHostToDevice));
    double k_ms = time_gpu([&] { launch(n, a, dx, dy); }, lg >= 26 ? 10 : 50, 3);

    auto e2e = [&](const float* hx, const float* hy, float* hr, int reps) {
      std::vector<double> t, h2d, d2h;
      for (int r = 0; r < reps + 1; ++r) {
        CK(cudaDeviceSynchronize());
        double t0 = wall_ms();
        CK(cudaMemcpy(dx, hx, size_t(n) * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dy, hy, size_t(n) * 4, cudaMemcpyHostToDevice));
        double t1 = wall_ms();
        launch(n, a, dx, dy);
        CK(cudaDeviceSynchronize());
        double t2 = wall_ms();
        CK(cudaMemcpy(hr, dy, size_t(n) * 4, cudaMemcpyDeviceToHost));
        double t3 = wall_ms();
        if (r == 0) continue;  // warm-up
        t.push_back(t3 - t0);
        h2d.push_back(t1 - t0);
        d2h.push_back(t3 - t2);
      }
      return std::vector<double>{median(t), median(h2d), median(d2h)};
    };
    int reps = lg >= 26 ? 3 : 10;
    auto pin = e2e(px, py, pr, reps);
    auto pag = e2e(gx.data(), gy.data(), gr.data(), reps);

    // chunked pipeline on pinned memory: overlap H2D, kernel and D2H across 4 streams
    std::vector<double> ts;
    for (int r = 0; r < reps + 1; ++r) {
      CK(cudaDeviceSynchronize());
      double t0 = wall_ms();
      int chunks = n >= (1 << 20) ? 16 : 1;
      int cs = n / chunks;
      for (int c = 0; c < chunks; ++c) {
        cudaStream_t s = st[c % 4];
        size_t off = size_t(c) * cs;
        CK(cudaMemcpyAsync(dx + off, px + off, size_t(cs) * 4, cudaMemcpyHostToDevice, s));
        CK(cudaMemcpyAsync(dy + off, py + off, size_t(cs) * 4, cudaMemcpyHostToDevice, s));
        launch(cs, a, dx + off, dy + off, s);
        CK(cudaMemcpyAsync(pr + off, dy + off, size_t(cs) * 4, cudaMemcpyDeviceToHost, s));
      }
      CK(cudaDeviceSynchronize());
      if (r) ts.push_back(wall_ms() - t0);
    }
    double streams_ms = median(ts);

    // correctness of the last pinned pipeline result
    long long bad = 0;
    for (int i = 0; i < n; ++i) {
      float ref = a * px[i] + py[i];
      if (pr[i] != ref) ++bad;
    }
    out.line(obj(kvs("kind", "saxpy"), kvi("n", n), kv("bytes", bytes), kv("kernel_ms", k_ms),
                 kv("kernel_gbs", bytes / k_ms / 1e6), kv("pinned_e2e_ms", pin[0]),
                 kv("pinned_h2d_ms", pin[1]), kv("pinned_d2h_ms", pin[2]),
                 kv("pinned_e2e_gbs", bytes / pin[0] / 1e6), kv("pageable_e2e_ms", pag[0]),
                 kv("pageable_h2d_ms", pag[1]), kv("pageable_d2h_ms", pag[2]),
                 kv("pageable_e2e_gbs", bytes / pag[0] / 1e6), kv("streams_e2e_ms", streams_ms),
                 kv("streams_e2e_gbs", bytes / streams_ms / 1e6),
                 kv("pinned_h2d_gbs", 8.0 * n / pin[1] / 1e6),
                 kv("pinned_d2h_gbs", 4.0 * n / pin[2] / 1e6),
                 kv("pageable_h2d_gbs", 8.0 * n / pag[1] / 1e6),
                 kv("pageable_d2h_gbs", 4.0 * n / pag[2] / 1e6), kvi("mismatch", bad)));
  }

  // launch overhead: empty-ish kernel back-to-back (throughput) and launch+sync (latency)
  double lt = time_gpu([&] { for (int i = 0; i < 1000; ++i) launch(1, a, dx, dy); }, 5, 1) / 1000;
  std::vector<double> ls;
  for (int i = 0; i < 1000; ++i) {
    double t0 = wall_ms();
    launch(1, a, dx, dy);
    CK(cudaDeviceSynchronize());
    ls.push_back(wall_ms() - t0);
  }
  out.line(obj(kvs("kind", "launch"), kv("back_to_back_us", lt * 1e3),
               kv("launch_sync_us_median", median(ls) * 1e3)));
  return 0;
}
