# CS149 Asst1：多核、SIMD、带宽与 GPU 对照（C++ / AVX-512 / CUDA）

对应 CS149 Fall 2024 Assignment 1「Analyzing Parallel Program Performance on a Quad-Core CPU」。六个 program 全部用
C++17 重写（`cpp/`），ISPC gang/task 用 AVX2/AVX-512 intrinsics 与原子计数器线程池手写；另用 CUDA（`gpu/a1_gpu.cu`）
在 RTX 5090 上回答同样的问题。所有 SIMD、多线程、GPU 结果与标量参考**逐位相等**（`-ffp-contract=off` / `-fmad=false`）。

完整报告（21 页，中文，含面试追问）：[`reports/cs149-asst1/cs149_asst1_report.pdf`](../../../reports/cs149-asst1/cs149_asst1_report.pdf)。

平台：

| | 服务器（主平台） | 本机（四核对照） |
| --- | --- | --- |
| CPU | 2 × Xeon Gold 6459C（Sapphire Rapids），cgroup 16 CPU，固定绑 socket 0 的 12 个空闲物理核 | Ice Lake 虚拟机，4 核 8 线程 |
| GPU | RTX 5090（sm_120，32 GB GDDR7） | 无 |
| 统计 | 3 轮独立实验的中位数，图中误差棒为最小–最大 | 单轮 |

## 一句话结论

- **负载均衡**：Mandelbrot 中间贵，12 线程连续块划分只有 5.89×，动态分块 11.1×。
- **SIMD 发散**：一个向量的代价由最慢的 lane 决定。sqrt 每 8 个元素放 1 个慢的，AVX2 只有 0.72×，比标量还慢；AVX-512 在 all_slow、one_slow_8、one_slow_16 上耗时完全相同（96.8 / 96.8 / 97.3 ms）。
- **延迟受限**：Mandelbrot 内层是依赖链，SMT 几乎把一个核当两个用；单线程交错两组 16 lane 向量，AVX-512 从 11.2× 提到 18.3×，超过向量宽度。
- **带宽**：saxpy 单线程 DRAM 23 GB/s、12 线程 173 GB/s；streaming store 省掉 write-allocate，DRAM 区间 +35%；跨 socket 访问 −30%。
- **Amdahl**：K-means 只并行最重的阶段只有 2.1×，三个阶段都并行并做循环交换与向量化后 11.7×。
- **GPU**：Mandelbrot 比最好的 12 核版本再快 60–100×；但 saxpy 一旦要过 PCIe，端到端只有 52 GB/s，比 CPU 还慢。

## 怎么跑

```bash
cd solutions/cs149/asst1-cpu-parallel
python scripts/build_cpp.py                    # g++ -O3 -march=native -ffp-contract=off，产物在 cpp/bin
python -m pytest -q tests                      # 含 test_cpp.py：六个 program 的正确性
python scripts/run_cpp_experiments.py --platform server --cores 0,1,2,3,4,6,10,11,14,16,19,23 --smt-offset 64
bash gpu/run_gpu.sh                            # 需要 CUDA；写 artifacts/raw/gpu/*.jsonl
python scripts/plot_cpp.py                     # 写 artifacts/figures/cpp_*.png|pdf
```

原始指标：`artifacts/raw/cpp/<platform>/*.jsonl`（`server`、`server_r2`、`server_r3` 为 3 轮，`server16` 为 16 核，`local4c` 为本机），
`artifacts/raw/gpu/*.jsonl`。

## 结果

### prog1：负载均衡

![views](artifacts/figures/cpp_mandel_views.png)

左：迭代数；右：每行工作量与 4/12 等分的连续块切分线。view 1 中间行跑满 256 次，12 等分时最重块是平均的 2.00 倍，所以连续块划分的加速比上限只有 6×。

![prog1 speedup](artifacts/figures/cpp_prog1_speedup.png)

| 线程 | view1 block | view1 interleaved | view1 dynamic | view2 block | view2 dynamic |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.99 | 2.00 | 2.00 | 1.68 | 1.99 |
| 3 | **1.64** | 2.99 | 3.00 | 2.18 | 2.98 |
| 8 | 4.07 | 7.93 | 7.74 | 4.17 | 6.79 |
| 12 | 5.89 | 9.10 | **11.08** | 5.78 | 9.52 |

3 线程比 2 线程还慢：三等分时中间那块独揽集合主体。隔行均衡了工作量，但仍是静态划分，任何一个核被其他租户抢占都会拖慢全体；动态分块同时解决两者。

![thread time](artifacts/figures/cpp_prog1_thread_time.png)

8 线程时每个线程的计算时间：墙钟由最慢线程决定。

![smt](artifacts/figures/cpp_smt.png)

同样线程数放在更少物理核上：6 线程挤进 3 个核（SMT）只比 6 个核慢 11%（标量），说明内核受依赖链延迟限制。
紫色的双向量交错版已经在单线程内填掉了空闲发射槽，从 SMT 得到的收益最少（view 2 上慢 45%）——**SMT 与 ILP 争夺同一批空闲槽**。

![chunk](artifacts/figures/cpp_prog1_chunk.png)

动态分块的粒度：1–16 行基本持平，100 行退化为连续块。注意这张图比线程扫描图低约 18%：它直接在刚唤醒的 12 个核上运行，
而线程扫描到 12 线程时核已预热（推测为频率爬升，未用计数器验证）。同一张图内的比较不受影响。

### prog2：fake vector intrinsics 与 lane 利用率

![prog2](artifacts/figures/cpp_prog2_utilization.png)

| 指数分布 | W=2 | W=4 | W=8 | W=16 |
| --- | ---: | ---: | ---: | ---: |
| uniform（官方） | 0.812 | 0.741 | 0.704 | 0.686 |
| constant | 0.938 | 0.938 | 0.938 | 0.938 |
| sorted | 0.938 | 0.938 | 0.938 | 0.937 |
| one_slow_per_vector | 0.711 | 0.584 | 0.521 | 0.490 |

宽度越大，组内最大指数越接近 9，利用率越低。constant 也只有 15/16：clamp 那条 `vmove` 一个 lane 都不启用。
排序后利用率回到 0.938（对应 GPU 的按工作量重排）。`arraySum` 指令数 = 1 + 2N/W + 2·log2 W，实测完全一致。

### prog3：SIMD gang、ILP 与 task

![simd](artifacts/figures/cpp_prog3_simd.png)

| | 标量 ms | AVX2 | AVX-512 | AVX-512 ×2 交错 | 模型 8·U8 / 16·U16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| view 1 | 332.2 | 7.42× | 11.21× | **18.30×** | 7.66 / 14.9 |
| view 2 | 185.9 | 5.96× | 8.31× | **12.31×** | 6.65 / 11.9 |

AVX2 接近模型（瓶颈就是发散）；AVX-512 远低于模型——16 lane 让依赖链变宽而没有变短。每次推进两组独立向量后超过 16×。

![tasks](artifacts/figures/cpp_prog3_tasks.png)

12 个工作线程：task 数等于核数时又回到连续块的不均衡，≥64 个 task 后到达平台。

![ladder](artifacts/figures/cpp_ladder.png)

加速阶梯：SIMD × ILP × 12 核 ≈ 139×（view 1），RTX 5090 再快约 60–80×。

![pin](artifacts/figures/cpp_pin.png)

测量陷阱：1 个 task 在新建工作线程上比主线程慢约 23%，与绑在 CPU 0、别的核或不绑核无关（假设被否定），与上面的「冷核」现象一致。
两种视角都多出 8.9 ms（38.6 对 29.7 ms，31.3 对 22.4 ms），工作量差 1.3 倍、多出的时间却相同，说明这是每次 launch 一个加性的固定代价。
本机 4 核对照中只有绑在 CPU 0 时慢（0.75 / 0.90），其他核与不绑核为 0.92–1.00，所以这是服务器宿主机特有的现象；A2 用 CPU 时间进一步把它定位为「执行变慢而不是等待」。

### prog4：Newton sqrt 的最好与最坏输入

![sqrt](artifacts/figures/cpp_prog4_sqrt.png)

| 输入 | 平均迭代 | 标量 ms | AVX2 | AVX-512 | AVX-512 + 12 线程 |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 5.35 | 742.7 | 3.98× | 8.36× | 68.5× |
| all_slow（最好） | 20.0 | 1305.5 | 5.09× | **13.48×** | **104.1×** |
| one_slow_8（最坏） | 2.50 | 183.3 | **0.72×** | 1.89× | 15.2× |
| one_slow_16 | 1.25 | 99.0 | 0.75× | 1.02× | 8.6× |

### prog5：saxpy 带宽

![sizes](artifacts/figures/cpp_prog5_sizes.png)

单线程：L1 约 229 GB/s → L2 125 → L3 35 → DRAM 18 GB/s。12 线程在 12.6 MB（落在 12 × 2 MB L2 内）超过 1 TB/s；最左端 12 线程很低——小问题不值得并行。
streaming store 在 DRAM 区间 +35%（≈16/12，省掉 write-allocate 读），在缓存区间反而慢 3 倍。

![threads](artifacts/figures/cpp_prog5_threads.png)

DRAM 真实带宽（含 write-allocate）：1 线程 23 GB/s，12 线程 173 GB/s，尚未饱和（单核受 Little 定律限制）。
NUMA：数据 first-touch 在 socket 0，线程全在 socket 1 时 123 GB/s（−30%）。

### prog6：K-means profiling

![kmeans](artifacts/figures/cpp_prog6_kmeans.png)

| 版本 | 改动 | 1 线程 | 12 线程 |
| --- | --- | ---: | ---: |
| baseline | 官方结构：k 外层、sqrt、串行 | 1.163 s | — |
| v1 | 并行 computeAssignments | 0.92× | 2.07×（Amdahl 上限 2.8×） |
| v2 | + centroids / cost 按点并行，线程私有部分和 | 0.92× | 8.82× |
| v3 | + 循环交换、平方距离、AVX-512 | 1.64× | **11.72×** |

所有版本的聚类分配与 baseline 完全一致。

### GPU 对照（RTX 5090）

![gpu mandel](artifacts/figures/cpp_gpu_mandel_shapes.png)

2D block 的 warp 效率更高（8×4：0.958 vs 32×1：0.940），但 32 线程的 block（红）受每 SM block 数上限限制，occupancy 不足；8×8 与 16×16 最快。

![gpu sqrt](artifacts/figures/cpp_gpu_sqrt.png)

sqrt 在 GPU 上所有输入都约 0.1 ms（1.54 TB/s，带宽受限，发散被隐藏）。每个元素重复求解 64 次后变成计算受限：
one_slow_8/16/32 与 all_slow 同为 5.03 ms（warp 宽 32），one_slow_64 约一半，排序后的 random 快 2.1 倍。

![bw](artifacts/figures/cpp_bw_cpu_vs_gpu.png)

saxpy：显存 1.55 TB/s、L2 6.1 TB/s，但含 PCIe 拷贝端到端只有 52 GB/s，低于 CPU 12 线程的 130 GB/s。
K-means assignment：AoS 3.74 ms → SoA 1.43 ms（合并访存）。

![gpu iters](artifacts/figures/cpp_gpu_iters.png)

## 面试讲法（C++/GPU 版）

**20 秒**：并行加速有三个杀手——负载不均（连续块 5.9× vs 动态 11.1×）、SIMD 发散（每 8 个混 1 个慢的，AVX2 比标量还慢）、
带宽（saxpy 单核 23 GB/s，12 核 173 GB/s，过 PCIe 只剩 52 GB/s）。另外 Mandelbrot 受延迟限制，SMT 和 ILP 都能接近翻倍，但二者不能相乘。

**追问**（详见报告第 11 节）：3 线程为什么比 2 线程慢？SIMD 为什么能超过 W？怎样构造 SIMD 比标量慢的输入？write-allocate 是什么？
K-means 为什么只有 2×？task 数设多少？GPU 发散和 CPU 发散有何不同？什么时候不该上 GPU？

**不能推出的结论**：
- 不能把「冷核」与「预热」两组数据混在一张图里比较；本文同一图内条件一致。
- 不能说 AVX-512 慢是因为降频：双向量交错版同样 512 bit，快 1.6 倍。
- 不能把多租户服务器上的绝对数字当作这颗 CPU 的规格；只看同条件下的相对关系。

## 文件（C++/GPU 版）

```text
cpp/common/common.h              # 计时、绑核（Linux/Windows）、线程、JSON 行输出
cpp/topology/                    # Windows 拓扑查询
cpp/prog1_mandelbrot_threads/    # block / interleaved / dynamic
cpp/prog2_vecintrin/             # cs149intrin.h（fake intrinsics + logger）与 clampedExp / arraySum
cpp/prog3_mandelbrot_simd/       # AVX2 / AVX-512 / AVX-512 ×2 gang，ISPC 式 task
cpp/prog4_sqrt/                  # 标量 / AVX2 / AVX-512 / 线程池，五种输入
cpp/prog5_saxpy/                 # scalar / simd / stream / read，常驻自旋线程池
cpp/prog6_kmeans/                # baseline 与 v1–v3
gpu/a1_gpu.cu, gpu/run_gpu.sh    # CUDA：mandel / sqrt / saxpy / kmeans
scripts/build_cpp.py, run_cpp_experiments.py, plot_cpp.py
tests/test_cpp.py
```

---

## 附录：Python/numpy 版（早期实现）

对应 CS149 Fall 2024 Assignment 1「Analyzing Parallel Program Performance on a Quad-Core CPU」。官方 starter 是
C++/ISPC，离线不可得。本目录用 Python 重写五个程序的核心问题：

- prog1 用进程池代替 `std::thread`，因为线程会被 GIL 串行化；
- prog2/3/4 用「gang 宽度 W 的 lane 掩码模型」计算 SIMD 利用率，并用 numpy 的「掩码执行 vs 活跃集压缩」实测发散的代价；
- prog5 用 numpy 和 torch 的 saxpy 测有效带宽。

机器：8 逻辑处理器的 Windows Server（WMI 报告 4 核 × 2 超线程，虚拟机拓扑，物理拓扑不确定），Python 3.12、
numpy 2.5、torch 2.6 CPU。测量期间机器上还有其他任务在跑：4 个 worker 以内的数字稳定，5–8 个 worker 的单点有明显
抖动。墙钟只看相对大小。

### 一句话结论

Mandelbrot 按连续块静态分配时，中间块的计算量是边缘块的 3.7 倍（8 worker 时 0.2333 s 对 0.0627 s），
4 worker 只有 2.93×；改成隔行分配是 3.93×，动态分块是 3.76×。SIMD 的代价是发散：每个 W=8 gang 只要有一个慢
lane，利用率就降到 0.163。numpy 掩码执行（相当于 gang 宽度 = 数组长度）在「每 16 个元素 1 个慢」的输入上和
「全部慢」一样贵（353 对 356 ns/元素），压缩活跃集后降到 29.4 ns。saxpy 在 DRAM 区间单线程 14.3 GB/s，
8 线程只到 47.8 GB/s，约 3 线程后基本饱和；12 MB 的工作集在 8 线程下可达 101 GB/s。

### 对应官方作业

| 官方部分 | 本目录实现 | 因 CPU/Python 而不同之处 |
| --- | --- | --- |
| prog1 mandelbrot_threads | `mandelbrot.py` + `parallel.py`：静态连续块、隔行、动态分块（16 行一块），1–8 个进程；记录每个 worker 的计算时间 | 进程不是线程：要 pickle 结果、要 IPC；内核是 numpy 向量化 + 每次迭代压缩活跃集，不是标量 C++ |
| prog2 vector intrinsics（CS149 fake intrinsics、`_cs149_cntbits`） | `simd.py::gang_utilization`：给定每元素迭代数，按 W 宽 gang 计算利用率 | 是模型，不是指令级模拟；不实现 clamp/exp 那道题 |
| prog3 ISPC（gang、task） | 同上模型用在 Mandelbrot 迭代数上；多核部分见 prog1 | 没有 ISPC 编译器；ISPC task ≈ 本目录的动态分块 |
| prog4 sqrt（Newton 法，输入决定迭代数） | `simd.py::sqrt_iterations` 完全照搬 starter 的迭代与阈值；输入模式 uniform / all_fast / all_slow / one_slow_per_gang | 官方要求构造让 ISPC 加速比最好/最差的输入，这里用模型算加速比，再用 numpy 掩码执行实测 |
| prog5 saxpy（带宽受限） | `saxpy.py`：numpy 两遍、torch 融合单遍（`torch.add(y, x, alpha=a)`），数组 4 KB–768 MB，线程 1–8 | 用 torch 内部线程和 numpy 分块线程代替 ISPC task |

### 怎么跑

```powershell
cd solutions/cs149/asst1-cpu-parallel
python -m pytest -q tests          # 15 passed
python scripts/run_experiments.py  # 约 3 分钟，写 artifacts/raw/*.csv 与 artifacts/figures/*.png
```

### 结果

#### 先修一个测量陷阱：numpy 内核对块大小敏感（`artifacts/raw/mandel_block_size.csv`）

同一幅 view1（1600×1200，256 次迭代）串行计算，只改每次 numpy 调用处理的行数：

| 每次调用行数 | 像素数 | 串行 s |
| ---: | ---: | ---: |
| 2 | 3200 | 2.6249 |
| 10 | 16000 | 1.2881 |
| 40 | 64000 | 1.2014 |
| 160 | 256000 | 1.5155 |
| 1200（整幅） | 1920000 | 3.6508 |

块太小时，每次迭代约 10 个 numpy 调用的固定开销占主导；块太大时，临时数组放不进 cache，每次迭代都要走内存。
第一版实验没控制这一点，结果出现了「view2 用 8 个 worker 加速 21 倍」的超线性假象：分块越细，单像素越便宜。
现在所有调度都按约 16384 像素一块（`BLOCK_POINTS`）切分，只比较行怎么分给 worker。

#### prog1：静态块 vs 隔行 vs 动态（`artifacts/raw/mandel_speedup.csv`，`mandel_work_imbalance.csv`）

加速比相对进程内串行（view1 1.2776 s，view2 2.9102 s），每个配置取 3 次最好。静态与隔行用 barrier 保证每块落在不同进程上。

| 视图 | worker | 静态块 | 隔行 | 动态（16 行） | 静态块：按迭代数算的理想上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| view1 | 2 | 1.999 | 1.976 | 1.893 | 1.995 |
| view1 | 3 | 2.098 | 2.936 | 2.846 | 1.664 |
| view1 | 4 | 2.931 | 3.929 | 3.76 | 2.481 |
| view1 | 8 | 4.967 | 4.081 | 5.072 | 4.14 |
| view2 | 4 | 3.713 | 4.147 | 4.063 | 3.428 |
| view2 | 8 | 4.981 | 5.313 | 6.058 | 6.854 |

- view1 静态块 3 worker 只比 2 worker 快一点（2.098 对 1.999）：中间那块包含整个 Mandelbrot 集主体。
  按迭代数算，最重的块是平均的 1.80 倍。
- 实测静态块比「迭代数理想上限」好（4 worker 2.931 对 2.481）：每个像素还有与迭代数无关的固定开销
  （坐标生成、压缩），让真实工作量比迭代数更均匀。
- view2 本身比较均匀（4 worker 时最重块是平均的 1.167 倍），三种调度差距小。
- 8 worker 隔行在 view1 只有 4.081：8 个隔行块按迭代数几乎完全均衡（最重/平均 = 1.003），但其中一个 worker
  用了 0.2922 s，其他约 0.16 s（`mandel_task_times.csv`）。这是外部负载造成的，不是工作量不均。动态分块对此不敏感，
  这正是动态调度的第二个好处。

![speedup](artifacts/figures/mandel_speedup.png)

![per-worker time](artifacts/figures/mandel_worker_time.png)

![view1](artifacts/figures/mandel_view1_counts.png)

#### prog2/3/4：SIMD 利用率模型（`artifacts/raw/simd_utilization.csv`）

每个 lane 的工作量 = 迭代数 + 1（最后一次判断）。利用率 = 有效 lane-迭代 / Σ(W × gang 内最大值)，模型加速比 = W × 利用率。

| 输入 | W=2 | W=4 | W=8 | W=16 | W=8 模型加速比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| mandel view1（行内相邻像素成 gang） | 0.9897 | 0.9762 | 0.957 | 0.9295 | 7.656 |
| mandel view2 | 0.9945 | 0.9874 | 0.9784 | 0.9672 | 7.827 |
| sqrt uniform（0.001 + 2.998·rand） | 0.8348 | 0.7098 | 0.6122 | 0.5347 | 4.897 |
| sqrt all_fast（全 1.0，0 次迭代） | 1.0 | 1.0 | 1.0 | 1.0 | 8.0 |
| sqrt all_slow（全 2.999，22 次迭代） | 1.0 | 1.0 | 1.0 | 1.0 | 8.0 |
| sqrt one_slow_per_gang（每 W 个 1 个 2.999） | 0.5217 | 0.2826 | 0.163 | 0.1033 | 1.304 |

- Mandelbrot 相邻像素的迭代数高度相关，发散代价很小，W=16 时利用率还有 0.93。
- one_slow_per_gang 是最坏输入：每个 gang 都要跑满 22 次迭代，有效工作却只有 1/W。官方 prog4 问的就是这种输入。
- all_slow 的利用率是 1，但它最慢。利用率高不等于跑得快。

![simd](artifacts/figures/simd_utilization.png)

#### 实测：标量 Python vs numpy 压缩 vs numpy 掩码（`artifacts/raw/simd_timing.csv`，ns/元素）

| 内核 / 输入 | 平均迭代 | Python 标量 | numpy 压缩活跃集 | numpy 掩码（不压缩） |
| --- | ---: | ---: | ---: | ---: |
| sqrt uniform | 5.362 | 929.91 | 247.02 | 448.307 |
| sqrt all_fast | 0.0 | 188.01 | 29.29 | 10.605 |
| sqrt all_slow | 22.0 | 3157.9 | 480.653 | 356.447 |
| sqrt one_slow_per_gang（W=16） | 1.375 | 355.3 | 29.441 | 353.147 |
| mandel view1（16384 像素一块） | 69.434 | 10104.82 | 649.485 | 1532.422 |

- 掩码执行：one_slow_per_gang 与 all_slow 几乎一样贵（353 对 356），平均迭代数却只有 1.375 对 22。这是 numpy 版的发散代价。
- 压缩活跃集（stream compaction）把发散的代价去掉了（29.4），但没有发散时它更贵（all_slow 481 对 356），
  因为每次迭代都要做一次 gather。GPU 上对应的取舍是 warp 发散 vs 重排/压缩。
- Python 标量比 numpy 慢 15.6 倍（mandel）。这主要是解释器开销，**不是** SIMD 宽度带来的加速。

#### prog5：saxpy 带宽（`artifacts/raw/saxpy_bandwidth.csv`，`saxpy_scaling.csv`）

有效带宽 = 3 × 4 B × N / 时间（读 x、读 y、写 out）。写 out 时，CPU 先要把目标 cache line 读进来
（write-allocate），所以真实内存流量是 4 × 4 B × N，CSV 里另有一列 ×4/3 的数字。numpy 没有融合的 axpy，
要分两遍（先乘、再加），实际流量更大，所以它的「有效带宽」更低。

| 工作集（3 个数组） | numpy 两遍 1 线程 | torch 融合 1 线程 | torch 融合 8 线程 |
| ---: | ---: | ---: | ---: |
| 192 KB | 34.375 | 43.424 | 45.271 |
| 768 KB | 48.496 | 71.442 | 28.949 |
| 12 MB | 23.106 | 37.203 | 101.068 |
| 192 MB | 9.897 | 14.829 | 51.429 |
| 768 MB | 10.286 | 14.754 | 51.193 |

N=2^25（384 MB，DRAM 区间）按线程数扩展（GB/s）：

| 线程 | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| torch 融合 | 14.335 | 27.162 | 39.455 | 39.186 | 41.327 | 47.791 |
| numpy 分块线程 | 9.937 | 18.713 | 25.326 | 31.848 | 40.085 | 41.199 |

对比 12 MB 工作集：torch 从 1 线程 33.949 升到 6 线程 159.868 GB/s。

- 本机单核吃不满 DRAM：torch 1 线程 14.3 GB/s，3 线程起基本停在 39–48 GB/s（5 线程那一点 27.6 是抖动）。这台服务器上「多线程对 DRAM 带宽
  完全无效」不成立，成立的是「很快饱和，而且饱和值远低于 cache 区间」。CS149 原作业的四核桌面机上，单核更接近饱和。
- 768 KB 时 8 线程（28.9）反而比 1 线程（71.4）慢：每次调用只有约 11 µs，线程唤醒和同步开销占了主导。
- saxpy 的算术强度是 2 FLOP / 12 B ≈ 0.17 FLOP/B，任何核数都是带宽受限。

![saxpy bandwidth](artifacts/figures/saxpy_bandwidth.png)

![saxpy scaling](artifacts/figures/saxpy_scaling.png)

### 面试讲法

**20 秒结论**：并行加速有三种杀手。第一是负载不均：Mandelbrot 中间贵，连续块静态分配时 4 核只有 2.9×，
隔行或动态分配能到 3.8–3.9×。第二是 SIMD 发散：一个 gang 的耗时由最慢的 lane 决定，每 8 个元素混 1 个慢的，
8 宽 SIMD 只剩 1.3×。第三是带宽：saxpy 每 12 字节只做 2 次浮点运算，加核很快撞上 DRAM 墙。

**白板**：
- 静态分配加速比上限 = 总工作 / 最大块工作；隔行把空间相关的代价打散。
- SIMD 利用率 = Σ work_i / Σ_gang (W · max work)；最坏情况是每个 gang 恰好 1 个慢 lane，利用率 ≈ (slow + (W−1)·fast) / (W·slow)。
- saxpy 流量：程序可见 12N 字节，加上 write-allocate 是 16N 字节；用 non-temporal / streaming store 可以省掉 write-allocate 那次读。
- 算术强度 2/12 FLOP/B；roofline 上的拐点 = 峰值 FLOPs / 峰值带宽，saxpy 远在拐点左边。

**不能推出的结论**：
- 不能推出「隔行在 8 worker 时比静态差」。view1 8 worker 隔行 4.081 是外部负载导致某个 worker 变慢（0.2922 s 对约 0.16 s），按迭代数它几乎完全均衡。
- 不能用 Python 标量对 numpy 的 15.6× 去估计 SIMD 加速。那是解释器开销。
- 不能把本机 DRAM 带宽的数字搬到别的机器，也不能据此判断这台虚拟机的物理拓扑。

**追问**：
1. 超线程对 Mandelbrot 有用吗？官方答案：超线程隐藏延迟，对计算密集、ILP 已经很高的标量循环帮助有限。
   本机 8 worker 动态分块为 5.072×/6.058×，高于 4，但测量期间机器不独占，又是虚拟机，分不清物理核和超线程。
2. 动态分块的块大小怎么选？块小，负载均衡好，但每块有调度/IPC 开销；块大则反之。
   这里 16 行 = 25600 像素，同时也照顾了 numpy 临时数组的 cache 大小（见块大小表）。
3. 为什么 saxpy 的 numpy 版比 torch 慢？numpy 两遍：`out = a*x` 一遍写，`out += y` 再读写一遍，流量约 20N 字节；
   torch 一遍融合。这就是算子融合省带宽的最小例子。

### 文件

```text
asst1-cpu-parallel/
├── code/cpupar/
│   ├── mandelbrot.py   # 标量/向量/掩码内核，静态块、隔行、动态分块，工作量统计
│   ├── parallel.py     # 进程池 + barrier（静态块一进程一块）+ 每任务计时
│   ├── simd.py         # gang 利用率模型、prog4 Newton sqrt、输入模式
│   └── saxpy.py        # numpy / torch saxpy、有效带宽、计时
├── tests/test_asst1.py # 15 个测试
├── scripts/run_experiments.py
└── artifacts/
    ├── raw/       # mandel_block_size / mandel_speedup / mandel_task_times / mandel_work_imbalance /
    │              # simd_utilization / simd_timing / saxpy_bandwidth / saxpy_scaling .csv
    └── figures/   # mandel_speedup / mandel_worker_time / mandel_view1_counts /
                   # simd_utilization / saxpy_bandwidth / saxpy_scaling .png
```
