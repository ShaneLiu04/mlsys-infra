# Module 01 — 为什么并行：性能模型与 roofline

- 课程出处：CS149 Fall 2024《Why Parallelism》 · CS229S Fall 2023《Hardware Aware Algorithm Design (10-04)》 · CMU 15-849《Intro to DL Systems》
- 必读论文：Williams, Waterman, Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, 2009；Amdahl, Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities, 1967；Gustafson, Reevaluating Amdahl's Law, 1988
- 对应实验：无（估算工具：[tools/mlsys_calc.py](../tools/mlsys_calc.py)）
- 状态：初稿

## 摘要

单核频率在功耗墙前停住了，性能只能靠并行和专用硬件拿。并行带来两个上限：串行部分（Amdahl）
和关键路径（span）。单个 kernel 的上限由 roofline 决定：算术强度低于拐点就受带宽限制，
高于拐点才受算力限制。面试里几乎所有「为什么慢」都能落到三类：compute-bound、memory-bound、
latency/overhead-bound。本讲给出判断它们的公式和读 profile 的顺序。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 加速比 `S = T1/TP`，效率 `E = S/P`。超线性加速通常来自 cache 容量变大，不是算法变快。
2. Amdahl：串行比例 `s` 决定上限 `1/s`。`s = 5%` 时无论多少核都不超过 20 倍。
3. Work/span：`TP ≥ max(T1/P, T∞)`，贪心调度达到 `TP ≤ T1/P + T∞`。并行度 `T1/T∞` 是可用核数的上限。
4. Roofline：`可达 FLOP/s = min(峰值算力, I × 带宽)`。H100 拐点 295 FLOP/B，A100 153 FLOP/B。
5. 数据搬运比计算贵。batch=1 的 LLM decode 算术强度约 1 FLOP/B，离拐点差两个数量级。

**必须能默写的公式**

```text
S = T1 / TP                     加速比；E = S / P 为并行效率
Amdahl:     S(P) = 1 / ( s + (1 − s)/P ),   S(∞) = 1/s        s = 串行比例（固定问题规模）
Gustafson:  S(P) = s + (1 − s)·P                               s = 并行机上测得的串行时间比例（问题随 P 放大）
Work/span:  max(T1/P, T∞) ≤ TP ≤ T1/P + T∞                     T1 = 总工作量，T∞ = 关键路径
Roofline:   F_achieved = min(π, I·β),   I = FLOPs / DRAM bytes,   ridge = π / β
Time:       T ≈ max( FLOPs/π , Bytes/β ) + T_overhead
Little:     bytes_in_flight = β × latency
Power:      P_dyn ≈ α·C·V²·f,  V ∝ f  ⇒  P_dyn ∝ f³
```

**一句话区分**

- latency vs throughput：一件事做完要多久 vs 单位时间做完多少件；GPU 牺牲前者换后者。
- Amdahl vs Gustafson：固定问题看能快多少 vs 固定时间看能做多大。
- compute-bound vs memory-bound：`I > ridge` vs `I < ridge`；前者靠 Tensor Core，后者靠少搬字节。
- memory-bound vs overhead-bound：带宽打满了 vs 算力和带宽都没打满，时间耗在 launch、同步、CPU 上。

## 学习目标

- 能从定义推出 Amdahl 与 Gustafson，并说清两者的假设差异；
- 能写出 work/span 的上下界，并解释贪心调度为何能达到上界；
- 能对任意算子算出 FLOPs、字节数和算术强度，在 A100/H100 上判断瓶颈；
- 能用功耗公式解释为什么多核取代了高主频；
- 能按「算力利用率 → 带宽利用率 → 空闲时间」三步读一份 profile。

## 先修知识

- 基本体系结构：cache、DRAM、时钟周期；bf16 每元素 2 字节，fp32 每元素 4 字节；
- 矩阵乘的 FLOPs：`M×K` 乘 `K×N` 需要 `2MNK` 次浮点运算（乘加各算一次）。

## 核心概念

### 1. latency 与 throughput

latency 是单个任务从开始到结束的时间，throughput 是单位时间完成的任务数。四台洗衣机并行，吞吐翻 4 倍，
但每一筐仍要洗 45 分钟。并行只改善 throughput，不缩短单个任务的 latency，除非任务本身能拆开。
LLM 服务里，TTFT 和每 token 延迟是 latency，tokens/s/GPU 是 throughput，batch 越大前者越差后者越好。

### 2. 加速比与效率

`S = T1/TP`。`T1` 必须是最好的串行实现，不是并行程序在 1 核上跑的时间。效率 `E = S/P`：8 卡 6 倍加速，效率 75%。
超线性加速（`S > P`）是真实现象：切分后每核数据装进了 cache。它不违反任何定律，只说明串行基线受访存拖累。

### 3. Amdahl 与 Gustafson

Amdahl 固定问题规模。程序中比例 `s` 的部分只能串行，剩下 `1−s` 被 P 核均分。
上限 `1/s` 与 P 无关：`s = 10%` 时 64 核只有 8.77 倍，无穷核也只有 10 倍。
Gustafson 固定运行时间，让问题规模随 P 增长；串行部分不随规模增长时，加速比近似线性。
ML 训练更像 Gustafson：卡数加倍时 global batch 也加倍。但通信和优化器步骤会以新的「串行比例」出现。

### 4. Work 与 span

把计算画成 DAG。`T1` 是所有节点工作量之和（work），`T∞` 是最长路径（span，关键路径）。
两个下界显然：P 个核每步最多做 P 份工作，所以 `TP ≥ T1/P`；关键路径上的节点必须串行，所以 `TP ≥ T∞`。
并行度 `T1/T∞` 表示「最多有多少核能被有效利用」。n 个数的树形求和：`T1 = n−1`，`T∞ = log2 n`。
`n = 2^20` 时并行度约 5.2 万，远大于任何 CPU 的核数。

### 5. Roofline 与算术强度

算术强度 `I = FLOPs / 从 DRAM（或 HBM）搬运的字节数`。注意分母是实际与慢存储交换的字节，不是算法读的总次数。
可达算力 `min(π, I·β)` 是一条折线：左边斜线是带宽屋顶，右边水平线是算力屋顶，交点就是 ridge `π/β`。
H100 bf16：`989e12 / 3.35e12 ≈ 295 FLOP/B`。A100：`312e12 / 2.04e12 ≈ 153 FLOP/B`。
同一个 kernel 换卡可能换瓶颈：`I = 200` 在 A100 上 compute-bound，在 H100 上 memory-bound。
多级 roofline：L2、shared memory 各有自己的带宽屋顶，分块（M07）就是把数据留在高带宽层，提高对 HBM 的 `I`。

### 6. 功耗墙与每次操作的能耗

动态功耗 `α·C·V²·f`，电压大致随频率线性，所以功耗随频率三次方增长：频率翻倍功耗约 8 倍，
同频率放两个核功耗 2 倍、可并行时吞吐也 2 倍。这是 2000 年代中期之后转向多核、SIMD 和加速器的物理原因。
另一面是能耗：从 DRAM 取一个字比做一次浮点运算贵数量级，Tensor Core、systolic array、算子融合都在提高每字节的运算次数。

### 7. 三类瓶颈与读 profile

- compute-bound：achieved FLOP/s 接近峰值。办法：低精度（bf16→fp8）、用 Tensor Core、减少冗余计算。
- memory-bound：achieved 带宽接近 HBM 峰值，FLOP/s 远低于峰值。办法：融合、量化权重、提高复用（更大 batch、分块）。
- latency/overhead-bound：两者都远未打满。时间耗在 kernel launch、Python 调度、同步、小 kernel、CPU 数据加载。
  办法：CUDA Graphs、`torch.compile`、合并小 kernel、增大每次 launch 的工作量。
读 profile 的顺序：先看 GPU timeline 有没有空洞（overhead），再看单 kernel 的 FLOP/s 和带宽各占峰值多少。

## 关键推导

### 推导 1：Amdahl

设串行时间 `T1 = 1`，其中 `s` 只能串行。P 核时 `TP = s + (1−s)/P`。

```text
S(P) = T1 / TP = 1 / ( s + (1−s)/P )
P → ∞:  S → 1/s
半效率点: S = P/2  ⇔  s + (1−s)/P = 2/P  ⇔  P = (1+s)/s ≈ 1/s
例: s = 0.05, P = 8  → 1/(0.05 + 0.11875) = 5.93
    s = 0.01, P = 1024 → 1/(0.01 + 0.000967) = 91.2
```

边界条件：假设并行部分完美均分、没有通信开销。真实系统的通信随 P 增长，会让曲线在某个 P 之后下降。

### 推导 2：Gustafson

在 P 核机器上测得运行时间为 1，其中串行 `s`、并行 `1−s`。同样的工作放到单核要 `s + (1−s)·P`。

```text
S(P) = s + (1−s)·P = P − s·(P − 1)
例: s = 0.05, P = 8 → 7.65
```

两者不矛盾：Amdahl 的 `s` 是相对单核时间，Gustafson 的 `s` 是相对并行机时间，问题规模也不同。

### 推导 3：矩阵乘与 elementwise 的算术强度

```text
GEMM  M=N=K=n, bf16, 理想复用（每个矩阵只读写一次）:
  FLOPs = 2n³,  Bytes = 2·(n² + n² + n²) = 6n²,  I = n/3
  n = 256 → 85；n = 512 → 171；n = 4096 → 1365  (FLOP/B)
Elementwise  z = x + y, bf16:
  FLOPs = 1/元素,  Bytes = 2+2+2 = 6/元素,  I = 1/6 ≈ 0.17
GEMV  y = W·x, W 为 d×d bf16:
  FLOPs = 2d²,  Bytes ≈ 2d²,  I ≈ 1
```

GEMM 的 `I` 随 n 线性增长，所以大矩阵乘一定 compute-bound；elementwise 和 GEMV 的 `I` 是常数，永远 memory-bound。

## 深入推导

### 推导 4：贪心调度的上界 `TP ≤ T1/P + T∞`（Brent 定理）

贪心调度：任何时刻只要有就绪任务且有空闲核，就立刻分配。把每个时间步分为两类：

- 完整步：P 个核都在干活。这类步最多 `T1/P` 个，因为总工作只有 `T1`。
- 不完整步：就绪任务少于 P，所有就绪任务都被执行。DAG 中所有当前「最深未完成」节点都就绪，
  所以每个不完整步都让剩余关键路径长度减 1。这类步最多 `T∞` 个。

```text
TP ≤ (#完整步) + (#不完整步) ≤ T1/P + T∞
结合下界: TP ≥ max(T1/P, T∞) ≥ (T1/P + T∞)/2
⇒ 贪心调度最多比最优差 2 倍；当 T1/T∞ ≫ P 时 TP ≈ T1/P，近似线性加速
```

工程含义：只要并行度远大于核数，调度器「别让核闲着」就够好。work stealing（M03）在期望意义上达到同一界。

### 推导 5：decode 的 batch 拐点

LLM decode 每步对每个权重做一次乘加。batch 中有 B 条序列，忽略 KV cache 和激活：

```text
FLOPs ≈ 2 · B · N_params
Bytes ≈ 2 · N_params           (bf16 权重只读一遍，被 B 条序列共享)
I ≈ B   FLOP/B
compute-bound 条件: B ≳ ridge  → H100 约 295，A100 约 153
```

加上 KV cache 后，每条序列还要读自己的 KV，字节数随 `B × seq_len` 增长，`I` 被拉低，长上下文 decode 很难达到拐点。
这是 M15、M16 里 GQA、KV 量化、PagedAttention 存在的根本原因。

### 推导 6：Little's law 与「为什么要很多线程」

带宽 β、延迟 L，要打满带宽必须同时有 `β·L` 字节在路上。CPU 一个 DDR5 通道约 45 GB/s、DRAM 延迟约 100 ns：

```text
bytes_in_flight = 45e9 B/s × 100e-9 s = 4500 B ≈ 70 条 64 B cache line
```

单个线程顺序依赖的访存一次只挂一个 miss，远不够。所以要用多核、硬件多线程、预取或 GPU 上的大量 warp 来凑并发访存（M02、M05）。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的数字。

```text
1) 4096³ bf16 GEMM 在 H100 上:
   FLOPs = 2·4096³ = 1.374e11;  compute 时间 = 1.374e11 / 989e12 = 0.139 ms
   Bytes = 6·4096² = 1.007e8;   memory 时间 = 1.007e8 / 3.35e12 = 0.030 ms
   I = 1365 > 295 → compute-bound，下限约 0.14 ms

2) 1e8 个 bf16 元素相加（H100）:
   Bytes = 6e8 → 6e8 / 3.35e12 = 179 µs
   FLOPs = 1e8 → 1e8 / 989e12 = 0.1 µs
   可达 FLOP/s = 0.17 × 3.35e12 ≈ 0.56 TFLOP/s，只有峰值的 0.06%

3) Llama-3-8B batch=1 decode 上限:
   权重 = 8.03e9 × 2 B = 16.06 GB
   H100: 16.06e9 / 3.35e12 = 4.79 ms/token → ≈ 209 token/s
   A100: 16.06e9 / 2.04e12 = 7.87 ms/token → ≈ 127 token/s

4) 同一个 I = 200 的 kernel:
   A100: 200 > 153 → compute-bound，上限 312 TFLOP/s
   H100: 200 < 295 → memory-bound，上限 200 × 3.35e12 = 670 TFLOP/s（峰值的 68%）

5) 训练 8B 模型 15T token，1024 张 H100，MFU 40%:
   FLOPs = 6ND = 6 × 8e9 × 15e12 = 7.2e23
   有效算力 = 1024 × 989e12 × 0.4 = 4.05e17 FLOP/s
   时间 = 7.2e23 / 4.05e17 = 1.78e6 s ≈ 20.6 天

6) Amdahl 套到训练步: 一步 100 ms，其中 10 ms 是单卡 CPU 数据预处理（不随卡数缩短）
   s = 0.1 → 无论加多少卡，单步最多快 10 倍；先把数据加载异步化，再谈扩卡
```

计算器的对应命令：

```bash
python tools/mlsys_calc.py hw
python tools/mlsys_calc.py decode --model llama3-8b --gpu h100 --seq 4096 --batch 1
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 拐点表 | [tools/mlsys_calc.py](../tools/mlsys_calc.py) 的 `hw` 子命令 | 输出 H100 295、A100 153，与速查表一致 |
| decode 上限 | 同一工具的 `decode` 子命令 | batch=1 时 token/s 与「权重字节 / 带宽」手算一致 |
| roofline 判断 | 下方 Python 片段 | 对 GEMM、elementwise、GEMV 给出正确的 bound 类型 |

```python
def roofline(flops, bytes_moved, peak_flops, bw):
    intensity = flops / bytes_moved
    t_compute, t_memory = flops / peak_flops, bytes_moved / bw
    bound = "compute" if t_compute >= t_memory else "memory"
    return intensity, max(t_compute, t_memory), bound

H100 = dict(peak_flops=989e12, bw=3.35e12)
n = 4096
print(roofline(2 * n**3, 6 * n**2, **H100))   # (1365.3, 1.39e-4, 'compute')
print(roofline(1e8, 6e8, **H100))             # (0.167, 1.79e-4, 'memory')
```

## 易错点与反例

- 「加 8 倍核就快 8 倍」：错。串行 5% 时 8 核只有 5.93 倍。
- 「用并行程序在 1 核上的时间当 T1」：加速比被夸大，因为并行版本自带同步和额外工作。
- 「算术强度按算法读了几次数据来算」：错。分母是和 HBM 实际交换的字节。分块后同一个 GEMM 的 `I` 会变。
- 「GPU 利用率 100% 说明算力打满」：`nvidia-smi` 的 utilization 只表示采样窗口内有 kernel 在跑，
  一个 memory-bound kernel 也会显示 100%。要看 MFU 或 achieved FLOP/s。
- 「memory-bound 就该换更大算力的卡」：H100 相对 A100 算力涨约 3.2 倍、带宽只涨约 1.6 倍，memory-bound kernel 只快约 1.6 倍。
- 「小 batch 推理 GPU 空转是因为算力不够」：反了，是带宽和 launch 开销限制，算力大量闲置。

## 实验或练习

- 假设：elementwise kernel 的耗时与元素数线性相关，且换算出的带宽接近 HBM 峰值；小尺寸时耗时不再下降。
- 配置：PyTorch 中对 `2^10` 到 `2^28` 个 bf16 元素做 `x + y`，用 CUDA event 计时，换算 `6N / t` 的带宽；
  再对 `n = 256…8192` 的方阵做 `torch.matmul`，换算 `2n³ / t` 的 TFLOP/s。
- 预期：大尺寸 elementwise 的带宽接近 HBM 峰值、TFLOP/s 很低；小尺寸耗时趋于常数（overhead-bound）。
  matmul 的 TFLOP/s 随 n 上升，在 n 足够大后接近 Tensor Core 峰值。把两组点画在同一张 roofline 图上。

## 高频问题

- 问：Amdahl 定律说了什么？
  答：串行比例 `s` 把加速比上限钉在 `1/s`，和核数无关。
- 问：什么是算术强度？
  答：FLOPs 除以与慢存储交换的字节数，单位 FLOP/B。
- 问：为什么 LLM decode 是 memory-bound？
  答：每读 2 字节权重只做 2 次运算（batch=1），`I ≈ 1`，远低于 295。
- 问：怎么判断一个程序是 overhead-bound？
  答：timeline 上 kernel 之间有大量空隙，achieved FLOP/s 和带宽都远低于峰值，增大问题规模耗时几乎不变。

## 面试追问链

1. 问：latency 和 throughput 有什么区别，GPU 偏向哪个？
   答：latency 是单任务耗时，throughput 是单位时间完成量。GPU 用大量线程和简单核心换吞吐，单线程延迟比 CPU 差。
2. 问：推一下 Amdahl。串行 5%，最多能快多少？
   答：`S = 1/(s + (1−s)/P)`，`P→∞` 得 `1/s = 20`。8 核时 5.93。
3. 问：那为什么超算还能用上万核？
   答：Gustafson：问题规模随核数放大，串行部分占比下降，`S = s + (1−s)P`。训练里对应 global batch 随卡数放大。
4. 问：写出 roofline，并判断 4096 方阵 bf16 GEMM 在 H100 上的瓶颈。
   答：`min(π, Iβ)`。`I = n/3 = 1365 > 295`，compute-bound，下限约 0.14 ms。
5. 问：同一个 kernel 在 A100 上 compute-bound，到 H100 上可能变成什么？
   答：H100 拐点更高（295 vs 153），`I` 在两者之间的 kernel 会变成 memory-bound，加速只有带宽比约 1.6 倍。
6. 问：batch=1 的 8B 模型 decode 在 H100 上最快多少 token/s？要 compute-bound 需要多大 batch？
   答：16 GB / 3.35 TB/s ≈ 4.8 ms，约 209 token/s。`I ≈ B`，需要 B 接近 295；算上 KV cache 读取还要更大。
7. 问：训练一步 profile 显示 MFU 只有 15%，你怎么排查？
   答：先看 timeline 空洞：数据加载、Python 开销、同步等待、通信未重叠。再看最耗时 kernel 的 FLOP/s 与带宽占比，
   区分是 attention/norm 等 memory-bound 算子拖累，还是 GEMM 形状太小没打满 Tensor Core。
8. 问：为什么行业从提主频转向多核和加速器？
   答：`P ∝ f³`，主频翻倍功耗约 8 倍；同功耗下多个低频核或专用单元给出更高吞吐，且数据搬运的能耗主导，专用硬件可以减少搬运。

## 自测题

1. 一个程序 20% 串行。求 4、16、∞ 核时的加速比。
   要点：`1/(0.2+0.8/4) = 2.5`；`1/(0.2+0.05) = 4`；`∞` 时 5。
2. 证明贪心调度满足 `TP ≤ T1/P + T∞`。
   要点：完整步不超过 `T1/P`；每个不完整步让剩余关键路径减 1。
3. `n = 512` 的 bf16 方阵乘法在 A100 和 H100 上分别是什么瓶颈？
   要点：`I = 171`；A100 上 `171 > 153` compute-bound；H100 上 `171 < 295` memory-bound。
4. 设计题：给你一张 H100，要让 8B 模型 decode 吞吐最大，batch 应该往哪个方向调，代价是什么？
   要点：增大 batch 提高 `I`，吞吐上升；代价是每 token 延迟变长、KV cache 显存线性增长。

## 关联笔记

- [M02](module-02-multicore-simd-threads.md)：用多核、SIMD、硬件多线程把峰值和带宽真正吃满。
- [M03](module-03-parallel-programming-scheduling.md)：work/span 与 work stealing 的调度界。
- [M05](module-05-gpu-architecture-cuda.md)：GPU 上的 roofline、warp 与延迟隐藏。
- [M15](module-15-inference-kv-speculative.md)：decode 的 memory-bound 分析与 batch 拐点。

## 本讲小结

并行的收益被串行部分和关键路径限制，kernel 的速度被 roofline 限制。判断瓶颈只需要三个数：FLOPs、字节数、
硬件拐点。H100 拐点约 295 FLOP/B，大 GEMM 远高于它，elementwise 和 batch=1 decode 远低于它。
读 profile 先找空洞，再看单 kernel 的算力和带宽占比，最后才谈换算法。

## 延伸阅读

- Williams, Waterman, Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, CACM 2009.
- Amdahl, Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities, AFIPS 1967.
- Gustafson, Reevaluating Amdahl's Law, CACM 1988.
- Horace He, Making Deep Learning Go Brrrr From First Principles, 2022, https://horace.io/brrr_intro.html
