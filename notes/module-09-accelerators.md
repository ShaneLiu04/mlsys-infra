# Module 09 — 专用硬件与加速器编程

- 课程出处：CS149 Fall 2024《Hardware Specialization》《Programming Specialized Hardware I/II》 · CMU 15-849《ML Hardware》（TPU）
- 必读论文：Jouppi et al.，In-Datacenter Performance Analysis of a Tensor Processing Unit，2017；Kung，Why Systolic Architectures?，1982；Chen et al.，Eyeriss，2016；Ragan-Kelley et al.，Halide，2013；Koeplinger et al.，Spatial，2018
- 对应实验：[CS149 A4 DNN 加速器](../solutions/cs149/asst4-dnn-accelerator/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst4/cs149_asst4_report.pdf)（实测：WMMA fp16 177.8 TFLOP/s（cuBLAS 的 86%）；分块流量模型与实测带宽差 2.3%）
- 状态：初稿

## 摘要

通用处理器把大部分能耗花在取指、译码、调度和搬数据上，真正做乘加的比例很小；而从 DRAM 取一个字的能耗比一次乘法高约两个数量级。
专用加速器的思路因此只有两条：把控制开销摊到大量运算上（宽 SIMD、矩阵单元），把数据留在离运算最近的地方反复使用（systolic array、
软件管理的 SRAM）。TPU v1 用 256×256 的 weight-stationary systolic array 做 int8 推理；AWS Trainium 用 128×128 的 Tensor Engine、
SBUF/PSUM 两级片上存储和显式 DMA，由程序员用 NKI 管理 tile。主线取舍：越专用越省能、越快，但对矩阵形状、batch 和编程者的要求越苛刻。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 能耗排序：DRAM 访问 ≫ 片上 SRAM 访问 ≫ 一次乘加；低精度乘法比 fp32 便宜得多。加速器的本质是提高每次取数的复用次数。
2. Systolic array 中数据在相邻 PE 间逐拍传递，一次从存储读入的值被一整行/列 PE 复用；N×N 阵列每个输入复用 N 次。
3. Weight-stationary：权重常驻 PE，激活横向流、部分和纵向流（TPU）；output-stationary：每个 PE 累加一个输出，A、B 都流动。
4. 阵列有填充/排空开销，流式维度长度 L 时利用率约 `L/(L+2N)`；小 batch 推理在大阵列上利用率很低。
5. Trainium/NKI：片上 SBUF 按 128 个 partition 组织，Tensor Engine 沿 partition 维做收缩，结果写到 PSUM 累加；HBM↔SBUF 全靠显式 DMA。

**必须能默写的公式**

```text
峰值 = PE 数 × 2 × 频率              TPU v1：65536 × 2 × 700 MHz ≈ 92 TOPS（int8）
TPU v1 拐点 = 92e12 / 34e9 ≈ 2700 ops/B ≈ 1350 MAC/权重字节
int8 MLP 强度（batch B）= 2B ops / 权重字节   → B ≥ 1350 才计算受限
N×N 阵列、K 维 K：output-stationary 周期 ≈ K + 2N − 2，利用率 ≈ K/(K + 2N)
片上复用：N×N 阵列中每个输入元素被 N 个 PE 使用
NKI 约束：partition 维 ≤ 128；nc_matmul 输出 tile ≤ 128 × 512（fp32 PSUM）
```

**一句话区分**

- GPU vs systolic ASIC：GPU 用大量线程和寄存器文件做通用调度，Tensor Core 是嵌在 SM 里的小矩阵单元；ASIC 把整块芯片面积给一个大阵列和 SRAM。
- 硬件 cache vs 软件 scratchpad：cache 自动替换，命中不可预测；scratchpad 由编译器/程序员显式搬运，可预测但编程难。
- Halide 算法 vs 调度：算法说「算什么」，调度说「按什么顺序、在哪里、用几个核算」，改调度不改结果。

## 学习目标

- 用能耗和控制开销解释为什么 ASIC 比 GPU、GPU 比 CPU 能效更高；
- 画出 weight-stationary 与 output-stationary systolic array 的数据流，推出周期数与利用率；
- 用 TPU v1 的参数画 roofline，解释为什么多数线上推理负载受访存限制；
- 说清 Trainium 的 SBUF/PSUM/partition 约束，写出 NKI 分块矩阵乘；
- 用 Halide 的 algorithm/schedule 分离解释「编程专用硬件」为什么需要 DSL。

## 先修知识

- [M07](module-07-matmul-hardware-aware.md) 的分块矩阵乘与强度；
- [M02](module-02-multicore-simd-threads.md) 的 SIMD 与多线程，作为通用处理器的对照。

## 核心概念

### 1. 为什么要专用化

一条 CPU 指令的大部分能耗在取指、译码、乱序调度、寄存器重命名上。SIMD 把这笔开销摊到 8–16 个 lane，GPU 摊到 32 个线程的 warp，
Tensor Core 摊到一整个小矩阵乘，systolic array 摊到上万个 PE。另一条线是数据：Horowitz（ISSCC 2014，45 nm）的数据给出，
读一次 DRAM 的能耗比一次 32-bit 浮点乘法高两个数量级以上，8-bit 整数乘法又比 fp32 乘法便宜一个数量级以上。
结论是：能效取决于「每从 DRAM 取一个字能做多少次运算」，这与 roofline 的算术强度是同一个量。

### 2. Systolic array

Kung 1982 的思路：数据像心跳一样在规则排列的 PE 之间逐拍流动，每个 PE 只和邻居通信，没有全局总线。
N×N 阵列算矩阵乘时，A 的一个元素从左边进来，向右经过 N 个 PE，被用 N 次；B 的元素向下经过 N 个 PE。
只有阵列边界需要读存储，内部 `N²` 个 PE 都只读邻居寄存器，片上带宽需求从 `O(N²)` 降到 `O(N)`。

### 3. 三种数据流

| 数据流 | 常驻在 PE 的量 | 流动的量 | 代表 |
| --- | --- | --- | --- |
| weight-stationary | 权重 | 激活横向、部分和纵向 | TPU v1 |
| output-stationary | 输出累加器 | A 横向、B 纵向 | 许多学术设计 |
| row-stationary | 卷积的一行权重和输入 | 部分和在 PE 间累加 | Eyeriss |

weight-stationary 适合权重复用大的推理（同一权重乘 batch 内每个样本）；output-stationary 让部分和不出 PE，适合 K 维长的场景。

### 4. TPU v1

TPU v1 是推理芯片：256×256 = 65536 个 8-bit MAC，700 MHz，峰值约 92 TOPS；片上 28 MiB 软件管理存储（24 MiB Unified Buffer + 4 MiB 累加器），
权重来自 8 GiB DDR3，带宽 34 GB/s。指令是 CISC 风格的少量大指令（读权重、矩阵乘、激活、写回），由主机经 PCIe 下发。
论文的核心观察：线上负载以 MLP 和 LSTM 为主，它们的权重复用由 batch 决定，而 p99 延迟约束（MLP0 为 7 ms）限制了 batch，
于是大部分应用落在 roofline 的斜线段，即受权重带宽限制；只有 CNN 接近计算屋顶。后续 TPU v2 起换成 HBM 并加入训练所需的浮点。

### 5. 软件管理的片上存储

GPU 的共享内存已经是 scratchpad，但还有 L1/L2 兜底。TPU 和 Trainium 的片上 SRAM 完全由软件管理：编译器或程序员决定何时把哪个 tile
从 HBM 搬进来、何时写回。好处是延迟确定、没有 tag 开销、可以做精确的双缓冲；代价是所有复用都要显式写出来，写错就是 DRAM 带宽墙。

### 6. AWS Trainium 与 NKI

Trn1 的 NeuronCore-v2 包含 Tensor Engine（128×128 systolic array）、Vector Engine、Scalar Engine、GpSimd Engine。
片上存储分两级：SBUF（24 MiB，128 个 partition，每 partition 192 KiB）存放输入输出 tile；PSUM（2 MiB，128 partition × 16 KiB）
是矩阵乘的累加缓冲。NKI（Neuron Kernel Interface）是 Python 嵌入的 kernel 语言：

- tile 的第 0 维是 partition 维，最大 128，对应阵列的一边；
- `nc_matmul` 沿 partition 维收缩：stationary 操作数 `[K≤128, M≤128]`，moving 操作数 `[K≤128, N≤512]`，结果 `[M, N]` 写入 PSUM；
- `nl.load/nl.store` 是 HBM↔SBUF 的 DMA，矩阵乘的左操作数要预先转置成「K 在 partition 维」；
- 不同 engine 异步并行，编译器根据依赖插入同步，程序员靠循环结构暴露并行（`affine_range` 表示迭代间无依赖）。

CS149 Fall 2024 的 A4 就是在这套硬件上把 DNN 层（矩阵乘、卷积）手工分块、安排 DMA，体会 tile 形状对利用率的影响。

### 7. Halide 与硬件 DSL

Halide 把图像/张量程序拆成 algorithm（纯函数定义）和 schedule（`split/reorder/tile/vectorize/parallel/compute_at`）。
同一算法换调度可以在局部性、并行度和重计算之间移动，不改变结果。Spatial（Stanford）更进一步面向 FPGA/CGRA：
显式声明 DRAM、SRAM、寄存器、FIFO，控制结构（`Pipe`、`Sequential`、`Stream`）直接映射到硬件流水。
NKI 处于两者之间：它暴露硬件存储层次，但仍是命令式循环。

## 关键推导

### 推导 1：systolic array 的周期与利用率

output-stationary，N×N 阵列，计算 `C[N,N] = A[N,K]·B[K,N]`。A 的第 i 行延迟 i 拍注入，B 的第 j 列延迟 j 拍注入（skew），
使 `A[i,k]` 与 `B[k,j]` 在第 `t = i + j + k` 拍于 PE(i,j) 相遇：

```text
最后一次相遇：t = (N−1) + (N−1) + (K−1)  →  总周期 = K + 2N − 2
有效 MAC = N²·K，可用 MAC = N²·(K + 2N − 2)
利用率 = K / (K + 2N − 2) ≈ K/(K + 2N)
N=128：K=128 → 33%；K=2048 → 89%；K=8 → 3%
```

weight-stationary 同理，流式维度换成激活的行数（batch×序列），再加上把权重装进阵列的 N 拍。所以小 batch decode 在大阵列上很亏，
解决办法是连续流过多个 tile 让填充/排空重叠（流水），或者用更小的阵列。

### 推导 2：TPU v1 roofline

```text
峰值 P = 65536 × 2 × 700e6 = 9.18e13 ops/s
权重带宽 β = 34e9 B/s
拐点 = P/β ≈ 2700 ops/B（论文以 MAC/权重字节计，约 1350）
int8 MLP，batch B：每个权重字节参与 B 次 MAC = 2B ops → 强度 2B ops/B
B < 1350 → 受权重带宽限制，可达算力 = 2B × 34e9
```

## 深入推导

### 代码：output-stationary 阵列的逐拍模拟

```python
import numpy as np

def os_systolic(A, B):
    """PE(i,j) 累加 C[i,j]；A 向右流，B 向下流；返回结果与周期数。"""
    n, K = A.shape; _, m = B.shape
    C = np.zeros((n, m)); a = np.zeros((n, m)); b = np.zeros((n, m))
    cycles = K + n + m - 2
    for t in range(cycles):
        a = np.roll(a, 1, axis=1); b = np.roll(b, 1, axis=0)     # 每拍向邻居传一格
        for i in range(n):
            k = t - i; a[i, 0] = A[i, k] if 0 <= k < K else 0.0  # 第 i 行延迟 i 拍
        for j in range(m):
            k = t - j; b[0, j] = B[k, j] if 0 <= k < K else 0.0  # 第 j 列延迟 j 拍
        C += a * b                                               # 所有 PE 同拍做一次 MAC
    return C, cycles

A, B = np.random.rand(8, 32), np.random.rand(32, 8)
C, cyc = os_systolic(A, B)
assert np.allclose(C, A @ B) and cyc == 32 + 2 * 8 - 2
```

模拟里只有边界的 `n + m` 个位置每拍读存储，内部全部是邻居传递，这就是 systolic 省带宽的来源。

### 代码：NKI 分块矩阵乘

```python
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def matmul_tiled(lhsT, rhs):            # lhsT: [K, M]（已转置，K 在 partition 维），rhs: [K, N]
    K, M = lhsT.shape
    _, N = rhs.shape
    out = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    TM, TK, TN = 128, 128, 512          # partition 上限 128；PSUM 一行最多 512 个 fp32
    for m in nl.affine_range(M // TM):
        for n in nl.affine_range(N // TN):
            acc = nl.zeros((TM, TN), nl.float32, buffer=nl.psum)
            for k in nl.affine_range(K // TK):
                a = nl.load(lhsT[k * TK:(k + 1) * TK, m * TM:(m + 1) * TM])   # DMA: HBM → SBUF
                b = nl.load(rhs[k * TK:(k + 1) * TK, n * TN:(n + 1) * TN])
                acc += nisa.nc_matmul(a, b)                                  # Tensor Engine，结果进 PSUM
            res = nl.copy(acc, dtype=out.dtype)                              # PSUM → SBUF
            nl.store(out[m * TM:(m + 1) * TM, n * TN:(n + 1) * TN], value=res)
    return out
```

这个版本每个 `(m, n)` 都重新加载 A、B tile，复用只发生在阵列内部。优化方向和 GPU 一样：把 `lhsT` 的一条 `[K, 128]` 列带一次性装进 SBUF，
在 n 循环里复用；让 DMA 与计算重叠；对卷积，把 `C·R·S` 映射到 partition 维收缩。

### 代码：Halide 的算法与调度

```cpp
Func blur_x("blur_x"), blur_y("blur_y");
Var x("x"), y("y"), xi("xi"), yi("yi");
blur_x(x, y) = (in(x - 1, y) + in(x, y) + in(x + 1, y)) / 3;          // 算法
blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;
blur_y.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);      // 调度
blur_x.compute_at(blur_y, x).vectorize(x, 8);                         // 在每个 tile 内按需计算中间结果
```

`compute_at` 把中间结果的生命周期缩到一个 tile，换来少量重叠区域的重计算，这与 FlashAttention 和算子融合是同一取舍。

## 白板估算

硬件参数见 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)；TPU v1 与 Trainium 的规格取自上文引用的论文与官方文档。

```text
1) TPU v1 上 int8 MLP，batch = 200
   强度 = 2·200 = 400 ops/B < 2700 → 受权重带宽限制
   可达 = 400 × 34e9 = 1.36e13 ops/s ≈ 峰值 92 TOPS 的 15%
   要到计算屋顶需 batch ≈ 1350

2) 128×128 阵列的填充开销（流式维度 L）
   decode，batch=8：     8/(8+256)     ≈ 3%
   prefill，L=2048：  2048/(2048+256)  ≈ 89%
   对比 H100：batch=8 的 GEMV 强度约 8 FLOP/B，也只能用到 989 TFLOP/s 的 8/295 ≈ 2.7%
   结论：小 batch 在任何矩阵硬件上都亏，只是亏的原因不同（阵列填充 vs HBM 带宽）

3) Trainium 上 4096×4096×4096 bf16 matmul 的 tile 账
   A、B 各 4096·4096·2 B = 33.5 MB > SBUF 24 MiB → 必须分块
   stationary tile 128×128 bf16 = 32 KB；moving tile 128×512 bf16 = 128 KB
   PSUM 输出 tile 128×512 fp32 = 256 KB = 每 partition 2 KB；PSUM 每 partition 16 KiB → 可同时开 8 个累加 tile
   tile 数 = (4096/128)·(4096/512)·(4096/128) = 32·8·32 = 8192 次 nc_matmul

4) 片上复用 vs DRAM 读取
   N=256 的阵列，每个从 SRAM 读出的激活被 256 个 PE 用掉
   若同一个权重 tile 再服务 batch B 行，DRAM 上的每个权重字节被用 B 次——强度就是 roofline 的横轴
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| NKI 分块、SBUF/PSUM | [CS149 A4](../solutions/cs149/asst4-dnn-accelerator/README.md) | 与 numpy 参考结果一致；记录不同 tile 形状下的耗时 |
| systolic 周期模型 | 本讲 `os_systolic` | 结果等于 `A @ B`，周期为 `K + 2N − 2` |
| roofline 判定 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `roofline_time` | 能复现本讲的「受哪个屋顶限制」结论 |

## 易错点与反例

- 常见错答：「TPU 比 GPU 快是因为频率高」。TPU v1 只有 700 MHz，优势来自 65536 个 MAC 和片上复用。
- 常见错答：「systolic array 什么形状都高效」。维度不是 128/256 的倍数要补零，流式维度短时填充开销占主导。
- 常见错答：「软件管理的 SRAM 只是更大的 cache」。它没有替换策略，不搬就没有；双缓冲、预取都要程序员写。
- NKI 里把 M 放在 partition 维做收缩：`nc_matmul` 沿 partition 维收缩，左操作数必须是 `[K, M]` 的转置布局。
- 把 TPU v1 的 92 TOPS 当浮点：那是 int8，v1 不支持训练。
- 认为 weight-stationary 一定最好：K 维很长、输出复用大时 output-stationary 少搬部分和。

## 实验或练习

- 假设：NKI 矩阵乘的吞吐随 moving tile 宽度（N 方向）从 128 增到 512 而上升，因为每次 DMA 进来的 stationary tile 被用得更多。
- 配置：M=N=K ∈ {512, 1024, 2048}，TN ∈ {128, 256, 512}，固定 TM=TK=128；记录 profiler 中 Tensor Engine 的活跃比例。
- 预期：TN=512 最好；把 A 的列带缓存在 SBUF 后进一步提升。具体数值以实测为准。

## 高频问题

- 问：为什么专用加速器能效高？
  答：控制开销摊到大量运算上，数据在片上反复复用，减少 DRAM 访问——后者是能耗大头。
- 问：systolic array 怎么省带宽？
  答：数据在相邻 PE 间传递，只有边界读存储，N×N 阵列每个输入复用 N 次。
- 问：TPU v1 论文的主要发现？
  答：线上推理受延迟约束，batch 有限，多数负载受权重带宽限制；只有 CNN 接近计算屋顶。
- 问：NKI 里 partition 维是什么？
  答：SBUF/PSUM 的第 0 维，最多 128，对应 Tensor Engine 的一边，矩阵乘沿它收缩。
- 问：Halide 为什么要分离算法和调度？
  答：同一算法在不同硬件上的最优循环结构不同，分离后只改调度就能移植和调优，结果不变。

## 面试追问链

1. 问：一次 DRAM 读取和一次乘法的能耗差多少，这说明了什么？
   答：两个数量级以上。能效由每次取数的复用次数决定，加速器设计的核心是复用而不是更多乘法器。
2. 问：推导 N×N output-stationary 阵列算一次 `N×K×N` 矩阵乘的周期数。
   答：skew 注入使 PE(i,j) 在 `i+j+k` 拍处理第 k 项，总周期 `K + 2N − 2`，利用率 `K/(K+2N−2)`。
3. 问：weight-stationary 和 output-stationary 各适合什么？
   答：前者权重复用大（大 batch 推理、卷积），权重只装一次；后者部分和不出 PE，K 维长时省掉部分和搬运。
4. 问：TPU v1 的拐点是多少，batch=200 的 MLP 能用到多少算力？
   答：92e12/34e9 ≈ 2700 ops/B；强度 400，可达约 13.6 TOPS，约 15%。
5. 问：Trainium 上写 matmul，为什么左操作数要转置？
   答：Tensor Engine 沿 partition 维收缩，K 必须在第 0 维；转置可以在 DMA 时做，也可以预先存成转置布局。
6. 问：GPU 的 Tensor Core 和 TPU 的 MXU 本质区别？
   答：Tensor Core 是每个 SM 内的小矩阵单元，由 warp 调度，靠寄存器和共享内存喂数；MXU 是整片的大阵列，靠大块 SRAM 与编译器静态调度。前者灵活，后者在规则大矩阵上能效更高。
7. 问：你要为一款 128×128 阵列的推理芯片部署 8B 模型的 decode，batch 只有 4，怎么提高利用率？
   答：阵列填充占主导，先合批（continuous batching 提高流式维度）；权重量化降低带宽；把多个 decode step 的 GEMV 流水排进阵列让填充重叠；必要时把阵列切成更小的子阵列并行服务不同层或请求。
8. 问：如果让你为新加速器设计编程接口，选 Halide 式 DSL、NKI 式底层接口还是直接接入 XLA/TVM？
   答：底层接口给专家调最热的 kernel（matmul、attention）；编译器后端覆盖长尾算子；DSL 的价值在于调度空间可被自动搜索。三层并存是现实答案，关键是 tile 和存储层次在 IR 里显式可见。

## 自测题

1. 计算 TPU v1 峰值并写出单位换算。
   要点：65536 × 2 × 7e8 = 9.18e13 ops/s。
2. 为什么 systolic array 的片上带宽需求是 `O(N)` 而不是 `O(N²)`？
   要点：只有边界 PE 读存储，内部 PE 读邻居寄存器。
3. 128×128 阵列，流式维度多长时利用率达到 80%？
   要点：`L/(L+256) = 0.8` → L = 1024。
4. 用 NKI 约束说明为什么 PSUM 输出 tile 最多 128×512。
   要点：partition 维 ≤ 128；PSUM 每 partition 一个 2 KB bank 放 512 个 fp32。
5. 设计题：把一个 3×3 卷积（C=K=128）映射到 128×128 阵列，收缩维放什么？
   要点：im2col 后收缩维 `C·R·S = 1152`，按 128 分 9 段在 PSUM 中累加；输出通道 K=128 占满另一边。

## 关联笔记

- [M07 矩阵乘与分块](module-07-matmul-hardware-aware.md)：systolic array 是硬化的分块 GEMM。
- [M10 ML 编译器](module-10-ml-compilers.md)：Halide 调度原语在 TVM/Ansor 中被自动搜索。
- [M05 GPU 架构与 CUDA](module-05-gpu-architecture-cuda.md)：GPU 作为对照的通用并行硬件。
- [M17 模型压缩](module-17-compression-quantization.md)：int8/fp8 量化与低精度硬件。

## 本讲小结

专用化的收益来自两件事：把控制开销摊薄，把数据留在片上复用。systolic array 用邻居传递实现 N 倍复用，代价是填充/排空和形状约束；
TPU v1 证明推理负载多数受权重带宽限制。Trainium/NKI 把 SBUF、PSUM、partition 维和 DMA 暴露给程序员，
分块就是性能。下一讲讨论如何让编译器自动完成这些分块和调度。

## 延伸阅读

- Jouppi et al., In-Datacenter Performance Analysis of a Tensor Processing Unit, ISCA 2017, https://arxiv.org/abs/1704.04760
- Kung, Why Systolic Architectures?, IEEE Computer 1982.
- Chen et al., Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for CNNs, ISCA 2016.
- Horowitz, Computing's Energy Problem (and what we can do about it), ISSCC 2014.
- Ragan-Kelley et al., Halide, PLDI 2013；Koeplinger et al., Spatial, PLDI 2018.
- AWS Neuron NKI 文档, https://awsdocs-neuron.readthedocs-hosted.com/
