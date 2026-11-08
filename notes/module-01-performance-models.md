# Module 01 — 为什么并行：性能模型与 roofline

- 课程出处：CS149 Fall 2024《Why Parallelism》 · CS229S Fall 2023《Hardware Aware Algorithm Design (10-04)》 · CMU 15-849《Intro to DL Systems》
- 必读论文：Williams, Waterman, Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, 2009；Amdahl, Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities, 1967；Gustafson, Reevaluating Amdahl's Law, 1988
- 对应实验：无（估算工具：[tools/mlsys_calc.py](../tools/mlsys_calc.py)）
- 状态：初稿

## 摘要

「这段代码为什么慢」是每次优化的起点，也是面试里最常见的问题。凭感觉猜，往往会猜错方向：
明明是带宽不够，却去换算力更强的卡；明明是 CPU 调度拖后腿，却去改 kernel。这一讲给出几个简单的模型，
让你在动手之前先算出瓶颈在哪里。

背景是单核主频在功耗墙前停住了，之后的性能只能靠并行和专用硬件来拿。但并行有两个绕不过去的上限：
程序里只能串行的那部分（Amdahl 定律），以及计算图里最长的那条依赖链（span，也叫关键路径）。
落到单个 kernel 上，上限由 roofline 决定：每搬一个字节做的运算太少，kernel 就被带宽卡住；
只有做得足够多，越过硬件的拐点，算力才成为瓶颈。

有了这些模型，几乎所有「为什么慢」都能归到三类：算力用满了（compute-bound）、带宽用满了（memory-bound），
或者两样都没用满、时间耗在启动和调度上（latency/overhead-bound）。本讲给出区分它们的公式，以及读 profile 的先后顺序。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 加速比 `S = T1/TP`，效率 `E = S/P`。偶尔会测到超过核数的超线性加速，这通常是因为切分后每个核的数据装进了 cache，
   而不是算法变快了。
2. Amdahl：只能串行的比例 `s` 把加速比上限钉在 `1/s`。`s = 5%` 时，无论多少核都快不过 20 倍。
3. Work/span：`TP ≥ max(T1/P, T∞)`，而贪心调度能做到 `TP ≤ T1/P + T∞`。所以并行度 `T1/T∞` 就是能有效利用的核数上限。
4. Roofline：`可达 FLOP/s = min(峰值算力, I × 带宽)`。两条屋顶的交点叫拐点，H100 是 295 FLOP/B，A100 是 153 FLOP/B。
5. 搬数据比算数据贵。batch=1 的 LLM decode 算术强度约 1 FLOP/B，离拐点差两个数量级，所以它几乎完全受带宽限制。

必须能默写的公式：

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

容易混的几对概念：

- latency 是一件事做完要多久，throughput 是单位时间做完多少件。GPU 牺牲前者来换后者。
- Amdahl 固定问题规模，问能快多少；Gustafson 固定运行时间，问能做多大。
- compute-bound 和 memory-bound 看 `I` 落在拐点哪一侧。`I > ridge` 时靠 Tensor Core 提速，`I < ridge` 时只能靠少搬字节。
- memory-bound 是带宽已经打满；overhead-bound 是算力和带宽都没打满，时间耗在 launch、同步和 CPU 上。

## 学习目标

- 能从定义推出 Amdahl 与 Gustafson，并说清两者的假设差异；
- 能写出 work/span 的上下界，并解释贪心调度为何能达到上界；
- 能对任意算子算出 FLOPs、字节数和算术强度，在 A100、H100 上判断瓶颈；
- 能用功耗公式解释为什么多核取代了高主频；
- 能从空闲时间、算力利用率、带宽利用率三个方面读一份 profile。

## 先修知识

- 基本体系结构：cache、DRAM、时钟周期；bf16 每元素 2 字节，fp32 每元素 4 字节；
- 矩阵乘的 FLOPs：`M×K` 乘 `K×N` 需要 `2MNK` 次浮点运算（乘和加各算一次）。

## 核心概念

### 1. latency 与 throughput

讲并行之前，先分清这两个词。很多「并行了为什么没变快」的困惑，都来自把它们混为一谈。
latency（延迟）是单个任务从开始到结束的时间，throughput（吞吐）是单位时间完成的任务数。

拿洗衣机打比方：四台洗衣机一起洗，吞吐翻了 4 倍，但每一筐衣服仍然要洗 45 分钟。
并行改善的是吞吐；除非单个任务本身能拆开，它的延迟不会缩短。

LLM 服务里两者都要看。首 token 延迟（TTFT）和每 token 延迟属于 latency，每张卡每秒生成的 token 数（tokens/s/GPU）属于 throughput。
batch 越大，吞吐越好，但每条请求等得越久。

### 2. 加速比与效率

衡量并行效果需要一把公平的尺子。加速比 `S = T1/TP`，其中 `T1` 必须是最好的串行实现，
而不是把并行程序放到 1 个核上跑的时间，因为并行版本自带同步和额外工作，拿它当基线会把加速比夸大。
效率 `E = S/P` 表示每个核平均发挥了多少：8 张卡跑出 6 倍加速，效率就是 75%。

有时会测到加速比超过核数（`S > P`），这叫超线性加速。它是真实现象，常见原因是数据切分后，每个核分到的那份正好装进了 cache。
这并不违反任何定律，只说明串行基线本身被访存拖慢了。

### 3. Amdahl 与 Gustafson

为什么加了很多核却快不了多少？Amdahl 定律的回答是：总有一部分工作只能串行。
假设问题规模固定，比例为 `s` 的部分只能串行，剩下的 `1−s` 被 P 个核均分。那么无论 P 多大，加速比都超不过 `1/s`。
`s = 10%` 时，64 核只有 8.77 倍，无穷多个核也只有 10 倍。

可超算明明用上了上万个核。Gustafson 换了一个问法：不固定问题规模，而是固定运行时间，让问题随 P 一起变大。
只要串行部分不随规模增长，它在总时间里的占比就越来越小，加速比于是接近线性。

ML 训练更像 Gustafson 的情形，因为卡数加倍时 global batch 通常也跟着加倍。
但这不等于没有上限：通信和优化器步骤会以新的「串行比例」重新出现。

### 4. Work 与 span

Amdahl 只把程序分成「串行」和「并行」两块，真实程序的依赖关系要细得多。
把计算画成一张 DAG（有向无环图，节点是计算，边是依赖），就能描述得更准确。
`T1` 是所有节点的工作量之和，叫 work；`T∞` 是图中最长路径的长度，叫 span，也就是关键路径。

从这两个量马上能得到两个下界。P 个核每一步最多做 P 份工作，所以 `TP ≥ T1/P`；
关键路径上的节点只能一个接一个做，所以 `TP ≥ T∞`。两者之比 `T1/T∞` 叫并行度，它表示最多有多少个核能被有效利用。

举个例子，n 个数的树形求和有 `T1 = n−1`、`T∞ = log2 n`。`n = 2^20` 时并行度约 5.2 万，
远大于任何 CPU 的核数，所以这类问题的瓶颈不会出在并行度上。

### 5. Roofline 与算术强度

同一张 GPU 上，大矩阵乘能跑到接近峰值，向量加法却连峰值的零头都不到。
roofline 模型解释了这个差距：kernel 能跑多快，取决于它每搬一个字节做多少次运算。

这个比值叫算术强度 `I`，等于 FLOPs 除以从 DRAM（GPU 上是 HBM）搬运的字节数。
分母是实际与慢存储交换的字节，不是算法读数据的总次数：同一份数据如果留在 cache 里被反复使用，只算一次。

可达算力是 `min(π, I·β)`，π 是峰值算力，β 是带宽。画出来是一条折线：左边的斜线是带宽屋顶，右边的水平线是算力屋顶，
交点就是拐点（ridge）`π/β`。拐点左边的 kernel 受带宽限制，右边的受算力限制。
H100 bf16 的拐点是 `989e12 / 3.35e12 ≈ 295 FLOP/B`，A100 是 `312e12 / 2.04e12 ≈ 153 FLOP/B`。

拐点随硬件变化，所以同一个 kernel 换卡后可能换了瓶颈。`I = 200` 的 kernel 在 A100 上是 compute-bound，到 H100 上就成了 memory-bound。

真实硬件还有多级 roofline，L2、shared memory 各有自己的带宽屋顶。M07 讲的分块，
就是把数据留在高带宽的层级里反复使用，从而提高对 HBM 的 `I`。

### 6. 功耗墙与每次操作的能耗

为什么 2000 年代中期以后，CPU 主频不再往上涨，行业转而堆核数、加 SIMD、做加速器？原因在功耗。

动态功耗约为 `α·C·V²·f`。提高频率时电压也要大致线性地跟着升高，所以功耗随频率的三次方增长：主频翻倍，功耗约涨 8 倍。
反过来，同样的频率放两个核，功耗只涨 2 倍，而程序能并行时吞吐也涨 2 倍。这就是转向多核、SIMD 和加速器的物理原因。

能耗是同一个问题的另一面。从 DRAM 取一个字比做一次浮点运算贵出数量级，
所以 Tensor Core、systolic array（脉动阵列）、算子融合这些设计都在做同一件事：让每个搬进来的字节多做几次运算。

### 7. 三类瓶颈与读 profile

有了 roofline，就能把「慢」分成三类。三类的对策完全不同，所以要先判断再动手。

compute-bound 的特征是 achieved FLOP/s（实测算力）接近峰值。这时可以降低精度（例如从 bf16 换成 fp8）、确保用上 Tensor Core、减少冗余计算。

memory-bound 的特征是实测带宽接近 HBM 峰值，而 FLOP/s 远低于峰值。这时多给算力也没用，只能少搬字节：
算子融合、量化权重，或者用更大的 batch 和分块提高数据复用。

latency/overhead-bound 的特征是算力和带宽都远没打满，时间耗在 kernel launch、Python 调度、同步、过多的小 kernel 或 CPU 数据加载上。
对策是 CUDA Graphs、`torch.compile`、合并小 kernel，或者让每次 launch 多做一些工作。

读 profile 的顺序也由此而来：先看 GPU timeline 上有没有空洞，有空洞就是 overhead 问题；
没有空洞，再看单个 kernel 的 FLOP/s 和带宽各占峰值多少。

## 关键推导

### 推导 1：Amdahl

设串行时间 `T1 = 1`，其中 `s` 只能串行。P 核时 `TP = s + (1−s)/P`，于是：

```text
S(P) = T1 / TP = 1 / ( s + (1−s)/P )
P → ∞:  S → 1/s
半效率点: S = P/2  ⇔  s + (1−s)/P = 2/P  ⇔  P = (1+s)/s ≈ 1/s
例: s = 0.05, P = 8  → 1/(0.05 + 0.11875) = 5.93
    s = 0.01, P = 1024 → 1/(0.01 + 0.000967) = 91.2
```

半效率点那一行值得记住：核数一旦超过 `1/s` 左右，每个核就只剩一半的效率。
这个推导假设并行部分能完美均分、没有通信开销。真实系统的通信随 P 增长，所以曲线在某个 P 之后反而会下降。

### 推导 2：Gustafson

这次从并行机出发：在 P 核机器上测得运行时间为 1，其中串行 `s`、并行 `1−s`。同样的工作放到单核上，要花 `s + (1−s)·P`。

```text
S(P) = s + (1−s)·P = P − s·(P − 1)
例: s = 0.05, P = 8 → 7.65
```

同样是 `s = 0.05`、8 核，Amdahl 给 5.93，Gustafson 给 7.65，但两者并不矛盾。
Amdahl 的 `s` 是相对单核时间的比例，Gustafson 的 `s` 是相对并行机时间的比例，而且两边的问题规模也不一样。

### 推导 3：矩阵乘与 elementwise 的算术强度

判断瓶颈的第一步是算 `I`。下面三种算子覆盖了 LLM 里最常见的情形：

```text
GEMM  M=N=K=n, bf16, 理想复用（每个矩阵只读写一次）:
  FLOPs = 2n³,  Bytes = 2·(n² + n² + n²) = 6n²,  I = n/3
  n = 256 → 85；n = 512 → 171；n = 4096 → 1365  (FLOP/B)
Elementwise  z = x + y, bf16:
  FLOPs = 1/元素,  Bytes = 2+2+2 = 6/元素,  I = 1/6 ≈ 0.17
GEMV  y = W·x, W 为 d×d bf16:
  FLOPs = 2d²,  Bytes ≈ 2d²,  I ≈ 1
```

GEMM 的 `I` 随 n 线性增长，所以足够大的矩阵乘一定是 compute-bound。
elementwise 和 GEMV 的 `I` 是常数，与规模无关，所以它们永远是 memory-bound。

## 深入推导

### 推导 4：贪心调度的上界 `TP ≤ T1/P + T∞`（Brent 定理）

下界告诉我们不可能比 `max(T1/P, T∞)` 更快，这里要回答的是：一个简单的调度器能离这个下界多近？

贪心调度的规则是：任何时刻只要有就绪任务、又有空闲核，就立刻分配。把每个时间步分成两类：

- 完整步：P 个核都在干活。总工作只有 `T1`，所以这类步最多 `T1/P` 个。
- 不完整步：就绪任务少于 P，所有就绪任务都被执行。DAG 中当前「最深未完成」的节点都是就绪的，
  所以每个不完整步都让剩余关键路径缩短 1。这类步最多 `T∞` 个。

```text
TP ≤ (#完整步) + (#不完整步) ≤ T1/P + T∞
结合下界: TP ≥ max(T1/P, T∞) ≥ (T1/P + T∞)/2
⇒ 贪心调度最多比最优差 2 倍；当 T1/T∞ ≫ P 时 TP ≈ T1/P，近似线性加速
```

对工程的含义是：只要并行度远大于核数，调度器做到「别让核闲着」就够好了。M03 的 work stealing 在期望意义上也达到同一个界。

### 推导 5：decode 的 batch 拐点

为什么 LLM 推理服务拼命把请求凑成大 batch？看 decode 的算术强度就明白了。
decode 每一步要对每个权重做一次乘加。batch 中有 B 条序列，先忽略 KV cache 和激活：

```text
FLOPs ≈ 2 · B · N_params
Bytes ≈ 2 · N_params           (bf16 权重只读一遍，被 B 条序列共享)
I ≈ B   FLOP/B
compute-bound 条件: B ≳ ridge  → H100 约 295，A100 约 153
```

权重只读一遍，却被 B 条序列共享，所以 batch 越大，`I` 越高。
但加上 KV cache 之后，每条序列还要读自己的 KV，字节数随 `B × seq_len` 增长，`I` 又被拉低，所以长上下文的 decode 很难达到拐点。
M15、M16 里的 GQA、KV 量化和 PagedAttention，针对的正是这部分字节。

### 推导 6：Little's law 与「为什么要很多线程」

单个线程读内存为什么跑不满带宽？Little's law 给出答案：带宽为 β、延迟为 L 时，要打满带宽，必须同时有 `β·L` 字节在路上。
以 CPU 的一个 DDR5 通道为例，带宽约 45 GB/s，DRAM 延迟约 100 ns：

```text
bytes_in_flight = 45e9 B/s × 100e-9 s = 4500 B ≈ 70 条 64 B cache line
```

一个访存前后相互依赖的线程，一次只能挂一个 miss，离 70 条差得很远。
所以要靠多核、硬件多线程、预取，或者 GPU 上的大量 warp 来凑出足够的并发访存（M02、M05）。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的数字。下面几道题覆盖了面试里最常被要求当场估算的场景。

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

下面的片段把 roofline 写成一个函数：分别算出计算时间和访存时间，取较大的那个，并报告是哪一项决定了时间。

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

- 「加 8 倍核就快 8 倍」：错。串行部分占 5% 时，8 核只有 5.93 倍。
- 「用并行程序在 1 核上的时间当 T1」：并行版本自带同步和额外工作，这样算出的加速比偏大。
- 「算术强度按算法读了几次数据来算」：错。分母是和 HBM 实际交换的字节，所以分块之后，同一个 GEMM 的 `I` 会变。
- 「GPU 利用率 100% 说明算力打满了」：`nvidia-smi` 的 utilization 只表示采样窗口内有 kernel 在跑，
  一个 memory-bound kernel 也会显示 100%。要看 MFU 或 achieved FLOP/s。
- 「memory-bound 就该换算力更大的卡」：H100 相对 A100 算力涨约 3.2 倍，带宽只涨约 1.6 倍，所以 memory-bound kernel 只快约 1.6 倍。
- 「小 batch 推理时 GPU 空转，是因为算力不够」：恰好相反。限制它的是带宽和 launch 开销，算力大量闲置。

## 实验或练习

这个练习的目的是亲手画出一张 roofline，看到两类算子分别落在哪条屋顶上。

- 假设：elementwise kernel 的耗时与元素数成线性关系，换算出的带宽接近 HBM 峰值；尺寸小到一定程度后，耗时不再下降。
- 配置：在 PyTorch 中对 `2^10` 到 `2^28` 个 bf16 元素做 `x + y`，用 CUDA event 计时，换算 `6N / t` 的带宽；
  再对 `n = 256…8192` 的方阵做 `torch.matmul`，换算 `2n³ / t` 的 TFLOP/s。
- 预期：大尺寸 elementwise 的带宽接近 HBM 峰值，TFLOP/s 很低；小尺寸时耗时趋于常数，说明进入了 overhead-bound。
  matmul 的 TFLOP/s 随 n 上升，n 足够大后接近 Tensor Core 峰值。最后把两组点画在同一张 roofline 图上。

## 高频问题

- 问：什么是算术强度？
  答：FLOPs 除以与慢存储（DRAM 或 HBM）交换的字节数，单位 FLOP/B。
- 问：roofline 的拐点怎么算？
  答：峰值算力除以带宽，`π/β`。H100 bf16 约 295 FLOP/B，A100 约 153 FLOP/B。
- 问：超线性加速违反 Amdahl 定律吗？
  答：不违反。它通常来自切分后每个核的数据装进了 cache，说明串行基线本身受访存拖累。
- 问：怎么判断一个程序是 overhead-bound？
  答：timeline 上 kernel 之间有大量空隙，achieved FLOP/s 和带宽都远低于峰值，而且增大问题规模时耗时几乎不变。

## 面试追问链

1. 问：latency 和 throughput 有什么区别，GPU 偏向哪个？
   答：latency 是单个任务的耗时，throughput 是单位时间的完成量。GPU 用大量线程和简单的核心换吞吐，所以单线程延迟比 CPU 差。
2. 问：推一下 Amdahl。串行部分占 5%，最多能快多少？
   答：`S = 1/(s + (1−s)/P)`，令 `P→∞` 得 `1/s = 20`。8 核时只有 5.93 倍。
3. 问：既然上限这么低，为什么超算还能用上万个核？
   答：因为问题规模也随核数放大了，这是 Gustafson 的视角：串行部分占比下降，`S = s + (1−s)P`。训练里对应的是 global batch 随卡数放大。
4. 问：写出 roofline，并判断 4096 方阵的 bf16 GEMM 在 H100 上的瓶颈。
   答：可达算力是 `min(π, Iβ)`。这里 `I = n/3 = 1365`，大于拐点 295，所以是 compute-bound，耗时下限约 0.14 ms。
5. 问：同一个 kernel 在 A100 上是 compute-bound，到 H100 上可能变成什么？
   答：H100 的拐点更高（295 对 153），`I` 落在两者之间的 kernel 会变成 memory-bound，换卡后只快带宽比那么多，约 1.6 倍。
6. 问：batch=1 的 8B 模型 decode，在 H100 上最快多少 token/s？瓶颈在哪里？
   答：每个 token 要把 16 GB 权重读一遍，16 GB / 3.35 TB/s ≈ 4.8 ms，约 209 token/s。
   瓶颈是带宽：每读 2 字节权重只做 2 次运算，`I ≈ 1`，远低于 295。
7. 问：训练时 profile 显示 MFU 只有 15%，你怎么排查？
   答：先看 timeline 上的空洞，常见来源是数据加载、Python 开销、同步等待和没能与计算重叠的通信。
   再看最耗时的 kernel 的 FLOP/s 与带宽占比，区分是 attention、norm 这类 memory-bound 算子在拖累，还是 GEMM 形状太小、没打满 Tensor Core。
8. 问：为什么行业从提主频转向多核和加速器？
   答：功耗随主频的三次方增长（`P ∝ f³`），主频翻倍功耗约涨 8 倍。同样的功耗下，多个低频核或专用单元能给出更高的吞吐。
   而且数据搬运的能耗占大头，专用硬件可以减少搬运。

## 自测题

1. 一个程序 20% 串行。求 4、16、∞ 核时的加速比。
   要点：`1/(0.2+0.8/4) = 2.5`；`1/(0.2+0.05) = 4`；`∞` 时 5。
2. 证明贪心调度满足 `TP ≤ T1/P + T∞`。
   要点：完整步不超过 `T1/P` 个；每个不完整步让剩余关键路径减 1。
3. `n = 512` 的 bf16 方阵乘法，在 A100 和 H100 上分别是什么瓶颈？
   要点：`I = 171`。A100 上 `171 > 153`，compute-bound；H100 上 `171 < 295`，memory-bound。
4. 设计题：给你一张 H100，要让 8B 模型的 decode 吞吐最大，batch 应该往哪个方向调？要调到多大才能 compute-bound，代价是什么？
   要点：增大 batch。`I ≈ B`，B 要接近 295 才到拐点，算上 KV cache 的读取还要更大。
   代价是每 token 延迟变长，KV cache 占用的显存随 batch 线性增长。

## 关联笔记

- [M02](module-02-multicore-simd-threads.md)：用多核、SIMD、硬件多线程把峰值和带宽真正吃满。
- [M03](module-03-parallel-programming-scheduling.md)：work/span 与 work stealing 的调度界。
- [M05](module-05-gpu-architecture-cuda.md)：GPU 上的 roofline、warp 与延迟隐藏。
- [M15](module-15-inference-kv-speculative.md)：decode 的 memory-bound 分析与 batch 拐点。
- [edge-deploy M01 端侧 roofline](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-01-edge-roofline.md)：同一模型在 4 核 CPU 上的实测版：fp32 拐点 11.5、int8 29.5，单核 DRAM 13.4 GB/s 受 Little 定律限制

## 本讲小结

这一讲要带走三点。并行的收益被串行部分和关键路径封顶，加核之前先算 `1/s` 和 `T1/T∞`。
单个 kernel 的速度被 roofline 封顶，判断瓶颈只需要三个数：FLOPs、字节数和硬件拐点；
H100 的拐点约 295 FLOP/B，大 GEMM 远在它右边，elementwise 和 batch=1 的 decode 远在它左边。
读 profile 时先找 timeline 上的空洞，再看单个 kernel 的算力和带宽占比，最后才考虑换算法。

## 延伸阅读

- Williams, Waterman, Patterson, Roofline: An Insightful Visual Performance Model for Multicore Architectures, CACM 2009.
- Amdahl, Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities, AFIPS 1967.
- Gustafson, Reevaluating Amdahl's Law, CACM 1988.
- Horace He, Making Deep Learning Go Brrrr From First Principles, 2022, https://horace.io/brrr_intro.html
