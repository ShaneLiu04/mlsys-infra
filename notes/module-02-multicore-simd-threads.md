# Module 02 — 多核、SIMD 与多线程

- 课程出处：CS149 Fall 2024《A Modern Multi-Core Processor》《Multi-Core Architecture Part II + ISPC Programming Abstractions》
- 必读论文：Pharr, Mark, ispc: A SPMD Compiler for High-Performance CPU Programming, 2012；Olukotun et al., The Case for a Single-Chip Multiprocessor, 1996
- 对应实验：[CS149 A1 CPU 并行](../solutions/cs149/asst1-cpu-parallel/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst1/cs149_asst1_report.pdf)（实测：12 核动态分配 11.1 倍；每 8 个混 1 个慢 lane，AVX2 只有标量的 0.72 倍）
- 状态：初稿

## 摘要

现代处理器的吞吐来自三种并行叠加：多核（线程级）、SIMD（数据级）、硬件多线程（隐藏延迟）。
超标量的指令级并行已接近天花板，所以芯片把晶体管从乱序、预测逻辑挪到更多核和更宽的向量单元上。
SIMD 的代价是分支发散：一个 gang 里只要有一条 lane 走慢路径，整组都等它。
硬件多线程不让单线程变快，只是在一个线程等内存时切到另一个，把延迟藏起来。
带宽受限的程序（saxpy）对以上三种手段都不敏感，这是本讲最重要的反例。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 峰值算力 = 核数 × SIMD 宽度 × 每周期 FMA 单元数 × 2 × 主频。少一个乘数都拿不到峰值。
2. SIMD 发散：if/else 两条路径都要执行，用 mask 关掉不该写的 lane。最坏情况利用率 `1/W`。
3. 硬件多线程的条件：线程数 `≥ 1 + L/C`（L 为停顿延迟，C 为两次停顿之间的计算周期）。
4. ISPC 是 SPMD 抽象、SIMD 实现：一个 gang 的 program instance 对应一条向量指令的 lane。
5. saxpy 每元素 2 FLOP、约 12–16 字节，算术强度约 0.125–0.17 FLOP/B，多核和 SIMD 都救不了。

**必须能默写的公式**

```text
Peak FLOP/cycle = cores × lanes × FMA_units × 2
SIMD 利用率     = 有效 lane·周期 / (W × 总周期)
发散期望时间    = c_a·(1 − (1−p)^W) + c_b·(1 − p^W)     p = 单 lane 走 a 分支的概率，W = 宽度
延迟隐藏线程数  N ≥ 1 + L / C                            L = 停顿周期，C = 每次停顿前的计算周期
saxpy 字节/元素 = 4(x) + 4(y) + 4(写 out) [+ 4 write-allocate 读 out]
saxpy 强度      = 2 / 16 = 0.125 FLOP/B（有 write-allocate），2 / 12 ≈ 0.17（streaming store）
```

**一句话区分**

- 超标量 vs SIMD：一个指令流里同时发多条不同指令 vs 一条指令作用于多个数据。
- 多核 vs 硬件多线程：多份执行单元同时算 vs 一份执行单元在多个上下文间切换。
- interleaved（细粒度）多线程 vs SMT：每周期只从一个线程取指 vs 同一周期从多个线程发射。
- explicit SIMD vs implicit SIMD：编译器生成向量指令（AVX/ISPC） vs 硬件把标量线程打包成 warp（GPU）。
- uniform vs varying：整个 gang 共享一个值 vs 每个 program instance 各有一个值。

## 学习目标

- 能画出一个核的取指/译码、执行单元、执行上下文，并解释三种并行分别复制了哪一块；
- 能写出峰值 FLOP/cycle 并解释为何实际程序拿不到；
- 能分析 SIMD 分支发散的代价，给出最好/最坏输入；
- 能用 `1 + L/C` 估算需要多少硬件线程才能隐藏内存延迟；
- 能写 ISPC 的 `foreach` 与 `launch`，解释 gang 内 interleaved 赋值为何更快；
- 能判断一个程序是否带宽受限，并说明此时加核、加 SIMD 的预期收益。

## 先修知识

- [M01](module-01-performance-models.md) 的 roofline 与 Little's law；
- 基本的 cache 层次：L1 ~4 周期、L3 ~40 周期、DRAM ~200–300 周期；C/C++ 指针与数组。

## 核心概念

### 1. 超标量与 ILP 的天花板

一个指令流中若相邻指令无依赖，超标量核可以在同一周期发射多条。乱序执行、分支预测、寄存器重命名都在挖 ILP。
但真实程序可挖的 ILP 有限，发射宽度继续加宽收益迅速递减，这些逻辑占了大量面积和功耗，却只服务一个指令流。

### 2. 多核：用面积换线程级并行

把复杂核换成多个简单核：单线程变慢一点，但总吞吐随核数增长。前提是软件提供多个独立指令流（线程）。
没有并行化的程序在多核上只用到一个核，这是「多核时代免费午餐结束」的含义。

### 3. SIMD：一份取指，多份 ALU

一条向量指令驱动 W 条 lane。AVX2 是 8 个 fp32 lane，AVX-512 是 16 个。取指译码的开销被 W 个数据摊薄。
条件分支是 SIMD 的敌人：硬件给每条 lane 一个 mask，先执行 if 路径（只写 mask 为真的 lane），
再执行 else 路径（mask 取反）。两段时间相加。若 8 条 lane 中只有 1 条走 1000 周期的慢路径，
其余 7 条在这 1000 周期里空等，利用率接近 1/8。

### 4. 硬件多线程：用上下文换延迟隐藏

核内保存多份执行上下文（寄存器、PC）。线程 A 发出 cache miss 后，核在下一周期切到线程 B。
只要可运行线程够多，ALU 永远有活干。代价：每个线程分到的寄存器和 cache 变少；单线程延迟变长
（它就绪时可能轮不到它）。Intel Hyper-Threading 是 SMT，每核 2 个线程。GPU 把这一招用到极致：
每个 SM 保存数十个 warp 的上下文（M05）。

### 5. 三者叠加

一个 8 核、AVX2、每核 2 个 FMA 单元的 CPU：

```text
8 cores × 8 lanes × 2 FMA × 2 FLOP = 256 FLOP/cycle
```

拿满需要：至少 8 个线程（每核一个），每个线程的内层循环被向量化，没有发散，有足够的独立 FMA
覆盖 FMA 延迟，数据在 L1 里。缺任何一项，实际性能按比例掉。

### 6. ISPC 编程模型

ISPC 用 SPMD 写法：程序员写一个 program instance 的代码，编译器把 `programCount` 个实例组成一个 gang，
映射到一条向量指令的 lane 上。`programIndex` 是 lane 编号。`uniform` 变量全 gang 共享，存在标量寄存器；
`varying`（默认）每个 lane 一个值，存在向量寄存器。`foreach` 把迭代交给编译器分配。
gang 只在一个核上运行；要用多核，必须用 `launch` 创建 task，由运行时的线程池执行。
抽象与实现要分开说：SPMD 是抽象，SIMD 指令是实现。

## 关键推导

### 推导 1：gang 内赋值方式为何影响性能

N 个元素给 8 个 lane。interleaved：lane `j` 处理 `j, j+8, j+16, …`。每次迭代 8 个 lane 访问连续 8 个元素，
一条 packed vector load 读 32 字节。blocked：lane `j` 处理 `[jN/8, (j+1)N/8)`。每次迭代 8 个 lane 访问
相隔 N/8 的地址，只能用 gather，每条 lane 可能触到不同 cache line。

```text
interleaved:  addr(lane j, iter k) = base + 4·(8k + j)    → 连续 32 B，1 条 vector load
blocked:      addr(lane j, iter k) = base + 4·(j·N/8 + k) → 8 个不同 cache line，gather
```

这和 GPU 的 memory coalescing（M05）是同一个道理：相邻 lane 访问相邻地址。

### 推导 2：延迟隐藏需要的线程数

每个线程执行 C 个周期计算后发出一次停顿 L 周期的访存。要让 ALU 不空，在一个线程等待的 L 周期内，
其余线程要提供 L 个周期的计算：

```text
(N − 1) · C ≥ L   ⇒   N ≥ 1 + L / C
例: L = 250 周期（DRAM），C = 25 周期  → N ≥ 11
    L = 40 周期（L3），   C = 25 周期  → N ≥ 2.6，3 个线程即可
```

线程数受上下文存储限制。CPU 每核只有 2 个硬件线程，所以 CPU 靠大 cache 和预取降低 L；
GPU 每 SM 几十个 warp，靠数量直接覆盖 L。

## 深入推导

### 推导 3：分支发散的期望代价

宽度 W 的 gang，每条 lane 独立以概率 p 走代价 `c_a` 的分支 a，否则走代价 `c_b` 的分支 b。
只要 gang 里至少一条 lane 走 a，整组就要执行 a 段；b 同理。

```text
E[T_gang] = c_a · (1 − (1−p)^W) + c_b · (1 − p^W)
E[useful]  = W · ( p·c_a + (1−p)·c_b )         （lane·周期）
利用率     = E[useful] / (W · E[T_gang])

例: W = 8, p = 1/8, c_a = 1000, c_b = 10
  1 − (7/8)^8 = 1 − 0.344 = 0.656
  1 − (1/8)^8 ≈ 1
  E[T_gang] ≈ 656 + 10 = 666 周期
  每 lane 平均有效 = 125 + 8.75 = 133.75 周期
  利用率 ≈ 133.75 / 666 ≈ 20%
```

W 越宽，`1 − (1−p)^W` 越接近 1，利用率越差。这解释了 A1 程序 2 中向量宽度从 2 增到 16 时 lane 利用率下降。

```python
def simd_util(W, p, ca, cb):
    t = ca * (1 - (1 - p) ** W) + cb * (1 - p ** W)
    useful = p * ca + (1 - p) * cb
    return useful / t

for W in (2, 4, 8, 16):
    print(W, round(simd_util(W, 1 / 8, 1000, 10), 3))
```

### 推导 4：saxpy 为何带宽受限

`out[i] = a·x[i] + y[i]`。fp32 每元素：读 x 4 B、读 y 4 B、写 out 4 B。普通 store 写一条不在 cache 中的
line 时，CPU 要先把它读进来（write-allocate），再多 4 B。

```text
Bytes/elem = 12（streaming store）或 16（write-allocate）
FLOP/elem  = 2
I = 0.17 或 0.125 FLOP/B
```

与 [M01](module-01-performance-models.md) 的 roofline 对照：CPU 的拐点在每字节数 FLOP 量级，saxpy 远在左边斜线上。
加线程只在单核带宽打不满 DRAM 时有用；DRAM 打满之后，再加核、加 SIMD 都不变快。
用 `_mm256_stream_ps` 等 non-temporal store 跳过 write-allocate，能把字节数从 16 降到 12，理论上快约 1.33 倍。

```cpp
#include <immintrin.h>
void saxpy_avx2(int n, float a, const float* x, const float* y, float* out) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vx, vy));
    }
    for (; i < n; ++i) out[i] = a * x[i] + y[i];
}
```

### 推导 5：ISPC foreach 与 task 的对应

```cpp
export void saxpy_ispc(uniform int N, uniform float scale,
                       uniform float X[], uniform float Y[], uniform float out[]) {
    foreach (i = 0 ... N) {            // 编译器按 gang 宽度 interleaved 分配
        out[i] = scale * X[i] + Y[i];  // i 是 varying，scale 是 uniform
    }
}

task void saxpy_task(uniform int N, uniform int span, uniform float scale,
                     uniform float X[], uniform float Y[], uniform float out[]) {
    uniform int start = taskIndex * span;
    uniform int end = min(N, start + span);
    foreach (i = start ... end) out[i] = scale * X[i] + Y[i];
}

export void saxpy_ispc_tasks(uniform int N, uniform float scale,
                             uniform float X[], uniform float Y[], uniform float out[]) {
    uniform int span = (N + 63) / 64;   // task 数远多于核数，便于负载均衡
    launch[64] saxpy_task(N, span, scale, X, Y, out);
}
```

`foreach` 只给 SIMD 并行；`launch` 才给多核并行。task 数取核数的数倍，让运行时在核间均衡。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的 CPU 数字。主频不在表中，下面设 `f = 3 GHz` 作为题设。

```text
1) 峰值: 8 核 × 8 lane(AVX2) × 2 FMA × 2 = 256 FLOP/cycle
         × 3e9 = 768 GFLOP/s

2) saxpy 上限（2 个 DDR5 通道，每通道 45 GB/s → 90 GB/s）:
   元素/s = 90e9 / 16 = 5.6e9
   FLOP/s = 2 × 5.6e9 = 11.25 GFLOP/s，只有峰值的 1.5%
   改用 streaming store: 90e9 / 12 × 2 = 15 GFLOP/s

3) 延迟隐藏: DRAM 约 250 周期，每次 miss 前 25 周期计算
   N ≥ 1 + 250/25 = 11 线程；CPU 每核 2 个 SMT 线程远远不够
   → CPU 必须靠 cache 命中（L1 ~4 周期）和预取

4) GPU 对照: 一个 warp 32 lane；H100 有 132 个 SM
   若每 SM 驻留 64 个 warp，全卡同时有 132 × 64 × 32 = 270,336 个线程上下文
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 线程级分解、负载不均 | [A1](../solutions/cs149/asst1-cpu-parallel/README.md) 程序 1（Mandelbrot 线程版） | 块状划分时各线程耗时差异大；interleaved 行划分后加速比接近线程数 |
| SIMD mask 与发散 | A1 程序 2（`CS149intrin` 向量化 `clampedExp`） | 输出正确；报告 lane 利用率随向量宽度增大而下降 |
| ISPC gang 与 task | A1 程序 3（Mandelbrot ISPC + tasks） | ISPC 相对串行接近 SIMD 宽度量级的加速；加 task 后再乘上核数量级 |
| 发散的最好/最坏输入 | A1 程序 4（Newton 迭代 sqrt） | 构造全部 lane 迭代数相同的输入得到最高 SIMD 加速，构造每 8 个里 1 个慢的输入使加速接近消失 |
| 带宽受限 | A1 程序 5（saxpy） | 加 task 后加速远低于核数；能用字节数解释 |

## 易错点与反例

- 「ISPC 的 gang 会自动用满多核」：错。gang 在一个核上；多核要靠 `launch` task。
- 「program instance 就是线程」：错。它是向量指令的一条 lane，没有独立 PC。
- 「SMT 让单线程更快」：反了。SMT 提高核吞吐，单线程可能更慢。
- 「向量宽度越宽越好」：发散代码中，宽度越宽，任一 lane 走慢路径的概率越高。
- 「saxpy 用 16 核应该快 16 倍」：DRAM 带宽被少数几个核就打满了，之后加核无效。
- 「blocked 赋值更自然所以更快」：gang 内 blocked 导致 gather；interleaved 才是连续访存。
  反过来，在多核之间按大块分配更好，因为每个核独占连续区域，cache 局部性好。

## 实验或练习

- 假设：Mandelbrot 不同区域迭代次数差别大，块状划分负载不均；saxpy 受带宽限制。
- 配置：A1 程序 1 用 2–8 线程分别做块状划分和 interleaved 划分，记录每个线程的耗时；
  程序 5 比较 ISPC 单核与 ISPC + tasks 的耗时，再用 `N × 16 B / t` 算出带宽。
- 预期：块状划分加速比随线程数增长不线性，中间带的线程最慢；interleaved 划分接近线性。
  saxpy 加 task 的收益远小于核数，换算出的带宽接近机器的内存带宽上限。

## 高频问题

- 问：SIMD 和多核的区别？
  答：SIMD 是一条指令多份数据，共享取指；多核是多个独立指令流，各有取指单元。
- 问：分支发散在 CPU SIMD 上怎么处理？
  答：两条路径都执行，用 mask 控制写回，耗时相加。
- 问：硬件多线程解决什么问题？
  答：隐藏访存延迟，提高执行单元利用率，不提高单线程速度。
- 问：为什么 saxpy 并行后几乎不加速？
  答：算术强度约 0.125 FLOP/B，被内存带宽限制，计算单元大部分在等数据。

## 面试追问链

1. 问：一个现代 CPU 核的峰值 FLOP/cycle 怎么算？
   答：lane 数 × FMA 单元数 × 2。AVX2、2 个 FMA 单元时每核 32 FLOP/cycle，再乘核数。
2. 问：SIMD 遇到 if/else 怎么办？最坏利用率多少？
   答：mask 执行两条路径。只有 1 条 lane 走长路径时，利用率接近 `1/W`。
3. 问：推一下需要多少硬件线程才能隐藏内存延迟。
   答：`(N−1)·C ≥ L`，`N ≥ 1 + L/C`。L = 250、C = 25 时要 11 个；CPU 只有 2 个，所以靠 cache。
4. 问：ISPC 里 interleaved 和 blocked 赋值哪个快，为什么？在多核间呢？
   答：gang 内 interleaved 快，相邻 lane 访问相邻地址，一条 packed load。多核间用大块，减少 cache line 共享和 false sharing。
5. 问：给你一个 saxpy，怎么证明它带宽受限，怎么优化？
   答：算每元素 16 B、2 FLOP；实测带宽接近 DRAM 峰值即证明。优化只能减字节：streaming store、融合到前后算子、降精度。
6. 问：GPU 的 warp 和 CPU 的 SIMD 有什么本质联系？
   答：都是一份取指驱动多条 lane。GPU 是 implicit SIMD：程序员写标量线程，硬件按 32 个打包，发散时同样 mask 串行化。
7. 问：在 LLM 推理里，哪些算子最像 saxpy？
   答：残差相加、RMSNorm、激活函数、RoPE 等 elementwise 或 reduction 算子，`I < 1 FLOP/B`。
   它们靠算子融合省掉中间张量的读写，而不是靠更多算力。
8. 问：MoE 路由、beam search 等数据相关的控制流，放到 GPU 上会遇到什么问题？
   答：同一 warp 的线程走不同分支或访问不同专家，造成发散和不连续访存。常见做法是先按专家或路径排序分组（M06 的 sort/scan），再让每组执行一致的代码。

## 自测题

1. 4 核、AVX-512（16 lane）、每核 2 个 FMA 单元，峰值 FLOP/cycle 是多少？
   要点：`4 × 16 × 2 × 2 = 256`。
2. W = 16、p = 1/16、`c_a = 100`、`c_b = 10`，求 SIMD 利用率。
   要点：`1 − (15/16)^16 ≈ 0.644`；`E[T] ≈ 64.4 + 10 = 74.4`；有效 `6.25 + 9.375 = 15.6`；利用率约 21%。
3. 为 A1 程序 4 构造让 ISPC 加速最大和最小的输入，并解释。
   要点：全部取收敛最慢的同一个值，无发散且计算密集；每 8 个里 1 个慢、其余收敛极快，发散最大。
4. 设计题：一个 CPU 推理服务对每个请求做 tokenization 和 embedding 查表，CPU 利用率高但吞吐低，你会先查什么？
   要点：查是否受内存延迟限制（随机查表 miss），看 IPC 与 cache miss 率；用批量化、预取、SMT、数据布局改进。

## 关联笔记

- [M01](module-01-performance-models.md)：roofline 与 Little's law 是本讲所有估算的基础。
- [M03](module-03-parallel-programming-scheduling.md)：多核之间如何分配工作、动态调度。
- [M04](module-04-locality-coherence-synchronization.md)：多核共享 cache line 的代价与 false sharing。
- [M05](module-05-gpu-architecture-cuda.md)：warp、SIMT 与大规模硬件多线程。

## 本讲小结

吞吐 = 核数 × SIMD 宽度 × 执行单元 × 主频，每一项都要软件配合才能兑现。SIMD 怕发散，多线程怕上下文不够，
多核怕负载不均。带宽受限程序对三者都不敏感，只能减少字节数。ISPC 把 SPMD 抽象映射到 SIMD，
GPU 把同一思想放大到上万个线程。

## 延伸阅读

- Pharr, Mark, ispc: A SPMD Compiler for High-Performance CPU Programming, InPar 2012.
- Olukotun et al., The Case for a Single-Chip Multiprocessor, ASPLOS 1996.
- Intel ISPC User's Guide, https://ispc.github.io/ispc.html
