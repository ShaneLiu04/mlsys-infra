# Module 02 — 多核、SIMD 与多线程

- 课程出处：CS149 Fall 2024《A Modern Multi-Core Processor》《Multi-Core Architecture Part II + ISPC Programming Abstractions》
- 必读论文：Pharr, Mark, ispc: A SPMD Compiler for High-Performance CPU Programming, 2012；Olukotun et al., The Case for a Single-Chip Multiprocessor, 1996
- 对应实验：[CS149 A1 CPU 并行](../solutions/cs149/asst1-cpu-parallel/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst1/cs149_asst1_report.pdf)（实测：12 核动态分配 11.1 倍；每 8 个混 1 个慢 lane，AVX2 只有标量的 0.72 倍）
- 状态：初稿

## 摘要

主频不再上涨之后，处理器的性能从哪里来？答案是三种并行叠加在一起：多核提供线程级并行；
SIMD 让一条指令同时处理多个数据；硬件多线程在一个线程等内存时切到另一个线程，把延迟藏起来。
超标量挖掘的指令级并行已经接近天花板，所以芯片把晶体管从乱序执行和分支预测逻辑挪到了更多的核和更宽的向量单元上。

这三种手段各有软肋。SIMD 怕分支发散：一组 lane 里只要有一条走慢路径，整组都要等它。
硬件多线程不会让单个线程变快，只是让执行单元少闲一会儿。本讲最重要的反例是 saxpy 这类受带宽限制的程序：
加核、加 SIMD、加线程对它都几乎没用，因为瓶颈根本不在计算。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 峰值算力 = 核数 × SIMD 宽度 × 每周期 FMA 单元数 × 2 × 主频。这是几个数相乘，软件少兑现任何一个，就拿不到峰值。
2. SIMD 遇到 if/else 时两条路径都要执行，用 mask 关掉不该写回的 lane。最坏情况下利用率只有 `1/W`。
3. 用硬件多线程隐藏延迟，线程数要满足 `≥ 1 + L/C`，其中 L 是停顿延迟，C 是两次停顿之间的计算周期。
4. ISPC 的抽象是 SPMD，实现是 SIMD：一个 gang 里的每个 program instance 对应一条向量指令的一条 lane。
5. saxpy 每个元素只做 2 FLOP，却要搬约 12–16 字节，算术强度约 0.125–0.17 FLOP/B，多核和 SIMD 都救不了它。

必须能默写的公式：

```text
Peak FLOP/cycle = cores × lanes × FMA_units × 2
SIMD 利用率     = 有效 lane·周期 / (W × 总周期)
发散期望时间    = c_a·(1 − (1−p)^W) + c_b·(1 − p^W)     p = 单 lane 走 a 分支的概率，W = 宽度
延迟隐藏线程数  N ≥ 1 + L / C                            L = 停顿周期，C = 每次停顿前的计算周期
saxpy 字节/元素 = 4(x) + 4(y) + 4(写 out) [+ 4 write-allocate 读 out]
saxpy 强度      = 2 / 16 = 0.125 FLOP/B（有 write-allocate），2 / 12 ≈ 0.17（streaming store）
```

容易混的几对概念：

- 超标量是在一个指令流里同时发射多条不同的指令；SIMD 是一条指令作用于多个数据。
- 多核是多份执行单元同时算；硬件多线程是一份执行单元在多个上下文之间切换。
- interleaved（细粒度）多线程每个周期只从一个线程取指；SMT 在同一个周期从多个线程发射指令。
- explicit SIMD 由编译器或程序员生成向量指令（AVX、ISPC）；implicit SIMD 由硬件把标量线程打包成 warp（GPU）。
- uniform 是整个 gang 共享的一个值；varying 是每个 program instance 各有一个值。

## 学习目标

- 能画出一个核的取指译码单元、执行单元和执行上下文，并解释三种并行分别复制了哪一块；
- 能写出峰值 FLOP/cycle，并解释为什么实际程序拿不到；
- 能分析 SIMD 分支发散的代价，给出最好和最坏的输入；
- 能用 `1 + L/C` 估算需要多少硬件线程才能隐藏内存延迟；
- 能写 ISPC 的 `foreach` 与 `launch`，解释 gang 内 interleaved 赋值为什么更快；
- 能判断一个程序是否受带宽限制，并说明此时加核、加 SIMD 的预期收益。

## 先修知识

- [M01](module-01-performance-models.md) 的 roofline 与 Little's law；
- 基本的 cache 层次：L1 ~4 周期、L3 ~40 周期、DRAM ~200–300 周期；C/C++ 指针与数组。

## 核心概念

### 1. 超标量与 ILP 的天花板

要理解今天的 CPU 为什么长这样，先要看单个指令流里还能挖出多少并行。
如果相邻的指令之间没有依赖，超标量（superscalar）核可以在同一个周期发射多条，这叫指令级并行（ILP）。
乱序执行、分支预测、寄存器重命名，都是为了挖出更多的 ILP。

问题在于真实程序里可挖的 ILP 有限。发射宽度继续加宽，收益很快递减，
而这些控制逻辑占了大量面积和功耗，却只服务一个指令流。于是芯片设计者开始把晶体管挪去做别的事。

### 2. 多核：用面积换线程级并行

第一个去处是多核：把一个复杂的核换成几个简单的核。每个核的单线程性能低一点，但总吞吐随核数增长。
前提是软件要提供多个独立的指令流，也就是线程。

一个没有并行化的程序，在多核机器上只用得到一个核。「多核时代免费午餐结束」说的就是这件事：硬件不再自动让老程序变快。

### 3. SIMD：一份取指，多份 ALU

第二个去处是加宽执行单元。SIMD（单指令多数据）让一份取指和译码驱动 W 条 lane（向量通道），每条 lane 处理一个数据。
AVX2 有 8 个 fp32 lane，AVX-512 有 16 个。取指译码的开销被 W 个数据摊薄，同样的面积就能放下更多 ALU。

代价出在条件分支上。所有 lane 共用一条指令流，遇到 if/else 时不能各走各的。
硬件给每条 lane 一个 mask（掩码）：先执行 if 路径，只写回 mask 为真的 lane；再把 mask 取反，执行 else 路径。两段时间相加。

所以只要有一条 lane 走慢路径，整组都得陪着它。8 条 lane 中只有 1 条走 1000 周期的慢路径时，
其余 7 条就在这 1000 周期里空等，利用率接近 1/8。

### 4. 硬件多线程：用上下文换延迟隐藏

前两招增加的是算力，但算力再多，等内存的时候也只能闲着。硬件多线程针对的就是这个问题。
核内保存多份执行上下文（寄存器和 PC），线程 A 发生 cache miss 后，核在下一个周期就切到线程 B 继续算。
只要可运行的线程足够多，ALU 就总有活干。

这招不是免费的。每个线程分到的寄存器和 cache 变少了；单个线程的延迟也会变长，因为它就绪时不一定马上轮得到。
Intel 的 Hyper-Threading 是 SMT（同时多线程），每核 2 个线程。
GPU 把这一招用到了极致：每个 SM 保存数十个 warp 的上下文（M05）。

### 5. 三者叠加

三种并行是相乘的关系，所以峰值很高，但也很难拿满。以一个 8 核、AVX2、每核 2 个 FMA 单元的 CPU 为例：

```text
8 cores × 8 lanes × 2 FMA × 2 FLOP = 256 FLOP/cycle
```

要拿到这个峰值，几个条件必须同时成立：至少 8 个线程，每核一个；每个线程的内层循环都被向量化，而且没有发散；
有足够多相互独立的 FMA 来覆盖 FMA 本身的延迟；数据都在 L1 里。缺了任何一项，实际性能就按比例往下掉。

### 6. ISPC 编程模型

手写向量 intrinsics 既难写又难读，ISPC 提供了一种更顺手的方式。它采用 SPMD（单程序多数据）的写法：
程序员只写一个 program instance（程序实例）的代码，编译器把 `programCount` 个实例组成一个 gang，映射到一条向量指令的各条 lane 上。
`programIndex` 就是 lane 编号。

变量分两种。`uniform` 变量整个 gang 共享一个值，放在标量寄存器里；`varying`（默认）变量每条 lane 一个值，放在向量寄存器里。
`foreach` 把循环迭代交给编译器去分配到各条 lane。

一个 gang 只在一个核上运行。要用上多核，必须用 `launch` 创建 task，由运行时的线程池去执行。
讲 ISPC 时要把抽象和实现分开说：SPMD 是给程序员的抽象，SIMD 指令是底下的实现。

## 关键推导

### 推导 1：gang 内赋值方式为何影响性能

同样是把 N 个元素分给 8 个 lane，分法不同，访存方式就完全不同。
interleaved 分法让 lane `j` 处理 `j, j+8, j+16, …`，每次迭代 8 个 lane 访问连续的 8 个元素，一条 packed vector load 读 32 字节就够了。
blocked 分法让 lane `j` 处理 `[jN/8, (j+1)N/8)`，每次迭代 8 个 lane 访问的地址相隔 N/8，只能用 gather，每条 lane 可能碰到不同的 cache line。

```text
interleaved:  addr(lane j, iter k) = base + 4·(8k + j)    → 连续 32 B，1 条 vector load
blocked:      addr(lane j, iter k) = base + 4·(j·N/8 + k) → 8 个不同 cache line，gather
```

这和 GPU 上的 memory coalescing（M05）是同一个道理：相邻的 lane 要访问相邻的地址。

### 推导 2：延迟隐藏需要的线程数

硬件多线程到底要几个线程才够？设每个线程算 C 个周期后发出一次访存，停顿 L 个周期。
要让 ALU 不空闲，在一个线程等待的这 L 个周期里，其余线程必须合起来提供 L 个周期的计算：

```text
(N − 1) · C ≥ L   ⇒   N ≥ 1 + L / C
例: L = 250 周期（DRAM），C = 25 周期  → N ≥ 11
    L = 40 周期（L3），   C = 25 周期  → N ≥ 2.6，3 个线程即可
```

线程数受上下文存储的限制。CPU 每核只有 2 个硬件线程，远不够覆盖 DRAM 延迟，所以 CPU 靠大 cache 和预取去降低 L；
GPU 每个 SM 有几十个 warp，靠数量直接覆盖 L。

## 深入推导

### 推导 3：分支发散的期望代价

前面说过最坏情况的利用率是 `1/W`，但实际输入是随机的，平均情况又如何？
设宽度为 W 的 gang 中，每条 lane 独立地以概率 p 走代价为 `c_a` 的分支 a，否则走代价为 `c_b` 的分支 b。
只要 gang 里至少有一条 lane 走 a，整组就要执行 a 段；b 同理。

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

W 越宽，`1 − (1−p)^W` 越接近 1，也就是几乎每个 gang 都会碰上慢路径，利用率随之变差。
A1 程序 2 中向量宽度从 2 增到 16 时 lane 利用率下降，原因就在这里。下面的片段可以直接算出不同宽度下的利用率：

```python
def simd_util(W, p, ca, cb):
    t = ca * (1 - (1 - p) ** W) + cb * (1 - p ** W)
    useful = p * ca + (1 - p) * cb
    return useful / t

for W in (2, 4, 8, 16):
    print(W, round(simd_util(W, 1 / 8, 1000, 10), 3))
```

### 推导 4：saxpy 为何带宽受限

`out[i] = a·x[i] + y[i]` 看起来是最适合并行的程序，却几乎加速不了，原因要从字节数里找。
fp32 下每个元素要读 x 4 B、读 y 4 B、写 out 4 B。普通 store 写一条不在 cache 中的 line 时，
CPU 要先把这条 line 读进来（write-allocate），于是又多出 4 B。

```text
Bytes/elem = 12（streaming store）或 16（write-allocate）
FLOP/elem  = 2
I = 0.17 或 0.125 FLOP/B
```

对照 [M01](module-01-performance-models.md) 的 roofline，CPU 的拐点在每字节几个 FLOP 的量级，saxpy 远在左边的斜线上。
加线程只在单个核的带宽打不满 DRAM 时有用；DRAM 一旦打满，再加核、加 SIMD 都不会变快。
能做的是减少字节：用 `_mm256_stream_ps` 这类 non-temporal store 跳过 write-allocate，把字节数从 16 降到 12，理论上快约 1.33 倍。

下面是 AVX2 版本，主循环每次处理 8 个元素，尾部用标量收尾：

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

同一个 saxpy 用 ISPC 写，可以清楚地看到哪一层给 SIMD 并行、哪一层给多核并行：

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

`foreach` 只提供 SIMD 并行，`launch` 才提供多核并行。task 数取核数的数倍，运行时就能在核之间做负载均衡。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的 CPU 数字。主频不在表中，下面设 `f = 3 GHz` 作为题设。
第 2 题最值得记住：一台峰值几百 GFLOP/s 的机器，跑 saxpy 只能用到百分之一出头。

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

- 「ISPC 的 gang 会自动用满多核」：错。一个 gang 只在一个核上跑，多核要靠 `launch` task。
- 「program instance 就是线程」：错。它是向量指令的一条 lane，没有独立的 PC。
- 「SMT 让单线程更快」：恰好相反。SMT 提高的是整个核的吞吐，单线程可能更慢。
- 「向量宽度越宽越好」：在有发散的代码里，宽度越宽，组里某条 lane 走慢路径的概率越高。
- 「saxpy 用 16 核应该快 16 倍」：少数几个核就把 DRAM 带宽打满了，之后再加核无效。
- 「blocked 赋值更自然，所以更快」：在 gang 内部，blocked 会导致 gather，interleaved 才是连续访存。
  但在多核之间正好反过来，按大块分配更好，因为每个核独占一段连续区域，cache 局部性好。

## 实验或练习

这个练习要验证两件事：Mandelbrot 的负载不均会拖慢块状划分，而 saxpy 无论怎么并行都会撞上带宽墙。

- 假设：Mandelbrot 不同区域的迭代次数差别大，块状划分负载不均；saxpy 受带宽限制。
- 配置：A1 程序 1 用 2–8 个线程，分别做块状划分和 interleaved 划分，记录每个线程的耗时；
  程序 5 比较 ISPC 单核与 ISPC + tasks 的耗时，再用 `N × 16 B / t` 算出带宽。
- 预期：块状划分的加速比不随线程数线性增长，负责中间带的线程最慢；interleaved 划分接近线性。
  saxpy 加 task 的收益远小于核数，换算出的带宽接近机器的内存带宽上限。

## 高频问题

- 问：SIMD 和多核的区别？
  答：SIMD 是一条指令处理多份数据，各 lane 共享取指；多核是多个独立的指令流，各有自己的取指单元。
- 问：硬件多线程解决什么问题？
  答：隐藏访存延迟，提高执行单元的利用率。它不提高单线程速度。
- 问：ISPC 的 `foreach` 和 `launch` 分别给出哪种并行？
  答：`foreach` 把迭代分给一个 gang 的各条 lane，是 SIMD 并行；`launch` 创建 task 分给多个核，是多核并行。
- 问：ISPC 里 `uniform` 和 `varying` 有什么区别？
  答：`uniform` 整个 gang 共享一个值，放在标量寄存器；`varying` 是默认类型，每条 lane 一个值，放在向量寄存器。

## 面试追问链

1. 问：一个现代 CPU 核的峰值 FLOP/cycle 怎么算？
   答：lane 数 × FMA 单元数 × 2。AVX2、2 个 FMA 单元时每核 32 FLOP/cycle，再乘以核数。
2. 问：SIMD 遇到 if/else 怎么办？最坏利用率是多少？
   答：用 mask 把两条路径都执行一遍，耗时相加。只有 1 条 lane 走长路径时，利用率接近 `1/W`。
3. 问：推一下需要多少硬件线程才能隐藏内存延迟。
   答：等待的 L 个周期要由其余线程的计算填满，`(N−1)·C ≥ L`，所以 `N ≥ 1 + L/C`。
   L = 250、C = 25 时要 11 个；CPU 每核只有 2 个，所以 CPU 主要靠 cache。
4. 问：ISPC 里 interleaved 和 blocked 赋值哪个快，为什么？在多核之间呢？
   答：gang 内 interleaved 快，因为相邻 lane 访问相邻地址，一条 packed load 就能读完。
   多核之间用大块，可以减少 cache line 共享和 false sharing。
5. 问：给你一个 saxpy，怎么证明它受带宽限制，又怎么优化？
   答：先算每个元素 16 B、2 FLOP；再测实际带宽，接近 DRAM 峰值就证明了。
   优化只能从减字节入手：streaming store、和前后算子融合、降低精度。
6. 问：GPU 的 warp 和 CPU 的 SIMD 有什么联系？
   答：都是一份取指驱动多条 lane。GPU 是 implicit SIMD：程序员写标量线程，硬件按 32 个一组打包，发散时同样用 mask 串行执行。
7. 问：在 LLM 推理里，哪些算子最像 saxpy？
   答：残差相加、RMSNorm、激活函数、RoPE 这类 elementwise 或 reduction 算子，`I < 1 FLOP/B`。
   它们靠算子融合省掉中间张量的读写，而不是靠更多算力。
8. 问：MoE 路由、beam search 这类数据相关的控制流，放到 GPU 上会遇到什么问题？
   答：同一个 warp 的线程走不同分支、访问不同专家，造成发散和不连续访存。
   常见做法是先按专家或路径排序分组（M06 的 sort 和 scan），再让每组执行一致的代码。

## 自测题

1. 4 核、AVX-512（16 lane）、每核 2 个 FMA 单元，峰值 FLOP/cycle 是多少？
   要点：`4 × 16 × 2 × 2 = 256`。
2. W = 16、p = 1/16、`c_a = 100`、`c_b = 10`，求 SIMD 利用率。
   要点：`1 − (15/16)^16 ≈ 0.644`；`E[T] ≈ 64.4 + 10 = 74.4`；有效 `6.25 + 9.375 = 15.6`；利用率约 21%。
3. 为 A1 程序 4 构造让 ISPC 加速最大和最小的输入，并解释。
   要点：全部取收敛最慢的同一个值，没有发散且计算密集；每 8 个里放 1 个慢的、其余收敛极快，发散最大。
4. 设计题：一个 CPU 推理服务对每个请求做 tokenization 和 embedding 查表，CPU 利用率高但吞吐低，你会先查什么？
   要点：先查是否受内存延迟限制（随机查表带来的 miss），看 IPC 与 cache miss 率；再用批量化、预取、SMT 和数据布局去改进。

## 关联笔记

- [M01](module-01-performance-models.md)：roofline 与 Little's law 是本讲所有估算的基础。
- [M03](module-03-parallel-programming-scheduling.md)：多核之间如何分配工作、动态调度。
- [M04](module-04-locality-coherence-synchronization.md)：多核共享 cache line 的代价与 false sharing。
- [M05](module-05-gpu-architecture-cuda.md)：warp、SIMT 与大规模硬件多线程。

## 本讲小结

吞吐等于核数、SIMD 宽度、执行单元数和主频的乘积，每一项都要软件配合才能兑现：
SIMD 怕发散，多线程怕上下文不够，多核怕负载不均。受带宽限制的程序对这三者都不敏感，唯一的办法是减少字节数。
ISPC 把 SPMD 抽象映射到 SIMD 指令上，GPU 则把同一个思想放大到上万个线程，这是下一步读 M05 的入口。

## 延伸阅读

- Pharr, Mark, ispc: A SPMD Compiler for High-Performance CPU Programming, InPar 2012.
- Olukotun et al., The Case for a Single-Chip Multiprocessor, ASPLOS 1996.
- Intel ISPC User's Guide, https://ispc.github.io/ispc.html
