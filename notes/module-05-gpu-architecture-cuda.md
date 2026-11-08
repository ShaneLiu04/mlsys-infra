# Module 05 — GPU 架构与 CUDA

- 课程出处：CS149 Fall 2024《GPU Architecture and CUDA Programming》 · CS229S Fall 2023《Hardware Aware Algorithm Design (10-04)》中的 GPU 存储层次
- 必读论文：Lindholm et al., NVIDIA Tesla: A Unified Graphics and Computing Architecture, 2008；Volkov, Better Performance at Lower Occupancy, GTC 2010；NVIDIA, H100 Tensor Core GPU Architecture Whitepaper, 2022
- 对应实验：[CS149 A3 scan 与 circle renderer](../solutions/cs149/asst3-scan-render/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst3/cs149_asst3_report.pdf)（实测：RTX 5090 saxpy kernel 1578 GB/s，算上 PCIe 只剩 53.8 GB/s）
- 状态：初稿

## 摘要

M02 讲了 CPU 上的三种并行：多核、SIMD 和硬件多线程。GPU 可以看成把这三招都推到极端的吞吐机器：
一张卡有上百个 SM（流式多处理器，相当于 GPU 的「核」），每个 SM 同时驻留几十个 warp，每个 warp 是 32 条 lane 的隐式 SIMD。
理解了这一点，CUDA 的大部分性能规则都能从 M02 推出来。

CUDA 用 grid、block、thread 三层抽象描述并行。block 之间互相独立，这一点很重要：正因为不依赖 block 的执行顺序，
同一个程序才能在 SM 数量不同的卡上扩展。

写 CUDA 时，性能主要取决于四件事：访存是否合并（coalescing）、shared memory 有没有 bank conflict、
驻留的 warp 够不够隐藏延迟（occupancy），以及计算是否落在 Tensor Core 上。
CS149 A3 的 scan 和 find_repeats 练的是并行原语；circle renderer 练的是在「保序」和「原子性」两个约束下怎么分块。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 层次从大到小是 grid、block、warp、thread。一个 block 最多 1024 个线程，落在同一个 SM 上，共享 shared memory，可以用 `__syncthreads` 同步；
   warp 是 32 个锁步执行的线程。
2. 每个 SM 的资源（A100 和 H100）：64K 个 32 位寄存器（256 KB）、最多 64 个 warp、最多 32 个 block；shared memory A100 是 164 KB，H100 是 228 KB。
3. coalescing：一个 warp 的 32 个 4 B 访问如果落在连续的 128 B 内，只需要 4 个 32 B sector；步长达到 8 个元素以上时，效率只剩 1/8。
4. shared memory 分成 32 个 bank，每个 4 B 宽。同一个 warp 里多个线程访问同一 bank 的不同地址会被串行化；访问同一个地址则是广播，不冲突。
5. Tensor Core 是 GEMM 的主力：A100 上 bf16 有 312 TFLOP/s，而不用 Tensor Core 的 fp32 只有 19.5 TFLOP/s，差 16 倍。

必须能默写的公式：

```text
全局线程号       i = blockIdx.x · blockDim.x + threadIdx.x
寄存器限制       warps_reg  = floor(65536 / (regs_per_thread × 32))
smem 限制        blocks_smem = floor(smem_per_SM / smem_per_block)
occupancy        = min(64, warps_reg, blocks_smem × warps_per_block, 32 × warps_per_block) / 64
合并效率         = 有用字节 / (触及的 32 B sector 数 × 32)
bank 号          bank = (byte_addr / 4) mod 32
延迟隐藏         每调度器 warp 数 ≥ L / C        L = 访存延迟周期，C = 每次访存间可发射的独立指令数
```

容易混的几对概念：

- SIMT 是程序员写标量线程、由硬件打包成 warp；SIMD 是程序员或编译器显式写向量指令。
- block 是调度到 SM 上的资源分配单位；warp 是真正一起取指执行的单位。
- shared memory 是程序员显式管理的 scratchpad；L1 是硬件管理的 cache。二者在同一块 SRAM 上按配置切分。
- occupancy 是驻留 warp 占上限的比例；利用率是执行单元实际忙碌的比例。高 occupancy 不等于高性能。
- `__syncthreads` 是 block 内的 barrier；kernel 边界才是全局 barrier，因为 kernel 内的 block 之间没有通用的同步手段。

## 学习目标

- 能把 CUDA 抽象映射到 SM、warp、寄存器和 shared memory 上；
- 能根据寄存器和 shared memory 的用量手算 occupancy；
- 能判断一个访存模式是否合并、shared memory 访问是否冲突，并给出修复；
- 能用 warp 数和 ILP 解释延迟隐藏；
- 能解释 streams、pinned memory 与 Tensor Core 的作用；
- 能说出 A3 circle renderer 的正确性约束与分块解法。

## 先修知识

- [M01](module-01-performance-models.md) 的 roofline，[M02](module-02-multicore-simd-threads.md) 的 SIMD 发散与硬件多线程；
- C/C++ 指针；
- 数据并行原语的概念（[M06](module-06-data-parallel-primitives.md)）。

## 核心概念

### 1. CUDA 抽象

CUDA 要解决的问题是：怎么让程序员描述「几十万个线程」的并行，又不用关心具体是哪张卡。它的答案是一个三层的抽象。

kernel 用 `kernel<<<grid, block, smem, stream>>>` 启动。grid 由 block 组成，block 由 thread 组成，维度可以是一维、二维或三维。
同一个 block 的线程保证在同一个 SM 上并发执行，因此可以通过 shared memory 交换数据，并用 `__syncthreads()` 同步。

不同 block 的执行顺序和是否并发，CUDA 不做任何保证。这看起来是限制，其实正是扩展性的来源：
同一个程序在 108 个 SM 的 A100 和 132 个 SM 的 H100 上都能跑，硬件按自己的 SM 数去调度 block 就行。
反过来，设计时就不能假设 block 之间能互相等待，否则被等的 block 可能还没被调度上来，程序就死锁了。

### 2. SM 与 warp

抽象落到硬件上是什么样？SM 是 GPU 的「核」。每个 SM 分成 4 个分区，每个分区有一个 warp 调度器，每个周期从就绪的 warp 里挑一个发射指令。

warp 是 32 个线程，共享一份取指，相当于一个 32 lane 的 SIMD 单元。一个 block 被拆成 `ceil(blockDim/32)` 个 warp；
block 大小不是 32 的倍数时，最后一个 warp 会有空闲的 lane。

### 3. SIMT 发散

既然 warp 本质上是 SIMD，M02 讲的分支发散问题在 GPU 上照样存在。同一个 warp 里的线程走不同分支时，
硬件用 active mask 依次执行每一条路径，代价和 CPU 上的 SIMD 发散完全一样。

Volta 之后有了独立线程调度，发散的路径可以交错推进，但同一时刻仍然只有一条路径在执行。
另外，也不能再假设 warp 内的线程会隐式同步，需要时要显式调用 `__syncwarp()`。

发散只发生在 warp 内部。如果分支按 warp 对齐（例如 `if (threadIdx.x / 32 == 0)`），同一个 warp 的线程走的是同一条路，就没有代价。

### 4. Occupancy

GPU 隐藏延迟靠的是「一个 warp 在等，就换另一个 warp 来算」，所以驻留的 warp 越多，可以切换的余地越大。
occupancy 衡量的就是这一点：SM 上驻留的 warp 数除以上限 64。

驻留多少 warp 受三项资源约束：寄存器、shared memory 和每个 SM 的 block 数上限。资源是按整个 block 分配的，
剩下的资源放不下一个完整的 block，就一个也放不下。

但 occupancy 只是隐藏延迟的手段，不是目标。如果每个线程自己就有很多独立的工作（ILP，指令级并行），低 occupancy 也能打满带宽。
GEMM kernel 常常用大量寄存器做分块，occupancy 只有 25% 左右，反而是最优的。

### 5. 访存合并

GPU 的显存带宽很高，但只有在访问模式合适时才拿得到。全局内存以 32 B 的 sector 为单位传输。
一个 warp 执行一条 load 指令时，硬件会把 32 个线程的地址合并成尽量少的 sector 请求。

所以访问模式决定了效率。线程 i 访问 `a[i]` 时，32 个线程正好覆盖连续的 128 B，即 4 个 sector，效率 100%。
访问 `a[2i]` 时要碰 8 个 sector，效率 50%。访问 `a[32i]`（比如按列读一个行主序矩阵）时，32 个 sector 每个只用到 4 B，效率只有 12.5%。

修复办法有两种：调整数据布局（例如把 AoS 改成 SoA，即把「结构体数组」改成「数组的结构体」），
或者先合并地把数据读进 shared memory，再在 shared memory 里按任意顺序访问。

### 6. Shared memory 与 bank conflict

shared memory 是每个 SM 上的片上 SRAM，延迟约 20–30 周期，带宽远高于 HBM，是 GPU 上做数据复用的主要场所。
为了让 32 个线程能同时访问，它被分成 32 个 bank，连续的 4 B 字依次落在相邻的 bank 上。

同一个 warp 的一次访问中，如果 k 个线程访问同一个 bank 的不同地址，这次访问就要分成 k 次完成，叫 k-way bank conflict。
经典的例子是按列读一个 32×32 的 float tile：地址步长是 32 个字，全部落在同一个 bank 上，形成 32-way conflict。
把每行补成 33 个元素，就能把它们错开。

### 7. 延迟隐藏、streams 与 Tensor Core

HBM 的访问延迟约 500 周期。一个 warp 发出 load 之后，调度器马上切到其他就绪的 warp。
这种切换没有开销，因为所有 warp 的上下文都常驻在寄存器文件里，不需要保存和恢复。

在 kernel 之外，还有主机和设备之间的拷贝要藏。streams 是有序的命令队列，不同 stream 里的拷贝和计算可以重叠执行；
要做异步的主机拷贝，主机端必须用 pinned（page-locked，锁页）内存。

最后是算力的来源。Tensor Core 是专做小矩阵乘累加的单元，一条 warp 级指令就完成一个 tile 的 MMA（矩阵乘累加）。
H100 还加入了 TMA（异步搬运 tile 的硬件单元）和 thread block cluster，让 block 之间可以访问彼此的 shared memory。
只有 GEMM 和 attention 这类算子能用上 Tensor Core，其余算子都在普通的 SIMT 单元上跑。

## 关键推导

### 推导 1：occupancy 手算（H100，每 SM 65536 寄存器、64 warp、228 KB smem）

面试里常被要求当场算 occupancy。方法是分别算出寄存器、shared memory 和 block 数上限各允许多少 warp，取最小的那个：

```text
例 1: 每线程 64 寄存器，block = 256 线程（8 warp）
  每 block 寄存器 = 64 × 256 = 16384 → 65536 / 16384 = 4 个 block → 32 warp → 50%
例 2: 每线程 32 寄存器 → 8 个 block → 64 warp → 100%
例 3: 每线程 40 寄存器，block = 1024 线程
  每 block 40960 寄存器 → floor(65536 / 40960) = 1 个 block → 32 warp → 50%
  改成 block = 256: 每 block 10240 → 6 个 block → 48 warp → 75%
例 4: block = 256 线程，每 block 48 KB smem
  floor(228 / 48) = 4 个 block → 32 warp → 50%（忽略每 block 的少量保留空间）
```

例 3 说明大 block 会因为整块分配而浪费资源：剩下的寄存器不够放第二个 1024 线程的 block，只好空着。
实际的编译器按 warp 粒度分配寄存器，所以真实数字会略有出入。

### 推导 2：延迟隐藏需要多少 warp

occupancy 要多高才够？这和 M02 的 `1 + L/C` 是同一个推理：

```text
每个调度器每周期发射一条指令。warp 发出 load 后还有 C 条不依赖该 load 的指令。
要在 L 周期内一直有活干: warps_per_scheduler ≥ L / C
L = 500, C = 40 → 12.5 → 13 个 warp/调度器 × 4 = 52 warp/SM → occupancy 81%
若每线程一次发 4 个独立 load（ILP），等效 C 变为 4 倍 → 13 warp/SM 就够
```

最后一行说明了为什么低 occupancy 也可以很快：靠 ILP 让每个 warp 自己多带几条独立指令，需要的 warp 数就成倍减少。

## 深入推导

### 推导 3：合并访存的有效带宽

把合并效率乘到 HBM 带宽上，就能看出访问模式写错的代价：

```text
H100 HBM 3.35 TB/s
连续:      效率 100% → 3.35 TB/s
步长 2:    50%      → 1.68 TB/s
步长 ≥ 8:  12.5%    → 0.42 TB/s
按列读 4096×4096 行主序 fp32 矩阵: 每次 warp 访问 32 行同一列，32 个 sector，退化到 1/8
```

矩阵转置同时需要按行读和按列写，没法两边都合并，所以要借 shared memory 中转。
标准做法是：block 合并地按行把数据读进 shared memory 的 tile，`__syncthreads()` 之后，再按列从 tile 里读出、合并地写回。

```cuda
#define TILE 32
__global__ void transpose(const float* in, float* out, int n) {
    __shared__ float tile[TILE][TILE + 1];          // +1 列填充，消除 bank conflict
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // 合并读
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];  // 合并写
}
```

### 推导 4：bank conflict 的计算

上面那行 `TILE + 1` 的作用，可以用 bank 号的公式直接算出来：

```text
float tile[32][32]，线程 t 读 tile[t][c]:
  字地址 = 32t + c → bank = (32t + c) mod 32 = c → 32 个线程同一 bank → 32-way，耗时 32 倍
float tile[32][33]，线程 t 读 tile[t][c]:
  字地址 = 33t + c → bank = (t + c) mod 32 → 32 个不同 bank → 无冲突
32 个线程读同一个字（tile[0][c]）: 广播，无冲突
```

### 推导 5：saxpy 与 PCIe

GPU 算得快，并不意味着把任务交给 GPU 就一定快，数据还得先送过去。saxpy 是最好的反例：

```cuda
__global__ void saxpy(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
// saxpy<<<(n + 255) / 256, 256>>>(n, a, d_x, d_y);
```

```text
n = 1e8 fp32: x、y 上传 + y 下载 = 3 × 400 MB = 1.2 GB
PCIe4 x16 32 GB/s: 1.2 / 32 = 37.5 ms
kernel 访存 1.2 GB / 3.35 TB/s = 0.36 ms
→ 端到端时间被 PCIe 主导，约 100 倍。A3 第一部分要求分别测 kernel 时间和含拷贝的时间，结论就是这个比值
```

kernel 本身只要零点几毫秒，拷贝却要几十毫秒。对应到 ML 系统里，权重和 KV cache 必须常驻 HBM；
把它们 offload 到主机内存的方案（M14）必须让 PCIe 传输和计算重叠，否则会慢两个数量级。

### 推导 6：circle renderer 的约束与分块

A3 第三部分给定一串按顺序排列的半透明圆，要求每个像素的颜色等于按输入顺序依次混合的结果。这里有两个约束：

- 原子性：同一个像素的读-改-写不能被打断；
- 保序：如果 i < j，圆 i 必须先于圆 j 作用到像素上。

最直接的写法是按圆并行，每个线程画一个圆，但它两个约束都违反了：多个线程会同时改同一个像素，而且先后顺序不确定。
可行的方案是换个方向，按像素分块：

```text
1. 图像切成 tile（例如 32×32 像素），每个 block 负责一个 tile，每个线程负责一个像素
2. block 内线程分批读取圆，判断与 tile 是否相交，得到 0/1 标志
3. 对标志做 shared memory 内的 exclusive scan，按 scan 结果把相交圆的下标紧凑写入 shared memory 列表（保持原顺序）
4. 每个线程按列表顺序依次混合自己的像素
```

这样每个像素只由一个线程写，原子性自然满足；相交圆的列表由 scan 紧凑生成，顺序与输入一致，保序也自然满足。
这里用到的正是 [M06](module-06-data-parallel-primitives.md) 的 stream compaction 原语。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)。第 1 题解释了为什么小 kernel 在 GPU 上跑不快，这是面试里很常见的追问。

```text
1) 全卡驻留线程上限
   H100: 132 SM × 64 warp × 32 = 270,336；A100: 108 × 64 × 32 = 221,184
   一个只有 1 万个线程的 kernel 只占 H100 驻留能力的 10000 / 270336 ≈ 3.7%，分到每个 SM 不到 3 个 warp
   → 不够隐藏延迟，小 kernel 往往是 latency/overhead-bound

2) 4096³ GEMM，A100
   fp32 SIMT: 1.374e11 / 19.5e12 = 7.0 ms
   bf16 Tensor Core: 1.374e11 / 312e12 = 0.44 ms → 16 倍

3) 合并访存: 一个按列读 fp32 的 kernel 在 H100 上有效带宽 3.35 × 1/8 ≈ 0.42 TB/s
   读 1 GB 需要 2.4 ms，而合并读只要 0.3 ms

4) shared memory 能放多大的 tile（H100，228 KB/SM，bf16）
   两个 128×64 的 tile: 2 × 128 × 64 × 2 B = 32 KB；双缓冲 64 KB → 每 SM 可驻留 3 个这样的 block

5) H2D 带宽差: 主机拷贝 1 GB 走 PCIe4 约 31 ms，同样字节在 HBM 内只需 0.3 ms
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| kernel 启动与合并访存 | [A3](../solutions/cs149/asst3-scan-render/README.md) 第一部分 CUDA saxpy | 分别报告 kernel 带宽与含拷贝带宽，前者接近 HBM、后者接近 PCIe |
| block 内同步与 scan | A3 第二部分 `exclusive_scan` | 任意长度（含非 2 的幂）正确；性能接近参考实现 |
| stream compaction | A3 第二部分 `find_repeats` | 输出下标与参考一致 |
| 保序 + 原子性 | A3 第三部分 circle renderer | 所有场景通过正确性检查；大场景耗时达到参考实现的评分线 |
| shared memory 分块 | 渲染器中按 tile 维护相交圆列表 | 列表由 shared memory scan 生成，保持输入顺序 |

## 易错点与反例

- 「`__syncthreads()` 可以放在 if 里」：如果只有部分线程到达，行为未定义，常见的结果是死锁。
- 「block 之间可以用全局变量自旋等待」：被等的 block 可能还没被调度上来，于是死锁。全局同步要拆成两个 kernel，或者用 cooperative launch。
- 「occupancy 100% 一定最快」：GEMM 用大量寄存器做分块，低 occupancy、高 ILP 反而更快。
- 「访问同一个 shared memory 地址会冲突」：同一地址是广播，不冲突；同一 bank 的不同地址才冲突。
- 「block 越大越好」：推导 1 的例 3 里，1024 线程的 block 只能放一个，occupancy 反而比 256 线程的 block 低。
- 「GPU 算得快，所以 saxpy 放到 GPU 上一定快」：算上 PCIe 拷贝，端到端可能比 CPU 还慢。
- 「warp 内的线程天然同步」：Volta 之后不再保证，依赖这一点的旧代码要加 `__syncwarp()`。

## 实验或练习

这个练习把本讲最重要的两条规则（合并访存和 bank conflict）变成可以测量的曲线。

- 假设：步长访问使有效带宽按 sector 利用率下降；shared memory 转置加上填充后，bank conflict 消失。
- 配置：写一个 kernel，以步长 1、2、4、8、16、32 读 fp32 数组，换算有效带宽；再实现三版转置：不用 shared memory、
  用 shared memory 但不填充、用 shared memory 并填充，用 Nsight Compute 看 `l1tex__data_bank_conflicts` 和 DRAM 吞吐。
- 预期：步长 1 接近 HBM 峰值，步长 8 以后降到约 1/8 并保持不变；三版转置依次变快，填充版本的 bank conflict 计数接近 0。

## 高频问题

- 问：warp 有多少线程？
  答：32。
- 问：什么是 occupancy，受哪些资源限制？
  答：驻留 warp 数占上限（64）的比例；受寄存器、shared memory 和每个 SM 的 block 数上限约束。
- 问：什么是 memory coalescing？
  答：一个 warp 的访问落在连续地址上时，硬件把它们合并成最少的 32 B sector 请求。
- 问：一个 warp 的线程都读 shared memory 的同一个地址，会有 bank conflict 吗？
  答：不会。同一地址是广播；只有同一 bank 的不同地址才会冲突。
- 问：哪些算子能用上 Tensor Core？
  答：GEMM 和 attention 这类以矩阵乘为主的算子；其余算子在普通的 SIMT 单元上跑。

## 面试追问链

1. 问：CUDA 的 grid、block、warp、thread 分别对应什么硬件？
   答：grid 对应整张卡；block 被调度到一个 SM 上，分配 shared memory 与寄存器；warp 是调度器一次发射的 32 线程组；thread 对应一条 lane。
2. 问：一个 kernel 每线程用 64 个寄存器、block 256 线程，H100 上 occupancy 多少？
   答：每个 block 要 16384 个寄存器，一个 SM 能放 4 个 block，共 32 个 warp，occupancy 50%。
3. 问：occupancy 低就一定慢吗？
   答：不一定。延迟可以靠 warp 数隐藏，也可以靠 ILP 隐藏。GEMM kernel 每个线程算多个输出、用很多寄存器，occupancy 低，却能打满 Tensor Core。
4. 问：按列读一个行主序矩阵，有效带宽损失多少？怎么修？
   答：每个线程碰到不同的 sector，效率只有 1/8。先合并地按行读进 shared memory，再按列从 shared memory 读，同时加填充避免 bank conflict。
5. 问：32×32 的 shared memory tile 按列访问为什么慢？算一下。
   答：地址是 `32t + c`，bank 全都是 `c`，32-way conflict，耗时 32 倍。改成 `[32][33]` 后 bank 变成 `(t + c) mod 32`，没有冲突。
6. 问：A3 的 circle renderer 怎么同时满足保序和原子性？
   答：按像素 tile 分 block，block 内用 shared memory scan 紧凑出相交圆的列表并保持顺序，每个线程独占一个像素、按顺序混合。
7. 问：为什么 FlashAttention 要用 shared memory 分块？
   答：标准 attention 要把 `N×N` 的分数矩阵写回 HBM，是 memory-bound 的。分块之后，Q、K、V 的 tile 在 shared memory 里完成 softmax 和乘法，
   HBM 流量从 O(N²) 降到 O(N²d²/M)（M 为 SRAM 大小），把算子从带宽受限推向算力受限。
8. 问：LLM decode 阶段 batch 小，GPU 利用率低，从架构角度怎么解释？有哪些办法？
   答：每一步每一层的 GEMV 只有少量 block，喂不满 132 个 SM；而且 `I ≈ batch`，远低于拐点 295。
   办法有：增大 batch（continuous batching）、用 split-K 让更多 block 参与、用 CUDA Graphs 去掉 launch 开销、量化权重减少字节。

## 自测题

1. A100 上一个 kernel 每线程 48 个寄存器、block 128 线程、每 block 16 KB smem，occupancy 是多少？
   要点：寄存器 `65536 / (48 × 128) = 10.67 → 10` 个 block；smem `164 / 16 = 10.25 → 10` 个 block；block 上限 32；
   10 × 4 = 40 warp，占 64 的 62.5%。
2. warp 中线程 t 访问 `a[3t]`（fp32），会碰到几个 32 B sector？
   要点：字节跨度 `3 × 31 × 4 + 4 = 376 B`，约 12 个 sector，效率约 1/3。
3. 为什么异步的 `cudaMemcpyAsync` 需要 pinned memory？
   要点：DMA 引擎需要物理地址固定的缓冲区；pageable 内存会被驱动先拷到内部的 pinned 缓冲区，于是失去了异步性。
4. 设计题：一个 RMSNorm kernel，每行 4096 个 bf16，batch × seq = 16384 行。怎么分配线程？
   要点：每行一个 block 或一个 warp；每个线程用向量化 load 读连续的 16 B；用 warp shuffle 做归约；整体是 memory-bound，目标是接近 HBM 带宽。
5. 解释为什么「不同 block 的执行顺序不确定」反而是好设计。
   要点：硬件可以按 SM 数量和资源自由调度，同一个程序能跨代扩展；代价是全局同步要拆成多个 kernel。

## 关联笔记

- [M02](module-02-multicore-simd-threads.md)：SIMD 发散与硬件多线程是 warp 的原型。
- [M04](module-04-locality-coherence-synchronization.md)：原子操作争用与私有化。
- [M06](module-06-data-parallel-primitives.md)：scan、compaction 在 CUDA 上的实现。
- [M07](module-07-matmul-hardware-aware.md)：用 shared memory 与寄存器分块写 GEMM。
- [M08](module-08-attention-flashattention.md)：FlashAttention 的 SRAM 分块。

## 本讲小结

GPU 用海量驻留的 warp 隐藏延迟，用 Tensor Core 提供算力。一个 CUDA 程序快不快，取决于访存是否合并、有没有 bank conflict、
occupancy 和 ILP 够不够隐藏延迟、是否用上了 Tensor Core，这几项都能手算。
block 互相独立是可扩展性的来源，也意味着全局同步要拆成多个 kernel。
A3 的渲染器展示了一个通用套路：用分块加 scan，把「保序 + 原子性」的问题变成每个线程独占一份输出的问题。

## 延伸阅读

- Lindholm et al., NVIDIA Tesla: A Unified Graphics and Computing Architecture, IEEE Micro 2008.
- Volkov, Better Performance at Lower Occupancy, GTC 2010.
- NVIDIA, CUDA C++ Programming Guide 与 CUDA C++ Best Practices Guide, https://docs.nvidia.com/cuda/
- NVIDIA, H100 Tensor Core GPU Architecture Whitepaper, 2022.
- Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022.
