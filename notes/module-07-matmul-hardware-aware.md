# Module 07 — 矩阵乘、分块与硬件感知算法

- 课程出处：CS149 Fall 2024《Efficiently Evaluating DNNs》 · CS229S Fall 2023《Hardware Aware Algorithm Design》(10-04) · CMU 15-849《ML Hardware》
- 必读论文：Hong & Kung，I/O Complexity: The Red-Blue Pebble Game，1981；Lavin & Gray，Fast Algorithms for Convolutional Neural Networks（Winograd），2016；Chowdhery et al.，PaLM（MFU 定义），2022
- 对应实验：[CS149 A4 DNN 加速器](../solutions/cs149/asst4-dnn-accelerator/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst4/cs149_asst4_report.pdf)（实测：SGEMM 0.94 → 49.4 TFLOP/s（cuBLAS 的 78%））
- 状态：初稿

## 摘要

打开任何一个深度网络的 profiler，你会发现时间大头都花在矩阵乘（GEMM）上：九成以上的 FLOPs 落在这里。
所以「算子快不快」基本等于「GEMM 快不快」。这一讲要回答的问题是：同样是 `2MNK` 次浮点运算，
为什么朴素的三重循环和 cuBLAS 能差出几十倍？配套实验里，一个 SGEMM 就是从 0.94 一步步优化到 49.4 TFLOP/s 的。

答案在数据复用。GEMM 要做 `2MNK` 次运算，涉及的数据却只有 `MK+KN+MN` 个元素。
计算量是三次方，数据量是二次方，所以每个元素理论上可以被用很多次。
这份潜力能不能兑现，取决于分块（tiling：把矩阵切成小块，让一块数据留在快存里被反复使用）。
tile 边长为 `b` 时，每读一个元素可以做 `b` 次乘加，因此算术强度和 tile 边长成正比。
但 tile 能开多大受快存容量限制，由这个约束可以推出通信量的下界 `Ω(N³/√M)`。

落到 GPU 上，分块分成三层：寄存器、共享内存、全局内存，最内层的形状由 Tensor Core 决定。
卷积可以用 im2col 改写成 GEMM；Winograd 和 FFT 则是用更多的加法换更少的乘法。

贯穿全讲的取舍有两条。第一，tile 越大强度越高，但它占用的寄存器和共享内存也越多，能同时运行的 CTA 越少，并行度就下降。
第二，batch 越小，GEMM 越接近 GEMV（矩阵乘向量），也就越受访存限制。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. GEMM `C[M,N] = A[M,K]·B[K,N]` 的 FLOPs 是 `2MNK`，因为一次乘加算 2 个 FLOP。bf16 方阵的算术强度约为 `n/3`。
2. 分块之后，一个 tile 的强度约为 `b·b'/(b+b')` FLOP/元素。方 tile 时是 `b/2` FLOP/元素，和 tile 边长成正比。
3. 快存容量 `M` 把 tile 边长限制在 `b ≈ √(M/3)`，于是总通信量是 `Θ(N³/√M)`。这正好是 Hong-Kung 下界，所以分块在渐近意义上已经最优。
4. 当 `M` 很小（比如 decode 时的 batch），GEMM 退化成 GEMV，强度约等于 batch 大小（FLOP/B）。在 H100 上，batch < 295 都受访存限制。
5. MFU = 模型理论 FLOPs / (时间 × 峰值)。HFU 额外把重计算也算进去，所以 HFU ≥ MFU。

**必须能默写的公式**

```text
FLOPs(GEMM)        = 2·M·N·K
Bytes(一次搬运)    = s·(M·K + K·N + M·N)              s = 每元素字节数
I(方阵 n, bf16)    = 2n³ / (2·3n²) = n/3  FLOP/B
I(tile BM×BN)      = BM·BN / (BM+BN)      FLOP/B（bf16，忽略 C 的写回）
通信量(分块)       = 2N³/b 元素，b ≈ √(M/3)  →  Θ(N³/√M)
GEMV(batch B)      I ≈ 2BNK / (2·NK) = B  FLOP/B（权重主导字节）
conv→GEMM          M' = N·P·Q,  N' = K_out,  K' = C·R·S
MFU = 6·N_param·tokens_per_s / (n_gpu · peak)；HFU 用 8N（全量重计算）
```

**一句话区分**

- MFU 和 HFU：MFU 只算模型「必须」做的 FLOPs；HFU 把重计算也算上，衡量的是硬件实际被用了多少。
- im2col 和 Winograd：im2col 不减少 FLOPs，只是把数据换成 GEMM 能吃的布局；Winograd 是真的减少了乘法次数。
- 寄存器 tile 和共享内存 tile：前者决定每条指令能复用多少数据，后者决定每次从 HBM/L2 读进来的数据能复用多少次。

## 学习目标

- 从 `2MNK` 和数据量推出 GEMM 的算术强度，并判断给定形状在 A100/H100 上受哪个屋顶限制；
- 推导分块后的强度与 tile 边长成正比，以及 `Θ(N³/√M)` 的通信量；
- 画出 GPU 上「grid → CTA tile → warp tile → MMA 指令」的三层分块，并说出每层的容量约束；
- 把卷积改写成 GEMM，算出 im2col 的显存放大倍数；
- 正确定义 MFU/HFU，并用它估算训练吞吐。

## 先修知识

- [M01](module-01-performance-models.md) 的 roofline 与拐点；
- [M05](module-05-gpu-architecture-cuda.md) 的 SM、warp、共享内存、coalescing。

## 核心概念

### 1. GEMM 的计算量与最小数据量

判断一个算子能跑多快，第一步是数清楚它要算多少、至少要搬多少。GEMM 的核心是 `C[i,j] += A[i,k]·B[k,j]`，
三重循环一共 `M·N·K` 次乘加，也就是 `2MNK` FLOPs。三个矩阵每个元素至少要碰一次，所以最少的数据量是 `MK+KN+MN` 个元素。

计算是三次方、数据是二次方，因此矩阵越大，每个字节摊到的计算越多，大 GEMM 天然是计算受限的。
举个例子：`n=4096` 的 bf16 方阵，强度是 `4096/3 ≈ 1365 FLOP/B`，远高于 H100 的拐点 295。
但要注意，这个值假设「每个元素只从 HBM 读一次」，是理想情况。朴素的三重循环离它很远，下一节解释原因。

### 2. 朴素循环为什么慢

按 `for i, for j, for k` 写，算一个 `C[i,j]` 要读 A 的一行和 B 的一列。
矩阵按行主序存放时，B 的一列在内存里是跨步分布的，每读一个 fp32 就浪费掉一整条 cache line。

更深层的问题是复用距离（同一个数据两次被使用之间隔了多少别的访问）。
B 的同一个元素要等算完一整行 C 才会被再次用到，而在这期间它早已被挤出了 cache。
结果是每次乘加都要从内存读约 2 个元素，强度只有约 0.25 FLOP/B（fp32），和上一节的 1365 差了几个数量级。

### 3. 分块（tiling / blocking）

既然问题出在复用距离太长，解决办法就是重排计算顺序，让一小块数据在被挤出之前用完。

具体做法是把 C 切成 `b×b` 的块，把 K 维切成长度为 `bk` 的段。对一个 C 块，每一步读入 A 的 `b×bk` 和 B 的 `bk×b`，
做 `2·b·b·bk` FLOPs，所以每读一个元素做 `2b²bk/(2b·bk) = b` FLOPs。
C 块一直常驻在快存（寄存器）里，直到 K 维全部累加完才写回。

这里有一个容易想错的地方：强度只取决于 C 块的边长 `b`，和 `bk` 无关，因为 `bk` 在分子分母里被约掉了。
`bk` 影响的是快存里 A、B 缓冲区的大小和流水线的深度。

### 4. GPU 的三层分块

GPU 的存储是分层的，每一层的容量和带宽都不同，所以分块也要一层套一层：
每一层的 tile 负责减少对上一层存储的读取。下表从外到内列出这三层。

| 层级 | 存放什么 | H100 容量约束 | 复用对象 |
| --- | --- | --- | --- |
| CTA tile（如 128×128×64） | A/B 的 tile，多级缓冲 | 共享内存 ≤ 228 KB/SM | 减少 HBM/L2 读取 |
| warp tile（如 64×64） | 从共享内存读入寄存器的片段 | 寄存器 256 KB/SM | 减少共享内存读取 |
| MMA 指令 | Tensor Core 一次算的小块 | 固定形状 | 一条指令内的乘加 |

算一下就知道 tile 为什么不能无限放大。一个 128×128 的 fp32 累加器是 64 KB，已经占掉寄存器文件的四分之一。
A、B 两个 128×64 的 bf16 tile 各 16 KB，做三级缓冲就要 96 KB 共享内存。
再放大的话，每个 SM 只能驻留一个 CTA，就没有别的 CTA 能在它等数据时顶上来掩盖延迟了。

### 5. Tensor Core 与 MMA 形状

Tensor Core 是固定形状的矩阵乘加单元：它一次只能算一个特定尺寸的小矩阵块，这个形状决定了最内层 tile 长什么样。

各代的形状不同。Volta 的 `HMMA` 由 m8n8k4 组合而成；Ampere 的 `mma.sync` 由一个 warp 发出，bf16 常用 m16n8k16；
Hopper 的 `wgmma` 由 4 个 warp 组成的 warpgroup 发出，形状是 m64nNk16（N 最大 256），而且操作数可以直接从共享内存读取。

形状固定带来的直接后果是：K 维或 N 维不是 8/16 的倍数时要做 padding，维度对齐到 64/128 才能把 Tensor Core 吃满。
为什么一定要用上 Tensor Core？因为不走 Tensor Core 的 fp32 在 A100 上只有 19.5 TFLOP/s，差了一个数量级。

### 6. 卷积变 GEMM：im2col

GEMM 已经被优化得这么好，最省事的做法就是把卷积也变成 GEMM。

设输入是 `[N,C,H,W]`，卷积核是 `[K,C,R,S]`，输出是 `[N,K,P,Q]`。
im2col 把每个输出位置对应的 `C·R·S` 个输入元素展开成一行，得到一个 `(N·P·Q) × (C·R·S)` 的矩阵，再乘以 `(C·R·S) × K` 的权重矩阵。
FLOPs 没有变化，仍是 `2·N·K·P·Q·C·R·S`。代价在显存：stride 为 1 时，每个输入元素被复制了约 `R·S` 次。
cuDNN 的 implicit GEMM 就是为了避开这个代价：它在加载 tile 时按需计算地址，并不真正把这个大矩阵物化出来。

### 7. Winograd 与 FFT

im2col 只改布局不省计算；如果想真正减少乘法，就要用代数变换。

Winograd `F(m×m, r×r)` 用 `(m+r−1)²` 次乘法算出 `m×m` 个输出。
例如 `F(2×2,3×3)` 用 16 次乘法代替 36 次，省 2.25×；`F(4×4,3×3)` 用 36 次代替 144 次，省 4×。
代价是变换过程中多出来的加法和数值误差，而且 tile 越大误差越大，在低精度下尤其明显。

FFT 卷积把复杂度降到 `O(n log n)`，但只有卷积核很大时才划算；在 3×3 的小核上不如 Winograd。

### 8. 层融合

GEMM 后面经常跟着 bias、GeLU、残差加法这类逐元素运算。如果分开跑，中间结果 `M×N` 要先写回 HBM，再被下一个 kernel 读出来。
这些逐元素算子的强度不到 1 FLOP/B，单独执行时永远受访存限制，算得再快也没用。

epilogue fusion（在 GEMM 的收尾阶段顺手做完后续运算）解决的就是这个问题：
趁累加器还在寄存器里，直接把逐元素运算做完，最后只写一次 HBM。对这类算子来说，融合是唯一的出路。

## 关键推导

### 推导 1：分块的通信量与 `√M`

这个推导回答的是：给定一块容量为 `M` 的快存，分块最多能把 HBM 流量压到多少？

设快存能放 `M` 个元素，使用 `b×b` 的 C 块、`b×bk` 的 A 块和 `bk×b` 的 B 块，三者要同时放得下，即 `b² + 2b·bk ≤ M`。
取 `bk≈b`，得到 `b ≈ √(M/3)`。然后逐项数读入量：

```text
C 块个数             (N/b)²
每块的 K 步数        N/bk
每步读入             2·b·bk 元素
总读入 Q             (N/b)² · (N/bk) · 2·b·bk = 2N³/b
代入 b = √(M/3)      Q = 2√3 · N³/√M = Θ(N³/√M)
强度                 2N³ FLOPs / Q = b  FLOP/元素
```

Hong-Kung 下界告诉我们，任何经典的 `O(N³)` 矩阵乘，通信量都至少是 `Ω(N³/√M)`。所以分块和最优解只差一个常数。
有两个边界情况：`b` 不能超过 N；当 `M ≥ 3N²` 时整个问题都装得进快存，通信量退化为读一遍数据的 `3N²`。

### 推导 2：GEMV 为什么访存受限

decode 时每个请求一次只出一个 token，所以 `M=B` 很小。这时 FLOPs 是 `2BNK`，数据量是 `s(BK + KN + BN)`。
因为 `B ≪ K,N`，字节数几乎全部来自权重 `KN`：

```text
I ≈ 2BNK / (2·KN) = B   FLOP/B（bf16）
```

H100 的拐点是 295，所以 decode 的 batch 要到约 295 才会变成计算受限。
这和 `tools/mlsys_calc.py` 里的 `critical_batch = ridge·bytes/2` 一致。
换个角度看，batch=1 时 Tensor Core 有 99% 的时间在等数据。

## 深入推导

### 从 HBM 到 SM：一个 CTA tile 需要多少带宽

上面的推导把快存看成一整块。真实 GPU 上，一个 CTA tile 的强度够不够喂饱 Tensor Core？算一下就知道。

设 CTA tile 为 `BM×BN`，K 段长 `bk`。每一步读入 `(BM+BN)·bk` 个 bf16，计算 `2·BM·BN·bk` FLOPs：

```text
I_CTA = 2·BM·BN·bk / (2·(BM+BN)·bk) = BM·BN/(BM+BN)  FLOP/B
128×128 → 64 FLOP/B；128×256 → 85 FLOP/B；256×256 → 128 FLOP/B
H100 满速需要的 L2→SM 带宽 = 989e12 / 64 ≈ 15.5 TB/s（128×128）
                              989e12 / 85 ≈ 11.6 TB/s（128×256）
```

这些强度都低于 HBM 拐点 295。这说明只靠「每个 CTA 各自从 HBM 读一次」是不够的，必须依赖 L2 命中：
相邻的 CTA 会用到 A 的同一个行块或 B 的同一个列块，第二次读就能从 L2 拿到。

所以高性能 GEMM 会做 CTA 光栅化 swizzle，也就是调整 CTA 的发射顺序，让同时运行的 CTA 访问相近的块。
Hopper 更进一步，用 cluster 内的 TMA multicast 把同一个 tile 广播给 2 个 CTA，相当于把 L2 读取再减半。
这也解释了为什么 H100 上主流的 tile 是 128×256，而不是 64×64。

### 代码：两级分块的 CUDA 骨架

下面这个 SGEMM 骨架同时做了两级分块：外层用共享内存，内层用寄存器。

```cuda
// C[M,N] = A[M,K] * B[K,N]，fp32，每个 CTA 算 BM×BN，每线程算 TM×TN
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[BK][BM];             // 转置存放，便于按列读
  __shared__ float Bs[BK][BN];
  float acc[TM][TN] = {0.f};               // 寄存器 tile：复用 TM·TN 次
  int tx = threadIdx.x % (BN / TN), ty = threadIdx.x / (BN / TN);
  const float* Ab = A + blockIdx.y * BM * K;
  const float* Bb = B + blockIdx.x * BN;
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < BM * BK; i += blockDim.x)
      As[i % BK][i / BK] = Ab[(i / BK) * K + k0 + i % BK];
    for (int i = threadIdx.x; i < BK * BN; i += blockDim.x)
      Bs[i / BN][i % BN] = Bb[(k0 + i / BN) * N + i % BN];
    __syncthreads();
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
      for (int m = 0; m < TM; ++m) a[m] = As[kk][ty * TM + m];
      for (int n = 0; n < TN; ++n) b[n] = Bs[kk][tx * TN + n];
      for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n) acc[m][n] += a[m] * b[n];   // TM+TN 次读，TM·TN 次 FMA
    }
    __syncthreads();
  }
  for (int m = 0; m < TM; ++m)
    for (int n = 0; n < TN; ++n)
      C[(blockIdx.y * BM + ty * TM + m) * N + blockIdx.x * BN + tx * TN + n] = acc[m][n];
}
```

两级各管一件事。外层的共享内存 tile 把 HBM 读取降到原来的 `1/BM` 和 `1/BN`。
内层的寄存器 tile 把共享内存读取从每次 FMA 2 次降到 `(TM+TN)/(TM·TN)` 次，`TM=TN=8` 时只有 0.25 次。
如果不做内层这一级，共享内存带宽会先成为瓶颈。

真正的 Tensor Core kernel 会把最内层换成 `mma`/`wgmma` 指令，并用 `cp.async` 或 TMA 做多级流水，让数据加载和计算重叠起来。

### MFU 的定义细节

MFU 是衡量训练效率最常用的数字，所以分子到底算哪些 FLOPs 必须说清楚。

训练一个 token 的模型 FLOPs 是 `6N`：前向 2N，反向 4N。长上下文时，PaLM 附录还会再加上注意力这一项，
每 token `12·L·d·T`（L 是层数，d 是模型宽度，T 是序列长度，按非因果计算）。
MFU 用的是 `6N`。如果开了全量激活重计算，前向要多跑一遍，硬件实际做的是 `8N`，这时 HFU = `8N·tps/peak`。

报告 MFU 的好处是它和实现方式无关。重计算做得越多，HFU 越好看，但 MFU 不会因此被刷高。

## 白板估算

下面用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 里的 H100 SXM 参数：bf16 989 TFLOP/s，HBM 3.35 TB/s，拐点 295。
四道题分别对应大方阵、decode、卷积和 MFU 反推。

```text
1) 方阵 GEMM n=4096，bf16
   FLOPs = 2·4096³ = 1.374e11
   Bytes = 2·3·4096² = 1.007e8 B ≈ 101 MB
   I = 1.374e11 / 1.007e8 ≈ 1365 FLOP/B  > 295  → 计算受限
   t_compute = 1.374e11 / 989e12 ≈ 0.139 ms；t_mem = 1.007e8 / 3.35e12 ≈ 30 µs
   方阵 I = n/3 = 295 → n ≈ 885：小于约 900 的方阵在 H100 上受访存限制

2) Llama-3-8B decode，batch=1，一层 MLP 的 up 投影 [1,4096]×[4096,14336]
   FLOPs = 2·4096·14336 = 1.17e8；权重 4096·14336·2 B = 117 MB
   I ≈ 1 FLOP/B；t = 117e6 / 3.35e12 ≈ 35 µs，而 t_compute ≈ 0.12 µs，差约 300 倍
   batch=256：I ≈ 256，仍略低于 295；batch≈300 才到拐点

3) 3×3 卷积 im2col，N=32, C=K=64, H=W=P=Q=56, bf16
   FLOPs = 2·32·64·56·56·64·9 ≈ 7.40e9
   im2col 矩阵 (32·3136)×(64·9) = 100352×576 ≈ 5.78e7 元素 ≈ 116 MB
   原输入 32·64·56·56·2 B ≈ 12.8 MB → 放大 9 倍（= R·S）

4) MFU 反推吞吐：8B 模型，8×H100，MFU 40%
   tokens/s = 0.4 · 8 · 989e12 / (6 · 8e9) = 3.165e15 / 4.8e10 ≈ 6.6e4 token/s
```

第 2 题最值得记住：同样一层 MLP，读权重的时间约是计算时间的 300 倍，这就是 decode 慢的根源。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| GEMM 强度与 roofline | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `gemm_intensity`、`roofline_time` | 方阵 4096 bf16 得 ≈1365 FLOP/B |
| decode 临界 batch | 同文件 `critical_batch` | H100 输出 ≈ 295 |
| 分块矩阵乘与卷积映射 | [CS149 A4](../solutions/cs149/asst4-dnn-accelerator/README.md) | tile 后的结果与参考实现一致，记录不同 tile 的耗时 |
| MFU | 同文件 `mfu` | 与手算 6N·tps/peak 一致 |

## 易错点与反例

- 常见错答：「GEMM 的 FLOPs 是 MNK」。乘和加各算一次，所以是 `2MNK`；厂商标的峰值也是按一次 FMA 算 2 FLOPs。
- 常见错答：「矩阵够大就一定计算受限」。`[1,4096]×[4096,14336]` 有 1.17e8 FLOPs，看起来不小，但 M=1，强度只有 1。
- 「tile 越大越好」不对。寄存器一旦溢出到 local memory，性能会断崖式下跌；tile 太大还会让 CTA 数少于 SM 数（wave quantization），最后一波有 SM 空转。
- 把强度写成 `b` 却忘了单位。`b` 的单位是 FLOP/元素，换成字节要除以元素大小：bf16 下是 `b/2` FLOP/B，fp32 下是 `b/4` FLOP/B。
- 以为 im2col 减少了计算。它只改变布局，FLOPs 一点没少，显存反而多了 `R·S` 倍。
- Winograd 在 int8/fp16 下误差会放大，所以量化推理里常常禁用大 tile 的 Winograd。
- MFU 的分子用了实测 FLOPs（含重计算），这样算出来的其实是 HFU。

## 实验或练习

这个练习用来亲手验证「小矩阵受访存限制、大矩阵受计算限制」。

- 假设：GEMM 吞吐随方阵边长先上升后走平，拐点在 n ≈ 900 附近（H100）或 n ≈ 460（A100，153·3）。
- 配置：`torch.matmul` 在 bf16 下扫 n ∈ {128, 256, 512, 1024, 2048, 4096, 8192}，记录 TFLOP/s；再固定 N=K=4096，扫 M ∈ {1, 8, 64, 512}。
- 预期：大 n 时吞吐接近峰值的 60%–80%；在 M 扫描中，吞吐大致随 M 线性增长，直到接近拐点。具体数值以自己机器的实测为准。

## 高频问题

- 问：GEMM 的算术强度怎么算？
  答：`2MNK / (s·(MK+KN+MN))`；bf16 方阵约为 `n/3`。
- 问：im2col 减少了卷积的计算量吗？
  答：没有。它只是把输入展开成 GEMM 的布局，FLOPs 不变，显存还多了约 `R·S` 倍；implicit GEMM 通过按需算地址避开这份显存。
- 问：Winograd 能省多少乘法，代价是什么？
  答：`F(2×2,3×3)` 省 2.25×，`F(4×4,3×3)` 省 4×；代价是额外的加法和数值误差，tile 越大、精度越低，误差越明显。
- 问：bias、GeLU 这类逐元素算子为什么要融合进 GEMM？
  答：它们的强度不到 1 FLOP/B，单独跑时时间全花在把 `M×N` 中间结果写回又读出上；epilogue fusion 在寄存器里做完，只写一次。
- 问：为什么报告训练效率时用 MFU 而不是 HFU？
  答：MFU 的分子只算模型必须做的 `6N`，和实现无关；HFU 会因为重计算变多而变好看，不能用来比较不同实现。

## 面试追问链

1. 问：一个 `[M,K]×[K,N]` 的 GEMM 有多少 FLOPs，最少要搬多少字节？
   答：`2MNK` FLOPs，最少 `s(MK+KN+MN)` 字节。二者之比就是理想强度，bf16 方阵为 `n/3`。
2. 问：朴素三重循环离这个理想强度差在哪里？
   答：复用距离太长。B 的元素在被再次使用前就已经被挤出 cache，每次乘加都要访问内存，强度掉到常数级。
3. 问：那分块之后通信量是多少？推导一下。
   答：C 块 `b×b`、K 步长 `bk`，总读入 `2N³/b`；快存约束给出 `b≈√(M/3)`，所以是 `Θ(N³/√M)`，与 Hong-Kung 下界同阶。
4. 问：既然 tile 越大强度越高，GPU 上为什么常用 128×128 或 128×256，而不是 512×512？
   答：128×128 的 fp32 累加器已经占 64 KB 寄存器，512×512 要 1 MB，远超每个 SM 的 256 KB；而且 CTA 数会太少，掩盖不了延迟。
5. 问：128×128 的 CTA tile 强度只有 64 FLOP/B，低于拐点，为什么 cuBLAS 还能跑到很高的利用率？
   答：这个强度是相对 L2 而言的。同时运行的 CTA 共享 A/B 块，L2 命中让 HBM 的实际流量远低于 CTA 的读取量；swizzle 和 TMA multicast 就是为了提高这种跨 CTA 的复用。
6. 问：decode 阶段 batch=1，怎样让 GEMM 不浪费 Tensor Core？
   答：增大 batch（continuous batching）；把权重量化以减少字节数（int4 权重让强度约 ×4）；用推测解码让一次前向处理多个 token。单个请求的延迟下界是读一遍权重的时间。
7. 问：你要为一个 3×3 卷积密集的新视觉模型选 kernel 方案，implicit GEMM 和 Winograd 怎么权衡？
   答：fp16/bf16、通道数大时，Winograd F(2,3) 省 2.25× 乘法，但增加了变换开销和误差；int8 量化部署倾向 implicit GEMM。先用 roofline 判断这一层是否计算受限，访存受限的层用 Winograd 也没用。
8. 问：你在 8×H100 上训练 8B 模型，只跑到 25% MFU，怎么排查？
   答：先看计算受限的 GEMM 是否太小（micro-batch 太小、序列太短、TP 切得太碎导致 N/K 太小）；再看逐元素算子有没有融合；再看通信是否没有和计算重叠；最后看 dataloader 和 Python 开销。每一步都用 profiler 的 kernel 时间占比来验证。

## 自测题

1. A100 bf16 下，方阵 GEMM 的 n 低于多少会受访存限制？
   要点：拐点 153，`n/3 = 153`，所以 n ≈ 460。
2. 证明方 tile 的强度与 `bk` 无关。
   要点：`2b²bk / (2b·bk) = b`，`bk` 被约掉；它只影响缓冲区大小和流水深度。
3. 一个 `[2048, 4096]×[4096, 4096]` 的 bf16 GEMM，在 H100 上的下界时间是多少？
   要点：FLOPs = 6.87e10，bytes = 2·(8.39e6+1.68e7+8.39e6) ≈ 67 MB；强度 ≈ 1024 > 295，计算受限；时间 ≈ 69 µs。
4. 推导 im2col 的矩阵形状与放大倍数。
   要点：形状是 `(N·P·Q)×(C·R·S)`；stride 1、same padding 时放大约 `R·S` 倍。
5. 设计题：给 Llama-3-8B decode 写一个 batch=4 的 GEMM，tile 怎么选？
   要点：M 维只有 4，应沿 N/K 切分以得到足够多的 CTA（split-K）；优化目标是 HBM 带宽，而不是 Tensor Core。

## 关联笔记

- [M01 性能模型与 roofline](module-01-performance-models.md)：拐点与强度的定义。
- [M05 GPU 架构与 CUDA](module-05-gpu-architecture-cuda.md)：共享内存、寄存器、warp 是三层分块的物理基础。
- [M08 Attention 与 FlashAttention](module-08-attention-flashattention.md)：把分块思想用到 softmax 上。
- [M09 专用硬件与加速器编程](module-09-accelerators.md)：systolic array 是硬化的分块 GEMM。
- [M15 Transformer 推理](module-15-inference-kv-speculative.md)：decode 的 GEMV 瓶颈。
- [edge-deploy M05 卷积 kernel](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-05-convolution-kernels.md)：CPU 上的 im2col、implicit GEMM、depthwise 布局与 Winograd 数值；M07 的双缓冲对应 cp.async 多级流水

## 本讲小结

这一讲要带走三点。第一，GEMM 的计算是 `2MNK`、数据是 `MK+KN+MN`，复用空间很大；分块把它兑现成和 tile 边长成正比的强度，通信量 `Θ(N³/√M)` 已经是下界。
第二，GPU 上的三层分块同时受寄存器、共享内存和 Tensor Core 形状约束，所以 tile 不是越大越好。
第三，小 batch 时 GEMM 退化成 GEMV，强度约等于 batch，只能靠增大 batch 或减少字节数来提速。

下一讲把同样的分块思想用到带 softmax 的 attention 上。

## 延伸阅读

- Hong & Kung, I/O Complexity: The Red-Blue Pebble Game, STOC 1981.
- Lavin & Gray, Fast Algorithms for Convolutional Neural Networks, CVPR 2016, https://arxiv.org/abs/1509.09308
- Chowdhery et al., PaLM: Scaling Language Modeling with Pathways（附录 MFU 定义）, 2022, https://arxiv.org/abs/2204.02311
- NVIDIA CUTLASS 文档（GEMM 分层分块）, https://github.com/NVIDIA/cutlass
- Chetlur et al., cuDNN: Efficient Primitives for Deep Learning, 2014, https://arxiv.org/abs/1410.0759
