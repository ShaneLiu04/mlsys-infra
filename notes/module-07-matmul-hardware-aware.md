# Module 07 — 矩阵乘、分块与硬件感知算法

- 课程出处：CS149 Fall 2024《Efficiently Evaluating DNNs》 · CS229S Fall 2023《Hardware Aware Algorithm Design》(10-04) · CMU 15-849《ML Hardware》
- 必读论文：Hong & Kung，I/O Complexity: The Red-Blue Pebble Game，1981；Lavin & Gray，Fast Algorithms for Convolutional Neural Networks（Winograd），2016；Chowdhery et al.，PaLM（MFU 定义），2022
- 对应实验：[CS149 A4 DNN 加速器](../solutions/cs149/asst4-dnn-accelerator/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst4/cs149_asst4_report.pdf)（实测：SGEMM 0.94 → 49.4 TFLOP/s（cuBLAS 的 78%））
- 状态：初稿

## 摘要

深度网络里九成以上的 FLOPs 落在 GEMM 上，所以「算子快不快」基本等于「GEMM 快不快」。
GEMM 的 FLOPs 是 `2MNK`，数据只有 `MK+KN+MN`，复用潜力巨大；能否兑现取决于分块：
tile 边长为 `b` 时每读一个元素做 `b` 次乘加，算术强度与 tile 边长成正比，而 tile 边长受快存容量限制，
由此得到通信下界 `Ω(N³/√M)`。GPU 上分块是三层的：寄存器、共享内存、全局内存，Tensor Core
决定了最内层的形状。卷积用 im2col 化成 GEMM，Winograd/FFT 用加法换乘法。主线取舍是：
tile 越大强度越高，但占用越多寄存器/共享内存、并行度越低；batch 越小越接近 GEMV，越受访存限制。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. GEMM `C[M,N] = A[M,K]·B[K,N]` 的 FLOPs 是 `2MNK`（一次乘加算 2 FLOPs），方阵强度 `≈ n/3`（bf16）。
2. 分块后每个 tile 的强度 ≈ `b·b'/(b+b')` FLOP/元素；方 tile 为 `b/2` FLOP/元素，和 tile 边长成正比。
3. 快存容量 `M` 限制 `b ≈ √(M/3)`，总通信量 `Θ(N³/√M)`，这也是 Hong-Kung 下界，分块是渐近最优的。
4. `M` 很小（decode 的 batch）时 GEMM 退化为 GEMV，强度 ≈ batch 大小 FLOP/B，H100 上 batch < 295 都受访存限制。
5. MFU = 模型理论 FLOPs / (时间 × 峰值)；HFU 额外计入重计算，所以 HFU ≥ MFU。

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

- MFU vs HFU：MFU 只算模型「必须」的 FLOPs；HFU 把重计算也算上，用来衡量硬件被用了多少。
- im2col vs Winograd：im2col 不减 FLOPs，只是换成 GEMM 布局；Winograd 真的减少乘法次数。
- 寄存器 tile vs 共享内存 tile：前者决定每条指令的复用，后者决定每次 HBM/L2 读取的复用。

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

`C[i,j] += A[i,k]·B[k,j]` 三重循环共 `M·N·K` 次乘加，即 `2MNK` FLOPs。三个矩阵至少各碰一次，
最少数据量是 `MK+KN+MN` 个元素。计算是三次方，数据是二次方，所以大 GEMM 天然计算受限。
方阵 `n=4096`、bf16：强度 `4096/3 ≈ 1365 FLOP/B`，远高于 H100 拐点 295。
但这只是「每个元素只从 HBM 读一次」的理想值，朴素三重循环达不到。

### 2. 朴素循环为什么慢

`for i, for j, for k` 中，算一个 `C[i,j]` 要读 A 的一行和 B 的一列。B 的列在行主序下跨步访问，
每读一个 fp32 浪费一整条 cache line。更根本的是复用距离：B 的同一元素要等算完一整行 C 才再次被用到，
中间已经被挤出 cache。结果是每次乘加读约 2 个元素，强度约 0.25 FLOP/B（fp32）。

### 3. 分块（tiling / blocking）

把 C 切成 `b×b` 的块，K 维切成 `bk` 的段。对一个 C 块，每步读 A 的 `b×bk` 和 B 的 `bk×b`，
做 `2·b·b·bk` FLOPs。每读一个元素做 `2b²bk/(2b·bk) = b` FLOPs。C 块常驻快存（寄存器）直到 K 维累加完。
要点：强度只取决于 C 块的边长 `b`，与 `bk` 无关；`bk` 只影响快存里 A/B 缓冲的大小和流水深度。

### 4. GPU 的三层分块

| 层级 | 存放什么 | H100 容量约束 | 复用对象 |
| --- | --- | --- | --- |
| CTA tile（如 128×128×64） | A/B 的 tile，多级缓冲 | 共享内存 ≤ 228 KB/SM | 减少 HBM/L2 读取 |
| warp tile（如 64×64） | 从共享内存读入寄存器的片段 | 寄存器 256 KB/SM | 减少共享内存读取 |
| MMA 指令 | Tensor Core 一次算的小块 | 固定形状 | 一条指令内的乘加 |

一个 128×128 的 fp32 累加器是 64 KB，正好占四分之一寄存器文件；A/B 两个 128×64 bf16 tile 各 16 KB，
三级缓冲就是 96 KB 共享内存。这就是为什么 tile 不能无限放大：再大就只能每个 SM 驻留一个 CTA，
没有别的 CTA 来掩盖延迟。

### 5. Tensor Core 与 MMA 形状

Tensor Core 是固定形状的矩阵乘加单元。Volta 的 `HMMA` 是 m8n8k4 的组合；Ampere 的 `mma.sync`
对 bf16 常用 m16n8k16，由一个 warp 发出；Hopper 的 `wgmma` 由 4 个 warp 组成的 warpgroup 发出，
形状 m64nNk16（N 最大 256），操作数可直接从共享内存读。形状意味着：K 维或 N 维不是 8/16 的倍数时要 padding，
维度对齐到 64/128 才能吃满。非 Tensor Core 的 fp32 只有 19.5 TFLOP/s（A100），差一个数量级。

### 6. 卷积变 GEMM：im2col

输入 `[N,C,H,W]`，卷积核 `[K,C,R,S]`，输出 `[N,K,P,Q]`。把每个输出位置对应的 `C·R·S` 个输入展开成一行，
得到 `(N·P·Q) × (C·R·S)` 的矩阵，乘以 `(C·R·S) × K` 的权重。FLOPs 不变：`2·N·K·P·Q·C·R·S`。
代价是输入被复制约 `R·S` 次（stride 1）。cuDNN 的 implicit GEMM 在加载 tile 时按需计算地址，不真正物化大矩阵。

### 7. Winograd 与 FFT

Winograd `F(m×m, r×r)` 用 `(m+r−1)²` 次乘法算出 `m×m` 个输出：`F(2×2,3×3)` 用 16 次乘法代替 36 次（2.25×），
`F(4×4,3×3)` 用 36 次代替 144 次（4×）。代价是变换的加法和数值误差，tile 越大误差越大，低精度下尤甚。
FFT 卷积把复杂度降到 `O(n log n)`，只在大卷积核时划算；3×3 小核上不如 Winograd。

### 8. 层融合

GEMM 后接 bias、GeLU、残差加法时，如果分开跑，中间结果 `M×N` 要写回 HBM 再读出来。
epilogue fusion 在累加器还在寄存器里时做完逐元素运算，只写一次。逐元素算子的强度不到 1 FLOP/B，
单独执行时永远访存受限，融合是它们唯一的出路。

## 关键推导

### 推导 1：分块的通信量与 `√M`

设快存能放 `M` 个元素，用 `b×b` 的 C 块、`b×bk` 的 A 块、`bk×b` 的 B 块，约束 `b² + 2b·bk ≤ M`，取 `bk≈b` 得 `b ≈ √(M/3)`。

```text
C 块个数             (N/b)²
每块的 K 步数        N/bk
每步读入             2·b·bk 元素
总读入 Q             (N/b)² · (N/bk) · 2·b·bk = 2N³/b
代入 b = √(M/3)      Q = 2√3 · N³/√M = Θ(N³/√M)
强度                 2N³ FLOPs / Q = b  FLOP/元素
```

Hong-Kung 下界说任何经典 `O(N³)` 矩阵乘的通信量都是 `Ω(N³/√M)`，所以分块只差常数。边界条件：
`b` 不能超过 N；当 `M ≥ 3N²` 整个问题装进快存，通信量退化为 `3N²`。

### 推导 2：GEMV 为什么访存受限

`M=B` 很小时，`2BNK` FLOPs，数据 `s(BK + KN + BN)`。`B ≪ K,N` 时权重 `KN` 主导：

```text
I ≈ 2BNK / (2·KN) = B   FLOP/B（bf16）
```

H100 拐点 295，所以 decode 的 batch 要到约 295 才计算受限。这与 `tools/mlsys_calc.py` 里
`critical_batch = ridge·bytes/2` 一致。batch=1 时 Tensor Core 有 99% 的时间在等数据。

## 深入推导

### 从 HBM 到 SM：一个 CTA tile 需要多少带宽

CTA tile `BM×BN`、K 段 `bk`，每步读 `(BM+BN)·bk` 个 bf16，算 `2·BM·BN·bk` FLOPs：

```text
I_CTA = 2·BM·BN·bk / (2·(BM+BN)·bk) = BM·BN/(BM+BN)  FLOP/B
128×128 → 64 FLOP/B；128×256 → 85 FLOP/B；256×256 → 128 FLOP/B
H100 满速需要的 L2→SM 带宽 = 989e12 / 64 ≈ 15.5 TB/s（128×128）
                              989e12 / 85 ≈ 11.6 TB/s（128×256）
```

这些强度都低于 HBM 拐点 295，说明「单个 CTA 读一次 HBM」不够，必须依赖 L2 命中：相邻 CTA 共享 A 的同一行块或 B 的同一列块。
所以高性能 GEMM 会做 CTA 光栅化 swizzle（让同时运行的 CTA 访问相近的块），Hopper 还用 cluster 内 TMA multicast
把同一 tile 广播给 2 个 CTA，等效把 L2 读取再减半。这解释了为什么 H100 上主流 tile 是 128×256 而不是 64×64。

### 代码：两级分块的 CUDA 骨架

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

外层共享内存 tile 把 HBM 读取降到 `1/BM` 和 `1/BN`，内层寄存器 tile 把共享内存读取从每 FMA 2 次降到
`(TM+TN)/(TM·TN)` 次。`TM=TN=8` 时是 0.25 次，这一级不做，共享内存带宽先成为瓶颈。
真正的 Tensor Core kernel 把最内层换成 `mma`/`wgmma`，并用 `cp.async`/TMA 做多级流水，让加载与计算重叠。

### MFU 的定义细节

训练一个 token 的模型 FLOPs 是 `6N`（前向 2N、反向 4N）。长上下文时 PaLM 附录再加注意力项，
每 token `12·L·d·T`（L 层、d 模型宽度、T 序列长度，按非因果计）。
MFU 用 `6N`；若开了全量激活重计算，前向多跑一遍，硬件实际做的是 `8N`，HFU = `8N·tps/peak`。
报告 MFU 的好处是和实现无关：重计算越多 HFU 越好看，但 MFU 不会被刷高。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的 H100 SXM：bf16 989 TFLOP/s，HBM 3.35 TB/s，拐点 295。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| GEMM 强度与 roofline | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `gemm_intensity`、`roofline_time` | 方阵 4096 bf16 得 ≈1365 FLOP/B |
| decode 临界 batch | 同文件 `critical_batch` | H100 输出 ≈ 295 |
| 分块矩阵乘与卷积映射 | [CS149 A4](../solutions/cs149/asst4-dnn-accelerator/README.md) | tile 后的结果与参考实现一致，记录不同 tile 的耗时 |
| MFU | 同文件 `mfu` | 与手算 6N·tps/peak 一致 |

## 易错点与反例

- 常见错答：「GEMM 的 FLOPs 是 MNK」。乘和加各算一次，是 `2MNK`；厂商峰值也按 FMA=2 FLOPs 计。
- 常见错答：「矩阵够大就一定计算受限」。`[1,4096]×[4096,14336]` 有 1.17e8 FLOPs，但 M=1，强度只有 1。
- 「tile 越大越好」不对：寄存器溢出到 local memory 会让性能断崖；tile 太大时 CTA 数少于 SM 数（wave quantization），尾波空转。
- 把强度写成 `b` 却忘了单位：`b` FLOP/元素，bf16 下是 `b/2` FLOP/B，fp32 下是 `b/4` FLOP/B。
- 认为 im2col 减少了计算：它只改变布局，FLOPs 一点没少，显存反而多了 `R·S` 倍。
- Winograd 在 int8/fp16 下误差会放大，大 tile 的 Winograd 常被量化推理禁用。
- MFU 分子误用实测 FLOPs（含重计算），得到的其实是 HFU。

## 实验或练习

- 假设：GEMM 吞吐随方阵边长先升后平，拐点在 n ≈ 900 附近（H100）或 n ≈ 460（A100，153·3）。
- 配置：`torch.matmul` 在 bf16 下扫 n ∈ {128, 256, 512, 1024, 2048, 4096, 8192}，记录 TFLOP/s；再扫 M ∈ {1, 8, 64, 512} 固定 N=K=4096。
- 预期：大 n 时接近峰值的 60%–80%；M 扫描中吞吐与 M 近似线性，直到接近拐点。具体数值以自己机器实测为准。

## 高频问题

- 问：GEMM 的算术强度怎么算？
  答：`2MNK / (s·(MK+KN+MN))`；方阵 bf16 是 `n/3`。
- 问：为什么分块能提高性能？
  答：C 块常驻快存，每读一个元素做 `b` 次乘加，强度与 tile 边长成正比。
- 问：tile 大小由什么决定？
  答：共享内存与寄存器容量、每 SM 需要的并发 CTA 数、Tensor Core 形状对齐、矩阵尺寸能否整除。
- 问：小 batch 推理为什么慢？
  答：退化成 GEMV，强度 ≈ batch，远低于拐点，时间等于读一遍权重。
- 问：MFU 和 HFU 的区别？
  答：MFU 只计模型理论 FLOPs，HFU 计入重计算等硬件实际执行的 FLOPs。

## 面试追问链

1. 问：一个 `[M,K]×[K,N]` 的 GEMM 有多少 FLOPs，最少要搬多少字节？
   答：`2MNK` FLOPs，最少 `s(MK+KN+MN)` 字节；二者之比是理想强度，方阵为 `n/3`（bf16）。
2. 问：朴素三重循环离理想强度差在哪？
   答：复用距离太长，B 的元素在被再次使用前已被挤出 cache，每次乘加都要读内存；强度掉到常数级。
3. 问：推导分块后的通信量。
   答：C 块 `b×b`、K 步 `bk`，总读入 `2N³/b`；快存约束 `b≈√(M/3)`，得 `Θ(N³/√M)`，与 Hong-Kung 下界同阶。
4. 问：GPU 上 tile 为什么常是 128×128 或 128×256，而不是 512×512？
   答：128×128 fp32 累加器已占 64 KB 寄存器，512×512 要 1 MB，远超 256 KB/SM；还会让 CTA 数太少、无法掩盖延迟。
5. 问：128×128 的 CTA tile 强度只有 64 FLOP/B，低于拐点，为什么 cuBLAS 还能跑到高利用率？
   答：tile 强度是相对 L2 的；同时运行的 CTA 共享 A/B 块，L2 命中让 HBM 实际流量远低于 CTA 读取量。swizzle 和 TMA multicast 就是为提高这种跨 CTA 复用。
6. 问：decode 阶段 batch=1，如何让 GEMM 不浪费 Tensor Core？
   答：增大 batch（continuous batching）、权重量化降低字节数（int4 权重使强度约 ×4）、推测解码让一次前向处理多个 token；单请求延迟下界是读一遍权重的时间。
7. 问：你要为一个新的 3×3 卷积密集的视觉模型选 kernel 方案，怎么权衡 implicit GEMM 与 Winograd？
   答：fp16/bf16 大通道数时 Winograd F(2,3) 省 2.25× 乘法但增加变换开销与误差；int8 量化部署倾向 implicit GEMM。先按 roofline 判断层是否计算受限，访存受限的层用 Winograd 也没用。
8. 问：你在 8×H100 上训练 8B 模型只跑到 25% MFU，排查顺序？
   答：先看是否计算受限的 GEMM 太小（micro-batch、序列长度、TP 切太碎导致 N/K 太小）；再看逐元素算子是否未融合；再看通信是否没被重叠；最后看 dataloader 与 Python 开销。每步都用 profiler 的 kernel 时间占比验证。

## 自测题

1. A100 bf16 下方阵 GEMM 的 n 低于多少会受访存限制？
   要点：拐点 153，`n/3 = 153` → n ≈ 460。
2. 证明方 tile 的强度与 `bk` 无关。
   要点：`2b²bk / (2b·bk) = b`，`bk` 被约掉；`bk` 只影响缓冲大小和流水。
3. 一个 `[2048, 4096]×[4096, 4096]` 的 bf16 GEMM 在 H100 上的下界时间？
   要点：FLOPs = 6.87e10，bytes = 2·(8.39e6+1.68e7+8.39e6) ≈ 67 MB；强度 ≈ 1024 > 295；时间 ≈ 69 µs。
4. 推导 im2col 的矩阵形状与放大倍数。
   要点：`(N·P·Q)×(C·R·S)`；stride 1、same padding 时放大约 `R·S`。
5. 设计题：给 Llama-3-8B decode 写一个 batch=4 的 GEMM，tile 怎么选？
   要点：M 维只有 4，应沿 N/K 切分以获得足够多 CTA（split-K），优化目标是 HBM 带宽而非 Tensor Core。

## 关联笔记

- [M01 性能模型与 roofline](module-01-performance-models.md)：拐点与强度的定义。
- [M05 GPU 架构与 CUDA](module-05-gpu-architecture-cuda.md)：共享内存、寄存器、warp 是三层分块的物理基础。
- [M08 Attention 与 FlashAttention](module-08-attention-flashattention.md)：把分块思想用到 softmax 上。
- [M09 专用硬件与加速器编程](module-09-accelerators.md)：systolic array 是硬化的分块 GEMM。
- [M15 Transformer 推理](module-15-inference-kv-speculative.md)：decode 的 GEMV 瓶颈。

## 本讲小结

GEMM 的 `2MNK` 对 `MK+KN+MN` 给出了复用空间，分块把它兑现成与 tile 边长成正比的强度，通信量 `Θ(N³/√M)` 是下界。
GPU 的三层分块受寄存器、共享内存和 Tensor Core 形状共同约束。小 batch 退化成 GEMV，强度约等于 batch，只能靠增大 batch 或减小字节数。
下一讲把同样的分块思想用在含 softmax 的 attention 上。

## 延伸阅读

- Hong & Kung, I/O Complexity: The Red-Blue Pebble Game, STOC 1981.
- Lavin & Gray, Fast Algorithms for Convolutional Neural Networks, CVPR 2016, https://arxiv.org/abs/1509.09308
- Chowdhery et al., PaLM: Scaling Language Modeling with Pathways（附录 MFU 定义）, 2022, https://arxiv.org/abs/2204.02311
- NVIDIA CUTLASS 文档（GEMM 分层分块）, https://github.com/NVIDIA/cutlass
- Chetlur et al., cuDNN: Efficient Primitives for Deep Learning, 2014, https://arxiv.org/abs/1410.0759
