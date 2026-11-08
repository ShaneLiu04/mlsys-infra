# Module 13 — 张量并行、流水线并行与自动并行

- 课程出处：CS229S Fall 2023《Parallelism Fundamentals》《Deepak Narayanan 客座：Efficient Large-Scale LM Training》 · CMU 15-849《Data and Model Parallelism》《Pipeline Parallelism》《Megatron-LM 客座》《GShard/GSPMD》
- 必读论文：Shoeybi et al., Megatron-LM, 2019；Huang et al., GPipe, 2019；Narayanan et al., PipeDream, 2019；Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, 2021；Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022；Jia et al., FlexFlow (SOAP), 2019；Xu et al., GSPMD, 2021；Zheng et al., Alpa, 2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：1F1B 气泡实测 0.343 / 0.245 / 0.233）
- 状态：初稿

## 摘要

数据并行（M12）有一个默认前提：每张卡都放得下整个模型。到了 70B 这个量级，这个前提就不成立了。混合精度训练状态一共 1.13 TB，而单卡只有 80 GB，所以只能把模型本身切开。

切法有两个方向。张量并行（tensor parallelism，TP）在层内切矩阵，每层前向 2 次、反向 2 次 all-reduce。这些通信卡在关键路径上，下一步计算必须等它完成，所以 TP 只能留在 NVLink 域内。
流水线并行（pipeline parallelism，PP）在层间切，只在级与级之间传边界激活，通信量小。它的代价是气泡（流水线启动和排空时有卡在空等），占比 (p−1)/(m+p−1)，另外还有激活显存的压力。

把 TP、PP 和数据并行（DP）组合起来就是 3D 并行。映射到机器上的规则按通信强度排：TP 在节点内，PP 跨节点，DP 在最外层。
手工选这些参数很费劲，自动并行（FlexFlow、GSPMD、Alpa）把它变成了代价模型上的优化问题。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. Megatron 的 MLP 第一个矩阵按列切。GeLU 是逐元素的，所以各卡可以各算各的，不用通信。第二个矩阵按行切，输出是部分和，要做一次 all-reduce。
2. 由此每个 Transformer 层前向有 2 次 all-reduce（attention 之后、MLP 之后），反向对称再 2 次，每次的张量大小是 b·s·h。
3. Sequence parallelism 把 all-reduce 拆成 reduce-scatter 加 all-gather。通信量不变，但 LayerNorm 和 dropout 的激活按 t 切开了。
4. GPipe 和 1F1B 的气泡一样大。1F1B 赢在显存：在途 micro-batch 从 m 降到 p，激活显存不再随 m 增长。
5. 交错（interleaved）调度把气泡除以 v，代价是 v 倍的点对点通信。

必须能默写的公式：

```text
TP 每层通信：4 次 all-reduce × (b·s·h·2 B)，每次每卡收发 2(t−1)/t × b·s·h·2 B
GPipe/1F1B 气泡占比：(p−1)/(m+p−1)；相对理想时间的比值：(p−1)/m
交错调度气泡：(p−1)/v / (m + (p−1)/v)
1F1B 第 i 级（从 1 计）在途 micro-batch ≤ p−i+1；GPipe 为 m
PP 每个边界每个 micro-batch：前向 b·s·h·2 B 激活，反向同样大小的梯度
3D 并行：world = t × p × d；每卡静态状态 ≈ 16N / (t·p)（不含 ZeRO）
```

一句话区分：

- TP 与 PP：TP 在层内切矩阵，每层都有通信卡在关键路径上；PP 在层间切，只传边界激活，代价是气泡。
- GPipe 与 1F1B：气泡相同，激活峰值分别是 m 份和 p 份。
- PipeDream 与 PipeDream-Flush：前者异步、不 flush，需要 weight stashing，语义已不再是同步 SGD；后者周期性 flush，语义与同步 SGD 一致。
- GSPMD 与 Alpa：GSPMD 由用户给分片注解、编译器负责传播；Alpa 自动搜索 inter-op 的切级和 intra-op 的分片。

## 学习目标

- 能在白板上画出 Megatron MLP 与 attention 的切法，并说清每个 all-reduce 的位置；
- 能写出 f/g 共轭算子的前向与反向；
- 推导 GPipe 气泡公式，解释 1F1B 为什么省显存但不省气泡；
- 给定模型大小和集群拓扑，设计 t、p、d 的取值并估算通信与显存；
- 说出 FlexFlow SOAP、GSPMD、Alpa 分别搜索什么空间。

## 先修知识

- M12：ring all-reduce 的通信量 2(p−1)/p·S，ZeRO 的三个阶段；
- M07：GEMM 的切块与算术强度；
- M05：NVLink 与 PCIe/IB 的带宽差距。

## 核心概念

### 1. 张量并行的 MLP

把一个 MLP 摊到 t 张卡上，第一个问题是怎么切才能让通信最少。

MLP 写成 Z = GeLU(X·A)·B，其中 X 的形状是 [b·s, h]，A 为 [h, 4h]，B 为 [4h, h]。
先把 A 按列切成 [A1, A2, …, At]，每张卡算 Yi = GeLU(X·Ai)。每张卡拿到的是完整的若干列，而 GeLU 逐元素作用，所以这一步不需要同步。
再把 B 按行切成 [B1; …; Bt]，每张卡算 Zi = Yi·Bi。这时每张卡手里只是部分和，Z = ΣZi，要用一次 all-reduce 加起来。

反过来切就会多一次通信。如果 A 按行切，X·A = X1·A1 + X2·A2，而 GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2)，所以 GeLU 之前就得先同步。

### 2. 张量并行的 attention

同样的思路可以搬到 attention 上。Q/K/V 投影按列切，等价于按头切：每张卡持有 a/t 个头。各个头的注意力计算互不相干，所以完全可以在本地做。
输出投影 Wo 按行切，结果是部分和，再做一次 all-reduce。

按头切也给 t 设了上限。在 GQA（多个 query 头共享一组 KV 头）下，KV 头只有 n_kv 个，Llama-3-70B 为 8。t > n_kv 时就得复制 KV 头，所以 t=8 是自然上限。

### 3. f/g 共轭算子

上面两处通信要落进 autograd，Megatron 用一对互为共轭的算子 f 和 g 来表达。

f 放在并行区域的入口。前向是恒等，把 X 原样交给每张卡；反向是 all-reduce，因为每张卡算出的 X 的梯度只是部分和。
g 放在并行区域的出口。前向是 all-reduce，反向是恒等。
一层有两个并行区域（attention 和 MLP），所以前向经过 2 次 g、反向经过 2 次 f，合计 4 次 all-reduce。

### 4. Sequence parallelism

TP 只切了矩阵乘那部分。TP 区域之外的 LayerNorm、dropout 和残差在每张卡上都重复计算，激活也重复存储，占 s·b·h 量级的若干份。模型其余部分被切得越细，这块重复的激活就越显眼。

Sequence parallelism（SP）在这些区域沿序列维切开：把 g 换成 reduce-scatter，让输出按序列切开；把 f 换成 all-gather。
在 ring 实现下，all-reduce 本来就等于 reduce-scatter 加 all-gather，所以通信量不变，而显存里这部分激活除以 t。

### 5. 流水线并行

TP 的通信太频繁，出不了节点。要继续往外扩，就换一个方向，按层切。

把 L 层切成 p 级（stage），每级 L/p 层。一个 mini-batch 再切成 m 个 micro-batch，让它们像流水线一样依次流过各级。几种调度的区别在于前向和反向怎么排：

- GPipe：先跑完全部 m 个前向，再跑全部反向，最后 flush、同步更新。启动和排空各浪费 p−1 个时隙。
- 1F1B（PipeDream-Flush）：第 i 级先预热 p−i+1 个前向，之后每做一个前向就做一个反向，激活可以尽早释放。
- PipeDream（异步）：不 flush，micro-batch 持续流动。同一个 micro-batch 的前向和反向必须用同一版权重，所以要做 weight stashing（把旧版本权重暂存起来），最多存 p 份权重。
- 交错调度：每张卡持有 v 个不相邻的层块（虚拟级）。每块更小，所以启动和排空的时间按 v 缩短。

### 6. 3D 并行与拓扑映射

三种并行怎么摆到具体的机器上？原则是按通信强度排序，让最重的通信走最快的链路。

TP 每层 4 次 all-reduce，通信量最大，又不能和下一层的计算完全重叠，所以放在 NVLink 内（t ≤ 8）。
PP 每个 micro-batch 只在级边界发一次激活，量小，可以跨节点走 IB。
DP 每步只有一次梯度 all-reduce，而且能和反向重叠，放在最外层。

落到进程编号上，就是让 TP 维变化最快，这样同一个 TP 组的进程会落在同一个节点上。

### 7. Context parallelism 与 expert parallelism

序列很长的时候，还可以沿第四个维度切。Context parallelism（也叫 ring attention）沿序列切：每张卡持有一段 Q，KV 块沿着环传递，每到一块就做一次 FlashAttention 式的分块更新。
只要算一个块的时间不小于传一个 KV 块的时间，通信就能被完全隐藏。128K 以上的长上下文训练靠的就是它。

Expert parallelism 则把 MoE 的专家分到不同的卡上，用 all-to-all 路由 token，见 M18。

### 8. 自动并行

手工定 t、p、d 已经要算半天，如果再细到每个算子怎么切，搜索空间很快就超出人力。自动并行把这件事交给搜索：

- FlexFlow 定义了 SOAP 空间：Sample（batch 维）、Operator（算子间）、Attribute（如空间维）、Parameter（权重维）。它用执行模拟器估计代价，用 MCMC 搜索。
- GShard / GSPMD：用户只在少数张量上标注它们在设备 mesh 上怎么分片，编译器把分片方式传播到全图，并插入需要的集合通信。
- Alpa 做两层搜索。inter-op 层用动态规划把图切成流水线级，并给每级分配一个子 mesh；intra-op 层在每个子 mesh 内用 ILP 选每个算子的分片。

这条线的起点是 Device placement（Mirhoseini et al., 2017），它用强化学习学算子放置，见 M10。

## 关键推导

### GPipe 气泡

这一节回答：流水线到底浪费了多少时间，又该怎么选 m。

设每个 micro-batch 在每一级的前向加反向时间为 t_mb，那么理想时间是 m·t_mb。
第一个 micro-batch 要经过 p−1 级才能到达最后一级，在这之前后面的级都在空等，前向启动浪费 (p−1)·t_f。反向排空是对称的，同样浪费 (p−1)·t_b。
所以总时间是 (m+p−1)·t_mb，气泡占比 (p−1)/(m+p−1)，相对理想时间的比值是 (p−1)/m。

看两个极端：m ≫ p 时气泡趋于零；m = p 时接近 50%。所以 m ≥ 4p 是常用的经验值。

### 1F1B 的显存上界

既然 1F1B 不省气泡，它省的是什么？答案是在途 micro-batch 的个数，也就是要同时存着的激活份数。

第 i 级在进入稳态前要先做 p−i+1 个前向，之后每完成一个反向才再做一个前向。
所以在途 micro-batch 数始终 ≤ p−i+1，第 1 级最多存 p 份激活。GPipe 则要存 m 份。
举个例子，m=32、p=8 时，第 1 级的激活从 32 份降到 8 份，而气泡仍是 7/39 = 18%。

### 交错调度

要真正压气泡，就得缩短每个时隙。交错调度让每张卡持有 v 个虚拟级，每个虚拟级的耗时是 t_mb/v。启动和排空仍然是 p−1 个时隙，但每个时隙缩短到原来的 1/v。
于是气泡比值变成 (p−1)/(v·m)。代价是每个 micro-batch 跨卡的次数乘以 v，点对点通信也乘以 v。

## 深入推导

### TP 通信与计算的比值

摘要里说 TP 只能留在 NVLink 域内，这一节把这句话算出来。

每层每个 micro-batch 的计算量（前向加反向）约为 6·(12h²)·b·s = 72·b·s·h² FLOP，这里 12h² 是每层的参数量（MHA 加 4h 宽的 MLP）。
每张卡算其中的 72·b·s·h²/t。通信是 4 次 all-reduce，每次每张卡发 2(t−1)/t·2·b·s·h 字节。
比值 = 计算时间 / 通信时间 = [72·b·s·h²/(t·F)] / [16·(t−1)/t·b·s·h / B] = 4.5·h·B / ((t−1)·F)。

b·s 被约掉了。也就是说，TP 的通信计算比只取决于 h 与硬件的带宽算力比，h 越大越划算。
代入 H100、h=8192、t=8、NVLink 450 GB/s：4.5×8192×450e9 / (7×989e12) ≈ 2.4，通信约为计算的 40%。
换成 IB 50 GB/s，比值只有 ≈ 0.27，通信是计算的 3.7 倍。这就是 TP 不出节点的定量理由。

### 为什么 f 的反向是 all-reduce

在并行区域入口，X 被复制到 t 张卡上，每张卡用 X 算出自己的 Yi。
按链式法则，dL/dX = Σi dL/dYi · dYi/dX。每张卡只有第 i 项，把它们加起来正好就是 all-reduce。
g 的前向把部分和加起来；反向时 dL/dZi = dL/dZ，各卡直接拿到完整的梯度，所以是恒等。

### 代码对应

下面这段把 f、g 写成自定义 autograd Function，再用它们拼出列并行和行并行的线性层：

```python
import torch, torch.distributed as dist
import torch.nn.functional as F

class CopyToTP(torch.autograd.Function):        # f
    @staticmethod
    def forward(ctx, x): return x
    @staticmethod
    def backward(ctx, g):
        g = g.clone(); dist.all_reduce(g, group=TP_GROUP); return g

class ReduceFromTP(torch.autograd.Function):    # g
    @staticmethod
    def forward(ctx, x):
        x = x.clone(); dist.all_reduce(x, group=TP_GROUP); return x
    @staticmethod
    def backward(ctx, g): return g

class ColumnParallelLinear(torch.nn.Module):    # W: [out/t, in]
    def __init__(self, d_in, d_out, t):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(d_out // t, d_in))
    def forward(self, x):
        return F.linear(CopyToTP.apply(x), self.w)        # 输出按列切，不通信

class RowParallelLinear(torch.nn.Module):       # W: [out, in/t]
    def __init__(self, d_in, d_out, t):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(d_out, d_in // t))
        self.b = torch.nn.Parameter(torch.zeros(d_out))
    def forward(self, x_shard):
        return ReduceFromTP.apply(F.linear(x_shard, self.w)) + self.b  # bias 在归约后加一次

def tp_mlp(x, fc1: ColumnParallelLinear, fc2: RowParallelLinear):
    return fc2(F.gelu(fc1(x)))
```

注意行并行层的 bias。如果在归约之前每张卡都加一遍，结果会多出 (t−1) 倍的 bias，这是手写 TP 时最常见的 bug。

## 白板估算

下面用 Llama-3-70B 走一遍完整的估算：先看静态显存要多少卡，再比较 TP 走 NVLink 和走 IB 的代价，最后给一个 64 卡的切法并核对气泡和 DP 通信。
硬件数字取自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 bf16 989 TFLOP/s，NVLink 450 GB/s，IB NDR 50 GB/s。模型 Llama-3-70B，N = 70.6 B，L = 80，h = 8192。

```text
静态显存：16 B × 70.6e9 = 1129.6 GB ≈ 1.13 TB，需要 ≥ 15 张 80 GB 卡才放得下，还没算激活

TP 单次 all-reduce（s=4096, b=1）：张量 4096×8192×2 B = 67.1 MB
  t=8 每卡收发 2×7/8×67.1 MB = 117.4 MB
  NVLink：117.4 MB / 450 GB/s = 0.26 ms；每 micro-batch 80 层 × 4 次 = 320 次 → 83.5 ms
  IB：117.4 MB / 50 GB/s = 2.35 ms；×320 → 751 ms
计算（t=8）：6 × 70.6e9 × 4096 / 8 = 2.17e14 FLOP，满峰 0.22 s，MFU 50% 时 0.44 s
  NVLink 下通信 ≈ 计算的 19%，可部分重叠；IB 下通信 > 计算，TP 跨节点不可行

64 张 H100（8 节点）的切法：t=8（节点内），p=4（跨节点），d=2
  每卡静态 1129.6 / 32 = 35.3 GB；剩余 ~45 GB 给激活（见 M14 的激活估算）
  每级 20 层；PP 边界每 micro-batch 传 67.1 MB，IB 上 1.34 ms，远小于一个 micro-batch 的计算

气泡：p=8, m=32 → 7/39 = 17.9%；v=4 交错 → 1.75/33.75 = 5.2%
      p=4, m=8  → 3/11 = 27.3%；m=32 → 3/35 = 8.6%
DP 梯度 all-reduce（d=2）：每卡参数 70.6e9/32 = 2.2e9 → bf16 梯度 4.4 GB
  每卡收发 2×1/2×4.4 GB = 4.4 GB，IB 上 88 ms，与反向重叠
```

这组数字里最该记住的是 TP 那一行的对比：同样 320 次 all-reduce，走 NVLink 是 83.5 ms，走 IB 是 751 ms，后者已经超过了计算本身。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| Column/Row parallel linear | [分布式训练实验](../solutions/cmu15849/distributed-training/README.md) 的 TP MLP | t=2 的输出与单卡 MLP 的最大误差在 bf16 容差内 |
| f/g 算子 | 同上，自定义 autograd Function | 梯度与单卡参考实现一致 |
| GPipe / 1F1B 调度 | 同上，流水线模拟 | 测得的空闲占比与 (p−1)/(m+p−1) 一致 |
| 气泡公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `pipeline_bubble` | `bubble --stages 8 --micro 32` 输出 0.179 |

## 易错点与反例

- 「TP 通信量随 batch 增大而变得不划算」：错。计算和通信都正比于 b·s，比值只由 h 和硬件决定。
- 「1F1B 减少气泡」：错。非交错的 1F1B 和 GPipe 气泡相同，它省的是激活显存。
- 「流水线级数越多越好」：p 增大会让气泡变大、每级层数变少，而且级数越多，越需要足够大的 m 才能填满流水线。
- 「SP 增加了通信」：在 ring 实现下并不增加，它只是把 all-reduce 拆成两半，放在不同的位置。
- 「PipeDream 等价于同步 SGD」：不等价。不同级用的权重版本不同，它只保证同一个 micro-batch 在同一级内前向和反向用的是同一版。
- TP 度超过 KV 头数时必须复制 KV；TP 度也不宜超过一个 NVLink 域（8 卡）。
- 把 DP 放在节点内、TP 跨节点是典型的错配：最大的通信走了最慢的链路。
- 交错调度要求 m 是 p 的整数倍（这是 Megatron 实现中的约束），否则调度排不规整。

## 实验或练习

这个练习要验证两件事：TP 的输出和单卡完全一致，以及气泡占比服从 (p−1)/(m+p−1)。

- 配置：用 2 卡或 CPU 上的 gloo 后端模拟 t=2；流水线取 p=4、m ∈ {4, 8, 16, 32}，记录每一级的空闲时间。
- 预期：m=4 时气泡 3/7 = 43%，m=32 时 3/35 = 8.6%；1F1B 的峰值激活数恒为 4，GPipe 的则随 m 线性增长。

## 高频问题

- 问：TP 一层几次 all-reduce？
  答：前向 2 次（attention 输出、MLP 输出），反向 2 次，共 4 次，每次 b·s·h 个元素。
- 问：PP 的通信量是多少？
  答：每个级边界、每个 micro-batch，前向发 b·s·h 个激活，反向收同样大小的梯度。这是点对点通信，与参数量无关。
- 问：交错调度的代价是什么？
  答：气泡除以 v，但点对点通信乘以 v；在 Megatron 里还要求 m 是 p 的整数倍。
- 问：为什么 DP 在最外层？
  答：DP 每步只同步一次梯度，还能和反向重叠，对带宽最不敏感。
- 问：GSPMD 要用户写什么？
  答：设备 mesh 的形状，以及少数关键张量（权重、输入）在 mesh 各维上的分片注解，其余由编译器传播。

## 面试追问链

1. 问：Megatron TP 的 MLP 怎么切？
   答：A 按列切，B 按行切。入口放 f（前向恒等、反向 all-reduce），出口放 g（前向 all-reduce、反向恒等）。这样一个 MLP 的前向只在末尾通信一次；反过来让 A 按行切，GeLU 之前就要多同步一次。
2. 问：推一下 TP 的通信计算比。
   答：每卡计算 72·b·s·h²/t，通信 4×2(t−1)/t×2·b·s·h 字节，比值是 4.5·h·B/((t−1)·F)，和 b·s 无关。H100 走 NVLink、h=8192、t=8 时约为 2.4。
3. 问：那为什么不把 t 开到 16，跨两个节点？
   答：一方面 t−1 变大，另一方面 B 从 450 GB/s 掉到 50 GB/s，比值会掉到 0.1 量级，通信是计算的 8 倍以上。另外 GQA 只有 8 个 KV 头，也切不开。
4. 问：气泡怎么推？怎么降？
   答：启动和排空各占 p−1 个时隙，所以占比是 (p−1)/(m+p−1)。降法有三种：增大 m，但受全局 batch 限制；交错调度把气泡除以 v，但通信乘以 v；zero-bubble 类调度把反向拆成输入梯度和权重梯度两部分，用来填空。
5. 问：增大 m 会不会把激活撑爆？
   答：GPipe 会，峰值是 m 份；1F1B 不会，峰值是 p 份。所以实践中都用 1F1B，m 只受全局 batch 约束。
6. 问：给你 64 张 H100 训 70B，怎么切？
   答：静态状态 1.13 TB，至少要 32 路模型并行才能留出激活空间。t=8 放在节点内，p=4 跨节点，d=2。这样每卡静态 35 GB，再配上 SP 和选择性重计算，把激活压到 ~11 GB。m 取 ≥ 16，让气泡低于 16%；DP 维上再用 ZeRO-1 省优化器状态。
7. 问：上下文从 4K 扩到 128K，要改什么？
   答：激活和 s 成正比，attention 部分和 s² 成正比。要加 context parallelism，把序列切到多卡上做 ring attention，KV 块在环上传、与计算重叠。TP 组不用动，CP 维通常也放在高带宽域内。
8. 问：自动并行能替代手工 3D 吗？
   答：在规整的 Transformer 上，Alpa 和 GSPMD 能找到接近 Megatron 的方案，它们的价值更多在异构模型和新硬件上。工业界多数仍用手工 3D 加少量注解，因为结果可预测、容易调试。

## 自测题

1. t=4 的 TP，h=4096、b=8、s=2048，单层前向两次 all-reduce，每卡共收发多少字节？
   要点：张量 8×2048×4096×2 B = 134 MB；每次 2×3/4×134 = 201 MB；两次 402 MB。
2. 证明 GPipe 的总时间是 (m+p−1)·t_mb。
   要点：最后一个 micro-batch 在第 m 个时隙进入第 1 级，再经过 p−1 级到达末级；反向对称。
3. p=8、m=16，v=2 交错后气泡是多少？
   要点：(7/2)/(16+3.5) = 17.9%；非交错时是 7/23 = 30.4%。
4. 设计题：只有 PCIe 互联的 8 张 4090 训 7B，选 TP 还是 PP？
   要点：PCIe 只有 32 GB/s，TP 每层 4 次 all-reduce 代价太高；PP 只传边界激活，所以优先 PP，再配 ZeRO 或 offload。
5. 用 all-reduce 的分解说明：SP 为什么能省显存，却不增加通信？
   要点：all-reduce = RS + AG，SP 只是把两半拆开放；LN/dropout 区的激活因此变成按序列切的 1/t。

## 关联笔记

- [M12 数据并行与 ZeRO](module-12-data-parallel-zero.md)：DP 维与 all-reduce 通信量
- [M14 显存优化](module-14-memory-optimization.md)：激活显存公式、选择性重计算
- [M18 MoE 与集群](module-18-architectures-moe-clusters.md)：expert parallelism 与 all-to-all
- [M10 ML 编译器](module-10-ml-compilers.md)：device placement 与图级优化
- [M16 推理服务](module-16-llm-serving.md)：推理侧的张量并行
- [edge-deploy M07 DMA 与双缓冲](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-07-dma-double-buffering.md)：k 级流水 Σs_i + (n − 1)·max s_i 与流水线气泡同形

## 本讲小结

TP 在层内切，通信计算比是 4.5·h·B/((t−1)·F)，这个比值决定了它只能留在 NVLink 域内。
PP 在层间切，气泡是 (p−1)/(m+p−1)。1F1B 把激活上界从 m 份降到 p 份，交错调度再把气泡除以 v。
3D 映射按通信强度排：TP 最内，PP 居中，DP 最外。自动并行把这些选择写成代价模型上的搜索，GSPMD 靠注解，Alpa 靠两层优化。

## 延伸阅读

- Shoeybi et al., Megatron-LM, 2019, https://arxiv.org/abs/1909.08053
- Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters, 2021, https://arxiv.org/abs/2104.04473
- Huang et al., GPipe, 2019, https://arxiv.org/abs/1811.06965
- Narayanan et al., PipeDream, 2019, https://dl.acm.org/doi/10.1145/3341301.3359646
- Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022, https://arxiv.org/abs/2205.05198
- Xu et al., GSPMD, 2021, https://arxiv.org/abs/2105.04663
- Zheng et al., Alpa, 2022, https://arxiv.org/abs/2201.12023
- Liu et al., Ring Attention with Blockwise Transformers, 2023, https://arxiv.org/abs/2310.01889
