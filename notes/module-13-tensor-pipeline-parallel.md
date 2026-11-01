# Module 13 — 张量并行、流水线并行与自动并行

- 课程出处：CS229S Fall 2023《Parallelism Fundamentals》《Deepak Narayanan 客座：Efficient Large-Scale LM Training》 · CMU 15-849《Data and Model Parallelism》《Pipeline Parallelism》《Megatron-LM 客座》《GShard/GSPMD》
- 必读论文：Shoeybi et al., Megatron-LM, 2019；Huang et al., GPipe, 2019；Narayanan et al., PipeDream, 2019；Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM, 2021；Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022；Jia et al., FlexFlow (SOAP), 2019；Xu et al., GSPMD, 2021；Zheng et al., Alpa, 2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：1F1B 气泡实测 0.343 / 0.245 / 0.233）
- 状态：初稿

## 摘要

数据并行要求每张卡放下整个模型，70B 模型的混合精度训练状态 1.13 TB，单卡 80 GB 放不下，只能把模型本身切开。
张量并行（TP）在层内切矩阵，每层前向 2 次、反向 2 次 all-reduce，通信在关键路径上，所以只能留在 NVLink 域内。
流水线并行（PP）在层间切，只传边界激活，通信小，但带来气泡 (p−1)/(m+p−1) 和激活显存压力。
3D 并行的映射规则是：TP 在节点内，PP 跨节点，DP 在最外层。自动并行（FlexFlow、GSPMD、Alpa）把这个手工搜索变成代价模型上的优化问题。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. Megatron MLP：第一个矩阵按列切（GeLU 前不用通信），第二个按行切（输出是部分和，需 all-reduce）。
2. 每个 Transformer 层：前向 2 次 all-reduce（attention 后、MLP 后），反向 2 次，每次张量大小 b·s·h。
3. Sequence parallelism 把 all-reduce 拆成 reduce-scatter + all-gather，通信量不变，LayerNorm/dropout 的激活按 t 切开。
4. GPipe 与 1F1B 气泡相同；1F1B 把在途 micro-batch 从 m 降到 p，激活显存与 m 无关。
5. 交错（interleaved）调度把气泡除以 v，代价是 v 倍的点对点通信。

**必须能默写的公式**

```text
TP 每层通信：4 次 all-reduce × (b·s·h·2 B)，每次每卡收发 2(t−1)/t × b·s·h·2 B
GPipe/1F1B 气泡占比：(p−1)/(m+p−1)；相对理想时间的比值：(p−1)/m
交错调度气泡：(p−1)/v / (m + (p−1)/v)
1F1B 第 i 级（从 1 计）在途 micro-batch ≤ p−i+1；GPipe 为 m
PP 每个边界每个 micro-batch：前向 b·s·h·2 B 激活，反向同样大小的梯度
3D 并行：world = t × p × d；每卡静态状态 ≈ 16N / (t·p)（不含 ZeRO）
```

**一句话区分**

- TP vs PP：层内切矩阵、通信每层都在关键路径 vs 层间切、只传边界、代价是气泡。
- GPipe vs 1F1B：同样的气泡，激活峰值 m 份 vs p 份。
- PipeDream vs PipeDream-Flush：异步无 flush、需要 weight stashing、语义不是同步 SGD vs 周期 flush、语义与同步 SGD 一致。
- GSPMD vs Alpa：用户给分片注解、编译器传播 vs 自动搜索 inter-op 切级与 intra-op 分片。

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

MLP 为 Z = GeLU(X·A)·B，X 形状 [b·s, h]，A 为 [h, 4h]，B 为 [4h, h]。
把 A 按列切成 [A1, A2, …, At]，每卡算 Yi = GeLU(X·Ai)。GeLU 是逐元素的，所以不需要同步。
把 B 按行切成 [B1; …; Bt]，每卡算 Zi = Yi·Bi，Z = ΣZi，需要一次 all-reduce。
反例：若 A 按行切，X·A = X1·A1 + X2·A2，而 GeLU(X1A1 + X2A2) ≠ GeLU(X1A1) + GeLU(X2A2)，GeLU 前就得同步，多一次通信。

### 2. 张量并行的 attention

Q/K/V 投影按列切，等价于按头切：每卡持有 a/t 个头，注意力计算完全本地。
输出投影 Wo 按行切，结果是部分和，一次 all-reduce。
GQA 下 KV 头只有 n_kv 个（Llama-3-70B 为 8），t > n_kv 时需要复制 KV 头，所以 t=8 是自然上限。

### 3. f/g 共轭算子

f 放在并行区域入口：前向恒等，反向 all-reduce（各卡对 X 的梯度是部分和）。
g 放在并行区域出口：前向 all-reduce，反向恒等。
一层有两个并行区域（attention、MLP），所以前向 2 次 g、反向 2 次 f，合计 4 次 all-reduce。

### 4. Sequence parallelism

TP 区域之外的 LayerNorm、dropout、残差在每卡上是重复计算，激活也是重复存储，占 s·b·h 量级的若干份。
SP 在这些区域沿序列维切开：g 换成 reduce-scatter（输出按序列切），f 换成 all-gather。
ring 实现下 all-reduce = reduce-scatter + all-gather，所以通信量不变，显存里那部分激活除以 t。

### 5. 流水线并行

把 L 层切成 p 级，每级 L/p 层。一个 mini-batch 切成 m 个 micro-batch。
GPipe：先跑完全部 m 个前向，再跑全部反向，最后 flush 同步更新。启动和排空各浪费 p−1 个时隙。
1F1B（PipeDream-Flush）：预热 p−i+1 个前向后，每做一个前向就做一个反向，激活尽早释放。
PipeDream（异步）：不 flush，持续流动。一个 micro-batch 的前向与反向必须用同一版权重，所以要 weight stashing，最多存 p 份权重。
交错调度：每卡持有 v 个不相邻的层块（虚拟级），每个块更小，启动排空的时间按 v 缩短。

### 6. 3D 并行与拓扑映射

TP 每层 4 次 all-reduce，通信量最大且不能和下一层计算完全重叠，放在 NVLink 内（t ≤ 8）。
PP 每个 micro-batch 只在边界发一次激活，量小，可以跨节点走 IB。
DP 每步一次梯度 all-reduce，可与反向重叠，放在最外层。
进程编号上 TP 维变化最快，保证同一 TP 组落在同一节点。

### 7. Context parallelism 与 expert parallelism

Context parallelism（ring attention）沿序列切，每卡持有一段 Q，KV 块沿环传递，每到一块就做一次 FlashAttention 的分块更新。
只要一个块的计算时间不小于传一个 KV 块的时间，通信可被完全隐藏。长上下文（128K 以上）训练靠它。
Expert parallelism 把 MoE 的专家分到不同卡，用 all-to-all 路由 token，见 M18。

### 8. 自动并行

FlexFlow 的 SOAP 空间：Sample（batch 维）、Operator（算子间）、Attribute（如空间维）、Parameter（权重维），用执行模拟器加 MCMC 搜索。
GShard / GSPMD：用户在少数张量上标注在设备 mesh 上的分片方式，编译器传播到全图并插入集合通信。
Alpa：两层搜索。inter-op 用动态规划把图切成流水线级并分配子 mesh；intra-op 在每个子 mesh 内用 ILP 选每个算子的分片。
Device placement（Mirhoseini et al., 2017）用 RL 学算子放置，是这条线的起点，见 M10。

## 关键推导

### GPipe 气泡

设每个 micro-batch 在每级的前向+反向时间为 t_mb。理想时间 m·t_mb。
第一个 micro-batch 要经过 p−1 级才到最后一级，前向启动浪费 (p−1)·t_f；反向排空同样浪费 (p−1)·t_b。
总时间 (m+p−1)·t_mb，气泡占比 (p−1)/(m+p−1)，相对理想时间的比值 (p−1)/m。
边界：m ≫ p 时气泡趋零；m = p 时接近 50%。所以 m ≥ 4p 是常用经验。

### 1F1B 的显存上界

第 i 级在稳态前要先做 p−i+1 个前向，之后每完成一个反向才再做一个前向。
所以在途 micro-batch 数恒 ≤ p−i+1，第 1 级最多 p 份激活。GPipe 要存 m 份。
m=32、p=8 时，第 1 级激活从 32 份降到 8 份；气泡仍是 7/39 = 18%。

### 交错调度

每卡 v 个虚拟级，每个虚拟级的耗时为 t_mb/v。启动排空仍是 p−1 个时隙，但每个时隙缩短为 1/v。
气泡比值 (p−1)/(v·m)。代价：每个 micro-batch 跨卡次数乘以 v，点对点通信乘以 v。

## 深入推导

### TP 通信与计算的比值

每层每个 micro-batch 计算量（前向+反向）约 6·(12h²)·b·s = 72·b·s·h² FLOP（12h² 为每层参数，MHA + 4h MLP）。
每卡算 72·b·s·h²/t。通信 4 次 all-reduce，每次每卡发 2(t−1)/t·2·b·s·h 字节。
比值 = 计算时间 / 通信时间 = [72·b·s·h²/(t·F)] / [16·(t−1)/t·b·s·h / B] = 4.5·h·B / ((t−1)·F)。
b·s 被约掉：TP 的通信计算比只取决于 h 与带宽算力比。h 越大越划算。
代入 H100、h=8192、t=8、NVLink 450 GB/s：4.5×8192×450e9 / (7×989e12) ≈ 2.4，通信约为计算的 40%。
换成 IB 50 GB/s：≈ 0.27，通信是计算的 3.7 倍。这就是 TP 不出节点的定量理由。

### 为什么 f 的反向是 all-reduce

并行区域入口处 X 被复制到 t 张卡，每卡用 X 算出自己的 Yi。
dL/dX = Σi dL/dYi · dYi/dX，每卡只有第 i 项，求和就是 all-reduce。
g 的前向把部分和加起来，反向时 dL/dZi = dL/dZ，各卡直接拿到完整梯度，所以恒等。

### 代码对应

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

bias 若在归约前每卡都加，结果会多出 (t−1) 倍 bias，这是手写 TP 最常见的 bug。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 bf16 989 TFLOP/s，NVLink 450 GB/s，IB NDR 50 GB/s。模型 Llama-3-70B，N = 70.6 B，L = 80，h = 8192。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| Column/Row parallel linear | [分布式训练实验](../solutions/cmu15849/distributed-training/README.md) 的 TP MLP | t=2 的输出与单卡 MLP 的最大误差在 bf16 容差内 |
| f/g 算子 | 同上，自定义 autograd Function | 梯度与单卡参考实现一致 |
| GPipe / 1F1B 调度 | 同上，流水线模拟 | 测得的空闲占比与 (p−1)/(m+p−1) 一致 |
| 气泡公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `pipeline_bubble` | `bubble --stages 8 --micro 32` 输出 0.179 |

## 易错点与反例

- 「TP 通信量随 batch 增大而变得不划算」：错。计算与通信都正比于 b·s，比值只由 h 和硬件决定。
- 「1F1B 减少气泡」：错。非交错 1F1B 与 GPipe 气泡相同，它省的是激活显存。
- 「流水线级数越多越好」：p 增大使气泡增大、每级层数减少，而且每级要有足够 m 才能填满。
- 「SP 增加了通信」：ring 实现下不增加，只是把 all-reduce 拆成两半放在不同位置。
- 「PipeDream 等价于同步 SGD」：不等价。不同级用的权重版本不同，只保证同一 micro-batch 在一级内前后向版本一致。
- TP 不能超过 KV 头数而不复制 KV；也不宜超过一个 NVLink 域（8 卡）。
- 把 DP 放在节点内、TP 跨节点是典型错配：最大的通信走了最慢的链路。
- 交错调度要求 m 是 p 的整数倍（Megatron 实现中的约束），否则调度不规整。

## 实验或练习

- 假设：TP 的输出与单卡完全一致；气泡占比服从 (p−1)/(m+p−1)。
- 配置：2 卡或 CPU gloo 后端模拟 t=2；流水线用 p=4、m ∈ {4, 8, 16, 32} 记录每级空闲时间。
- 预期：m=4 时气泡 3/7 = 43%，m=32 时 3/35 = 8.6%；1F1B 的峰值激活数恒为 4，GPipe 随 m 线性增长。

## 高频问题

- 问：TP 一层几次 all-reduce？
  答：前向 2 次（attention 输出、MLP 输出），反向 2 次，共 4 次，每次 b·s·h 个元素。
- 问：为什么第一个矩阵按列切？
  答：按列切后每卡得到完整的部分列，GeLU 逐元素可以本地做；按行切得到部分和，GeLU 前必须先归约。
- 问：PP 的通信量是多少？
  答：每个级边界每个 micro-batch 前向发 b·s·h 个激活，反向收同样大小的梯度，点对点，与参数量无关。
- 问：为什么 DP 在最外层？
  答：DP 每步一次梯度同步，可以和反向重叠，对带宽最不敏感。
- 问：GSPMD 用户要写什么？
  答：设备 mesh 形状，以及少数关键张量（权重、输入）在 mesh 维度上的分片注解，其余由编译器传播。

## 面试追问链

1. 问：Megatron TP 的 MLP 怎么切？
   答：A 列切、B 行切；入口 f（前向恒等、反向 all-reduce），出口 g（前向 all-reduce、反向恒等）。一个 MLP 前向只在末尾通信一次。
2. 问：推一下 TP 的通信计算比。
   答：计算 72·b·s·h²/t，通信 4×2(t−1)/t×2·b·s·h 字节，比值 4.5·h·B/((t−1)·F)，与 b·s 无关。H100 NVLink、h=8192、t=8 时约 2.4。
3. 问：那为什么不把 t 开到 16 跨两个节点？
   答：t−1 变大，同时 B 从 450 GB/s 掉到 50 GB/s，比值掉到 0.1 量级，通信是计算的 8 倍以上。另外 GQA 的 8 个 KV 头也切不开。
4. 问：气泡怎么推？怎么降？
   答：启动排空各 p−1 个时隙，占比 (p−1)/(m+p−1)。降法：增大 m（受全局 batch 限制）、交错调度除以 v（通信乘 v）、zero-bubble 类调度把反向拆成输入梯度与权重梯度填空。
5. 问：增大 m 会不会把激活撑爆？
   答：GPipe 会，峰值 m 份；1F1B 不会，峰值 p 份，所以实践都用 1F1B，m 只受全局 batch 约束。
6. 问：给你 64 张 H100 训 70B，怎么切？
   答：静态 1.13 TB，至少 32 路模型并行才能留出激活空间。t=8 在节点内，p=4 跨节点，d=2。每卡静态 35 GB，配 SP + 选择性重计算把激活压到 ~11 GB。m 取 ≥ 16 让气泡低于 16%，DP 维上 ZeRO-1 再省优化器状态。
7. 问：上下文从 4K 扩到 128K，要改什么？
   答：激活与 s 成正比，attention 部分与 s² 成正比。加 context parallelism，把序列切到多卡做 ring attention，KV 块在环上传、与计算重叠；TP 组不变，CP 维通常也放在高带宽域内。
8. 问：自动并行能替代手工 3D 吗？
   答：Alpa/GSPMD 在规整的 Transformer 上能找到接近 Megatron 的方案，价值在异构模型和新硬件。工业界多数仍用手工 3D + 少量注解，因为可预测、易调试。

## 自测题

1. t=4 的 TP，h=4096、b=8、s=2048，单层前向两次 all-reduce 每卡共收发多少字节？
   要点：张量 8×2048×4096×2 B = 134 MB；每次 2×3/4×134 = 201 MB；两次 402 MB。
2. 证明 GPipe 的总时间是 (m+p−1)·t_mb。
   要点：最后一个 micro-batch 在第 m 个时隙进入第 1 级，再经 p−1 级到达末级；反向对称。
3. p=8、m=16，v=2 交错后气泡是多少？
   要点：(7/2)/(16+3.5) = 17.9%；非交错 7/23 = 30.4%。
4. 设计题：只有 PCIe 互联的 8 张 4090 训 7B，选 TP 还是 PP？
   要点：PCIe 32 GB/s，TP 每层 4 次 all-reduce 代价高；PP 只传边界激活，优先 PP + ZeRO/offload。
5. 解释 SP 为什么能省显存却不增加通信。
   要点：all-reduce = RS + AG；LN/dropout 区的激活变成按序列切的 1/t。

## 关联笔记

- [M12 数据并行与 ZeRO](module-12-data-parallel-zero.md)：DP 维与 all-reduce 通信量
- [M14 显存优化](module-14-memory-optimization.md)：激活显存公式、选择性重计算
- [M18 MoE 与集群](module-18-architectures-moe-clusters.md)：expert parallelism 与 all-to-all
- [M10 ML 编译器](module-10-ml-compilers.md)：device placement 与图级优化
- [M16 推理服务](module-16-llm-serving.md)：推理侧的张量并行

## 本讲小结

TP 切层内，通信计算比 4.5·h·B/((t−1)·F) 决定了它只能留在 NVLink 域。
PP 切层间，气泡 (p−1)/(m+p−1)，1F1B 把激活上界从 m 份降到 p 份，交错再把气泡除以 v。
3D 映射按通信强度排序：TP 最内、PP 居中、DP 最外。
自动并行把这些选择写成代价模型上的搜索，GSPMD 靠注解，Alpa 靠两层优化。

## 延伸阅读

- Shoeybi et al., Megatron-LM, 2019, https://arxiv.org/abs/1909.08053
- Narayanan et al., Efficient Large-Scale Language Model Training on GPU Clusters, 2021, https://arxiv.org/abs/2104.04473
- Huang et al., GPipe, 2019, https://arxiv.org/abs/1811.06965
- Narayanan et al., PipeDream, 2019, https://dl.acm.org/doi/10.1145/3341301.3359646
- Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022, https://arxiv.org/abs/2205.05198
- Xu et al., GSPMD, 2021, https://arxiv.org/abs/2105.04663
- Zheng et al., Alpa, 2022, https://arxiv.org/abs/2201.12023
- Liu et al., Ring Attention with Blockwise Transformers, 2023, https://arxiv.org/abs/2310.01889
