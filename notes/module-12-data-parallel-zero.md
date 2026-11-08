# Module 12 — 数据并行训练、all-reduce 与 ZeRO

- 课程出处：CS229S Fall 2023《Parallelism Fundamentals》、Ce Zhang 客座（去中心化训练） · CMU 15-849《Data and Model Parallelism》《Zero Redundancy Optimizer》
- 必读论文：Rajbhandari et al.，ZeRO，2020；Li et al.，PyTorch Distributed（DDP），2020；Zhao et al.，PyTorch FSDP，2023；Patarasuk & Yuan，Bandwidth Optimal All-reduce Algorithms，2009；Lin et al.，Deep Gradient Compression，2018；Yuan et al.，Decentralized Training of Foundation Models in Heterogeneous Environments，2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：ZeRO 分片显存与公式差 0.099%，分片 AdamW 逐位相等）
- 状态：初稿

## 摘要

一张卡训练太慢，最直接的办法是多买几张卡、每张卡处理不同的数据。这就是数据并行。
这一讲要回答两个问题：卡多了以后，通信会不会成为瓶颈？显存又该怎么办？

数据并行让每张卡持有完整的模型、处理不同的样本，每一步用 all-reduce（把所有卡上的张量求和，再让每张卡都拿到结果）把梯度求平均。
先说通信。ring all-reduce 每张卡收发 `2(p−1)/p·S` 字节，这个量和卡数几乎无关，所以数据并行的通信是可以扩展的。
DDP 还会把梯度分桶，在反向传播还没结束时就开始通信，把通信藏到计算后面。

真正的问题在显存。混合精度 Adam 每个参数要 16 字节，而每张卡都存一份完整副本，这是纯粹的冗余。
ZeRO 分阶段把优化器状态、梯度和参数切分到各张卡上：ZeRO-1/2 的通信量和普通数据并行一样，ZeRO-3 多 50%。
配套实验里，ZeRO 分片后的实测显存和公式只差 0.099%，分片 AdamW 的结果和不分片的逐位相等。

如果卡之间的网络很慢，还可以用梯度压缩和去中心化训练来降低通信。
贯穿全讲的取舍是：切得越碎，显存越省，但通信越多，对网络也越敏感。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. all-reduce = reduce-scatter + all-gather。ring 实现每卡收发 `2(p−1)/p·S`，时间是 `2(p−1)α + 2(p−1)/p·S/β`（α 是每条消息的延迟，β 是链路带宽）。
2. 混合精度 Adam 的静态显存是 16Ψ 字节：bf16 权重 2Ψ，bf16 梯度 2Ψ，fp32 主权重、m、v 共 12Ψ。7B 模型就是 112 GB。
3. ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 再切参数。通信量分别是 2Ψ、2Ψ、3Ψ（按元素数），所以 ZeRO-3 是 DP 的 1.5 倍。
4. DDP 按 bucket（默认 25 MB）在反向过程中逐个发起 all-reduce，和剩下的反向计算重叠。bucket 太小，时间被延迟 α 支配；太大，能重叠的部分就少了。
5. bus bandwidth = algbw × 2(p−1)/p。它衡量 all-reduce 有没有跑满链路，可以直接和 NVLink 的单向带宽比较。

**必须能默写的公式**

```text
α-β 模型：发送 n 字节耗时 α + n/β
ring all-reduce：T = 2(p−1)·α + 2(p−1)/p · S/β
reduce-scatter / all-gather（各一半）：T = (p−1)·α + (p−1)/p · S/β
tree（流水化、双二叉树）：T ≈ 2·log₂p·α + 2·S/β（近似）
busbw = (S / T) · 2(p−1)/p
ZeRO 每卡显存（K = 12）：
  DP     (2 + 2 + K)Ψ          = 16Ψ
  ZeRO-1 4Ψ + KΨ/N
  ZeRO-2 2Ψ + (2 + K)Ψ/N
  ZeRO-3 (2 + 2 + K)Ψ/N        = 16Ψ/N
通信（每卡、元素数）：DP 2Ψ；ZeRO-1/2 2Ψ（RS Ψ + AG Ψ）；ZeRO-3 3Ψ（前向 AG + 反向 AG + RS）
```

**一句话区分**

- all-reduce 和 reduce-scatter + all-gather：两者结果相同；拆开之后的中间状态是「每张卡持有 1/p 的和」，这正是 ZeRO 需要的分片。
- DDP 和 FSDP：DDP 每张卡有完整参数，只同步梯度；FSDP（ZeRO-3）连参数也分片，用到时临时 all-gather 回来。
- ring 和 tree：ring 带宽最优，但延迟随 p 线性增长；tree 的延迟是 `log p`，大消息时要用双二叉树才能接近带宽最优。

## 学习目标

- 定义五种集合通信原语，写出 ring 实现并推导其 α-β 代价；
- 比较 ring、tree、双二叉树的延迟与带宽项，解释 bus bandwidth；
- 说清 DDP 的梯度分桶与计算通信重叠；
- 推导 ZeRO 各阶段的显存与通信，解释 ZeRO-3 为什么是 1.5 倍；
- 说清 FSDP 的 unit 包装、预取和峰值显存；
- 评估梯度压缩（量化、top-k + error feedback）与去中心化训练的适用条件。

## 先修知识

- [M06](module-06-data-parallel-primitives.md) 的 reduce/scan 与 parameter server 对照；
- [M11](module-11-frameworks-autodiff.md) 的反向传播、6ND 与训练显存。

## 核心概念

### 1. 数据并行

数据并行是最简单的多卡训练方式，几乎所有大规模训练都以它为基础，再叠加别的并行。

p 张卡各持一份模型，每张卡处理 `B/p` 个样本。反向得到局部梯度之后，所有卡把梯度求平均，再各自做完全相同的优化器更新。
数学上，这等价于用全局 batch B 做单机 SGD（前提是 BN 这类跨样本的算子要另外同步）。
计算量随卡数线性扩展，而每张卡的通信量保持恒定，所以它扩展得很好。

### 2. 集合通信原语

多卡之间交换数据有几种固定的模式，叫集合通信原语。后面讲的 DDP、ZeRO、MoE 都是这几种原语的组合，所以先把它们认清：

| 原语 | 输入（每卡） | 输出（每卡） |
| --- | --- | --- |
| broadcast | root 有 S | 全部有 S |
| reduce | 各有 S | root 有 ΣS |
| all-gather | 各有 S/p 的一片 | 全部有完整 S |
| reduce-scatter | 各有 S | 各有 ΣS 的 1/p |
| all-reduce | 各有 S | 全部有 ΣS |

其中最重要的关系是 all-reduce = reduce-scatter + all-gather。另外还有 all-to-all（MoE 和 embedding 会用到），它让每张卡给其他每张卡发不同的数据。

### 3. ring、tree 与 NCCL

同一个 all-reduce 可以有不同的实现，它们在「延迟」和「带宽」两项上各有取舍。

ring 把 p 张卡连成一个环，数据切成 p 块。reduce-scatter 走 p−1 步，all-gather 再走 p−1 步，每一步每张卡只发一块 `S/p`。
它的带宽项 `2(p−1)/p·S/β` 已经是下界，因为每张卡至少要把自己数据的 (p−1)/p 发出去，再把结果收回来。
但它的延迟项 `2(p−1)α` 随 p 线性增长。

tree 先 reduce 到根，再 broadcast 下去，延迟只有 `O(log p)`。但在朴素的二叉树里，内部节点要同时接收两个孩子的数据，所以带宽只能用到一半。

NCCL 的双二叉树解决了这个问题：两棵互补的二叉树各承担一半数据，每个节点在一棵树里是叶子，在另一棵树里是内部节点。
这样延迟是 `log p`，带宽也接近用满。NCCL 会根据消息大小和拓扑自动选择 ring 或 tree，在多 NIC 的机器上还会同时开多个 ring（channel）。

### 4. DDP：分桶与重叠

如果等反向全部算完再一次性 all-reduce，通信时间就会完整地加到每一步上。DDP 的目标是把它藏进反向计算里。

DDP 在每个参数上挂一个 autograd hook。反向传播按层倒序产出梯度，一旦某个 bucket 里的梯度装满，就立即异步发起这个 bucket 的 all-reduce，
同时反向继续计算前面的层。

最后一个 bucket 的通信没有后续计算可以掩盖，所以 bucket 的顺序要按「反向产出梯度的顺序」来排。
`bucket_cap_mb` 默认是 25；设置 `gradient_as_bucket_view=True` 可以让梯度直接成为 bucket 的视图，省掉一次拷贝。

### 5. ZeRO

通信解决了，显存却没有。7B 模型光静态状态就要 112 GB，一张 80 GB 的卡放不下，而数据并行要求每张卡都放一整份。

ZeRO-DP 的出发点是一个简单的观察：数据并行里，每张卡存的优化器状态、梯度和参数完全一样，这是 N 倍的冗余。它分三个阶段消除这份冗余：

- ZeRO-1：每张卡只存 1/N 的 fp32 优化器状态，只负责更新自己那一份参数。梯度做 reduce-scatter 后，每张卡拿到自己分片的梯度和，更新完再 all-gather 出 bf16 参数。
- ZeRO-2：梯度也只保留自己的分片（reduce-scatter 之后丢掉其余部分），显存进一步下降，通信量仍和 DP 相同。
- ZeRO-3：参数也分片。前向算到某一层时，all-gather 这一层的参数，用完就释放；反向时再 all-gather 一次，算完梯度后 reduce-scatter。

### 6. FSDP

ZeRO-3 的思路讲清楚了，落到 PyTorch 里就是 FSDP。它的工程细节决定了峰值显存和通信效率。

FSDP 按 wrap policy 划分 unit（通常每个 Transformer block 一个），把每个 unit 的参数拼成一个 FlatParameter 再分片。
前向时，先 all-gather 当前 unit，计算，然后释放（`reshard_after_forward`）；反向时，再 all-gather 一次，计算梯度，然后 reduce-scatter。

为了不让通信卡住计算，FSDP 会预取：前向时提前发出下一个 unit 的 all-gather，反向时提前发出前一个 unit 的 all-gather。
所以峰值显存约等于分片后的全部状态，再加上 1–2 个 unit 的完整参数和梯度。
unit 太大，峰值显存就高；unit 太小，通信消息变多，延迟项就变大。

### 7. 梯度压缩

前面的估算都假设卡之间有 NVLink 这样的高速链路。如果网络很慢，通信藏不住了，一个自然的想法是少发一点。

- 量化：1-bit SGD、QSGD（用随机量化保持无偏），以及 1-bit Adam（先用全精度 Adam 预热、冻结方差，再压缩动量并做误差补偿）；
- 稀疏化：只发 top-k（如 0.1%–1%）的分量，没发出去的部分累加进本地残差，下一步再发（error feedback），否则收敛会变差；
- 低秩：PowerSGD 用低秩矩阵近似梯度，可以直接对低秩因子做 all-reduce。

压缩不是免费的。压缩和解压本身要花计算；top-k 选出的索引每张卡都不同，不能直接 all-reduce，只能 all-gather。
所以在 NVLink 上通常不划算，只有在慢网络上才有意义。

### 8. 去中心化与异构训练（Ce Zhang 客座）

再进一步，如果连「每一步全局同步一次」都负担不起，比如卡分布在不同地域，就需要去掉全局 all-reduce。

去中心化 SGD 让每个节点只和邻居做 gossip 平均，单步通信量与 p 无关，对慢节点也不敏感；代价是各个节点上的模型之间存在共识误差。

在异构、跨地域的网络上，Yuan et al.（NeurIPS 2022）把训练切成数据并行组和流水线组，按网络拓扑搜索任务分配，让通信量大的交换尽量走快链路。
CocktailSGD 则组合随机稀疏化、top-k 和量化，用于慢网络上的微调。
这些工作的共同结论是：网络越慢，越需要压缩和拓扑感知的调度。

### 9. 大 batch 训练

数据并行还有一个容易忽略的副作用：卡数翻倍，全局 batch 也翻倍。batch 变大之后，学习率怎么调、收敛会不会变差？

线性缩放规则是：batch 乘 k，学习率也乘 k，并配合 warmup（Goyal et al. 2017）。
但这有上限。超过临界 batch 之后，多加的样本几乎不再减少所需的步数，等于在浪费算力。
LARS/LAMB 按层归一化更新幅度，可以把可用的 batch 再推高一些。

## 关键推导

### 推导 1：ring all-reduce 的代价

这个推导解释为什么 ring 的通信量和卡数几乎无关。把它一步步数出来：

```text
数据 S 切成 p 块，每块 S/p
reduce-scatter：p−1 步，每步每卡发 S/p、收 S/p 并累加 → (p−1)·(α + S/(pβ))
all-gather：    p−1 步，每步转发一块已完成的和         → (p−1)·(α + S/(pβ))
合计 T = 2(p−1)·α + 2(p−1)/p · S/β
p → ∞：带宽项 → 2S/β，与 p 无关；延迟项 ∝ p
```

为什么说这已经是最优？每张卡的 S 里有 (p−1)/p 要贡献给别的卡的分片，所以至少要发出 `(p−1)/p·S`；
它还要收到其余 (p−1)/p 的结果，所以至少再收 `(p−1)/p·S`。
在全双工链路上，「发」和「收」可以并行，但 reduce-scatter 和 all-gather 之间有依赖，所以 ring 达到的 `2(p−1)/p·S/β` 是带宽最优的。

### 推导 2：ZeRO-3 为什么是 1.5 倍

ZeRO-1/2 不增加通信，ZeRO-3 却多了 50%。逐项对比三种情况就能看出多在哪里：

```text
DP：     all-reduce 梯度 = RS(Ψ) + AG(Ψ)                       = 2Ψ
ZeRO-1/2：梯度 RS(Ψ) → 本地更新自己分片 → AG 参数(Ψ)            = 2Ψ
ZeRO-3： 前向 AG 参数(Ψ) + 反向 AG 参数(Ψ) + 梯度 RS(Ψ)           = 3Ψ = 1.5 × DP
```

ZeRO-3 多出来的那次 all-gather，是因为前向用完就把参数释放了，反向还得再取一次。
如果显存允许把参数保留下来（`reshard_after_forward=False`，也就是 ZeRO-2 式的行为），通信量就回到 2Ψ。

## 深入推导

### 代码：ring all-reduce 的逐步模拟

下面这段模拟把 ring 的两个阶段写成了两个循环，可以对照推导 1 看每一步发了什么：

```python
import numpy as np

def ring_allreduce(xs):
    p = len(xs)
    chunks = [np.array_split(x.astype(float), p) for x in xs]
    for s in range(p - 1):                                    # reduce-scatter
        sent = [chunks[r][(r - s) % p].copy() for r in range(p)]
        for r in range(p):
            src = (r - 1) % p                                 # 从左邻居收
            chunks[r][(src - s) % p] += sent[src]
    # 此时 rank r 持有第 (r+1)%p 块的完整和
    for s in range(p - 1):                                    # all-gather
        sent = [chunks[r][(r + 1 - s) % p].copy() for r in range(p)]
        for r in range(p):
            src = (r - 1) % p
            chunks[r][(src + 1 - s) % p] = sent[src]
    return [np.concatenate(c) for c in chunks]

xs = [np.random.rand(1003) for _ in range(8)]
assert all(np.allclose(o, sum(xs)) for o in ring_allreduce(xs))
```

每一步每个 rank 只发一块 `S/p`，2(p−1) 步加起来是 `2(p−1)/p·S`，和公式逐项对应。
如果只跑前半段，得到的就是 reduce-scatter：每个 rank 持有一块完整的和。ZeRO 的分片正是从这里来的。

### 代码：DDP 分桶与压缩 hook

DDP 允许为每个 bucket 注册一个通信 hook。下面这个 hook 先把梯度压成 bf16 再 all-reduce：

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def bf16_compress_hook(state, bucket):
    buf = bucket.buffer()                                     # 这个 bucket 的扁平梯度
    t = buf.to(torch.bfloat16).div_(dist.get_world_size())
    fut = dist.all_reduce(t, async_op=True).get_future()     # 异步：反向继续算前面的层
    return fut.then(lambda f: buf.copy_(f.value()[0]))       # 解压回原 dtype

model = DDP(model.cuda(), bucket_cap_mb=25, gradient_as_bucket_view=True)
model.register_comm_hook(state=None, hook=bf16_compress_hook)
```

PyTorch 已经内置了等价的 `default_hooks.bf16_compress_hook` 和 `powerSGD_hook`。这里写成显式版本，是为了看清「每个 bucket 对应一个异步 all-reduce」。

### 代码：top-k 稀疏化与误差反馈

error feedback 的实现很短，关键在于没发出去的部分不能丢，要留到下一步：

```python
def topk_with_error_feedback(grad, residual, ratio=0.01):
    acc = grad + residual                                     # 加上历史上没发出去的部分
    k = max(1, int(acc.numel() * ratio))
    idx = acc.abs().flatten().topk(k).indices
    vals = acc.flatten()[idx]
    residual = acc.clone(); residual.view(-1)[idx] = 0        # 发出的清零，其余留到下一步
    return idx, vals, residual                                # idx/vals 各卡不同 → 需要 all-gather
```

## 白板估算

参数取 H100 SXM 80 GB，NVLink4 450 GB/s（单向），IB NDR 50 GB/s/卡（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）；计算器 [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `train-mem`、`allreduce` 用的是同一组公式。
六道题依次回答：显存够不够、通信要多久、能不能藏住、跨节点怎么办、压缩值不值、实测带宽怎么看。

```text
1) 7B（Ψ = 7e9），8 卡，混合精度 Adam，不含激活
   DP     16·7e9                   = 112 GB      放不下
   ZeRO-1 4·7e9 + 12·7e9/8 = 28 + 10.5 = 38.5 GB
   ZeRO-2 2·7e9 + 14·7e9/8 = 14 + 12.25 = 26.25 GB
   ZeRO-3 112/8                    = 14 GB

2) 7B bf16 梯度 all-reduce，8×H100，节点内
   S = 14e9 B；每卡收发 2·(7/8)·14e9 = 24.5e9 B
   T = 24.5e9 / 450e9 ≈ 54 ms
   ZeRO-3：3 次各 (7/8)·14e9 = 12.25e9 → 36.75e9 / 450e9 ≈ 82 ms

3) 这些通信能否被藏住：每卡 16 条 × 4096 token = 65,536 token/步，MFU 40%
   计算 = 6·7e9·65536 / (0.4·989e12) = 2.75e15 / 3.96e14 ≈ 6.96 s
   54 ms / 6.96 s ≈ 0.8% → 节点内数据并行的通信几乎免费

4) 跨 2 节点 16 卡，层次化 all-reduce
   节点内 RS：(7/8)·14e9 / 450e9 ≈ 27 ms → 每卡持 1.75e9 B 的分片和
   跨节点：同号卡两两 all-reduce，2·(1/2)·1.75e9 / 50e9 = 35 ms（8 张 NIC 并行）
   节点内 AG：≈ 27 ms                                   合计 ≈ 89 ms（未计流水重叠）
   常见错答：「16 卡 ring 全走 IB：2·(15/16)·14e9/50e9 ≈ 525 ms」——那是只开一个 ring、只用一张 NIC 的情形

5) top-k 1%（值 bf16 2 B + 索引 int32 4 B）
   每卡要发的稀疏梯度 0.01·7e9·6 B = 0.42e9 B；all-gather 每卡收 (p−1)·0.42e9
   p = 8：2.94e9 B，比 ring 的 24.5e9 B 少约 8 倍
   p = 64：26.5e9 B，已超过 ring 的 2·(63/64)·14e9 ≈ 27.6e9 B 的 96% → 压缩优势随 p 消失

6) bus bandwidth：若 8 卡上 14e9 B 的 all-reduce 实测 60 ms，algbw = 14e9 / 0.06 ≈ 233 GB/s；busbw = 233 · 2·7/8 ≈ 408 GB/s ≈ 450 的 91%
```

第 3 题是整讲最重要的结论之一：节点内的梯度同步只占一步时间的 0.8%，所以节点内的数据并行，通信几乎是免费的，真正的问题是显存。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| ring all-reduce / RS / AG | [15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md) | 模拟结果等于逐元素求和；记录每步字节数等于 `S/p` |
| ZeRO 显存账 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `training_memory_bytes` | `--params 7e9 --dp 8 --zero 3` 输出 14 GB |
| all-reduce 时间 | 同文件 `ring_allreduce_time` | `--bytes 14e9 --gpus 8 --bw 450e9` 约 54 ms |
| DDP 重叠 | 同实验，profiler 时间线 | 反向期间可见 NCCL kernel 与计算 kernel 并行 |

## 易错点与反例

- 常见错答：「all-reduce 的通信量随卡数线性增长」。ring 每卡是 `2(p−1)/p·S`，p 从 8 增加到 1024，也只是从 1.75S 涨到约 2S；随卡数增长的是延迟项。
- 常见错答：「ZeRO 都会增加通信」。ZeRO-1/2 与 DP 相同（RS + AG 正好等于一次 all-reduce），只有 ZeRO-3 多 50%。
- 常见错答：「ZeRO-3 就是模型并行」。ZeRO-3 在计算每一层时参数是完整的（已经 all-gather 回来），计算本身没有切分；张量并行才会把单个 GEMM 切开。
- 把 NVLink 的 900 GB/s（双向）代进 ring 公式。公式里的 β 是单向带宽，H100 应该用 450。
- bucket 设得很小（如 1 MB）。消息数会上千，`2(p−1)α` 累积起来反而压过带宽项。
- 梯度累积时每个 micro-batch 都做 all-reduce。在不是最后一步的 micro-batch 上，应该用 `no_sync()` 跳过同步。
- top-k 不做 error feedback。长期被丢弃的小分量会造成偏差，收敛变差。
- 扩卡时学习率不变。全局 batch 变大、步数变少，等于学习率相对变小，最后收敛到更差的点。

## 实验或练习

这个练习把本讲的两个 U 形取舍（消息大小、bucket 大小）放到真机上看。

- 假设：在单机多卡上，all-reduce 的 busbw 随消息大小上升，在数十 MB 之后饱和；DDP 的 step 时间对 `bucket_cap_mb` 呈 U 形。
- 配置：用 `nccl-tests` 的 `all_reduce_perf -b 8 -e 1G -f 2`；用 DDP 训练 GPT-2 124M，`bucket_cap_mb ∈ {1, 5, 25, 100, 1000}`；再用 FSDP 对比 `reshard_after_forward` 开和关时的显存与吞吐。
- 预期：小消息时 busbw 远低于链路带宽（延迟主导）；bucket 极小和极大都更慢；关闭 reshard 后显存升高、通信减少约 1/3。具体数值以实测为准。

## 高频问题

- 问：bus bandwidth 是什么？为什么不直接看 algbw？
  答：algbw 是 `S/T`；busbw 再乘上 `2(p−1)/p`，换算成每条链路实际承载的带宽，所以可以直接和 NVLink 单向带宽比较，看有没有跑满。
- 问：梯度累积时 DDP 要注意什么？
  答：不是最后一个 micro-batch 时用 `no_sync()` 跳过 all-reduce，否则每个 micro-batch 都白白同步一次。
- 问：DDP 怎么把通信藏起来？
  答：梯度分桶，某个 bucket 的梯度一凑齐就异步 all-reduce，和前面层的反向计算重叠。
- 问：ZeRO 的三个阶段各切什么？
  答：ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 再切参数；每卡显存从 16Ψ 降到 16Ψ/N。
- 问：ZeRO-3 的通信为什么是 1.5 倍？
  答：前向和反向各 all-gather 一次参数，再 reduce-scatter 一次梯度，共 3Ψ，而 DP 是 2Ψ。

## 面试追问链

1. 问：定义 all-reduce，并说出它的两种分解。
   答：每卡输入 S，输出全体之和。它可以分解为 reduce + broadcast（树形），或者 reduce-scatter + all-gather（ring），后者带宽最优。
2. 问：推导 ring all-reduce 的 α-β 时间。
   答：切成 p 块，RS 和 AG 各走 p−1 步，每步 `α + S/(pβ)`，合计 `2(p−1)α + 2(p−1)/p·S/β`。
3. 问：卡数到 1024 时，ring 会出什么问题？怎么办？
   答：延迟项 `2(p−1)α` 线性增长，小消息会被延迟支配。办法有：用双二叉树（延迟 `log p`）、层次化 all-reduce（节点内走 NVLink、节点间走 IB）、增大 bucket。
4. 问：7B 模型用 8 卡训练，ZeRO 各阶段每卡显存是多少？
   答：DP 112 GB；ZeRO-1 38.5 GB；ZeRO-2 26.25 GB；ZeRO-3 14 GB。另外还要加上激活。
5. 问：既然 ZeRO-2 通信不变、显存也大降了，什么时候必须上 ZeRO-3？
   答：当 `2Ψ + 14Ψ/N` 加上激活仍然放不下，也就是连 bf16 参数本身的 2Ψ 都成了负担时（例如 70B 的 bf16 参数就有 140 GB）。代价是多 50% 通信：节点内的 NVLink 一般扛得住，跨节点就要算一算能不能被计算掩盖。
6. 问：FSDP 的 unit 怎么划分？预取有什么代价？
   答：通常每个 Transformer block 一个 unit。预取让下一个 unit 的 all-gather 和当前计算重叠，代价是同时驻留两个 unit 的完整参数，峰值显存上升。
7. 问：梯度压缩在什么场景下值得做？
   答：通信无法被计算掩盖、而且链路慢的时候（跨节点低带宽、跨地域）。在 NVLink 上，54 ms 对 7 s 的计算可以忽略，压缩只会增加计算和收敛风险；top-k 还要 all-gather，卡数一大优势就消失了。
8. 问：你要在 4 个机房、每个机房 8×H100、机房之间只有 10 Gb/s 网络的条件下微调 7B 模型，怎么设计？
   答：机房内用 FSDP/ZeRO 加 NVLink 做数据并行。机房之间不要每一步都做全精度 all-reduce（14 GB 在 1.25 GB/s 上要十秒级），改成本地走多步再同步（local SGD / 去中心化 gossip），或者强压缩（量化 + top-k + error feedback）；按拓扑把通信量大的切分放在机房内。验收看收敛曲线，而不是吞吐。

## 自测题

1. 8 卡 ring all-reduce 一共几步？每步每卡发多少？
   要点：`2(p−1) = 14` 步，每步 `S/8`。
2. 推导 ZeRO-1 的每卡显存公式，并代入 Ψ=70e9、N=64。
   要点：`4Ψ + 12Ψ/N = 280 GB + 13.1 GB ≈ 293 GB`，单卡放不下，必须配合 ZeRO-3 或张量并行。
3. 证明 reduce-scatter 的带宽下界是 `(p−1)/p·S/β`。
   要点：每张卡最终的分片依赖其余 p−1 张卡各 `S/p` 的数据，所以至少要收 `(p−1)S/p`。
4. 已知 all-reduce 的 busbw 为 400 GB/s，16 卡同步 2 GB 梯度要多久？
   要点：`T = 2·(15/16)·2e9 / 400e9 ≈ 9.4 ms`。
5. 设计题：DDP 里最后一个 bucket 的通信无法隐藏，怎样缩短这段尾巴？
   要点：让最后发出的 bucket 尽量小（第一层参数单独分桶），或者把下一步的前向和上一步的通信重叠（ZeRO/FSDP 的参数预取思路）。

## 关联笔记

- [M11 深度学习框架与自动微分](module-11-frameworks-autodiff.md)：反向产出梯度的顺序决定 bucket 顺序。
- [M13 张量并行与流水线并行](module-13-tensor-pipeline-parallel.md)：数据并行与 TP/PP 组合成 3D 并行。
- [M14 显存优化](module-14-memory-optimization.md)：ZeRO-Offload/Infinity 把分片后的状态再放到 CPU/NVMe。
- [M06 数据并行原语](module-06-data-parallel-primitives.md)：reduce/scan 与 parameter server 的对照。
- [M18 MoE 与集群](module-18-architectures-moe-clusters.md)：all-to-all 与集群网络拓扑。

## 本讲小结

数据并行的通信是可以扩展的：ring all-reduce 每卡约 2S，DDP 用分桶把它藏进反向计算。显存却不能扩展：16Ψ 的冗余副本要靠 ZeRO 逐级切掉，
ZeRO-1/2 不增加通信，ZeRO-3 多 50%。在慢网络上可以用压缩和去中心化来减少通信，代价是收敛。

数据并行切不开单层的计算，下一讲用张量并行和流水线并行来切模型本身。

## 延伸阅读

- Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, SC 2020, https://arxiv.org/abs/1910.02054
- Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training, VLDB 2020, https://arxiv.org/abs/2006.15704
- Zhao et al., PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, VLDB 2023, https://arxiv.org/abs/2304.11277
- Patarasuk & Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, JPDC 2009.
- Lin et al., Deep Gradient Compression, ICLR 2018；Tang et al., 1-bit Adam, ICML 2021；Vogels et al., PowerSGD, NeurIPS 2019.
- Yuan et al., Decentralized Training of Foundation Models in Heterogeneous Environments, NeurIPS 2022, https://arxiv.org/abs/2206.01288
- Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017, https://arxiv.org/abs/1706.02677
