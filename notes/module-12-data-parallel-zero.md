# Module 12 — 数据并行训练、all-reduce 与 ZeRO

- 课程出处：CS229S Fall 2023《Parallelism Fundamentals》、Ce Zhang 客座（去中心化训练） · CMU 15-849《Data and Model Parallelism》《Zero Redundancy Optimizer》
- 必读论文：Rajbhandari et al.，ZeRO，2020；Li et al.，PyTorch Distributed（DDP），2020；Zhao et al.，PyTorch FSDP，2023；Patarasuk & Yuan，Bandwidth Optimal All-reduce Algorithms，2009；Lin et al.，Deep Gradient Compression，2018；Yuan et al.，Decentralized Training of Foundation Models in Heterogeneous Environments，2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：ZeRO 分片显存与公式差 0.099%，分片 AdamW 逐位相等）
- 状态：初稿

## 摘要

数据并行让每张卡持有完整模型、处理不同样本，每步用 all-reduce 把梯度求平均。ring all-reduce 每卡收发 `2(p−1)/p·S`，
与卡数几乎无关，所以数据并行的通信可以扩展；DDP 再把梯度分桶，在反向还没结束时就开始通信，把通信藏到计算后面。数据并行的真正问题是显存：混合精度 Adam 每参数 16 字节，每张卡都存一份完整副本是纯冗余。ZeRO 按阶段把优化器状态、梯度、参数切到各卡，
ZeRO-1/2 通信量不变，ZeRO-3 多 50%。跨慢网络时再用梯度压缩和去中心化训练降低通信。主线取舍：切得越碎显存越省，但通信越多、对网络越敏感。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. all-reduce = reduce-scatter + all-gather；ring 实现每卡收发 `2(p−1)/p·S`，时间 `2(p−1)α + 2(p−1)/p·S/β`。
2. 混合精度 Adam 静态显存 16Ψ 字节：bf16 权重 2Ψ + bf16 梯度 2Ψ + fp32 主权重/m/v 12Ψ；7B 就是 112 GB。
3. ZeRO-1 切优化器状态，ZeRO-2 再切梯度，ZeRO-3 再切参数；通信量分别是 2Ψ、2Ψ、3Ψ（元素数），ZeRO-3 是 DP 的 1.5 倍。
4. DDP 按 bucket（默认 25 MB）在反向过程中逐个发起 all-reduce，与剩余反向计算重叠；bucket 太小受延迟 α 支配，太大重叠变少。
5. bus bandwidth = algbw × 2(p−1)/p，是衡量 all-reduce 是否跑满链路的指标，可直接与 NVLink 单向带宽比较。

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

- all-reduce vs reduce-scatter + all-gather：结果相同；拆开后中间态「每卡持有 1/p 的和」正是 ZeRO 需要的分片。
- DDP vs FSDP：DDP 每卡完整参数，只同步梯度；FSDP（ZeRO-3）参数也分片，用时临时 all-gather。
- ring vs tree：ring 带宽最优、延迟随 p 线性增长；tree 延迟 `log p`，大消息时双二叉树才能接近带宽最优。

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

p 张卡各持一份模型，各处理 `B/p` 个样本，反向得到局部梯度后求平均，再各自做相同的优化器更新。
数学上等价于全局 batch B 的单机 SGD（前提是 BN 之类的跨样本算子另行同步）。计算随卡数线性扩展，通信量每卡恒定。

### 2. 集合通信原语

| 原语 | 输入（每卡） | 输出（每卡） |
| --- | --- | --- |
| broadcast | root 有 S | 全部有 S |
| reduce | 各有 S | root 有 ΣS |
| all-gather | 各有 S/p 的一片 | 全部有完整 S |
| reduce-scatter | 各有 S | 各有 ΣS 的 1/p |
| all-reduce | 各有 S | 全部有 ΣS |

all-reduce = reduce-scatter + all-gather。另有 all-to-all（MoE、embedding 用），每卡给每卡发不同数据。

### 3. ring、tree 与 NCCL

ring：p 卡连成环，数据切 p 块，reduce-scatter 走 p−1 步，all-gather 再走 p−1 步，每步每卡发一块 `S/p`。带宽项 `2(p−1)/p·S/β` 是下界（每卡至少要把自己数据的 (p−1)/p 发出去并收回结果），但延迟项 `2(p−1)α` 随 p 线性增长。
tree：reduce 到根再 broadcast，延迟 `O(log p)`；朴素二叉树中内部节点要接收两个孩子的数据，带宽只有一半。
NCCL 的双二叉树：两棵互补的二叉树各承担一半数据，每个节点在一棵树里是叶子、在另一棵里是内部节点，于是延迟 `log p` 且带宽接近满。NCCL 根据消息大小和拓扑自动选 ring 或 tree，在多 NIC 的机器上同时开多个 ring/channel。

### 4. DDP：分桶与重叠

DDP 在每个参数上挂 autograd hook。反向按层倒序产出梯度，梯度装满一个 bucket 就立即异步发起该 bucket 的 all-reduce，
同时反向继续计算前面的层。最后一个 bucket 的通信无法被隐藏，所以 bucket 顺序按「反向产出顺序」排。`bucket_cap_mb` 默认 25；`gradient_as_bucket_view=True` 让梯度直接是 bucket 的视图，省一次拷贝。

### 5. ZeRO

ZeRO-DP 的观察：数据并行里每张卡存的优化器状态、梯度、参数完全一样，是 N 倍冗余。

- ZeRO-1：每卡只存 1/N 的 fp32 状态，只更新自己那份参数；梯度 reduce-scatter 后各卡拿到自己分片的和，更新后 all-gather bf16 参数。
- ZeRO-2：梯度也只保留自己分片（reduce-scatter 后丢掉其余），显存再降；通信与 DP 相同。
- ZeRO-3：参数也分片；前向到某层时 all-gather 该层参数，用完释放；反向再 all-gather 一次，算完梯度 reduce-scatter。

### 6. FSDP

PyTorch FSDP 是 ZeRO-3 的实现。按 wrap policy（通常每个 Transformer block 一个 unit）把参数拼成 FlatParameter 并分片。前向：all-gather 当前 unit → 计算 → 释放（`reshard_after_forward`）；反向：再 all-gather → 计算梯度 → reduce-scatter。
预取：前向时提前发下一个 unit 的 all-gather，反向时提前发前一个 unit 的 all-gather。峰值显存 ≈ 分片后的全部状态 + 1–2 个 unit 的完整参数与梯度。unit 太大，峰值显存高；太小，通信消息多、延迟项大。

### 7. 梯度压缩

- 量化：1-bit SGD、QSGD（随机量化保持无偏）、1-bit Adam（先用全精度 Adam 预热、冻结方差，再压缩动量并做误差补偿）；
- 稀疏化：只发 top-k（如 0.1%–1%）分量，未发送的部分累加进本地残差下步再发（error feedback），否则收敛会变差；
- 低秩：PowerSGD 用低秩近似梯度矩阵，可以直接 all-reduce 因子。
代价：压缩/解压计算，top-k 的索引各卡不同、不能直接 all-reduce（要 all-gather），在 NVLink 上通常不划算，在慢网络上才有意义。

### 8. 去中心化与异构训练（Ce Zhang 客座）

去中心化 SGD 去掉全局 all-reduce，每个节点只与邻居做 gossip 平均，单步通信与 p 无关、对慢节点不敏感，代价是模型间存在共识误差。
在异构、跨地域网络上，Yuan et al.（NeurIPS 2022）把训练切成数据并行组与流水线组，按网络拓扑搜索任务分配，让通信量大的交换尽量走快链路；CocktailSGD 组合随机稀疏化、top-k 与量化做慢网微调。结论：网络越慢越需要压缩与拓扑感知调度。

### 9. 大 batch 训练

数据并行扩卡等于扩全局 batch。线性缩放规则：batch 乘 k，学习率乘 k，配合 warmup（Goyal et al. 2017）。超过临界 batch 后，多加的样本几乎不减少所需步数，相当于浪费算力；LARS/LAMB 按层归一化更新幅度，把可用 batch 再推高一些。

## 关键推导

### 推导 1：ring all-reduce 的代价

```text
数据 S 切成 p 块，每块 S/p
reduce-scatter：p−1 步，每步每卡发 S/p、收 S/p 并累加 → (p−1)·(α + S/(pβ))
all-gather：    p−1 步，每步转发一块已完成的和         → (p−1)·(α + S/(pβ))
合计 T = 2(p−1)·α + 2(p−1)/p · S/β
p → ∞：带宽项 → 2S/β，与 p 无关；延迟项 ∝ p
```

下界论证：每张卡的 S 中有 (p−1)/p 要贡献给别人的分片，至少发出 `(p−1)/p·S`；它也要收到其余 (p−1)/p 的结果，至少再收 `(p−1)/p·S`。
在全双工链路上「发」和「收」可并行，但 reduce-scatter 和 all-gather 有依赖，ring 达到 `2(p−1)/p·S/β` 是带宽最优的。

### 推导 2：ZeRO-3 为什么是 1.5 倍

```text
DP：     all-reduce 梯度 = RS(Ψ) + AG(Ψ)                       = 2Ψ
ZeRO-1/2：梯度 RS(Ψ) → 本地更新自己分片 → AG 参数(Ψ)            = 2Ψ
ZeRO-3： 前向 AG 参数(Ψ) + 反向 AG 参数(Ψ) + 梯度 RS(Ψ)           = 3Ψ = 1.5 × DP
```

ZeRO-3 多出的那次 all-gather 是因为前向用完就释放了参数，反向还要再取一次。若显存允许保留（`reshard_after_forward=False`，即 ZeRO-2 式行为），就回到 2Ψ。

## 深入推导

### 代码：ring all-reduce 的逐步模拟

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

每一步每个 rank 只发一块 `S/p`，2(p−1) 步合计 `2(p−1)/p·S`，与公式逐项对应；只跑前半段就是 reduce-scatter，这正是 ZeRO 的分片来源。

### 代码：DDP 分桶与压缩 hook

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

PyTorch 内置了等价的 `default_hooks.bf16_compress_hook` 与 `powerSGD_hook`；写成显式版本是为了看清「每个 bucket 一个异步 all-reduce」。

### 代码：top-k 稀疏化与误差反馈

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

H100 SXM 80 GB，NVLink4 450 GB/s（单向），IB NDR 50 GB/s/卡（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）；计算器 [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `train-mem`、`allreduce` 用同一组公式。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| ring all-reduce / RS / AG | [15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md) | 模拟结果等于逐元素求和；记录每步字节数等于 `S/p` |
| ZeRO 显存账 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `training_memory_bytes` | `--params 7e9 --dp 8 --zero 3` 输出 14 GB |
| all-reduce 时间 | 同文件 `ring_allreduce_time` | `--bytes 14e9 --gpus 8 --bw 450e9` 约 54 ms |
| DDP 重叠 | 同实验，profiler 时间线 | 反向期间可见 NCCL kernel 与计算 kernel 并行 |

## 易错点与反例

- 常见错答：「all-reduce 通信量随卡数线性增长」。ring 每卡 `2(p−1)/p·S`，p 从 8 到 1024 只从 1.75S 涨到约 2S；增长的是延迟项。
- 常见错答：「ZeRO 都会增加通信」。ZeRO-1/2 与 DP 相同（RS + AG 正好等于 all-reduce），只有 ZeRO-3 多 50%。
- 常见错答：「ZeRO-3 = 模型并行」。ZeRO-3 每层计算时参数是完整的（all-gather 回来），计算不切分；张量并行才把单个 GEMM 切开。
- 把 NVLink 900 GB/s（双向）代进 ring 公式：公式里的 β 是单向带宽，H100 用 450。
- bucket 设得很小（如 1 MB）：消息数上千，`2(p−1)α` 累积起来压过带宽项。
- 梯度累积时每个 micro-batch 都 all-reduce：应在非最后一步用 `no_sync()` 跳过同步。
- top-k 不做 error feedback：被长期丢弃的小分量造成偏差，收敛变差。
- 扩卡时学习率不变：全局 batch 变大、步数变少，等于学习率相对变小，收敛到更差的点。

## 实验或练习

- 假设：在单机多卡上，all-reduce 的 busbw 随消息大小上升并在数十 MB 后饱和；DDP 的 step 时间对 `bucket_cap_mb` 呈 U 形。
- 配置：`nccl-tests` 的 `all_reduce_perf -b 8 -e 1G -f 2`；DDP 训练 GPT-2 124M，`bucket_cap_mb ∈ {1, 5, 25, 100, 1000}`；FSDP 对比 `reshard_after_forward` 开/关的显存与吞吐。
- 预期：小消息 busbw 远低于链路带宽（延迟主导）；bucket 极小与极大都更慢；关闭 reshard 显存升高、通信减少约 1/3。具体数值以实测为准。

## 高频问题

- 问：ring all-reduce 每卡收发多少字节？
  答：`2(p−1)/p·S`，接近 2S，与卡数几乎无关。
- 问：all-reduce 和 reduce-scatter、all-gather 的关系？
  答：all-reduce = reduce-scatter + all-gather，两半各 `(p−1)/p·S`。
- 问：DDP 怎么把通信藏起来？
  答：梯度分桶，某个 bucket 的梯度一齐就异步 all-reduce，与前面层的反向计算重叠。
- 问：ZeRO 三个阶段各切什么？
  答：1 切优化器状态，2 再切梯度，3 再切参数；显存从 16Ψ 降到 16Ψ/N。
- 问：ZeRO-3 的通信为什么是 1.5 倍？
  答：前向和反向各 all-gather 一次参数，再 reduce-scatter 梯度，共 3Ψ 对 DP 的 2Ψ。

## 面试追问链

1. 问：定义 all-reduce，说出它的两种分解。
   答：每卡输入 S、输出全体之和；= reduce + broadcast（树形）或 reduce-scatter + all-gather（ring），后者带宽最优。
2. 问：推导 ring all-reduce 的 α-β 时间。
   答：切 p 块，RS 与 AG 各 p−1 步，每步 `α + S/(pβ)`，合计 `2(p−1)α + 2(p−1)/p·S/β`。
3. 问：卡数到 1024 时 ring 有什么问题，怎么办？
   答：延迟项 `2(p−1)α` 线性增长，小消息被延迟支配；用双二叉树（延迟 `log p`）、层次化 all-reduce（节点内 NVLink、节点间 IB）、增大 bucket。
4. 问：7B 模型 8 卡训练，ZeRO 各阶段每卡显存多少？
   答：DP 112 GB；ZeRO-1 38.5 GB；ZeRO-2 26.25 GB；ZeRO-3 14 GB，另加激活。
5. 问：既然 ZeRO-2 通信不变、显存已大降，什么时候必须上 ZeRO-3？
   答：当 `2Ψ + 14Ψ/N` 加激活仍放不下，即 bf16 参数本身 2Ψ 都成负担时（如 70B 的 bf16 参数 140 GB）；代价是多 50% 通信，节点内 NVLink 一般扛得住，跨节点要算是否能被计算掩盖。
6. 问：FSDP 的 unit 怎么划分，预取有什么代价？
   答：通常每个 Transformer block 一个 unit；预取让下一个 unit 的 all-gather 与当前计算重叠，代价是同时驻留两个 unit 的完整参数，峰值显存上升。
7. 问：梯度压缩在什么场景下值得做？
   答：通信无法被计算掩盖且链路慢时（跨节点低带宽、跨地域）。NVLink 上 54 ms 对 7 s 的计算可以忽略，压缩只会增加计算和收敛风险；top-k 还要 all-gather，卡数大时优势消失。
8. 问：你要在 4 个机房、每个机房 8×H100、机房间只有 10 Gb/s 的网络上微调 7B 模型，怎么设计？
   答：机房内用 FSDP/ZeRO 加 NVLink 做数据并行；机房间不要每步全精度 all-reduce（14 GB 在 1.25 GB/s 上要十秒级），改成本地多步后同步（local SGD / 去中心化 gossip）或强压缩（量化 + top-k + error feedback）；按拓扑把通信量大的切分放在机房内。用收敛曲线而不是吞吐来验收。

## 自测题

1. 8 卡 ring all-reduce 共几步，每步每卡发多少？
   要点：`2(p−1) = 14` 步，每步 `S/8`。
2. 推导 ZeRO-1 的每卡显存公式，代入 Ψ=70e9、N=64。
   要点：`4Ψ + 12Ψ/N = 280 GB + 13.1 GB ≈ 293 GB`，单卡放不下，必须配合 ZeRO-3 或张量并行。
3. 证明 reduce-scatter 的带宽下界是 `(p−1)/p·S/β`。
   要点：每卡最终分片依赖其余 p−1 卡各 `S/p` 的数据，至少收 `(p−1)S/p`。
4. 已知 all-reduce 的 busbw 为 400 GB/s，16 卡同步 2 GB 梯度要多久？
   要点：`T = 2·(15/16)·2e9 / 400e9 ≈ 9.4 ms`。
5. 设计题：DDP 中最后一个 bucket 的通信无法隐藏，如何缩短这段尾巴？
   要点：让最后发出的 bucket 尽量小（第一层参数单独分桶），或者把下一步前向与上一步通信重叠（ZeRO/FSDP 的参数预取思路）。

## 关联笔记

- [M11 深度学习框架与自动微分](module-11-frameworks-autodiff.md)：反向产出梯度的顺序决定 bucket 顺序。
- [M13 张量并行与流水线并行](module-13-tensor-pipeline-parallel.md)：数据并行与 TP/PP 组合成 3D 并行。
- [M14 显存优化](module-14-memory-optimization.md)：ZeRO-Offload/Infinity 把分片后的状态再放到 CPU/NVMe。
- [M06 数据并行原语](module-06-data-parallel-primitives.md)：reduce/scan 与 parameter server 的对照。
- [M18 MoE 与集群](module-18-architectures-moe-clusters.md)：all-to-all 与集群网络拓扑。

## 本讲小结

数据并行的通信可以扩展：ring all-reduce 每卡约 2S，DDP 用分桶把它藏进反向计算。显存不能扩展：16Ψ 的冗余副本由 ZeRO 逐级切掉，
ZeRO-1/2 不加通信，ZeRO-3 加 50%。慢网络上的手段是压缩和去中心化，但要用收敛代价换通信。数据并行切不开单层计算，下一讲用张量并行和流水线并行切模型本身。

## 延伸阅读

- Rajbhandari et al., ZeRO: Memory Optimizations Toward Training Trillion Parameter Models, SC 2020, https://arxiv.org/abs/1910.02054
- Li et al., PyTorch Distributed: Experiences on Accelerating Data Parallel Training, VLDB 2020, https://arxiv.org/abs/2006.15704
- Zhao et al., PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel, VLDB 2023, https://arxiv.org/abs/2304.11277
- Patarasuk & Yuan, Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations, JPDC 2009.
- Lin et al., Deep Gradient Compression, ICLR 2018；Tang et al., 1-bit Adam, ICML 2021；Vogels et al., PowerSGD, NeurIPS 2019.
- Yuan et al., Decentralized Training of Foundation Models in Heterogeneous Environments, NeurIPS 2022, https://arxiv.org/abs/2206.01288
- Goyal et al., Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour, 2017, https://arxiv.org/abs/1706.02677
