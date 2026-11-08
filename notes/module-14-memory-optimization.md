# Module 14 — 显存优化：重计算、混合精度、offload

- 课程出处：CS229S Fall 2023《Tim Dettmers 客座：8-bit Methods for Efficient Deep Learning》《Parallelism Fundamentals》 · CMU 15-849《Tensor Rematerialization》《ZeRO-Offload / ZeRO-Infinity》
- 必读论文：Chen et al., Training Deep Nets with Sublinear Memory Cost, 2016；Jain et al., Checkmate, 2020；Kirisame et al., Dynamic Tensor Rematerialization, 2021；Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022；Micikevicius et al., Mixed Precision Training, 2018；Micikevicius et al., FP8 Formats for Deep Learning, 2022；Ren et al., ZeRO-Offload, 2021；Rajbhandari et al., ZeRO-Infinity, 2021；Dettmers et al., 8-bit Optimizers via Block-wise Quantization, 2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：16 B/参数实测 16.00–16.32；全量重计算 +18.9% 时间，batch 8 → 24）
- 状态：初稿

## 摘要

训练时报 OOM，第一反应往往是「模型太大」。但真正占显存的东西有好几块，每块的应对手段都不一样，所以先要分清是哪一块爆了。

训练显存由五块组成：权重、梯度、优化器状态、激活、临时缓冲与碎片。
前三块是静态的，和 batch 无关。在混合精度 Adam 下合计 16 B/参数，靠 ZeRO（M12）和 offload 来处理。
激活随 batch 和序列长度线性增长，其中 attention 部分随 s² 增长。对大模型来说它往往是最大的一块，靠重计算来处理。

重计算用算力换显存。全量重计算大约多出 1/3 的计算；选择性重计算只重算 attention 里 s² 大小的那部分，代价小得多。
混合精度则是用低精度做矩阵乘、用高精度存主权重。三种低精度格式缺的东西不一样：fp16 范围小，需要 loss scaling；bf16 范围够，不需要；fp8 两头都紧，需要逐张量或逐块的 scaling factor。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 混合精度 Adam 每参数 16 B：bf16 权重 2、bf16 梯度 2、fp32 主权重 4、m 4、v 4。激活另算。
2. 不做重计算、不做并行时，每层激活 ≈ s·b·h·(34 + 5·a·s/h) 字节。其中 5·a·s/h 这一项来自 attention 的 s² 矩阵，序列一长它就占大头。
3. √n 分段检查点把激活显存降到 O(√n)，代价是多跑一次前向，总计算约 +33%。
4. fp16 的动态范围小（最大 65504），小梯度会下溢成 0，所以必须做 loss scaling。bf16 的指数位和 fp32 一样，不需要。
5. 8-bit Adam 把 m、v 从 8 B 压到 2 B，每参数从 16 B 降到 10 B。

必须能默写的公式：

```text
静态显存：(2 + 2 + 12) · N 字节；8-bit Adam：(2 + 2 + 4 + 2) · N
每层激活（Korthikanti 近似，16 位激活）：
  无并行：            s·b·h·(34 + 5·a·s/h)
  TP=t：              s·b·h·(10 + 24/t + 5·a·s/(h·t))
  TP + SP：           s·b·h·(34 + 5·a·s/h) / t
  TP + SP + 选择性重算：s·b·h·34 / t
  全量重算：          每层只存输入 2·s·b·h
√n 检查点：每 k 层存一次，显存 ∝ n/k + k，k = √n 时最小为 2√n
全量重算计算量：前向 2N + 反向 4N + 重算 2N = 8N/token，比 6N 多 33%
```

一句话区分：

- fp16 与 bf16：fp16 是 5 位指数、10 位尾数，范围小、精度高；bf16 是 8 位指数、7 位尾数，范围与 fp32 相同。
- e4m3 与 e5m2：前者精度优先（最大 448），用于前向；后者范围优先（最大 57344），用于梯度。
- Checkmate 与 DTR：前者离线用 ILP 求最优重算计划；后者在运行时贪心驱逐，不需要静态图。
- ZeRO-3 与 ZeRO-Offload：前者把状态切到其他 GPU 上，后者把状态搬到 CPU 内存里。

## 学习目标

- 列出训练显存的五个组成部分，并能为给定模型算出每一块；
- 推导 Transformer 每层激活的 34·sbh 与 5·a·s²·b 两项的来源；
- 解释 √n 检查点、Checkmate、DTR、选择性重计算的区别；
- 说清 loss scaling 的动机与动态调整规则，fp32 主权重为什么必要；
- 估算 ZeRO-Offload 的 PCIe 与 CPU 内存带宽开销。

## 先修知识

- M11：反向传播需要保存哪些前向张量；
- M12：ZeRO 1/2/3 切分的是哪几块；
- M13：TP、SP、1F1B 的在途 micro-batch 数。

## 核心概念

### 1. 显存的五个组成

要优化显存，先得把账分清楚。五块里每一块的增长规律不同：

- 权重、梯度、优化器状态：与 batch 无关，合计 16 B/参数。7B 模型就是 107.8 GB，单卡 80 GB 放不下。
- 激活：反向传播需要的前向中间结果，正比于 b·s·L。
- 临时缓冲：GEMM 的 workspace、通信 bucket，以及 logits。logits 的大小是 b·s·V，词表 V=128256 时相当可观。
- 碎片：caching allocator 里已经保留、但用不上的块。nvidia-smi 看到的是 reserved，不是 allocated。

### 2. 激活为什么这么大

很多人以为大模型的显存主要花在参数上，但在长序列训练里，激活常常比参数还大。原因是反向传播要用到几乎每一步前向的中间结果。

一层 Transformer 要存的东西如下（16 位，MHA，4h 宽的 MLP）：

- attention 块共 11·sbh + 5·a·s²·b：QKV 输入 2sbh、Q 与 K 4sbh、softmax 输出 2as²b、dropout mask as²b、dropout 输出 2as²b、V 2sbh、输出投影输入 2sbh、dropout mask sbh。
- MLP 块共 19·sbh：输入 2sbh、GeLU 输入 8sbh、GeLU 输出 8sbh、dropout mask sbh。
- 两个 LayerNorm 的输入共 4·sbh。

加起来是 34·sbh + 5·a·s²·b。
这个式子只当量级估算用。FlashAttention 不物化 s² 矩阵，所以 5·a·s/h 这一项实际上会消失；Llama 类结构（GQA、门控 MLP、没有 dropout）的系数也不一样。

### 3. 重计算家族

激活大的根本原因是「存下来等反向用」。重计算（rematerialization）换一种思路：不存，反向需要时再算一遍。几种做法的区别在于决定「存哪些、重算哪些」的方式：

- Chen et al.：把 n 层分成 √n 段，只存段边界；反向时逐段重跑前向。显存是 O(√n)，计算多一次前向。
- Checkmate：把「第 t 步是否存储或重算节点 i」写成 0-1 变量，在显存约束下最小化总计算，用 ILP 求解。它适合静态图的离线规划。
- DTR：运行时显存不够了就驱逐张量，用 cost / (memory × staleness) 这类启发式挑驱逐对象，需要时再重算。它适合动态图。
- 选择性重计算：只重算 attention 里 s² 大小、但算起来便宜的部分（softmax、dropout），MLP 的激活照存。

### 4. 混合精度

低精度能省显存和带宽，但会带来两类数值问题：范围不够时数会下溢或溢出，精度不够时小的更新会被舍入掉。混合精度的各个部件分别对付这两件事。

先说范围。fp16 的最小正规数约为 6.1e-5，而很多梯度比它还小，直接下溢成 0。loss scaling 的办法是把 loss 乘以 S，让梯度整体放大，更新前再除以 S。
S 是动态调整的：一旦出现 inf/NaN，就跳过本步并把 S 减半；连续若干步（PyTorch 默认 2000）没有溢出，就把 S 翻倍。

再说精度。bf16 在 1.0 附近的间隔是 2⁻⁷ ≈ 0.0078，lr·g = 1e-5 这样的更新加上去会被直接舍入掉。所以主权重要用 fp32 来累加更新。

fp8 把这两个问题都推到了极限。H100 的 fp8 稠密算力是 1979 TFLOP/s，是 bf16 的 2 倍，但范围太窄，必须给每个张量配一个 scaling factor，常用 amax 历史做 delayed scaling；更细的做法是按 tile 或 block 配 scale。

### 5. Offload

卡少、显存又不够的时候，还可以把一部分状态搬到 GPU 之外的存储上。

ZeRO-Offload 把 fp32 主权重、m、v 和 Adam 更新都放到 CPU，前向和反向留在 GPU。每步 GPU→CPU 传 2N 字节的梯度，CPU→GPU 传 2N 字节的新权重。论文报告单卡可以训 13B 量级的模型。
ZeRO-Infinity 再往下一层，用到 NVMe。它的关键设计是 bandwidth-centric partitioning：每个参数切到所有 DP rank 上，各 rank 从自己的 PCIe 读一片，再做 all-gather，这样聚合带宽随卡数增长。

offload 的瓶颈是 PCIe 和 CPU 内存带宽。所以它适合「显存不够但卡少」的场景，不适合追求吞吐的大集群。

### 6. Caching allocator 与碎片

有时候 OOM 报出来，空闲显存的总量明明够。这要从 PyTorch 怎么管显存说起。

PyTorch 不会每次都调 cudaMalloc，而是先保留大块，再切分着用，释放的块按 stream 缓存起来。
大小不一的张量交替分配和释放，就会产生碎片：空闲的总量够，但没有一块连续的空间足够大，于是报 OOM。

诊断时看 memory_allocated 和 memory_reserved 的差，用 max_memory_allocated 看峰值。缓解办法有三个：开 expandable_segments、固定形状（padding 到几个桶）、避免在循环里反复创建变长张量。

### 7. 8-bit 优化器

16 B 里，Adam 的 m、v 就占了 8 B。能不能把它们存得更省？

8-bit 优化器把 m、v 用 int8 存，按块（block 2048）量化，每块一个 absmax scale。这样一个异常值只影响它所在的那一块。
量化码表用 dynamic 量化，以覆盖跨度很大的数值。更新时先反量化到 fp32，算完再量化回去。embedding 层要配合 stable embedding 才能保持稳定。

## 关键推导

### √n 检查点

这一节回答：检查点设多密，显存最省？

设 n 层，每 k 层存一次输入，共 n/k 个检查点。反向处理某一段时，先重跑这段的前向，临时存下 k 层的激活。
所以峰值 ∝ n/k + k。对 k 求导得 k = √n，此时峰值是 2√n 层的激活。代价是每层的前向多跑一次。

代入 n=80，k≈9，峰值约为 18 层的激活，比存 80 层降了 4.4 倍。如果递归地分段，还能把显存降到 O(log n)，但计算会变成 O(n log n)。

### 全量重计算的 33%

每个 token 的训练计算是 6N：前向 2N，反向 4N（对输入和对权重各 2N）。全量重算多出一次前向 2N，总共 8N，比 6N 多 1/3。

这会影响你怎么读 MFU。MFU 按 6N 计，所以开了全量重算后 MFU 读数会下降；但硬件实际的利用率（HFU）并没有降。

## 深入推导

### 激活随并行方式的变化

上一讲（M13）的 TP、SP 会切掉一部分激活。这一节算清楚每种组合下还剩多少，以及为什么选择性重算的性价比这么高。

TP=t 时，attention 头和 MLP 中间维被切开。被切的有 24·sbh（Q/K/V 4、V 2、输出投影输入 2、MLP 中间 16），以及全部的 5·a·s²·b。
没被切的是 LN、dropout 和各块的输入，共 10·sbh。于是得到 s·b·h·(10 + 24/t + 5·a·s/(h·t))。
SP 把剩下的 10·sbh 也沿序列切开，于是整体除以 t：s·b·h·(34 + 5·a·s/h)/t。
选择性重算再去掉 5·a·s²·b，剩下 34·s·b·h/t。

被去掉的这一项占比随 s 增长。s=4096、a=64、h=8192 时 5·a·s/h = 160，占总量的 160/194 = 82%。
而重算它只需要重做 softmax 和 dropout 这类逐元素操作，FLOP 很少。所以选择性重算用很小的重算代价，消掉了大部分激活。

### fp16 的数值边界与 loss scaling

fp16 的位分配是 1 符号 + 5 指数 + 10 尾数：最大 65504，最小正规数 2⁻¹⁴ ≈ 6.1e-5，最小次正规数 2⁻²⁴ ≈ 6e-8。
梯度乘以 S = 2¹⁶ 后，原本 1e-8 的梯度变成 6.6e-4，回到可表示的范围。但 S 太大的话，有些梯度会超过 65504 而溢出，这时就跳过这一步、把 S 减半。

bf16 有 8 位指数，范围和 fp32 一致（约 3.4e38），所以不需要 S。但它的尾数只有 7 位，所以累加和主权重必须用 fp32。

下面的代码把这两件事放在一起：用 checkpoint 只包住 attention 来做选择性重算，用 GradScaler 做 fp16 的动态 loss scaling。

```python
import torch
from torch.utils.checkpoint import checkpoint

class Block(torch.nn.Module):
    def __init__(self, attn, mlp, ln1, ln2):
        super().__init__()
        self.attn, self.mlp, self.ln1, self.ln2 = attn, mlp, ln1, ln2
    def forward(self, x):
        # 选择性重算：只把 attention 包进 checkpoint，MLP 激活照常保存
        x = x + checkpoint(self.attn, self.ln1(x), use_reentrant=False)
        return x + self.mlp(self.ln2(x))

scaler = torch.cuda.amp.GradScaler()          # 仅 fp16 需要；bf16 直接 autocast
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()             # 梯度乘 S
    scaler.step(opt)                          # 先反缩放；有 inf 则跳过
    scaler.update()                           # 溢出则 S 减半，连续无溢出则加倍
print(torch.cuda.max_memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30)
```

## 白板估算

这里做两组估算。第一组接着 M13 的 70B 切法，看激活到底放不放得下、每种手段能省多少；第二组看单卡全参微调 7B 有哪些路可走。
硬件数字取自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 80 GB，PCIe 4.0 32 GB/s，DDR5 单通道 40–50 GB/s。

```text
Llama-3-70B 形状近似：s=4096, b=1, h=8192, a=64, L=80
  s·b·h = 33.55e6；5·a·s/h = 5×64×4096/8192 = 160
  无并行每层：33.55e6 × 194 = 6.51 GB；80 层 521 GB
  TP=8 + SP：6.51 / 8 = 0.81 GB/层
  再加选择性重算：33.55e6 × 34 / 8 = 143 MB/层
  全量重算：每层只存 2·s·b·h = 67 MB（SP 下再除以 8）
M13 的切法 t=8, p=4：每级 20 层，1F1B 第 1 级在途 4 个 micro-batch
  TP+SP：0.81 GB × 20 × 4 = 65 GB → 加 35.3 GB 静态，超 80 GB
  +选择性重算：143 MB × 80 = 11.4 GB → 总计约 47 GB，放得下

7B 全参微调单卡：16 × 6.74e9 = 107.8 GB > 80 GB
  8-bit Adam：10 × 6.74e9 = 67.4 GB，激活只剩约 12 GB
  ZeRO-Offload：GPU 只留 bf16 权重 13.5 GB
    梯度下行 13.5 GB / 32 GB/s = 0.42 s，新权重上行 0.42 s
    CPU Adam 每参数读写约 28 B（fp32 p/m/v 读写 24 + 梯度 2 + 新 bf16 权重 2）
    28 × 6.74e9 = 189 GB；假设 8 通道 DDR5 ≈ 360 GB/s → 0.52 s/步
全量重算代价：8N/6N = 1.33；在 MFU 50% 的 70B 训练上，等于每步多 1/3 的时间
```

第一组里最有说服力的是 70B 那两行：只开 TP+SP 时总量超过 80 GB；加上选择性重算后约 47 GB，就放得下了。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 显存分账 | [分布式训练实验](../solutions/cmu15849/distributed-training/README.md) 的显存统计 | 静态部分与 16N 的误差小于 5% |
| 检查点 | 同上，`torch.utils.checkpoint` 开关对比 | 峰值激活下降，loss 曲线与不开时一致 |
| 混合精度 | 同上，fp32 / bf16 / fp16+GradScaler 三组 | fp16 不开 scaler 时出现梯度为 0 或不收敛 |
| 静态显存公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `training_memory_bytes` | `train-mem --params 7e9` 总计 112 GB |

## 易错点与反例

- 「激活和模型大小成正比」：只对了一部分。激活 ∝ L·h·b·s，和 b·s 是线性关系，attention 部分还和 s² 成正比，所以小模型配长序列也能把显存撑爆。
- 「bf16 训练不需要 fp32」：主权重和优化器状态仍然要 fp32，否则小的更新会被舍入掉。
- 「loss scaling 改变优化结果」：不会。先缩放再反缩放，数学上等价，作用只是避免下溢。
- 「重计算不改变 MFU」：按 6N 计的 MFU 会下降，HFU 不变。
- 「nvidia-smi 显示 79 GB，说明模型用了 79 GB」：那是 reserved，里面包括缓存和碎片。
- 「offload 总是划算」：PCIe 32 GB/s 比 HBM 慢 100 倍，计算量小的步会被传输主导。
- 「FlashAttention 和选择性重算要二选一」：FlashAttention 本身就在反向时重算 attention 分块，已经消除了 s² 激活，两者的目标是重合的。
- fp8 不是把 bf16 直接 cast 过去：没有 scaling factor 会大量下溢或溢出。

## 实验或练习

这个练习验证一个假设：检查点让峰值激活随层数从 O(n) 变为 O(√n)，步时间增加约 1/3。

- 配置：24 层的小 Transformer，s=1024，分别测无检查点、每层检查点、每 5 层检查点三种情况的峰值显存与步时间。
- 预期：每层检查点的峰值最低，但多跑一次完整前向；每 5 层（≈√24）是显存和时间之间的折中。

## 高频问题

- 问：混合精度 Adam 每参数几字节？
  答：16。bf16 权重和梯度各 2，fp32 主权重、m、v 各 4。
- 问：为什么 fp16 要 loss scaling 而 bf16 不要？
  答：fp16 指数只有 5 位，小梯度会下溢；bf16 指数有 8 位，范围和 fp32 一样。
- 问：选择性重计算重算的是什么？
  答：attention 里 s² 大小的 softmax、dropout 等。它们占的显存大，但 FLOP 少。
- 问：ZeRO-Offload 每步 PCIe 传多少？
  答：梯度 2N 下行，更新后的 bf16 权重 2N 上行。
- 问：OOM 了，但空闲显存总量是够的，怎么回事？
  答：是碎片。看 reserved 与 allocated 的差，开 expandable_segments 或固定张量形状。

## 面试追问链

1. 问：训练显存由哪几块组成？
   答：权重、梯度、优化器状态是静态的，合计 16N；激活正比于 b·s·L；另外还有临时缓冲与碎片。
2. 问：推一下每层激活里的 34·sbh。
   答：attention 块 11（QKV 输入 2、Q/K 4、V 2、投影输入 2、mask 1），MLP 块 19（输入 2、GeLU 输入和输出各 8、mask 1），LN 4。另外还有 5·a·s²·b 的 softmax/dropout 项。
3. 问：√n 检查点是怎么来的？代价多大？
   答：峰值 ∝ n/k + k，取 k = √n 最小。代价是多一次前向，计算从 6N 变成 8N，多 33%。
4. 问：既然全量重算也能省显存，为什么 Megatron 默认用选择性重计算？
   答：s=4096 时 s² 项占激活的 82%，而它的 FLOP 只是逐元素操作。只重算这一项就能拿走大部分显存，开销只有几个百分点；全量重算则要多付 33%。
5. 问：fp8 训练要注意什么？
   答：前向用 e4m3，梯度用 e5m2；每个张量或每个块配一个 scale，用 amax 历史来估计；主权重和累加仍用高精度；对异常值敏感的层（比如输出层）常保留 bf16。
6. 问：一张 80 GB 的卡全参微调 7B，你怎么排？
   答：16N = 108 GB，不够。先上 8-bit Adam 降到 67 GB，再用 bf16 加每层检查点压激活。还不够就用 ZeRO-Offload 把优化器搬到 CPU，代价是约 1 s/步的 PCIe 和 CPU 带宽开销；或者干脆改用 LoRA（M17）。
7. 问：70B 在 64 卡上用 t=8、p=4，激活放不下，按什么顺序加手段？
   答：先开 SP，它没有额外通信；再开选择性重算，代价几个百分点；仍然不够，再上全量重算或减小 micro-batch；最后才考虑 offload。每一步都先用上面的公式算一遍，再动手改。

## 自测题

1. s=2048、b=4、h=4096、a=32，单层不做重算的激活有多少？
   要点：sbh = 33.55e6；5·a·s/h = 80；33.55e6 × 114 = 3.83 GB。
2. 证明 loss scaling 不改变 SGD 更新。
   要点：∇(S·L) = S·∇L，更新前除以 S；只要没有溢出，结果相同。
3. bf16 主权重 w=1.0，lr·g = 3e-3，更新后是多少？
   要点：间隔 2⁻⁷ ≈ 0.0078，3e-3 小于半个间隔 0.0039，舍入后 w 仍是 1.0。
4. 设计题：用 8 张 4090（24 GB）训 7B。
   要点：静态 108 GB，ZeRO-3 切 8 份，每份 13.5 GB；加每层检查点；PCIe 互联下 all-gather 是瓶颈，可以考虑 8-bit Adam 或 LoRA。
5. DTR 和 Checkmate 各适合什么框架？
   要点：Checkmate 需要静态图来离线求解；DTR 在运行时决策，适合 eager 模式。

## 关联笔记

- [M12 数据并行与 ZeRO](module-12-data-parallel-zero.md)：静态显存的切分
- [M13 张量与流水线并行](module-13-tensor-pipeline-parallel.md)：SP 与 1F1B 在途 micro-batch
- [M08 FlashAttention](module-08-attention-flashattention.md)：不物化 s² 矩阵
- [M17 压缩与 PEFT](module-17-compression-quantization.md)：数值格式、QLoRA 的分页优化器
- [edge-deploy M04 静态内存规划](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-04-static-memory-planning.md)：MCU 上的激活下界、arena 规划与 patch 重算，对应训练侧的激活重计算

## 本讲小结

静态显存是 16N，激活是每层 s·b·h·(34 + 5·a·s/h)。遇到 OOM，先用这两个式子算清楚是哪块爆了，再决定用什么手段。
重计算是用 FLOP 换字节：全量重算多 33%，选择性重算只付出逐元素操作的重算代价。
混合精度要把范围和精度分开处理：fp16 靠 loss scaling 补范围，bf16 靠 fp32 主权重补精度，fp8 两头都要 scale。offload 则用 PCIe 和 CPU 内存换显存，适合卡少的场景。

## 延伸阅读

- Chen et al., Training Deep Nets with Sublinear Memory Cost, 2016, https://arxiv.org/abs/1604.06174
- Jain et al., Checkmate, 2020, https://arxiv.org/abs/1910.02653
- Kirisame et al., Dynamic Tensor Rematerialization, 2021, https://arxiv.org/abs/2006.09616
- Korthikanti et al., Reducing Activation Recomputation, 2022, https://arxiv.org/abs/2205.05198
- Micikevicius et al., Mixed Precision Training, 2018, https://arxiv.org/abs/1710.03740
- Micikevicius et al., FP8 Formats for Deep Learning, 2022, https://arxiv.org/abs/2209.05433
- Ren et al., ZeRO-Offload, 2021, https://arxiv.org/abs/2101.06840
- Rajbhandari et al., ZeRO-Infinity, 2021, https://arxiv.org/abs/2104.07857
- Dettmers et al., 8-bit Optimizers via Block-wise Quantization, 2022, https://arxiv.org/abs/2110.02861
