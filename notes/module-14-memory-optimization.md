# Module 14 — 显存优化：重计算、混合精度、offload

- 课程出处：CS229S Fall 2023《Tim Dettmers 客座：8-bit Methods for Efficient Deep Learning》《Parallelism Fundamentals》 · CMU 15-849《Tensor Rematerialization》《ZeRO-Offload / ZeRO-Infinity》
- 必读论文：Chen et al., Training Deep Nets with Sublinear Memory Cost, 2016；Jain et al., Checkmate, 2020；Kirisame et al., Dynamic Tensor Rematerialization, 2021；Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022；Micikevicius et al., Mixed Precision Training, 2018；Micikevicius et al., FP8 Formats for Deep Learning, 2022；Ren et al., ZeRO-Offload, 2021；Rajbhandari et al., ZeRO-Infinity, 2021；Dettmers et al., 8-bit Optimizers via Block-wise Quantization, 2022
- 对应实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf)（实测：16 B/参数实测 16.00–16.32；全量重计算 +18.9% 时间，batch 8 → 24）
- 状态：初稿

## 摘要

训练显存由五块组成：权重、梯度、优化器状态、激活、临时缓冲与碎片。前三块是静态的，混合精度 Adam 下 16 B/参数，由 ZeRO 和 offload 处理。
激活随 batch、序列长度线性增长（attention 部分随 s² 增长），对大模型往往是最大的一块，由重计算处理。
重计算用算力换显存：全量重计算多约 1/3 的计算，选择性重计算只重算 attention 的 s² 部分，代价小得多。
混合精度用低精度做矩阵乘、高精度存主权重；fp16 需要 loss scaling，bf16 不需要，fp8 需要逐张量或逐块的 scaling factor。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 混合精度 Adam：bf16 权重 2 + bf16 梯度 2 + fp32 主权重 4 + m 4 + v 4 = 16 B/参数，激活另算。
2. 每层激活（无重计算、无并行）≈ s·b·h·(34 + 5·a·s/h) 字节，其中 5·a·s/h 项来自 attention 的 s² 矩阵。
3. √n 分段检查点：显存 O(√n)，代价是多一次前向（总计算约 +33%）。
4. fp16 动态范围小（最大 65504），梯度会下溢，必须 loss scaling；bf16 指数位与 fp32 相同，不需要。
5. 8-bit Adam 把 m、v 从 8 B 降到 2 B，每参数 16 B → 10 B。

**必须能默写的公式**

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

**一句话区分**

- fp16 vs bf16：5 位指数 10 位尾数、范围小精度高 vs 8 位指数 7 位尾数、范围与 fp32 相同。
- e4m3 vs e5m2：精度优先（最大 448），用于前向 vs 范围优先（最大 57344），用于梯度。
- Checkmate vs DTR：离线 ILP 求最优重算计划 vs 运行时贪心驱逐，无需静态图。
- ZeRO-3 vs ZeRO-Offload：把状态切到其他 GPU vs 把状态搬到 CPU 内存。

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

权重、梯度、优化器状态：与 batch 无关，16 B/参数。7B 是 107.8 GB，单卡 80 GB 放不下。
激活：反向需要的前向中间结果，正比于 b·s·L。
临时缓冲：GEMM workspace、通信 bucket、logits（b·s·V，V=128256 时很大）。
碎片：caching allocator 里已保留但不能复用的块，nvidia-smi 看到的是 reserved 而不是 allocated。

### 2. 激活为什么这么大

一层 Transformer 存的东西（16 位，MHA，4h MLP）：
attention 块 11·sbh + 5·a·s²·b：QKV 输入 2sbh、Q 与 K 4sbh、softmax 输出 2as²b、dropout mask as²b、dropout 输出 2as²b、V 2sbh、输出投影输入 2sbh、dropout mask sbh。
MLP 块 19·sbh：输入 2sbh、GeLU 输入 8sbh、GeLU 输出 8sbh、dropout mask sbh。
两个 LayerNorm 输入 4·sbh。合计 34·sbh + 5·a·s²·b。
FlashAttention 不物化 s² 矩阵，5·a·s/h 项实际上消失；Llama 类结构（GQA、门控 MLP、无 dropout）的系数也不同，所以这个式子只当量级估算用。

### 3. 重计算家族

Chen et al.：把 n 层分成 √n 段，只存段边界；反向时逐段重跑前向。显存 O(√n)，计算多一次前向。
Checkmate：把图上的「第 t 步是否存储/重算节点 i」写成 0-1 变量，在显存约束下最小化总计算，用 ILP 求解，适合静态图离线规划。
DTR：运行时显存不够时驱逐张量，按 cost / (memory × staleness) 这类启发式选择驱逐对象，需要时再重算，适合动态图。
选择性重计算：只重算 attention 里 s² 大小、算力便宜的部分（softmax、dropout），MLP 激活照存。

### 4. 混合精度

fp16 最小正规数约 6.1e-5，很多梯度比它小，直接下溢成 0。loss scaling 把 loss 乘 S，梯度整体放大，更新前再除以 S。
动态 loss scaling：出现 inf/NaN 就跳过本步并把 S 减半；连续若干步（PyTorch 默认 2000）无溢出就翻倍。
fp32 主权重：bf16 在 1.0 附近的间隔是 2⁻⁷ ≈ 0.0078，lr·g = 1e-5 的更新加上去会被舍入掉。主权重用 fp32 累加更新。
fp8：H100 的 fp8 稠密算力 1979 TFLOP/s，是 bf16 的 2 倍。范围太窄，每个张量配一个 scaling factor，常用 amax 历史做 delayed scaling；更细的做法按 tile 或 block 配 scale。

### 5. Offload

ZeRO-Offload：fp32 主权重、m、v 与 Adam 更新放 CPU，前反向在 GPU。每步 GPU→CPU 传 2N 字节梯度，CPU→GPU 传 2N 字节新权重。论文报告单卡可训 13B 量级模型。
ZeRO-Infinity：再往下到 NVMe。bandwidth-centric partitioning：每个参数切到所有 DP rank 上，各自从自己的 PCIe 读一片再 all-gather，聚合带宽随卡数增长。
offload 的瓶颈是 PCIe 和 CPU 内存带宽，适合「显存不够但卡少」的场景，不适合追求吞吐的大集群。

### 6. Caching allocator 与碎片

PyTorch 不每次调 cudaMalloc，而是保留大块再切分，按 stream 缓存释放的块。
碎片来自大小不一的张量交替分配释放：总空闲够，但没有一块连续够大，报 OOM。
诊断：memory_allocated 与 memory_reserved 的差；max_memory_allocated 看峰值。缓解：expandable_segments、固定形状（padding 到桶）、避免在循环里反复创建变长张量。

### 7. 8-bit 优化器

m、v 用 int8 存，按块（block 2048）量化，每块一个 absmax scale，异常值只影响自己那一块。
用 dynamic 量化码表覆盖大跨度数值；更新时反量化到 fp32、计算、再量化。embedding 层配合 stable embedding 保持稳定。

## 关键推导

### √n 检查点

n 层，每 k 层存一次输入，共 n/k 个检查点。反向处理某段时，重跑该段前向，临时存 k 层的激活。
峰值 ∝ n/k + k，对 k 求导得 k = √n，峰值 2√n 层。代价：每层前向多跑一次。
n=80 时 k≈9，峰值约 18 层的激活，比 80 层降 4.4 倍。递归分段可到 O(log n) 显存，计算 O(n log n)。

### 全量重计算的 33%

每 token 训练计算 6N：前向 2N、反向 4N（对输入和对权重各 2N）。全量重算多一次前向 2N，总计 8N，比 6N 多 1/3。
MFU 按 6N 计，所以全量重算会让 MFU 读数下降，而硬件实际利用率（HFU）不降。

## 深入推导

### 激活随并行方式的变化

TP=t 时，attention 头与 MLP 中间维被切开：被切的有 24·sbh（Q/K/V 4、V 2、输出投影输入 2、MLP 中间 16）和全部 5·a·s²·b。
未被切的是 LN、dropout、各块输入，共 10·sbh。得到 s·b·h·(10 + 24/t + 5·a·s/(h·t))。
SP 把这 10·sbh 也沿序列切开，于是全部除以 t：s·b·h·(34 + 5·a·s/h)/t。
选择性重算再去掉 5·a·s²·b：34·s·b·h/t。这一项占比随 s 增长：s=4096、a=64、h=8192 时 5·a·s/h = 160，占总量 160/194 = 82%。
所以选择性重算用很少的重算 FLOP（softmax/dropout 是逐元素的）消掉了大部分激活。

### fp16 的数值边界与 loss scaling

fp16：1 符号 + 5 指数 + 10 尾数，最大 65504，最小正规 2⁻¹⁴ ≈ 6.1e-5，最小次正规 2⁻²⁴ ≈ 6e-8。
梯度乘 S = 2¹⁶ 后，原本 1e-8 的梯度变成 6.6e-4，回到可表示范围。若 S 太大，部分梯度超 65504 溢出 → 跳步、减半。
bf16 有 8 位指数，范围与 fp32 一致（约 3.4e38），不需要 S，但尾数只有 7 位，所以累加与主权重必须 fp32。

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

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 80 GB，PCIe 4.0 32 GB/s，DDR5 单通道 40–50 GB/s。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 显存分账 | [分布式训练实验](../solutions/cmu15849/distributed-training/README.md) 的显存统计 | 静态部分与 16N 的误差小于 5% |
| 检查点 | 同上，`torch.utils.checkpoint` 开关对比 | 峰值激活下降，loss 曲线与不开时一致 |
| 混合精度 | 同上，fp32 / bf16 / fp16+GradScaler 三组 | fp16 不开 scaler 时出现梯度为 0 或不收敛 |
| 静态显存公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `training_memory_bytes` | `train-mem --params 7e9` 总计 112 GB |

## 易错点与反例

- 「激活和模型大小成正比」：只部分正确。激活 ∝ L·h·b·s，与 b·s 线性，attention 部分与 s² 成正比；小模型长序列也能爆显存。
- 「bf16 训练不需要 fp32」：主权重和优化器状态仍需 fp32，否则小更新被舍入掉。
- 「loss scaling 改变优化结果」：不改变。缩放后再反缩放，数学上等价，只是避免下溢。
- 「重计算不改变 MFU」：按 6N 计的 MFU 会下降，HFU 不变。
- 「nvidia-smi 显示 79 GB，说明模型用了 79 GB」：那是 reserved，含缓存与碎片。
- 「offload 总是划算」：PCIe 32 GB/s 比 HBM 慢 100 倍，计算量小的步会被传输主导。
- 「FlashAttention 和选择性重算二选一」：FlashAttention 本身在反向重算 attention 分块，已经消除了 s² 激活，两者目标重合。
- fp8 不是把 bf16 直接 cast：没有 scaling factor 会大量下溢或溢出。

## 实验或练习

- 假设：checkpoint 让峰值激活随层数从 O(n) 变为 O(√n)，步时间增加约 1/3。
- 配置：24 层小 Transformer，s=1024，分别测 无检查点 / 每层检查点 / 每 5 层检查点 的峰值显存与步时间。
- 预期：每层检查点峰值最低但多一次完整前向；每 5 层（≈√24）在显存和时间之间折中。

## 高频问题

- 问：混合精度 Adam 每参数几字节？
  答：16。bf16 权重与梯度各 2，fp32 主权重、m、v 各 4。
- 问：为什么 fp16 要 loss scaling 而 bf16 不要？
  答：fp16 指数只有 5 位，小梯度下溢；bf16 指数 8 位，范围同 fp32。
- 问：选择性重计算重算什么？
  答：attention 中 s² 大小的 softmax、dropout 等，它们显存大、FLOP 少。
- 问：ZeRO-Offload 每步 PCIe 传多少？
  答：梯度 2N 下行，更新后的 bf16 权重 2N 上行。
- 问：OOM 但总空闲显存够，怎么回事？
  答：碎片。看 reserved 与 allocated 的差，开 expandable_segments 或固定张量形状。

## 面试追问链

1. 问：训练显存由哪几块组成？
   答：权重、梯度、优化器状态（静态 16N），激活（∝ b·s·L），临时缓冲与碎片。
2. 问：推一下每层激活的 34·sbh。
   答：attention 11（QKV 输入 2、Q/K 4、V 2、投影输入 2、mask 1），MLP 19（输入 2、GeLU 输入输出各 8、mask 1），LN 4；另有 5·a·s²·b 的 softmax/dropout 项。
3. 问：√n 检查点怎么来的？代价多大？
   答：峰值 ∝ n/k + k，k = √n。计算多一次前向，6N → 8N，+33%。
4. 问：为什么 Megatron 默认选择性重计算而不是全量？
   答：s=4096 时 s² 项占激活 82%，但其 FLOP 只是逐元素操作；只重算它就能拿走大部分显存，开销只有几个百分点，全量要 33%。
5. 问：fp8 训练要注意什么？
   答：前向 e4m3、梯度 e5m2；每张量或每块一个 scale，用 amax 历史估计；主权重和累加仍高精度；对异常值敏感的层（如输出层）常保留 bf16。
6. 问：一张 80 GB 卡全参微调 7B，你怎么排？
   答：16N = 108 GB 不够。先 8-bit Adam 到 67 GB，加 bf16 + 每层检查点压激活；仍不够就 ZeRO-Offload 优化器到 CPU，代价约 1 s/步的 PCIe 和 CPU 带宽；或改 LoRA（M17）。
7. 问：70B 在 64 卡上用 t=8、p=4，激活放不下，按什么顺序加手段？
   答：先 SP（零额外通信）；再选择性重算（几个百分点）；仍不够再全量重算或减小 micro-batch；最后才考虑 offload。每一步都用上面的公式先算再改。

## 自测题

1. s=2048、b=4、h=4096、a=32，单层无重算激活多少？
   要点：sbh = 33.55e6；5·a·s/h = 80；33.55e6 × 114 = 3.83 GB。
2. 证明 loss scaling 不改变 SGD 更新。
   要点：∇(S·L) = S·∇L，更新前除以 S；只要不溢出，结果相同。
3. bf16 主权重 w=1.0，lr·g = 3e-3，更新后是多少？
   要点：间隔 2⁻⁷ ≈ 0.0078，3e-3 小于半个间隔 0.0039，舍入后 w 仍是 1.0。
4. 设计题：8 张 4090（24 GB）训 7B。
   要点：静态 108 GB，ZeRO-3 切 8 份 13.5 GB；加每层检查点；PCIe 互联下 all-gather 是瓶颈，考虑 8-bit Adam 或 LoRA。
5. DTR 和 Checkmate 各适合什么框架？
   要点：Checkmate 需静态图离线求解；DTR 运行时决策，适合 eager 模式。

## 关联笔记

- [M12 数据并行与 ZeRO](module-12-data-parallel-zero.md)：静态显存的切分
- [M13 张量与流水线并行](module-13-tensor-pipeline-parallel.md)：SP 与 1F1B 在途 micro-batch
- [M08 FlashAttention](module-08-attention-flashattention.md)：不物化 s² 矩阵
- [M17 压缩与 PEFT](module-17-compression-quantization.md)：数值格式、QLoRA 的分页优化器
- [edge-deploy M04 静态内存规划](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-04-static-memory-planning.md)：MCU 上的激活下界、arena 规划与 patch 重算，对应训练侧的激活重计算

## 本讲小结

静态显存 16N，激活 s·b·h·(34 + 5·a·s/h) 每层，两者都要先算再决定手段。
重计算是用 FLOP 换字节：全量 +33%，选择性只付出逐元素重算。
混合精度的要点是范围与精度分开处理：fp16 补范围靠 loss scaling，bf16 补精度靠 fp32 主权重，fp8 两头都要 scale。
offload 用 PCIe 与 CPU 内存换显存，适合卡少的场景。

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
