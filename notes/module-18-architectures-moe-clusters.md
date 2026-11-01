# Module 18 — 高效架构、MoE、集群与数据

- 课程出处：CS229S Fall 2023《Attention-Free Architectures》《William Fedus 客座：Mixture of Experts》《Scheduling ML Clusters》《Data Pipelines》 · CMU 15-849《Graph Neural Networks》《ML for Systems》《Federated Learning》
- 必读论文：Katharopoulos et al., Transformers are RNNs (Linear Attention), 2020；Gu et al., S4, 2022；Gu & Dao, Mamba, 2023；Shazeer et al., Sparsely-Gated MoE, 2017；Lepikhin et al., GShard, 2021；Fedus et al., Switch Transformer, 2022；Hoffmann et al., Chinchilla, 2022；Xiao et al., Gandiva, 2018；Gu et al., Tiresias, 2019；Qiao et al., Pollux, 2021；Kraska et al., The Case for Learned Index Structures, 2018；McMahan et al., FedAvg, 2017；Hamilton et al., GraphSAGE, 2017
- 对应实验：无（估算可用 [mlsys_calc.py](../tools/mlsys_calc.py)）
- 状态：初稿

## 摘要

前面的模块假设模型是稠密 Transformer。本讲从三个方向改这个假设：换掉 O(T²) 的 attention（线性 attention、SSM），换掉「每个 token 用全部参数」（MoE），以及从单个任务扩到整个集群和数据流水线。
SSM 有两种等价形式：训练用卷积（FFT，O(T log T)），推理用递推（每 token O(1)、状态大小固定），所以没有随 T 增长的 KV cache。
MoE 每个 token 只激活 k 个专家，参数量随专家数增长而每 token FLOPs 基本不变；代价是 all-to-all 通信、负载不均和更大的显存。
集群层面，深度学习作业需要 gang scheduling，调度器的目标从「公平分 GPU」演进到「最大化 goodput」。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. SSM 递推 x_k = Ā·x_{k−1} + B̄·u_k、y_k = C·x_k；LTI 时可展开成卷积核 K̄ = (CB̄, CĀB̄, …)，用 FFT 训练。
2. Mamba 让 Δ、B、C 依赖输入（selective），不再是卷积，训练改用硬件感知的并行 scan，推理每 token O(1)。
3. MoE：top-k 门控，专家容量 = capacity_factor × T·k/E，溢出 token 被丢弃（走残差）；Switch 负载均衡损失 α·E·Σ f_i·P_i。
4. Expert parallelism 每个 MoE 层前向 2 次 all-to-all（dispatch、combine），反向再 2 次，每次每卡发 T·k·h·bytes 的 (E_p−1)/E_p。
5. Chinchilla：计算最优约 20 token/参数，C ≈ 6ND。

**必须能默写的公式**

```text
线性 attention：softmax(QKᵀ)V → φ(Q)·(φ(K)ᵀV)，O(T·d²)；递推态 S_t = S_{t−1} + φ(k_t)·v_tᵀ
SSM 离散化（ZOH）：Ā = exp(ΔA)，B̄ = (ΔA)⁻¹(exp(ΔA) − I)·ΔB
卷积形式：y = K̄ * u，K̄_j = C·Ā^j·B̄；FFT 卷积 O(T log T)
MoE 输出：y = Σ_{i∈TopK} g_i(x)·E_i(x)，g = softmax(TopK(x·W_g))
专家容量：C_e = CF × T × k / E
负载均衡：L_aux = α · E · Σ_i f_i · P_i（f_i 实际分到的比例，P_i 平均门控概率）
all-to-all 每卡每次：T·k·h·bytes × (E_p − 1)/E_p
Chinchilla：D_opt ≈ 20·N，C = 6·N·D → N_opt ≈ sqrt(C/120)
FedAvg：w ← Σ_k (n_k/n)·w_k；每轮每客户端通信 2 × 模型大小
```

**一句话区分**

- S4 vs Mamba：线性时不变、可用卷积训练 vs 输入相关的选择性参数、用并行 scan 训练。
- MoE 参数 vs FLOPs：总参数 E 份专家都要存 vs 每 token 只算 k 份。
- Gandiva vs Tiresias vs Pollux：时间片与迁移 vs 不知作业时长时按已获服务量排优先级 vs 联合调整资源与 batch size 最大化 goodput。
- 数据并行 vs 联邦学习：数据可随意搬、通信廉价 vs 数据不出端、通信昂贵、非 IID。

## 学习目标

- 推导 SSM 的递推与卷积两种形式，说明各自用于训练还是推理；
- 比较 KV cache 与 SSM 状态的显存随序列长度的增长；
- 写出 top-k 门控、容量与负载均衡损失，估算 expert parallelism 的通信量；
- 用 Chinchilla 规则估算给定算力的最优模型与数据量；
- 说出 DL 集群调度的核心问题与 Gandiva、Tiresias、Pollux 的思路；
- 知道 learned index、联邦学习、GNN 采样的系统要点。

## 先修知识

- M06：scan 原语（Mamba 的并行 scan）；
- M13：expert parallelism 的位置与 3D 并行；
- M15：KV cache 与 decode 的字节瓶颈。

## 核心概念

### 1. 线性 attention

softmax attention 必须先算 T×T 矩阵。若把 softmax 换成核函数 φ(q)·φ(k)ᵀ，可以先算 φ(K)ᵀV（d×d），复杂度 O(T·d²)。
因果情形下它就是一个 RNN：状态 S_t 是 d×d 矩阵，每 token 更新一次。代价是表达力：固定大小的状态难以精确回忆任意远处的 token。

### 2. SSM：S4 与 Mamba

连续系统 x'(t) = A·x + B·u、y = C·x，离散化后得到递推。A、B、C 不随时间变就是 LTI 系统，输出是输入与核 K̄ 的卷积。
训练：长度 T 的卷积用 FFT 是 O(T log T)，全序列并行。推理：用递推，每 token O(1) 计算、状态大小固定。
S4 用 HiPPO 初始化 A 以记住长历史，并用结构化 A 高效计算 K̄。
Mamba：Δ、B、C 由当前输入算出，模型能选择记住或忽略某个 token。时变后卷积形式失效，改用在 SRAM 里完成的并行 scan（与 M06 的 scan 同构），不把展开的状态写回 HBM。
Hyena 等长卷积模型走另一条路：用隐式参数化的长卷积 + 门控替代 attention。

### 3. MoE

把 FFN 换成 E 个专家，路由器对每个 token 选 top-k 个。Switch Transformer 用 k=1，GShard 用 k=2。
参数随 E 线性增长，每 token FLOPs 只随 k 增长。Mixtral 8×7B 总参数 46.7B，每 token 激活 12.9B。
负载不均：路由器容易坍缩到少数专家。辅助损失 α·E·Σ f_i·P_i 在均匀分布时取最小值 α（α=0.01 常用）。f_i 不可导，梯度经 P_i 回传。
容量：每个专家每批最多处理 CF × T·k/E 个 token，超出的被丢弃（只走残差）。CF 大则丢得少但显存与填充浪费多，训练常用 1.0–1.25。

### 4. Expert parallelism

E 个专家分到 E_p 张卡。每张卡上的 token 先按路由结果 all-to-all 发给目标专家（dispatch），专家算完再 all-to-all 发回（combine）。
通信量与专家参数无关，只与 T·k·h 成正比。跨节点的 all-to-all 受 IB 带宽限制，是 MoE 训练与服务的主瓶颈。
推理时 MoE 的显存按总参数算、带宽按激活参数算：大 batch 下不同 token 激活不同专家，权重读取接近全量。

### 5. Scaling laws

C ≈ 6ND（M11）。Chinchilla 在固定 C 下拟合最优 N、D，得到两者大致同比例增长，约 20 token/参数。
这是训练算力最优，不是推理成本最优：要部署的模型常远超 20 token/参数（Llama-3-8B 训了 15T token，约 1900 token/参数），用训练算力换更小的推理模型。

### 6. DL 集群调度

gang scheduling：一个分布式作业的所有 worker 必须同时拿到 GPU，否则已启动的 worker 空等 all-reduce。
局部性：同一作业的 GPU 尽量在同一节点（NVLink）或同一机架。
Gandiva：利用训练的迭代可预测性做时间片轮转、作业迁移和内省式打包。
Tiresias：作业时长未知，按已获服务量（GPU 数 × 时间）做多级反馈队列（2D-LAS），近似短作业优先。
Pollux：作业吞吐与统计效率都随 batch size 变化，调度器联合选择 GPU 数和 batch size，最大化 goodput = 吞吐 × 统计效率。
异构：同一作业在 A100、H100 上的加速比不同，Gavel 等按「有效吞吐」归一化后再分配。

### 7. 数据流水线

弱监督（Snorkel）：用多个有噪声的标注函数加标签模型替代人工标注。数据选择与去重：质量过滤、近似去重决定预训练效果。
dataloader：文本 token 读取量极小，瓶颈在 tokenization 与打包；图像视频的瓶颈是 CPU 解码与增强。手段：预处理离线化、多 worker、pinned memory、预取、GPU 解码。

### 8. ML for systems、联邦学习、GNN

Learned index：把有序键到位置的映射看作 CDF，用小模型预测位置，再在误差界内局部搜索。
Chip placement：用 RL 加图神经网络放置宏单元，与 M10 的 device placement 同一思路。
FedAvg：每轮选一部分客户端，各自本地训练若干 epoch，服务器按样本数加权平均。难点：非 IID、掉线、通信、隐私（安全聚合、差分隐私）。
GNN：消息传递本质是稀疏矩阵乘（SpMM）。全图训练时 L 层邻域呈指数膨胀，GraphSAGE 按层固定采样数（如 25、10）控制每个种子节点的计算量。

## 关键推导

### SSM 递推与卷积等价

x_0 = B̄·u_0，x_1 = Ā·B̄·u_0 + B̄·u_1，…，x_k = Σ_{j=0}^{k} Ā^{k−j}·B̄·u_j。
y_k = C·x_k = Σ_j C·Ā^{k−j}·B̄·u_j = (K̄ * u)_k，K̄_j = C·Ā^j·B̄。
前提是 Ā、B̄、C 与 k 无关；Mamba 让它们随输入变化，于是只剩递推（可并行 scan）一种高效算法。

### 负载均衡损失的最小值

Σ f_i = 1、Σ P_i = 1。若路由可微地使 f_i 与 P_i 一致，Σ f_i·P_i = Σ P_i² ≥ 1/E（Cauchy–Schwarz），均匀时取等。
乘 E 归一化后最小值为 α，与专家数无关，便于跨配置使用同一个 α。

## 深入推导

### MoE 的通信与计算比

每卡 T 个 token，h 维，top-k，bf16。一次 all-to-all 每卡发出 2·T·k·h·(E_p−1)/E_p 字节，一层前后向共 4 次。
计算（每层专家部分）：6 × P_expert × k × T FLOP（P_expert 为一个专家参数量）。
比值 = 6·P_expert·k·T/F ÷ 8·T·k·h·(E_p−1)/(E_p·B) = 0.75·P_expert·B·E_p / (h·F·(E_p−1))。T、k 约掉，和 TP 一样只取决于结构与硬件。
门控 FFN 的 P_expert = 3·h·d_ff，比值 ≈ 2.25·d_ff·B / F（E_p 大时）。d_ff=14336、H100：NVLink 下 ≈ 14.7，IB 下 ≈ 1.6。跨节点 EP 时通信接近计算的 60%，必须与计算重叠或用分层 all-to-all。

```python
import torch
import torch.nn.functional as F

def topk_gating(x, w_gate, k=2, capacity_factor=1.25, alpha=0.01):
    """x: [T, h]；w_gate: [h, E]。返回专家分配、门控权重、辅助损失。"""
    T, E = x.shape[0], w_gate.shape[1]
    probs = F.softmax(x @ w_gate, dim=-1)                       # [T, E]
    gate, expert = probs.topk(k, dim=-1)                        # [T, k]
    gate = gate / gate.sum(-1, keepdim=True)
    capacity = int(capacity_factor * T * k / E)
    onehot = F.one_hot(expert, E).reshape(T * k, E)             # 按 token 顺序排队
    pos = (onehot.cumsum(0) - 1) * onehot                       # 每个 token 在专家队列里的位置
    keep = ((pos < capacity) & (onehot > 0)).any(-1).reshape(T, k)   # 超容量的丢弃，走残差
    f = F.one_hot(expert[:, 0], E).float().mean(0)              # 实际分到的比例（按 top-1 统计）
    P = probs.mean(0)                                           # 平均门控概率
    aux = alpha * E * (f * P).sum()
    return expert, gate * keep, aux

def ssm_recurrent(u, A_bar, B_bar, C):
    """推理模式：u [T]，状态 x [N]；每步 O(N)，与 T 无关。"""
    x = torch.zeros(A_bar.shape[0]); ys = []
    for u_t in u:
        x = A_bar @ x + B_bar * u_t
        ys.append(C @ x)
    return torch.stack(ys)
```

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 989 TFLOP/s、NVLink 450 GB/s、IB NDR 50 GB/s；Llama-3-8B KV 128 KB/token。

```text
MoE all-to-all（Mixtral 形状：h=4096, d_ff=14336, E=8, k=2；每卡 T=8192 token；EP=8；bf16）
  一次 dispatch：8192 × 2 × 4096 × 2 B = 134 MB，离开本卡 7/8 → 117 MB
  每层前向 2 次 + 反向 2 次 → 470 MB/卡
  NVLink：117 MB / 450 GB/s = 0.26 ms × 4 = 1.04 ms/层；32 层 33 ms
  IB：117 MB / 50 GB/s = 2.35 ms × 4 = 9.4 ms/层；32 层 301 ms
  专家计算：单专家 3 × 4096 × 14336 = 176 M 参数；6 × 176e6 × 2 × 8192 = 1.73e13 FLOP → 满峰 17.5 ms/层
  → IB 上通信约为满峰计算的 54%，NVLink 上 6%
专家容量：CF=1.25 → 1.25 × 8192 × 2 / 8 = 2560 token/专家

SSM 状态 vs KV（d=4096，Mamba 扩展 2 倍、N=16）
  每层状态 2 × 4096 × 16 = 131072 个元素，与 T 无关
  Llama-3-8B 每层每 token KV：2 × 8 × 128 = 2048 个元素
  → SSM 状态等于 64 个 token 的 KV；T=8K 时 KV 是它的 128 倍

Scaling：C = 1e24 → N ≈ sqrt(1e24/120) = 91 B，D ≈ 1.8 T
  70B Chinchilla 最优：1.4 T token，C = 6 × 70e9 × 1.4e12 = 5.9e23 FLOP
  H100 MFU 40%：每卡 3.96e14 FLOP/s → 5.9e23 / 3.96e14 = 1.49e9 s ≈ 41 万 GPU·h
  Llama-3-8B 15T：7.2e23 → 约 50.5 万 GPU·h
数据：8B 在 H100 MFU 40% → 3.96e14 / (6 × 8.03e9) = 8200 token/s/卡，按 4 B/token 仅 33 KB/s
FedAvg：100 M 参数 fp32 = 400 MB，每轮每客户端上下行 800 MB；100 轮 80 GB
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 训练算力 6ND | [mlsys_calc.py](../tools/mlsys_calc.py) `train_flops` | 8B × 15T 输出 7.2e23 |
| MFU | 同上 `mfu` | 给定 token/s 与卡数，结果与 6N·tok/s/(峰值·卡数) 一致 |
| 通信带宽 | 同上 `hw` 与 `allreduce` | 与本讲 NVLink、IB 数字一致 |
| top-k 门控 | 本讲 `topk_gating` | 均匀路由时 aux = α；超容量 token 的权重为 0 |

## 易错点与反例

- 「MoE 省显存」：错。显存按总参数，Mixtral 要存 46.7B；省的是每 token 计算。
- 「MoE 推理一定快」：小 batch 时只读激活专家，快；大 batch 时几乎读全部专家，带宽按总参数算，再加 all-to-all。
- 「SSM 训练也是逐 token 递推」：LTI 的 S4 用 FFT 卷积并行训练；Mamba 用并行 scan，同样不是逐 token 串行。
- 「线性 attention 与 softmax attention 等价」：只是近似，固定状态在精确回忆类任务上明显弱。
- 「负载均衡损失越大越好」：α 过大会压过主损失，路由退化为均匀随机。
- 「Chinchilla 说模型训 20 token/参数最好」：只是训练算力最优；考虑推理成本时应训更多 token。
- 「集群调度和 CPU 调度一样」：DL 作业需要 gang、对放置敏感、可迁移且吞吐随资源非线性变化。
- 「FedAvg 等价于大 batch SGD」：本地多步更新加非 IID 数据会产生客户端漂移，不等价。

## 实验或练习

- 假设：top-k 门控在无辅助损失时出现专家坍缩；加 α=0.01 后各专家负载接近 T·k/E。
- 配置：随机初始化 E=8 的路由器，在合成数据上训练若干步，记录每个专家的 token 数与丢弃率，对比有无辅助损失、CF ∈ {1.0, 1.25, 2.0}。
- 预期：无辅助损失时少数专家吃掉大部分 token、丢弃率高；CF 增大降低丢弃但增加填充浪费。

## 高频问题

- 问：为什么 MoE 增参数不增 FLOPs？
  答：每 token 只经过 k 个专家，FLOPs 与 k 成正比；参数与 E 成正比。
- 问：容量因子的作用？
  答：限制每个专家每批处理的 token 数，保证张量形状固定；超出的 token 丢弃，CF 控制丢弃与浪费的折中。
- 问：SSM 推理为什么省显存？
  答：状态大小固定，不随序列长度增长，不需要 KV cache。
- 问：gang scheduling 是什么？
  答：分布式作业的所有 worker 同时分配资源，避免部分 worker 空等。
- 问：Chinchilla 的结论？
  答：固定算力下参数与数据同比例扩，约 20 token/参数，C ≈ 6ND。

## 面试追问链

1. 问：Mamba 和 Transformer 推理的显存差在哪？
   答：Transformer 的 KV 随 T 线性增长；Mamba 每层固定状态（d=4096 时约 13 万个元素），等于 64 个 token 的 KV。
2. 问：推一下 SSM 递推和卷积为什么等价。
   答：展开递推 x_k = Σ Ā^{k−j}B̄u_j，y = C·x 得卷积核 K̄_j = CĀ^jB̄；前提是参数不随时间变。
3. 问：Mamba 为什么不能用卷积训练？用什么代替？
   答：选择性参数随输入变，不是 LTI；改用并行 scan，在 SRAM 内完成，不物化全部状态。
4. 问：MoE 的 all-to-all 量多大？与什么有关？
   答：每次每卡 T·k·h·bytes × (E_p−1)/E_p，前后向共 4 次，与专家参数无关。Mixtral 形状每卡 8192 token 时一次 117 MB，IB 上 2.35 ms。
5. 问：EP 放在节点内还是跨节点？
   答：通信计算比 ≈ 2.25·d_ff·B/F，NVLink 约 15、IB 约 1.6。能放进节点就放节点内；专家多到放不下时用分层 all-to-all（先节点内聚合再跨节点）并与计算重叠。
6. 问：MoE 服务和稠密服务有什么不同？
   答：显存按总参数，所以要更多卡或量化；小 batch 带宽按激活参数，大 batch 接近总参数；专家负载不均导致尾延迟；EP 的 all-to-all 进入每层关键路径，TP 与 EP 的组合要按拓扑选。
7. 问：要给 1000 卡集群设计训练作业调度器，你关心什么？
   答：gang 分配与拓扑感知放置（同节点、同交换机）；作业时长未知时按已获服务量排优先级（Tiresias 思路）；支持检查点与迁移以整理碎片；对弹性作业联合调整卡数与 batch size（Pollux 思路）；异构卡按有效吞吐分配；故障时快速从检查点恢复。

## 自测题

1. E=64、k=2、T=4096、CF=1.0，每个专家容量多少？
   要点：4096 × 2 / 64 = 128。
2. C = 6e23，按 Chinchilla 最优 N 和 D？
   要点：N = sqrt(6e23/120) = sqrt(5e21) ≈ 71 B，D ≈ 1.4 T。
3. 证明 Σ f_i·P_i 在 f = P 时的最小值是 1/E。
   要点：Σ P_i² ≥ (Σ P_i)²/E = 1/E。
4. 线性 attention 在 T=8192、d=128 时比 softmax attention 省多少 FLOPs（每头）？
   要点：softmax ≈ 4·T²·d = 3.4e10；线性 ≈ 4·T·d² = 5.4e8，约 64 倍（T/d）。
5. 设计题：联邦学习中上传带宽受限，怎么减通信？
   要点：增加本地步数减少轮数、更新量化/稀疏化、只训部分参数（如 LoRA），并注意非 IID 下的漂移。

## 关联笔记

- [M13 张量与流水线并行](module-13-tensor-pipeline-parallel.md)：EP 在 3D 并行中的位置
- [M15 推理](module-15-inference-kv-speculative.md)：KV cache 与 SSM 状态的对比
- [M06 数据并行原语](module-06-data-parallel-primitives.md)：并行 scan
- [M10 ML 编译器](module-10-ml-compilers.md)：device placement 与 ML for systems
- [M16 推理服务](module-16-llm-serving.md)：MoE 与集群级服务调度

## 本讲小结

SSM 用固定状态替代 KV cache，训练走卷积或并行 scan，推理走递推。
MoE 用 all-to-all 换参数规模：参数随 E、FLOPs 随 k，通信计算比只看 d_ff 与带宽算力比。
Chinchilla 给训练算力最优点，部署模型常故意过训。
集群调度的核心是 gang、拓扑与 goodput；联邦学习和 GNN 的系统难点分别是通信与邻域膨胀。

## 延伸阅读

- Katharopoulos et al., Transformers are RNNs, 2020, https://arxiv.org/abs/2006.16236
- Gu et al., Efficiently Modeling Long Sequences with Structured State Spaces (S4), 2022, https://arxiv.org/abs/2111.00396
- Gu & Dao, Mamba, 2023, https://arxiv.org/abs/2312.00752
- Fedus et al., Switch Transformer, 2022, https://arxiv.org/abs/2101.03961
- Lepikhin et al., GShard, 2021, https://arxiv.org/abs/2006.16668
- Hoffmann et al., Training Compute-Optimal LLMs (Chinchilla), 2022, https://arxiv.org/abs/2203.15556
- Gu et al., Tiresias, NSDI 2019, https://www.usenix.org/conference/nsdi19/presentation/gu
- Qiao et al., Pollux, OSDI 2021, https://www.usenix.org/conference/osdi21/presentation/qiao
- Kraska et al., The Case for Learned Index Structures, 2018, https://arxiv.org/abs/1712.01208
- McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, 2017, https://arxiv.org/abs/1602.05629
- Hamilton et al., Inductive Representation Learning on Large Graphs (GraphSAGE), 2017, https://arxiv.org/abs/1706.02216
