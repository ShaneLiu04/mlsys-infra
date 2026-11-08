# Module 18 — 高效架构、MoE、集群与数据

- 课程出处：CS229S Fall 2023《Attention-Free Architectures》《William Fedus 客座：Mixture of Experts》《Scheduling ML Clusters》《Data Pipelines》 · CMU 15-849《Graph Neural Networks》《ML for Systems》《Federated Learning》
- 必读论文：Katharopoulos et al., Transformers are RNNs (Linear Attention), 2020；Gu et al., S4, 2022；Gu & Dao, Mamba, 2023；Shazeer et al., Sparsely-Gated MoE, 2017；Lepikhin et al., GShard, 2021；Fedus et al., Switch Transformer, 2022；Hoffmann et al., Chinchilla, 2022；Xiao et al., Gandiva, 2018；Gu et al., Tiresias, 2019；Qiao et al., Pollux, 2021；Kraska et al., The Case for Learned Index Structures, 2018；McMahan et al., FedAvg, 2017；Hamilton et al., GraphSAGE, 2017
- 对应实验：无（估算可用 [mlsys_calc.py](../tools/mlsys_calc.py)）
- 状态：初稿

## 摘要

前面的模块都默认模型是稠密 Transformer，而且只关心一个训练或推理任务。这一讲从三个方向松动这些假设。

第一个方向是换掉 O(T²) 的 attention，代表是线性 attention 和 SSM（状态空间模型）。SSM 有两种等价的写法：训练时写成卷积，用 FFT 算，复杂度 O(T log T)；推理时写成递推，每个 token O(1)，状态大小固定。所以它没有随 T 增长的 KV cache。

第二个方向是不再让每个 token 用全部参数，代表是 MoE（mixture of experts）。每个 token 只激活 k 个专家，所以参数量可以随专家数增长，而每 token 的 FLOPs 基本不变。代价是 all-to-all 通信、负载不均和更大的显存。

第三个方向是把视野从单个任务扩到整个集群和数据流水线。深度学习作业需要 gang scheduling（一个作业的所有 worker 同时拿到资源），调度器的目标也从「公平地分 GPU」演进到了「最大化 goodput」。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. SSM 的递推是 x_k = Ā·x_{k−1} + B̄·u_k、y_k = C·x_k。参数不随时间变（LTI）时，它可以展开成卷积核 K̄ = (CB̄, CĀB̄, …)，用 FFT 训练。
2. Mamba 让 Δ、B、C 依赖输入（selective），于是不再是卷积。训练改用硬件感知的并行 scan，推理仍是每 token O(1)。
3. MoE 用 top-k 门控。专家容量 = capacity_factor × T·k/E，超出容量的 token 被丢弃（只走残差）。Switch 的负载均衡损失是 α·E·Σ f_i·P_i。
4. Expert parallelism 每个 MoE 层前向 2 次 all-to-all（dispatch 和 combine），反向再 2 次。每次每卡发出 T·k·h·bytes 的 (E_p−1)/E_p。
5. Chinchilla：计算最优约为 20 token/参数，C ≈ 6ND。

必须能默写的公式：

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

一句话区分：

- S4 与 Mamba：S4 线性时不变，可以用卷积训练；Mamba 的参数随输入变化（选择性），用并行 scan 训练。
- MoE 的参数与 FLOPs：参数要存全部 E 个专家，FLOPs 每 token 只算 k 个。
- Gandiva、Tiresias 与 Pollux：Gandiva 做时间片和迁移；Tiresias 在不知道作业时长时按已获服务量排优先级；Pollux 联合调整资源和 batch size，最大化 goodput。
- 数据并行与联邦学习：前者数据可以随意搬、通信便宜；后者数据不出端、通信昂贵，而且数据非 IID。

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

softmax attention 的 O(T²) 来自它必须先算出 T×T 的注意力矩阵。能不能绕开这个矩阵？

如果把 softmax 换成核函数 φ(q)·φ(k)ᵀ，矩阵乘就可以换个结合顺序：先算 φ(K)ᵀV，得到一个 d×d 的矩阵，复杂度变成 O(T·d²)。
在因果情形下，它其实就是一个 RNN：状态 S_t 是一个 d×d 矩阵，每来一个 token 更新一次。
代价是表达力。固定大小的状态很难精确回忆任意远处的某个 token。

### 2. SSM：S4 与 Mamba

SSM 想要两全其美：训练时像 CNN 一样全序列并行，推理时像 RNN 一样每步 O(1)。

它从一个连续系统出发：x'(t) = A·x + B·u，y = C·x，离散化之后得到递推。如果 A、B、C 不随时间变化，这就是一个 LTI（线性时不变）系统，输出等于输入与核 K̄ 的卷积。
于是训练时，长度为 T 的卷积用 FFT 算，O(T log T)，整条序列并行；推理时用递推，每个 token O(1) 计算，状态大小固定。

S4 用 HiPPO 初始化 A，让模型能记住很长的历史，并用结构化的 A 高效计算 K̄。

Mamba 的改进是让 Δ、B、C 由当前输入算出来，这样模型可以选择记住或忽略某个 token。但参数一旦随时间变化，卷积形式就失效了。Mamba 改用在 SRAM 里完成的并行 scan（和 M06 的 scan 同构），不把展开的状态写回 HBM。

Hyena 等长卷积模型走的是另一条路：用隐式参数化的长卷积加门控来替代 attention。

### 3. MoE

稠密模型想加参数，每个 token 的计算量就得跟着涨。MoE 想把这两件事解耦。

它把 FFN 换成 E 个专家，由一个路由器给每个 token 挑 top-k 个专家。Switch Transformer 用 k=1，GShard 用 k=2。
这样参数随 E 线性增长，而每 token 的 FLOPs 只随 k 增长。比如 Mixtral 8×7B 总参数 46.7B，每个 token 只激活 12.9B。

MoE 有两个绕不开的工程问题。

第一个是负载不均：路由器很容易坍缩到少数几个专家上。办法是加辅助损失 α·E·Σ f_i·P_i，它在均匀分布时取最小值 α（常用 α=0.01）。其中 f_i 不可导，梯度经由 P_i 回传。

第二个是容量：为了让张量形状固定，每个专家每批最多处理 CF × T·k/E 个 token，超出的被丢弃，只走残差。CF 大，丢的 token 少，但显存和填充浪费多。训练时 CF 常用 1.0–1.25。

### 4. Expert parallelism

专家多了，一张卡放不下，就要把它们分到不同的卡上。这带来了一种新的通信模式。

设 E 个专家分到 E_p 张卡上。每张卡上的 token 先按路由结果做一次 all-to-all，发给目标专家所在的卡（dispatch）；专家算完后，再做一次 all-to-all 把结果发回来（combine）。
通信量与专家的参数量无关，只和 T·k·h 成正比。跨节点的 all-to-all 受 IB 带宽限制，是 MoE 训练与服务的主要瓶颈。

推理时还有一个容易忽略的点：MoE 的显存要按总参数算，带宽则按激活参数算。但在大 batch 下，不同 token 会激活不同的专家，权重读取量就接近全量了。

### 5. Scaling laws

给定一笔训练算力，模型该多大、数据该多少？

训练算力 C ≈ 6ND（见 M11）。Chinchilla 在固定 C 下拟合最优的 N 和 D，发现两者大致同比例增长，约 20 token/参数。

要注意这是训练算力最优，不是推理成本最优。要部署的模型常常远超 20 token/参数，比如 Llama-3-8B 训了 15T token，约 1900 token/参数。这是用更多的训练算力，换一个推理时更小的模型。

### 6. DL 集群调度

到了集群层面，问题变成：许多用户、许多作业共享一大批 GPU，怎么分？深度学习作业有几个和普通 CPU 作业不同的特点，调度器得专门对付。

第一是 gang scheduling：一个分布式作业的所有 worker 必须同时拿到 GPU，否则先启动的 worker 只能空等 all-reduce。
第二是局部性：同一个作业的 GPU 最好在同一节点（走 NVLink）或同一机架。

几个代表性的调度器各有侧重：

- Gandiva：利用训练迭代的可预测性，做时间片轮转、作业迁移和内省式打包。
- Tiresias：作业时长事先未知，于是按已获得的服务量（GPU 数 × 时间）做多级反馈队列（2D-LAS），近似短作业优先。
- Pollux：作业的吞吐和统计效率都随 batch size 变化。它联合选择 GPU 数和 batch size，最大化 goodput = 吞吐 × 统计效率。

此外还有异构问题：同一个作业在 A100 和 H100 上的加速比不同，Gavel 等系统按「有效吞吐」归一化之后再分配。

### 7. 数据流水线

GPU 再快，数据喂不上也是白搭。数据这一侧既有「数据从哪来、质量如何」的问题，也有「怎么及时送到 GPU」的问题。

前者的代表是弱监督：Snorkel 用多个有噪声的标注函数，加一个标签模型，来替代人工标注。数据选择和去重也很关键，质量过滤和近似去重决定了预训练的效果。

后者是 dataloader。文本的 token 读取量极小，瓶颈在 tokenization 和打包；图像和视频的瓶颈则是 CPU 解码与数据增强。常用手段有：预处理离线化、多 worker、pinned memory、预取、GPU 解码。

### 8. ML for systems、联邦学习、GNN

最后是几个课程里涉及的交叉方向，每个只记它的系统要点：

- Learned index：把「有序的键到位置」的映射看作一个 CDF，用小模型预测位置，再在误差界内做局部搜索。
- Chip placement：用强化学习加图神经网络放置宏单元，和 M10 的 device placement 是同一个思路。
- FedAvg：每轮选一部分客户端，各自在本地训练若干 epoch，服务器按样本数加权平均。难点在于非 IID、客户端掉线、通信成本和隐私（安全聚合、差分隐私）。
- GNN：消息传递本质上是稀疏矩阵乘（SpMM）。全图训练时，L 层的邻域会指数膨胀；GraphSAGE 按层固定采样数（比如 25、10），控制每个种子节点的计算量。

## 关键推导

### SSM 递推与卷积等价

SSM 能「训练用卷积、推理用递推」，靠的是这两种形式在数学上等价。把递推展开一下就能看出来：

x_0 = B̄·u_0，x_1 = Ā·B̄·u_0 + B̄·u_1，…，x_k = Σ_{j=0}^{k} Ā^{k−j}·B̄·u_j。
所以 y_k = C·x_k = Σ_j C·Ā^{k−j}·B̄·u_j = (K̄ * u)_k，其中 K̄_j = C·Ā^j·B̄。

这个推导的前提是 Ā、B̄、C 与 k 无关。Mamba 让它们随输入变化，于是高效的算法就只剩递推这一种（可以用并行 scan 加速）。

### 负载均衡损失的最小值

为什么辅助损失要写成 α·E·Σ f_i·P_i 这个样子？这一节说明它在均匀路由时最小，而且最小值和专家数无关。

已知 Σ f_i = 1、Σ P_i = 1。假设路由可微地使 f_i 和 P_i 一致，则 Σ f_i·P_i = Σ P_i² ≥ 1/E（由 Cauchy–Schwarz 不等式），均匀分布时取等号。
乘以 E 归一化之后，最小值就是 α，与专家数无关。这样不同配置可以用同一个 α。

## 深入推导

### MoE 的通信与计算比

M13 推过 TP 的通信计算比，这里用同样的方法看 expert parallelism：EP 能不能跨节点？

设每张卡有 T 个 token，隐藏维 h，top-k，bf16。一次 all-to-all 每张卡发出 2·T·k·h·(E_p−1)/E_p 字节，一层前向加反向共 4 次。
每层专家部分的计算是 6 × P_expert × k × T FLOP，其中 P_expert 是一个专家的参数量。
比值 = 6·P_expert·k·T/F ÷ 8·T·k·h·(E_p−1)/(E_p·B) = 0.75·P_expert·B·E_p / (h·F·(E_p−1))。

T 和 k 被约掉了。和 TP 一样，这个比值只取决于模型结构和硬件。
对门控 FFN，P_expert = 3·h·d_ff，所以比值 ≈ 2.25·d_ff·B / F（E_p 较大时）。代入 d_ff=14336 和 H100：NVLink 下 ≈ 14.7，IB 下 ≈ 1.6。
也就是说，跨节点做 EP 时，通信接近计算的 60%，必须和计算重叠，或者用分层 all-to-all。

下面的代码实现了带容量限制和辅助损失的 top-k 门控，以及 SSM 的推理模式（逐 token 递推）：

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

这组估算覆盖本讲的四个主题：Mixtral 形状下 all-to-all 和专家计算各花多少时间、SSM 状态和 KV cache 差多少、按 Chinchilla 规则训一个模型要多少 GPU 时，以及数据和联邦学习的带宽量级。
硬件数字取自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 989 TFLOP/s、NVLink 450 GB/s、IB NDR 50 GB/s；Llama-3-8B KV 128 KB/token。

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

数据那一行值得多看一眼：一张卡每秒只需要约 33 KB 的文本 token。所以文本训练的 dataloader 瓶颈不在读盘，而在 tokenization 和打包。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 训练算力 6ND | [mlsys_calc.py](../tools/mlsys_calc.py) `train_flops` | 8B × 15T 输出 7.2e23 |
| MFU | 同上 `mfu` | 给定 token/s 与卡数，结果与 6N·tok/s/(峰值·卡数) 一致 |
| 通信带宽 | 同上 `hw` 与 `allreduce` | 与本讲 NVLink、IB 数字一致 |
| top-k 门控 | 本讲 `topk_gating` | 均匀路由时 aux = α；超容量 token 的权重为 0 |

## 易错点与反例

- 「MoE 省显存」：错。显存按总参数算，Mixtral 要存 46.7B；省的是每个 token 的计算。
- 「MoE 推理一定快」：小 batch 时只读被激活的专家，确实快；大 batch 时几乎要读全部专家，带宽按总参数算，还要加上 all-to-all。
- 「SSM 训练也是逐 token 递推」：LTI 的 S4 用 FFT 卷积并行训练；Mamba 用并行 scan，同样不是逐 token 串行。
- 「线性 attention 和 softmax attention 等价」：它只是近似，固定大小的状态在精确回忆类任务上明显更弱。
- 「负载均衡损失越大越好」：α 太大会压过主损失，路由退化成均匀随机。
- 「Chinchilla 说模型训 20 token/参数最好」：那只是训练算力最优；考虑推理成本时应该训更多 token。
- 「集群调度和 CPU 调度一样」：DL 作业需要 gang、对放置敏感、可以迁移，而且吞吐随资源的变化是非线性的。
- 「FedAvg 等价于大 batch SGD」：本地多步更新加上非 IID 数据会造成客户端漂移，两者并不等价。

## 实验或练习

这个练习验证一个假设：没有辅助损失时，top-k 门控会出现专家坍缩；加上 α=0.01 之后，各专家的负载接近 T·k/E。

- 配置：随机初始化一个 E=8 的路由器，在合成数据上训练若干步，记录每个专家分到的 token 数和丢弃率；对比有无辅助损失，以及 CF ∈ {1.0, 1.25, 2.0}。
- 预期：没有辅助损失时，少数专家吃掉大部分 token，丢弃率高；CF 增大会降低丢弃率，但填充浪费会增加。

## 高频问题

- 问：为什么 MoE 增加参数却不增加 FLOPs？
  答：每个 token 只经过 k 个专家，FLOPs 与 k 成正比；参数则与 E 成正比。
- 问：容量因子有什么作用？
  答：它限制每个专家每批处理的 token 数，保证张量形状固定。超出的 token 被丢弃，CF 控制的是丢弃与浪费之间的折中。
- 问：线性 attention 的代价是什么？
  答：表达力。它用一个固定大小的 d×d 状态概括全部历史，难以精确回忆任意远处的 token，在回忆类任务上明显弱于 softmax attention。
- 问：gang scheduling 是什么？
  答：分布式作业的所有 worker 同时分配资源，避免部分 worker 空等。
- 问：Chinchilla 的结论是什么？
  答：固定算力下，参数和数据应同比例扩大，约 20 token/参数，C ≈ 6ND。

## 面试追问链

1. 问：Mamba 和 Transformer 推理时的显存差在哪？
   答：Transformer 的 KV 随 T 线性增长；Mamba 每层是固定大小的状态（d=4096 时约 13 万个元素），只相当于 64 个 token 的 KV。
2. 问：推一下 SSM 的递推和卷积为什么等价。
   答：展开递推得到 x_k = Σ Ā^{k−j}B̄u_j，代入 y = C·x，就得到卷积核 K̄_j = CĀ^jB̄。前提是参数不随时间变化。
3. 问：那 Mamba 为什么不能用卷积训练？用什么代替？
   答：它的选择性参数随输入变化，不再是 LTI 系统。改用并行 scan，在 SRAM 内完成，不把全部状态物化出来。
4. 问：MoE 的 all-to-all 通信量有多大？和什么有关？
   答：每次每卡 T·k·h·bytes × (E_p−1)/E_p，前后向共 4 次，和专家参数量无关。Mixtral 形状、每卡 8192 token 时，一次是 117 MB，IB 上要 2.35 ms。
5. 问：EP 放在节点内还是跨节点？
   答：通信计算比 ≈ 2.25·d_ff·B/F，NVLink 下约 15，IB 下约 1.6。能放进节点就放节点内；专家多到放不下时，用分层 all-to-all（先在节点内聚合，再跨节点），并和计算重叠。
6. 问：MoE 服务和稠密模型服务有什么不同？
   答：显存按总参数算，所以要更多卡或者做量化；小 batch 时带宽按激活参数算，大 batch 时接近总参数；专家负载不均会拉高尾延迟；EP 的 all-to-all 落在每一层的关键路径上，TP 和 EP 怎么组合要按拓扑来选。
7. 问：要给一个 1000 卡的集群设计训练作业调度器，你关心什么？
   答：首先是 gang 分配和拓扑感知放置（同节点、同交换机）。作业时长未知时，按已获服务量排优先级（Tiresias 的思路）。支持检查点和迁移，用来整理碎片。对弹性作业，联合调整卡数和 batch size（Pollux 的思路）。异构卡按有效吞吐分配。出故障时能快速从检查点恢复。

## 自测题

1. E=64、k=2、T=4096、CF=1.0，每个专家的容量是多少？
   要点：4096 × 2 / 64 = 128。
2. C = 6e23，按 Chinchilla 规则，最优的 N 和 D 是多少？
   要点：N = sqrt(6e23/120) = sqrt(5e21) ≈ 71 B，D ≈ 1.4 T。
3. 证明 Σ f_i·P_i 在 f = P 时的最小值是 1/E。
   要点：Σ P_i² ≥ (Σ P_i)²/E = 1/E。
4. 线性 attention 在 T=8192、d=128 时比 softmax attention 省多少 FLOPs（每个头）？
   要点：softmax ≈ 4·T²·d = 3.4e10；线性 ≈ 4·T·d² = 5.4e8，约 64 倍（即 T/d）。
5. 设计题：联邦学习中上传带宽受限，怎么减少通信？
   要点：增加本地步数来减少轮数；对更新做量化或稀疏化；只训练一部分参数（比如 LoRA）。同时要注意非 IID 下的漂移。

## 关联笔记

- [M13 张量与流水线并行](module-13-tensor-pipeline-parallel.md)：EP 在 3D 并行中的位置
- [M15 推理](module-15-inference-kv-speculative.md)：KV cache 与 SSM 状态的对比
- [M06 数据并行原语](module-06-data-parallel-primitives.md)：并行 scan
- [M10 ML 编译器](module-10-ml-compilers.md)：device placement 与 ML for systems
- [M16 推理服务](module-16-llm-serving.md)：MoE 与集群级服务调度

## 本讲小结

SSM 用固定大小的状态替代 KV cache：训练走卷积或并行 scan，推理走递推。
MoE 用 all-to-all 换参数规模。参数随 E 增长，FLOPs 随 k 增长，通信计算比只看 d_ff 与硬件的带宽算力比。
Chinchilla 给出的是训练算力的最优点，要部署的模型常常故意过训。
集群调度的核心是 gang、拓扑和 goodput；联邦学习和 GNN 的系统难点分别是通信和邻域膨胀。

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
