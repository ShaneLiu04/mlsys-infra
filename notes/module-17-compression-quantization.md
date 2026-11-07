# Module 17 — 模型压缩：量化、剪枝、蒸馏、PEFT

- 课程出处：CS229S Fall 2023《Memory Efficient Neural Networks》《Adapting LLMs》 · CMU 15-849《Model Compression》
- 必读论文：Han et al., Deep Compression, 2016；Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference, 2018；Dettmers et al., LLM.int8(), 2022；Xiao et al., SmoothQuant, 2023；Frantar et al., GPTQ, 2023；Lin et al., AWQ, 2024；Frankle & Carbin, The Lottery Ticket Hypothesis, 2019；Hinton et al., Distilling the Knowledge in a Neural Network, 2015；Hu et al., LoRA, 2022；Dettmers et al., QLoRA, 2023
- 对应实验：[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)（实测：GPT-2 W3：RTN ppl 65.9 → GPTQ 37.2；Wanda 50% 稀疏 47.2 对幅值剪枝 538）
- 状态：初稿

## 摘要

推理 decode 由字节决定（M15），所以压缩的首要收益是少读字节：权重 int4 让 batch=1 的 decode 上界接近 bf16 的 4 倍。
量化把浮点映射到少量离散值，核心参数是 scale、zero-point 和粒度（逐张量、逐通道、逐组）。LLM 的难点在激活：少数通道的异常值比其他大两个数量级，逐张量 int8 会把正常值压成几个台阶。
weight-only（GPTQ、AWQ）解决访存受限的 decode；W8A8（SmoothQuant）让 GEMM 用低精度 tensor core，解决算力受限的 prefill。
剪枝和蒸馏减少参数或换小模型；LoRA/QLoRA 不压缩推理模型，而是压缩微调时的显存。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 指数位决定范围，尾数位决定精度：bf16 是 8/7，fp16 是 5/10，e4m3 是 4/3（最大 448），e5m2 是 5/2（最大 57344）。
2. 对称量化 s = max|x| / (2^{b−1} − 1)，非对称 s = (max − min)/(2^b − 1)、z = round(−min/s)；最大舍入误差 s/2。
3. 激活异常值让逐张量 int8 失效：LLM.int8 把 |x| ≥ 6 的特征维拆出来用 fp16 算；SmoothQuant 用 s_j 把难度从激活挪到权重。
4. 2:4 结构化稀疏：每 4 个元素至少 2 个为零，sparse tensor core 理论 2 倍，非结构化稀疏在 GPU 上很难提速。
5. LoRA：W = W0 + (α/r)·B·A，参数 r·(d + k)，推理时合并进 W0 无额外延迟；QLoRA = NF4 底座 + 双重量化 + 分页优化器。

**必须能默写的公式**

```text
量化：q = clamp(round(x/s) + z, q_min, q_max)；反量化：x̂ = s·(q − z)
对称 int8：s = max|x| / 127，z = 0
非对称 uint8：s = (max − min)/255，z = round(−min/s)
舍入误差：|x − x̂| ≤ s/2，均匀假设下方差 s²/12
SmoothQuant：s_j = max|X_j|^α / max|W_j|^{1−α}，Y = (X·diag(s)⁻¹)·(diag(s)·W)
STE：前向 q(x)，反向 ∂q/∂x ≈ 1（在裁剪区间内），区间外为 0
蒸馏：L = (1−λ)·CE(y, σ(z_s)) + λ·T²·KL(σ(z_t/T) ‖ σ(z_s/T))
LoRA：h = W0·x + (α/r)·B·(A·x)，B ∈ R^{d×r} 初始化为 0，A ∈ R^{r×k}
每权重比特（逐组）：b + 16/g（fp16 scale，组大小 g）；int4、g=128 → 4.125
```

**一句话区分**

- PTQ vs QAT：训练后用校准数据定 scale vs 训练中插入伪量化、用 STE 回传梯度。
- GPTQ vs AWQ：用二阶信息逐列量化并补偿误差 vs 按激活幅度找重要通道、先缩放再量化。
- weight-only vs W8A8：省带宽、decode 受益 vs 省算力、prefill 受益。
- 结构化 vs 非结构化剪枝：删整行/整头、直接提速 vs 删单个权重、需要稀疏 kernel。

## 学习目标

- 说清各数值格式的位分配与适用场景；
- 手算一组数的 absmax 与非对称量化，并给出误差上界；
- 解释激活异常值为何破坏 int8，比较 LLM.int8、SmoothQuant、AWQ、GPTQ 的应对方式；
- 推导蒸馏损失中的 T² 因子；
- 计算 LoRA 参数量与 QLoRA 的显存占用。

## 先修知识

- M14：fp16/bf16/fp8 的范围与精度；
- M15：decode 的字节瓶颈；
- M07：tensor core 与 GEMM。

## 核心概念

### 1. 数值格式

fp32 1/8/23；tf32 1/8/10；bf16 1/8/7；fp16 1/5/10；fp8 e4m3 1/4/3、e5m2 1/5/2；int8 [−128, 127]；int4 [−8, 7]。
浮点的相对误差恒定（由尾数决定），整数的绝对误差恒定（由 scale 决定）。权重近似正态分布，大部分值集中在零附近，所以 NF4 按正态分位点放置 16 个值，比均匀 int4 更省误差。

### 2. 线性量化的粒度

逐张量：一个 scale，最省，最受异常值影响。逐通道：每个输出通道一个 scale，权重量化的默认选择。
逐组：每 g 个连续元素一个 scale（g=128 常见），int4 必用，额外开销 16/g 比特。
激活通常按 token 或按张量量化，因为通道维 scale 无法在矩阵乘后提出来（除非像 SmoothQuant 那样并进权重）。

### 3. k-means 量化

Deep Compression：剪枝 + 权重共享（k-means 聚类，存聚类中心码本 + 每个权重的索引）+ Huffman 编码，AlexNet 压缩 35 倍、VGG-16 压缩 49 倍。
非线性码本精度高，但推理要查表，GPU 上不如线性量化好做 kernel。

### 4. PTQ 与 QAT

PTQ：拿几百条校准样本统计激活范围，定 scale，不更新权重。便宜，8 bit 基本无损，4 bit 以下损失明显。
QAT：前向插入伪量化（量化再反量化），round 的导数几乎处处为零，用 STE 近似为 1。成本是一次（部分）训练。

### 5. LLM 量化方法

LLM.int8：逐向量（行/列）量化 + 混合精度分解。约 6.7B 起出现系统性的异常特征维，幅度 ≥ 6 的维度用 fp16 矩阵乘，其余 int8。
SmoothQuant：激活难量化、权重好量化。对每个输入通道 j 除以 s_j，权重对应行乘以 s_j，数学等价，α=0.5 常用。之后 W8A8 都能逐张量量化。
GPTQ：逐层最小化 ‖WX − ŴX‖²，用校准激活的 Hessian H = 2XXᵀ，一次量化一列，把误差按 H⁻¹ 分摊到尚未量化的列。175B 模型数小时量化完成。
AWQ：约 1% 的权重通道很重要，重要性由对应激活幅度决定。不做混合精度，而是给这些通道乘一个 s > 1 再量化，相对误差变小，激活侧除以 s。
KV cache 量化：int8/fp8 把 KV 减半；更低比特时 key 按通道、value 按 token 量化效果更好（KIVI 的观察）。

### 6. 剪枝

幅度剪枝：删绝对值最小的权重，再微调恢复。敏感度分析：逐层扫描剪枝率与精度，敏感层少剪。
非结构化：50% 稀疏在 GPU 上几乎没有加速，索引开销和不规则访存吃掉收益。
结构化：删通道、头、层，得到更小的稠密模型，直接提速但精度损失大。
2:4：每 4 个连续权重至少 2 个为零，Ampere 起的 sparse tensor core 用 2 bit 元数据跳过零，理论 2 倍算力，实测收益小于 2 倍。
Lottery ticket：迭代幅度剪枝后，把剩余权重回到初始值（或早期值）重训，能达到原精度，说明稠密网络里存在可训练的稀疏子网络。

### 7. 知识蒸馏

学生拟合教师的软标签 σ(z_t/T)。T > 1 让分布变平，暴露类别之间的相似度（暗知识）。
软目标的梯度随 T 变成 1/T² 量级，乘 T² 使软硬两项的梯度量级与 T 无关。

### 8. PEFT：LoRA 与 QLoRA

LoRA：冻结 W0，学低秩增量 BA。B 初始化为 0，训练开始时模型与原模型完全相同。
d=k=4096、r=16：2·d·r = 131072，相对 d² = 16.8 M 是 0.78%。优化器状态只为 LoRA 参数分配。
推理时 W = W0 + (α/r)·BA 合并，无额外延迟；多租户时不合并，按请求动态加（M16 的 S-LoRA）。
QLoRA：底座 NF4 量化存储、计算时反量化到 bf16；双重量化把每 64 个权重一个 fp32 scale（0.5 bit/参数）再量化成 8 bit（每 256 个 scale 共享一个 fp32），降到约 0.127 bit/参数；分页优化器用统一内存在显存尖峰时把优化器状态换到 CPU。

## 关键推导

### 量化误差与异常值

对称 int8，s = max|x|/127，误差 ≤ s/2。正常值在 ±1，若同一张量有一个 60 的异常值：s = 60/127 = 0.472。
0.3 → round(0.635) = 1 → 0.472，相对误差 57%；±1 之间只剩 {−2, −1, 0, 1, 2} 五个台阶。
逐通道（对激活即逐输入维）能救，但矩阵乘 Y = XW 的求和维正是这个维度，scale 无法在累加后提出来。这就是 SmoothQuant 把 scale 并入权重的原因。

### 蒸馏的 T²

p_i = softmax(z_s/T)_i，q_i = softmax(z_t/T)_i。∂KL/∂z_s,i = (p_i − q_i)/T。
T 大时 p_i ≈ (1 + z_s,i/T)/K（logits 零均值），于是 p_i − q_i ≈ (z_s,i − z_t,i)/(K·T)，梯度 ≈ (z_s,i − z_t,i)/(K·T²)。
乘 T² 后梯度量级与 T 无关；T→∞ 时等价于匹配 logits 的 MSE。

## 深入推导

### GPTQ 的误差补偿

目标：min ‖WX − ŴX‖²，逐行独立。对一行 w，损失是二次型 (w − ŵ)ᵀH(w − ŵ)，H = 2XXᵀ。
量化第 q 个元素引入误差 δ = w_q − quant(w_q)。OBS 给出最优补偿：其余未量化元素更新 Δw = −δ/[H⁻¹]_qq · H⁻¹_{:,q}。
GPTQ 的工程化：所有行用相同的列顺序，H⁻¹ 只算一次；按 128 列分块延迟更新，变成 GEMM；用 Cholesky 分解保证数值稳定。
结果：每层只需一次前向收集 X，量化是单次的（one-shot），不需要反向传播。

### W4A16 的 decode 收益与上限

decode 读权重 b/8 字节/参数。int4 + g=128 的 fp16 scale：4.125 bit，约 bf16 的 1/3.9。
batch 小时步时间 ∝ 权重字节，加速接近 3.9 倍；batch 增大后 KV 与反量化开销占比上升，收益下降。
到 prefill（算力受限）时，W4A16 要先反量化成 bf16 再用 bf16 tensor core，没有算力收益，甚至更慢。prefill 要靠 W8A8（int8 或 fp8）。

```python
import torch

def quant_absmax_group(w: torch.Tensor, bits=4, group=128):
    """对称逐组量化。w: [out, in]，in 可被 group 整除。"""
    qmax = 2 ** (bits - 1) - 1
    wg = w.reshape(w.shape[0], -1, group)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax).to(torch.int8)
    return q, scale.to(torch.float16)

def dequant(q, scale, shape):
    return (q.float() * scale.float()).reshape(shape)

class FakeQuantSTE(torch.autograd.Function):          # QAT：前向量化，反向直通
    @staticmethod
    def forward(ctx, x, scale, qmax):
        ctx.save_for_backward(x); ctx.lim = scale * qmax
        return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= ctx.lim), None, None

class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features) / r)
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))   # 初始增量为 0
        self.scaling = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling
    @torch.no_grad()
    def merge(self):                                   # 推理前合并，无额外延迟
        self.base.weight += self.scaling * self.B @ self.A
```

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 80 GB、3.35 TB/s、bf16 989、fp8 1979 TFLOP/s；Llama-3-70B 70.6 B、KV 320 KB/token。

```text
absmax int8 手算：x = [0.5, −1.2, 3.0, 0.1]，s = 3.0/127 = 0.0236
  q = [21, −51, 127, 4]；x̂ = [0.496, −1.205, 3.0, 0.094]；最大误差 ≤ s/2 = 0.0118

Llama-3-70B 权重：bf16 141.2 GB；int8 70.6 GB；int4 + g128 scale：70.6e9 × 4.125/8 = 36.4 GB
  int4 单卡 H100：KV 余量 72 − 36.4 = 35.6 GB（90% 可用）
    8K 序列 2.68 GB → 13 条；KV int8 后 1.34 GB → 26 条
  batch=1 decode 上界：36.4 GB / 3.35 TB/s = 10.9 ms → 92 token/s（单卡）
  对照 bf16 4 卡 TP：10.5 ms → 95 token/s，但用了 4 张卡
Llama-3-8B int4：8.03e9 × 4.125/8 = 4.14 GB → 1.24 ms → 约 800 token/s 上界（bf16 为 208）
W8A8（fp8）prefill：算力上限 1979 vs 989 TFLOP/s，prefill 理论快 2 倍

LoRA，Llama-2-7B（q/k/v/o 都是 4096×4096），r=16：
  每矩阵 2 × 4096 × 16 = 131072；每层 4 个 → 524288；32 层 16.8 M 参数（0.25%）
  显存：bf16 底座 13.5 GB + LoRA 16.8M × 16 B = 0.27 GB + 激活；全参需 107.8 GB
QLoRA 双重量化：0.5 bit → 8/64 + 32/(64×256) = 0.127 bit，省 0.373 bit/参数
  65B：省 65e9 × 0.373 / 8 = 3.0 GB；NF4 底座 65e9 × 4.127/8 = 33.5 GB，单张 48 GB 卡可微调
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| absmax 逐组量化 | [编译与压缩实验](../solutions/cmu15849/compiler-compression/README.md) 的量化模块 | 反量化误差逐元素 ≤ s/2；group 越小误差越小 |
| 异常值实验 | 同上，向激活注入单通道异常值 | 逐张量 int8 误差显著上升，SmoothQuant 后恢复 |
| 剪枝 | 同上，幅度剪枝 + 敏感度曲线 | 输出每层剪枝率 vs 精度表 |
| LoRA | 同上，LoRALinear | 初始化时输出与底座完全相同；merge 前后输出一致 |

## 易错点与反例

- 「int4 一定比 int8 快 2 倍」：只在访存受限时近似成立；prefill 中 W4A16 要反量化，没有算力收益。
- 「bf16 比 fp16 精度高」：反了。bf16 范围大、精度低（7 位尾数 vs 10 位）。
- 「逐通道量化激活就能解决异常值」：激活的异常值在矩阵乘的求和维上，scale 提不出来，需要 SmoothQuant 式的迁移。
- 「50% 非结构化稀疏 = 2 倍加速」：GPU 上通常几乎无加速；2:4 才有硬件支持。
- 「LoRA 推理更慢」：合并后与原模型完全相同；只有不合并的多租户场景才有额外计算。
- 「QLoRA 是 4 bit 训练」：计算在 bf16 中进行，4 bit 只用于存储底座；LoRA 参数与梯度是 bf16。
- 「蒸馏的 T² 是经验系数」：它来自梯度随 T 的 1/T² 缩放，是推导出来的。
- 「GPTQ 需要重训」：不需要，只要少量校准数据前向。

## 实验或练习

- 假设：逐组量化误差随组变小而下降；注入异常值后逐张量 int8 误差升高一个数量级以上。
- 配置：取一个小 Transformer 的某层权重与激活；比较逐张量/逐通道/逐组（g=32、128）在 int8 与 int4 下的 ‖WX − ŴX‖ 相对误差；对激活第 0 维乘 50 后重测。
- 预期：int4 逐张量误差最大，g=32 最小；异常值让逐张量激活量化严重劣化，SmoothQuant（α=0.5）缓解。

## 高频问题

- 问：对称和非对称量化的区别？
  答：对称 z=0，适合以零为中心的权重；非对称有 zero-point，适合 ReLU 后全非负的激活，多用一个台阶的范围。
- 问：为什么 LLM 的激活难量化？
  答：少数通道有比其他大约两个数量级的异常值，逐张量 scale 被它主导，其余值只剩几个台阶。
- 问：GPTQ 和 AWQ 都是 weight-only，有什么不同？
  答：GPTQ 用二阶信息补偿量化误差；AWQ 用激活幅度找重要通道并做等价缩放，不需要反向，泛化好。
- 问：LoRA 为什么 B 初始化为 0？
  答：使初始增量为 0，模型从预训练状态出发；A 随机以打破对称。
- 问：2:4 稀疏是什么？
  答：每 4 个连续权重至少 2 个零，硬件用元数据跳过，理论 2 倍算力。

## 面试追问链

1. 问：bf16、fp16、fp8 怎么选？
   答：训练用 bf16（范围同 fp32，免 loss scaling）；fp16 精度更高但需 loss scaling；fp8 前向 e4m3、梯度 e5m2，配 scaling factor。
2. 问：手写对称 int8 量化，误差上界多少？
   答：s = max|x|/127，q = round(x/s)，x̂ = s·q，误差 ≤ s/2。
3. 问：为什么逐张量 int8 在 LLM 激活上失效？怎么修？
   答：异常值把 s 撑大，正常值只剩几个台阶。修法：LLM.int8 拆出异常维走 fp16；SmoothQuant 用 s_j 把异常迁到权重后再 W8A8；或只量化权重。
4. 问：weight-only 和 W8A8 分别加速什么？
   答：weight-only 减少每步读的字节，加速访存受限的 decode；W8A8 让 GEMM 走 fp8/int8 tensor core，加速算力受限的 prefill 和大 batch decode。
5. 问：蒸馏为什么乘 T²？
   答：软目标梯度 ≈ (z_s − z_t)/(K·T²)，乘 T² 保持与硬标签项同量级，换 T 时不用重调 λ。
6. 问：70B 要单卡 H100 上线，怎么做？代价是什么？
   答：int4 + g128 权重 36.4 GB，剩 35.6 GB 给 KV，8K 序列约 13 条，KV int8 后 26 条；batch=1 上界约 92 token/s。代价：4 bit 精度损失要在任务集上评估；prefill 无算力收益；长上下文高并发仍不够，需多卡。
7. 问：一个团队要为 200 个客户各微调一个 8B 模型，预算有限，怎么设计训练与上线？
   答：训练用 LoRA 或 QLoRA（底座冻结、只训 r=16 适配器，单卡即可）；上线共享一份底座，适配器每个几十 MB，用 S-LoRA/Punica 式批处理混合服务；底座可再做 weight-only int4 提高并发，但需在每个客户的评测集上验证量化 + LoRA 组合的精度。

## 自测题

1. 非对称 uint8 量化 [−1, 3]，s 和 z 是多少？0 映射到几？
   要点：s = 4/255 = 0.0157，z = round(1/0.0157) = 64；0 → 64，正好可表示，零点无误差。
2. int4、组大小 64、fp16 scale，每权重多少比特？
   要点：4 + 16/64 = 4.25。
3. d=8192、k=8192、r=8，LoRA 参数占比？
   要点：8 × 16384 = 131072，/ 67.1 M = 0.2%。
4. 证明 SmoothQuant 的变换数学等价。
   要点：XW = X·diag(s)⁻¹·diag(s)·W，逐通道缩放互相抵消。
5. 设计题：一个 CNN 要部署到只支持 int8 的边缘芯片，PTQ 掉点 3%，怎么办？
   要点：逐通道权重、更好的校准（百分位裁剪）、敏感层保留高精度；仍不够做 QAT（STE）。

## 关联笔记

- [M14 显存优化](module-14-memory-optimization.md)：fp16/bf16/fp8 的训练侧用法与 8-bit 优化器
- [M15 推理](module-15-inference-kv-speculative.md)：decode 由字节决定，量化直接缩短步时间
- [M16 推理服务](module-16-llm-serving.md)：多 LoRA 服务与 KV 量化带来的并发
- [M10 ML 编译器](module-10-ml-compilers.md)：量化算子的融合与代码生成
- [edge-deploy M02–M03、M08](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-03-calibration-localization.md)：端侧的整数契约、校准与误差定位，以及同一 GPT-2 协议下的 Q8_0 / Q4_0 困惑度

## 本讲小结

量化的三个旋钮：比特数、scale/zero-point、粒度；误差上界 s/2，异常值通过 s 放大所有值的误差。
weight-only 省字节救 decode，W8A8 省算力救 prefill，两者解决不同的瓶颈。
剪枝要有硬件支持的结构（2:4、整头整通道）才能提速；蒸馏的 T² 来自梯度缩放。
LoRA 把微调显存从 16N 降到底座 2N 加极少的可训练参数，QLoRA 再把底座压到约 0.52 字节/参数。

## 延伸阅读

- Han et al., Deep Compression, 2016, https://arxiv.org/abs/1510.00149
- Jacob et al., Integer-Arithmetic-Only Inference, 2018, https://arxiv.org/abs/1712.05877
- Dettmers et al., LLM.int8(), 2022, https://arxiv.org/abs/2208.07339
- Xiao et al., SmoothQuant, 2023, https://arxiv.org/abs/2211.10438
- Frantar et al., GPTQ, 2023, https://arxiv.org/abs/2210.17323
- Lin et al., AWQ, 2024, https://arxiv.org/abs/2306.00978
- Frankle & Carbin, The Lottery Ticket Hypothesis, 2019, https://arxiv.org/abs/1803.03635
- Hinton et al., Distilling the Knowledge in a Neural Network, 2015, https://arxiv.org/abs/1503.02531
- Hu et al., LoRA, 2022, https://arxiv.org/abs/2106.09685
- Dettmers et al., QLoRA, 2023, https://arxiv.org/abs/2305.14314
