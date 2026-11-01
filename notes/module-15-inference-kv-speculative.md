# Module 15 — Transformer 推理：KV cache 与推测解码

- 课程出处：CS229S Fall 2023《Analyzing the Performance of Transformers (10-09)》《Project 2：KV caching and speculative decoding in nanoGPT》
- 必读论文：Pope et al., Efficiently Scaling Transformer Inference, 2022；Shazeer, Fast Transformer Decoding (MQA), 2019；Ainslie et al., GQA, 2023；DeepSeek-AI, DeepSeek-V2 (MLA), 2024；Leviathan et al., Fast Inference from Transformers via Speculative Decoding, 2023；Chen et al., Accelerating LLM Decoding with Speculative Sampling, 2023；Cai et al., Medusa, 2024；Li et al., EAGLE, 2024
- 对应实验：[CS229S P2 KV cache 与推测解码](../solutions/cs229s/project2-kv-speculative/README.md)
- 实验报告：[报告 PDF](../reports/cs229s-project2/cs229s_project2_report.pdf)（实测：gpt2-xl decode 20.6 → 4.22 ms；推测解码贪心 1.63 倍）
- 状态：初稿

## 摘要

自回归生成分两段：prefill 一次处理整个 prompt，矩阵乘的 M 维是 prompt 长度，算力受限；decode 每步只生成一个 token，M 维是 batch，访存受限。
KV cache 把历史 token 的 K、V 存下来，使每步 decode 的计算从 O(T²) 降到 O(T)，代价是显存 2·L·n_kv·d_head·T·B·bytes。
decode 对权重的算术强度约等于 batch size，H100 上要 batch ≈ 300 才到拐点，所以推理优化的主线是「把 batch 做大」和「把每步的字节变少」。
推测解码用小模型起草 γ 个 token、大模型一次前向验证，接受规则 min(1, p/q) 保证输出分布与大模型完全相同。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 每 token 前向 FLOPs ≈ 2N + 4·L·d·T，第二项是 attention 对 T 个历史 token 的点积与加权和。
2. KV cache = 2·L·n_kv·d_head·T·B·bytes；Llama-3-8B 每 token 128 KB，70B 每 token 320 KB。
3. decode 权重部分强度 = 2NB / 2N = B FLOP/B；attention 部分强度 = n_heads/n_kv，与 batch 无关。
4. TTFT 由 prefill 决定（算力），TPOT 由 decode 决定（带宽）；batch=1 时 TPOT 下界 = 权重字节 / HBM 带宽。
5. 推测解码每次目标前向期望产出 (1−α^{γ+1})/(1−α) 个 token，加速比再除以 (γ·c + 1)。

**必须能默写的公式**

```text
每 token FLOPs：2N + 4·L·d·T                    （N 参数量，T 当前上下文长度）
KV 字节：2 · L · n_kv · d_head · T · B · bytes
decode 步时间下界：max( (2NB + 4LdTB)/F , (bytes·N + KV)/BW )
临界 batch：B* ≈ F / BW × bytes / 2 = 拐点（bf16）
接受概率：min(1, p(x)/q(x))；拒绝后从 norm(max(0, p − q)) 重采样
α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)
E[tokens/目标前向] = (1 − α^{γ+1}) / (1 − α)
加速比 = E / (γ·c + 1)，c = 草稿一次前向 / 目标一次前向
```

**一句话区分**

- prefill vs decode：M 维是 prompt 长度、算力受限 vs M 维是 batch、访存受限。
- MQA vs GQA vs MLA：所有头共享 1 组 KV vs 每组头共享 1 组 KV vs 缓存一个低秩 latent，用时再投影。
- 推测解码 vs Medusa：独立草稿模型 vs 在目标模型上加多个预测头。
- top-k vs top-p：保留固定 k 个 vs 保留累计概率达到 p 的最小集合。

## 学习目标

- 推导每 token FLOPs 与 KV cache 字节数，并能代入任意模型；
- 解释 prefill 与 decode 的瓶颈差异，算出临界 batch；
- 证明推测采样的输出分布等于目标分布；
- 推导期望接受长度与加速比，并找到最优 γ；
- 说清 MQA/GQA/MLA 各省了什么、付出了什么。

## 先修知识

- M01：roofline 与拐点；
- M07：GEMM 算术强度 2mnk / bytes(mk + kn + mn)；
- M08：attention 的计算与 FlashAttention。

## 核心概念

### 1. 两个阶段

prefill：输入 [B, T_prompt] 个 token，每个线性层是 [B·T_prompt, d] × [d, d'] 的 GEMM，强度随 T_prompt 增长，2K prompt 时远高于拐点。
decode：输入 [B, 1]，线性层是 [B, d] × [d, d']，每个权重读一次只做 2B 次 FLOP。
没有 KV cache 时，每步都要重算整个前缀，生成 T 个 token 的总计算是 O(T²) 次前向 token；有了 cache 每步只算一个新 token。

### 2. KV cache 的大小

每层每 token 存 K 和 V，各 n_kv·d_head 个元素。Llama-3-8B：2 × 32 × 8 × 128 × 2 B = 131072 B = 128 KB。
同样形状的 MHA（Llama-2-7B，32 个 KV 头）是 512 KB，GQA 省 4 倍。
Llama-3-70B：2 × 80 × 8 × 128 × 2 B = 320 KB，8K 上下文一条序列 2.68 GB。

### 3. 减少 KV 的结构

MQA：所有 query 头共享一组 K/V，KV 除以 n_heads，质量有损。
GQA：n_heads 个 query 头分成 n_kv 组，每组共享 K/V。KV 除以 n_heads/n_kv，质量接近 MHA。
MLA（DeepSeek-V2）：缓存一个 512 维的压缩 latent 加 64 维的 RoPE key，推理时把上投影矩阵吸收进 Q 和输出投影，缓存远小于 MHA。

### 4. 延迟指标

TTFT（time to first token）：排队 + prefill。TPOT（time per output token）：decode 步时间。
端到端延迟 ≈ TTFT + (n_out − 1) × TPOT。交互场景常见 SLO 是 TTFT 数百毫秒、TPOT 数十毫秒。

### 5. 推测解码

小模型 q 自回归起草 γ 个 token x1…xγ。大模型 p 对前缀加这 γ 个 token 做一次前向，得到 γ+1 个位置的分布。
逐个检查：以 min(1, p(xi)/q(xi)) 接受；第一次拒绝时从 norm(max(0, p − q)) 采样一个替换 token 并停止；全部接受则从第 γ+1 个位置的 p 再采一个 bonus token。
一次目标前向至少产出 1 个 token，所以最坏情况不会比普通 decode 少产出，只多付草稿成本。

### 6. 变体

Medusa：目标模型最后一层接多个头，分别预测 t+1、t+2…，候选组织成树，用 tree attention 一次验证。
EAGLE：在目标模型倒数第二层的特征空间上训练一个轻量自回归草稿头，接受率高于 token 级草稿。
Lookahead decoding：用 Jacobi 迭代生成 n-gram 候选，不需要草稿模型。
贪心解码下验证退化为「草稿 token 等于目标 argmax 就接受」。

### 7. 采样

temperature：softmax(z/τ)，τ→0 退化为贪心。top-k：只保留概率最高的 k 个再归一化。
top-p（nucleus）：按概率降序累加，保留累计 ≥ p 的最小集合。
推测解码时 p 和 q 必须是施加同样温度与截断之后的分布，否则无损性不成立。

## 关键推导

### 每 token FLOPs

每个参数在一次矩阵向量乘中参与一次乘加，2N。
attention 每层：q·Kᵀ 对 T 个 key 各做 d 次乘加，2·T·d；权重乘 V 同样 2·T·d；合计 4·L·d·T。
GQA 不改变这一项（query 头仍是 n_heads·d_head = d），只改变 KV 的字节。

### decode 的算术强度

权重部分：FLOPs 2N·B，字节 2N（bf16），强度 B。
attention 部分：每条序列 FLOPs 4·L·d·T，读 KV 4·L·n_kv·d_head·T 字节，强度 d/(n_kv·d_head) = n_heads/n_kv。Llama-3-8B 为 4，与 batch 无关。
所以加大 batch 能救权重部分，救不了 attention 部分；长上下文大 batch 时 KV 读取成为主瓶颈。这是 GQA/MLA 和 KV 量化存在的理由。

## 深入推导

### 推测采样的无损性

记目标分布 p、草稿分布 q。输出 token 为 x 的概率由两部分组成：
草稿给出 x 且被接受：q(x)·min(1, p(x)/q(x)) = min(q(x), p(x))。
草稿被拒（与草稿给出什么无关）的总概率：β = Σ_y q(y)·(1 − min(1, p(y)/q(y))) = Σ_y (q(y) − min(p(y), q(y))) = 1 − Σ_y min(p, q)。
因为 Σp = Σq = 1，Σ max(0, p − q) = Σ(p − min(p, q)) = 1 − Σ min(p, q) = β。
拒绝后以 max(0, p(x) − q(x)) / β 采样，贡献 β × max(0, p(x) − q(x))/β = max(0, p(x) − q(x))。
合计 min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x)。证毕。每个位置都成立，按位置归纳得到整段序列的分布与目标模型逐 token 采样相同。

### 期望长度与加速比

假设每个草稿 token 被接受的概率独立且都为 α。第 i 个 token 被接受的概率 α^i。
产出数 = 接受数 + 1（替换或 bonus）。E = 1 + α + α² + … + α^γ = (1 − α^{γ+1})/(1 − α)。
一轮时间 = γ 次草稿前向 + 1 次目标前向 = (γ·c + 1)·T_target。加速比 = E/(γ·c + 1)。
前提：目标模型验证 γ+1 个 token 的时间约等于生成 1 个 token。decode 访存受限时成立；大 batch 下 decode 已接近算力受限，验证 γ+1 个 token 的成本上升，加速比缩水。

```python
import torch

def speculative_step(p_logits, q_probs, draft, temp=1.0):
    """p_logits: [γ+1, V] 目标模型；q_probs: [γ, V] 草稿分布；draft: [γ] 草稿 token。"""
    p = torch.softmax(p_logits / temp, dim=-1)
    out = []
    for i, x in enumerate(draft.tolist()):
        if torch.rand(()) < torch.clamp(p[i, x] / q_probs[i, x], max=1.0):
            out.append(x)                                   # 接受
            continue
        resid = torch.clamp(p[i] - q_probs[i], min=0.0)     # norm(max(0, p − q))
        out.append(torch.multinomial(resid / resid.sum(), 1).item())
        return out                                          # 首次拒绝即停
    out.append(torch.multinomial(p[-1], 1).item())          # 全部接受：bonus token
    return out

def attend_with_cache(q, k_new, v_new, cache, layer):
    """q/k_new/v_new: [B, H, 1, d]；cache 只追加，不重算前缀。"""
    k = torch.cat([cache.k[layer], k_new], dim=2); v = torch.cat([cache.v[layer], v_new], dim=2)
    cache.k[layer], cache.v[layer] = k, v
    return torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1) @ v
```

torch.cat 每步重新分配整块 cache，是 O(T²) 的拷贝；真实实现预分配 [B, H, T_max, d] 或用分页块（M16）。

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 989 TFLOP/s、3.35 TB/s、80 GB；Llama-3-8B 8.03 B、KV 128 KB/token；Llama-3-70B 70.6 B、KV 320 KB/token。

```text
Llama-3-8B，batch=1：权重 16.06 GB；TPOT 下界 16.06 / 3.35e3 GB/s = 4.8 ms → 约 208 token/s
  prefill 2048 token：2 × 8.03e9 × 2048 = 3.29e13 FLOP → 满峰 33 ms，MFU 50% 时 66 ms
  临界 batch ≈ 989e12 / 3.35e12 = 295

Llama-3-70B bf16，权重 141.2 GB，8K 序列 KV = 8192 × 327680 B = 2.68 GB
  假设每卡 90% HBM 可用（72 GB），其余给激活与 workspace
  2×H100 TP：每卡权重 70.6 GB；KV 余量 2×72 − 141.2 = 2.8 GB → 只能放 1 条 8K 序列
  4×H100 TP：每卡 35.3 GB；KV 余量 288 − 141.2 = 146.8 GB → 54 条
  8×H100 TP：每卡 17.65 GB；KV 余量 576 − 141.2 = 434.8 GB → 162 条
  batch=1 decode 上界（只计权重、忽略通信）：
    2 卡 141.2 GB / 6.7 TB/s = 21.1 ms → 47 token/s
    4 卡 / 13.4 TB/s = 10.5 ms → 95 token/s
    8 卡 / 26.8 TB/s = 5.27 ms → 190 token/s
  8 卡，batch=64，每条 8K：读 141.2 + 64×2.68 = 313 GB → 11.7 ms/步
    聚合 64 / 11.7 ms ≈ 5480 token/s，每条 85 token/s
    计算 2×70.6e9×64 + 4×80×8192×8192×64 = 1.04e13 FLOP / 7.9e15 = 1.3 ms，确认访存受限

推测解码 α=0.8, γ=4, c=0.05：
  E = (1 − 0.8⁵)/0.2 = (1 − 0.32768)/0.2 = 3.36
  加速比 = 3.36 / (4×0.05 + 1) = 3.36 / 1.2 = 2.80
  γ=8：E = (1 − 0.8⁹)/0.2 = 4.33，/1.4 = 3.09；γ=10：4.57/1.5 = 3.05 → 最优 γ 在 8–9
  α=0.6, γ=4：E = 2.31，加速比 1.92
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| KV cache | [P2 实验](../solutions/cs229s/project2-kv-speculative/README.md) 的 nanoGPT 生成 | 开启 cache 后输出与不开逐 token 一致，长序列吞吐明显提升 |
| 推测采样 | 同上，草稿小模型 + 目标模型 | 大样本下输出 token 频率与目标模型直接采样一致 |
| 接受率与加速比 | 同上，统计 α 与每轮产出 | 实测每轮产出与 (1−α^{γ+1})/(1−α) 接近 |
| 公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `spec`、`decode`、`kv` | `spec --alpha 0.8 --gamma 4` 输出 3.362 与 2.801 |

## 易错点与反例

- 「decode 慢是因为 FLOPs 多」：错。8B 模型每 token 只有 16 GFLOP，H100 上 16 µs，实际要 4.8 ms，瓶颈是读 16 GB 权重。
- 「batch 大了 decode 就算力受限」：只对权重部分成立。KV 部分强度恒为 n_heads/n_kv，长上下文下仍访存受限。
- 「推测解码是近似」：采样版用 min(1, p/q) 与残差重采样是严格无损的；只有改了接受规则（如放宽阈值）才有损。
- 「拒绝后从 p 重采样」：错，要从 max(0, p − q) 归一化后采样，否则分布偏向草稿已经覆盖过的 token。
- 「γ 越大越好」：E 有上界 1/(1−α)，草稿成本线性增长，存在最优 γ。
- 「推测解码在高负载服务里也加速」：大 batch 下目标模型已接近算力受限，验证多个 token 不再免费，收益下降甚至为负。
- 「GQA 减少 FLOPs」：几乎不减少，减少的是 KV 显存与带宽。
- 「TTFT 只和 prompt 长度有关」：还包括排队时间，高负载下排队常常主导。

## 实验或练习

- 假设：KV cache 让每步时间与上下文长度近似无关（短上下文时），推测解码实测产出符合 α 公式。
- 配置：nanoGPT 124M 作目标、更小模型作草稿；生成 512 token，记录每步时间、α、每轮产出；γ ∈ {2, 4, 8}。
- 预期：不开 cache 时每步时间随 T 线性增长；开 cache 后基本平坦；γ 增大时每轮产出增加但趋于饱和。

## 高频问题

- 问：KV cache 公式？
  答：2·L·n_kv·d_head·T·B·bytes，2 是 K 和 V。
- 问：为什么 decode 访存受限？
  答：每读一个权重只做 2B 次 FLOP，batch 远小于拐点（H100 约 295）。
- 问：TTFT 和 TPOT 分别由什么决定？
  答：TTFT 由排队和 prefill 算力决定；TPOT 由每步需读的权重与 KV 字节除以带宽决定。
- 问：推测解码的接受概率？
  答：min(1, p(x)/q(x))；拒绝后从 norm(max(0, p − q)) 重采样。
- 问：α 和两个分布什么关系？
  答：α = Σ min(p, q) = 1 − 总变差距离。

## 面试追问链

1. 问：解释 prefill 和 decode 的区别。
   答：prefill 的 GEMM M 维是 prompt 长度，算力受限；decode 的 M 维是 batch，每步读全部权重和 KV，访存受限。
2. 问：推 decode 的算术强度和临界 batch。
   答：权重部分 2NB/2N = B，临界 batch = 拐点 ≈ 295（H100 bf16）；attention 部分强度 n_heads/n_kv，与 batch 无关。
3. 问：证明推测解码无损。
   答：P(x) = q·min(1, p/q) + β·max(0, p−q)/β = min(p, q) + max(0, p − q) = p。
4. 问：α=0.8、c=0.05，γ 取多少？
   答：γ=4 时 2.80 倍，γ=8 时 3.09 倍，γ=10 降到 3.05，取 8 左右。实际还要看验证 γ+1 个 token 是否仍免费。
5. 问：GQA、MLA、KV int8 量化，各自的代价是什么？
   答：GQA 改结构，需从训练开始或 uptrain；MLA 改结构，实现复杂，RoPE 需解耦；KV int8 无需重训，KV 减半，有轻微精度损失和反量化开销。
6. 问：70B 模型上线，4 卡还是 8 卡 H100？
   答：4 卡每卡权重 35 GB，KV 余量 147 GB，可放 54 条 8K 序列，单流上界 95 token/s；8 卡余量 435 GB、162 条、190 token/s。按 SLO 选：TPOT 要求严或并发高选 8 卡；成本优先选 4 卡。还要计入 TP 每层 2 次小 all-reduce 的延迟，8 卡时更明显。
7. 问：服务负载很高时还要不要开推测解码？
   答：低负载（batch 小、访存受限）开，收益接近公式；高负载 batch 已接近临界值时关掉或降低 γ，把算力留给更多请求。可按当前 batch 动态切换。

## 自测题

1. Llama-2-7B 一条 4K 序列 KV 多大？
   要点：512 KB × 4096 = 2 GiB（2.15 GB）。
2. 8B 模型、H100、上下文 4K、batch=32，decode 一步下界？
   要点：权重 16.06 GB + KV 32×4096×128 KB = 17.2 GB，共 33.3 GB / 3.35 TB/s ≈ 9.9 ms。
3. 证明 α = 1 − TV(p, q)。
   要点：TV = ½Σ|p − q| = Σ max(0, p − q) = 1 − Σ min(p, q)。
4. α=0.9、γ=4，期望产出？
   要点：(1 − 0.9⁵)/0.1 = (1 − 0.59)/0.1 = 4.10。
5. 设计题：为什么草稿模型要和目标模型共享 tokenizer？
   要点：接受检验按 token 比较 p(x) 与 q(x)，词表不同则 x 无法对齐。

## 关联笔记

- [M08 FlashAttention](module-08-attention-flashattention.md)：prefill 的 attention kernel
- [M16 LLM 推理服务](module-16-llm-serving.md)：KV 显存的分页管理与调度
- [M17 量化](module-17-compression-quantization.md)：weight-only 量化与 KV 量化减少每步字节
- [M01 性能模型](module-01-performance-models.md)：roofline 与拐点

## 本讲小结

推理的核心不等式是 decode 强度 ≈ batch 远小于拐点，所以 decode 由字节决定。
KV cache 用 2·L·n_kv·d_head·T·B·bytes 的显存换掉了 O(T²) 重算，GQA/MLA/量化在压这一项。
推测解码用多余的算力换步数，min(1, p/q) 加残差采样保证无损，加速比 (1−α^{γ+1})/((1−α)(γc+1))。
所有优化都要回答一个问题：这一步读了多少字节。

## 延伸阅读

- Pope et al., Efficiently Scaling Transformer Inference, 2022, https://arxiv.org/abs/2211.05102
- Ainslie et al., GQA, 2023, https://arxiv.org/abs/2305.13245
- DeepSeek-AI, DeepSeek-V2, 2024, https://arxiv.org/abs/2405.04434
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding, 2023, https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling, 2023, https://arxiv.org/abs/2302.01318
- Cai et al., Medusa, 2024, https://arxiv.org/abs/2401.10774
- Li et al., EAGLE, 2024, https://arxiv.org/abs/2401.15077
- Fu et al., Lookahead Decoding, 2024, https://arxiv.org/abs/2402.02057
