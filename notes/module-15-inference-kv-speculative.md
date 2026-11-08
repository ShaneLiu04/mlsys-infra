# Module 15 — Transformer 推理：KV cache 与推测解码

- 课程出处：CS229S Fall 2023《Analyzing the Performance of Transformers (10-09)》《Project 2：KV caching and speculative decoding in nanoGPT》
- 必读论文：Pope et al., Efficiently Scaling Transformer Inference, 2022；Shazeer, Fast Transformer Decoding (MQA), 2019；Ainslie et al., GQA, 2023；DeepSeek-AI, DeepSeek-V2 (MLA), 2024；Leviathan et al., Fast Inference from Transformers via Speculative Decoding, 2023；Chen et al., Accelerating LLM Decoding with Speculative Sampling, 2023；Cai et al., Medusa, 2024；Li et al., EAGLE, 2024
- 对应实验：[CS229S P2 KV cache 与推测解码](../solutions/cs229s/project2-kv-speculative/README.md)
- 实验报告：[报告 PDF](../reports/cs229s-project2/cs229s_project2_report.pdf)（实测：gpt2-xl decode 20.6 → 4.22 ms；推测解码贪心 1.63 倍）
- 状态：初稿

## 摘要

用过大模型的人都见过这个现象：一大段 prompt 很快就读完了，回答却是一个字一个字往外吐。这一讲解释为什么，以及怎么让它快起来。

自回归生成分两段。prefill 一次处理整个 prompt，矩阵乘的 M 维是 prompt 长度，计算量大，所以是算力受限。decode 每步只生成一个 token，M 维只有 batch 那么大，每读一遍权重只做很少的计算，所以是访存受限。

KV cache 把历史 token 的 K、V 存下来，使每步 decode 的计算从 O(T²) 降到 O(T)。代价是显存：2·L·n_kv·d_head·T·B·bytes。
decode 对权重的算术强度（每读一个字节做多少次浮点运算）约等于 batch size，而 H100 要 batch ≈ 300 才到拐点。所以推理优化有两条主线：把 batch 做大，把每步要读的字节变少。

推测解码走的是第三条路：用小模型先起草 γ 个 token，大模型一次前向验证。接受规则 min(1, p/q) 保证输出分布和大模型完全相同。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 每个 token 的前向 FLOPs ≈ 2N + 4·L·d·T。第二项是 attention 对 T 个历史 token 做的点积与加权和。
2. KV cache = 2·L·n_kv·d_head·T·B·bytes。Llama-3-8B 每 token 128 KB，70B 每 token 320 KB。
3. decode 时权重部分的强度 = 2NB / 2N = B FLOP/B，所以加 batch 有用；attention 部分的强度 = n_heads/n_kv，与 batch 无关。
4. TTFT 由 prefill 决定，看算力；TPOT 由 decode 决定，看带宽。batch=1 时 TPOT 的下界 = 权重字节 / HBM 带宽。
5. 推测解码每次目标前向期望产出 (1−α^{γ+1})/(1−α) 个 token，再除以 (γ·c + 1) 才是加速比。

必须能默写的公式：

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

一句话区分：

- prefill 与 decode：前者 M 维是 prompt 长度，算力受限；后者 M 维是 batch，访存受限。
- MQA、GQA 与 MLA：MQA 让所有头共享一组 KV；GQA 让每组头共享一组 KV；MLA 只缓存一个低秩 latent，用的时候再投影回去。
- 推测解码与 Medusa：前者用独立的草稿模型，后者在目标模型上加几个预测头。
- top-k 与 top-p：前者保留固定的 k 个候选，后者保留累计概率达到 p 的最小集合。

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

同一个模型、同一块 GPU，prefill 和 decode 的瓶颈却完全不同。原因在矩阵乘的形状上。

prefill 的输入是 [B, T_prompt] 个 token，每个线性层都是 [B·T_prompt, d] × [d, d'] 的 GEMM。它的强度随 T_prompt 增长，2K 的 prompt 时远高于拐点。
decode 的输入是 [B, 1]，线性层变成 [B, d] × [d, d']。每个权重读一次，只做 2B 次 FLOP。

KV cache 解决的是另一个浪费。没有它的话，每步都要把整个前缀重算一遍，生成 T 个 token 的总计算是 O(T²) 次前向 token；有了 cache，每步只算一个新 token。

### 2. KV cache 的大小

KV cache 省了计算，但占显存，而且随上下文长度和 batch 线性增长。算一下就知道它有多大。

每层每个 token 存 K 和 V，各 n_kv·d_head 个元素。Llama-3-8B 是 2 × 32 × 8 × 128 × 2 B = 131072 B = 128 KB。
同样形状的 MHA（比如 Llama-2-7B，有 32 个 KV 头）是 512 KB，所以 GQA 省了 4 倍。
Llama-3-70B 是 2 × 80 × 8 × 128 × 2 B = 320 KB，一条 8K 上下文的序列就要 2.68 GB。

### 3. 减少 KV 的结构

既然 KV 这么占地方，一个思路是从模型结构上让它变小。三种做法的区别是共享得多狠、付出了什么：

- MQA：所有 query 头共享一组 K/V，KV 除以 n_heads，但质量有损失。
- GQA：n_heads 个 query 头分成 n_kv 组，每组共享 K/V。KV 除以 n_heads/n_kv，质量接近 MHA。
- MLA（DeepSeek-V2）：缓存一个 512 维的压缩 latent，加一个 64 维的 RoPE key。推理时把上投影矩阵吸收进 Q 和输出投影，所以缓存远小于 MHA。

### 4. 延迟指标

用户感受到的「快」分两种：多久看到第一个字，以及之后的字吐得有多快。

TTFT（time to first token）等于排队时间加 prefill 时间。TPOT（time per output token）是每个 decode 步的时间。
端到端延迟 ≈ TTFT + (n_out − 1) × TPOT。交互场景常见的 SLO 是 TTFT 几百毫秒、TPOT 几十毫秒。

### 5. 推测解码

decode 访存受限，意味着大模型每步读了一遍全部权重，却只算了一个 token，算力大量闲置。推测解码就是想把这部分闲置的算力用起来：既然验证 γ+1 个 token 和生成 1 个 token 读的权重一样多，那就一次多验证几个。

具体流程是这样的。小模型 q 自回归地起草 γ 个 token x1…xγ，大模型 p 对「前缀加这 γ 个 token」做一次前向，得到 γ+1 个位置上的分布。
然后逐个检查：以 min(1, p(xi)/q(xi)) 的概率接受。第一次被拒绝时，从 norm(max(0, p − q)) 里采样一个替换 token，然后停止。如果全部接受，就从第 γ+1 个位置的 p 再采一个 bonus token。

一次目标前向至少产出 1 个 token，所以最坏情况下产出不会比普通 decode 少，只是多付了草稿的成本。

### 6. 变体

独立的草稿模型要另外训练、另外部署，于是有了几种不用它的做法：

- Medusa：在目标模型的最后一层接多个头，分别预测 t+1、t+2…。候选组织成树，用 tree attention 一次验证。
- EAGLE：在目标模型倒数第二层的特征空间上训练一个轻量的自回归草稿头，接受率比 token 级的草稿更高。
- Lookahead decoding：用 Jacobi 迭代生成 n-gram 候选，不需要草稿模型。

在贪心解码下，验证规则会退化成「草稿 token 等于目标模型的 argmax 就接受」。

### 7. 采样

最后一步是从分布里挑 token。几种常见的采样方式：

- temperature：用 softmax(z/τ) 调节分布的尖锐程度，τ→0 时退化为贪心。
- top-k：只保留概率最高的 k 个，再归一化。
- top-p（nucleus）：按概率从高到低累加，保留累计 ≥ p 的最小集合。

这和推测解码有一个交点：p 和 q 必须是施加了同样温度和截断之后的分布，否则无损性就不成立。

## 关键推导

### 每 token FLOPs

先数清楚 decode 一个 token 到底要算多少，后面才能判断它是被算力还是被带宽卡住。

每个参数在一次矩阵向量乘里参与一次乘加，贡献 2N。
attention 每层：q·Kᵀ 对 T 个 key 各做 d 次乘加，是 2·T·d；注意力权重乘 V 同样是 2·T·d。所有层合计 4·L·d·T。
GQA 不改变这一项，因为 query 头的总维度仍是 n_heads·d_head = d，它改变的只是 KV 的字节数。

### decode 的算术强度

有了 FLOPs，再除以读的字节，就得到强度。这里要把权重和 attention 分开看，因为它们对 batch 的反应不一样。

权重部分：FLOPs 是 2N·B，字节是 2N（bf16），强度就是 B。
attention 部分：每条序列的 FLOPs 是 4·L·d·T，要读的 KV 是 4·L·n_kv·d_head·T 字节，强度是 d/(n_kv·d_head) = n_heads/n_kv。Llama-3-8B 为 4，和 batch 无关。

所以加大 batch 能救权重部分，救不了 attention 部分。上下文一长、batch 一大，读 KV 就成了主瓶颈。GQA、MLA 和 KV 量化存在的理由就在这里。

## 深入推导

### 推测采样的无损性

推测解码听起来像是「让小模型替大模型做决定」，很容易被误认为是近似。这一节证明它是严格无损的。

记目标分布为 p，草稿分布为 q。最终输出 token 为 x 的概率由两部分组成。
第一部分是草稿给出 x 且被接受：q(x)·min(1, p(x)/q(x)) = min(q(x), p(x))。
第二部分来自拒绝。草稿被拒的总概率（与草稿给出什么无关）是 β = Σ_y q(y)·(1 − min(1, p(y)/q(y))) = Σ_y (q(y) − min(p(y), q(y))) = 1 − Σ_y min(p, q)。
因为 Σp = Σq = 1，所以 Σ max(0, p − q) = Σ(p − min(p, q)) = 1 − Σ min(p, q) = β。
拒绝后以 max(0, p(x) − q(x)) / β 采样，这部分的贡献是 β × max(0, p(x) − q(x))/β = max(0, p(x) − q(x))。

两部分合计 min(p(x), q(x)) + max(0, p(x) − q(x)) = p(x)，证毕。每个位置都成立，按位置归纳，整段序列的分布就和目标模型逐 token 采样相同。

### 期望长度与加速比

无损性说明它不会变差，这一节算它能快多少。

假设每个草稿 token 被接受的概率相互独立，且都是 α，那么第 i 个 token 被接受的概率是 α^i。
每轮产出的 token 数 = 接受数 + 1（替换 token 或 bonus token），所以 E = 1 + α + α² + … + α^γ = (1 − α^{γ+1})/(1 − α)。
一轮的时间是 γ 次草稿前向加 1 次目标前向，即 (γ·c + 1)·T_target。于是加速比 = E/(γ·c + 1)。

这里藏着一个前提：目标模型验证 γ+1 个 token 的时间约等于生成 1 个 token 的时间。decode 访存受限时这成立；但大 batch 下 decode 已经接近算力受限，验证 γ+1 个 token 的成本就会上升，加速比随之缩水。

下面的代码实现了上面的接受与重采样规则，也给了一个最朴素的 KV cache 追加写法：

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

这个 cache 写法只用来说明原理。torch.cat 每步都会重新分配整块 cache，累计是 O(T²) 的拷贝；真实实现会预分配 [B, H, T_max, d]，或者用分页块（M16）。

## 白板估算

这一节用三组估算把上面的公式落到具体模型上：8B 单卡的 decode 和 prefill、70B 要几张卡才放得下足够多的 KV、推测解码的 γ 该取多少。
硬件和模型数字取自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 989 TFLOP/s、3.35 TB/s、80 GB；Llama-3-8B 8.03 B、KV 128 KB/token；Llama-3-70B 70.6 B、KV 320 KB/token。

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

70B 那组里最值得看的是最后一行：8 卡、batch=64 时，计算只要 1.3 ms，读字节却要 11.7 ms。即使 batch 已经不小，decode 仍然是访存受限。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| KV cache | [P2 实验](../solutions/cs229s/project2-kv-speculative/README.md) 的 nanoGPT 生成 | 开启 cache 后输出与不开逐 token 一致，长序列吞吐明显提升 |
| 推测采样 | 同上，草稿小模型 + 目标模型 | 大样本下输出 token 频率与目标模型直接采样一致 |
| 接受率与加速比 | 同上，统计 α 与每轮产出 | 实测每轮产出与 (1−α^{γ+1})/(1−α) 接近 |
| 公式 | [mlsys_calc.py](../tools/mlsys_calc.py) `spec`、`decode`、`kv` | `spec --alpha 0.8 --gamma 4` 输出 3.362 与 2.801 |

## 易错点与反例

- 「decode 慢是因为 FLOPs 多」：错。8B 模型每 token 只有 16 GFLOP，在 H100 上只要 16 µs，实际却要 4.8 ms，瓶颈是读 16 GB 的权重。
- 「batch 大了 decode 就算力受限」：只对权重部分成立。KV 部分的强度恒为 n_heads/n_kv，长上下文下仍然访存受限。
- 「推测解码是近似」：采样版用 min(1, p/q) 加残差重采样，是严格无损的；只有改了接受规则（比如放宽阈值）才会有损。
- 「拒绝后从 p 重采样」：错，要从 max(0, p − q) 归一化后采样，否则分布会偏向草稿已经覆盖过的 token。
- 「γ 越大越好」：E 有上界 1/(1−α)，而草稿成本线性增长，所以存在一个最优 γ。
- 「推测解码在高负载服务里也能加速」：大 batch 下目标模型已经接近算力受限，验证多个 token 不再免费，收益会下降，甚至为负。
- 「GQA 减少 FLOPs」：几乎不减少，它减少的是 KV 的显存和带宽。
- 「TTFT 只和 prompt 长度有关」：还包括排队时间，高负载下排队常常是大头。

## 实验或练习

这个练习验证两个假设：在上下文较短时，KV cache 让每步时间和上下文长度基本无关；推测解码的实测产出符合 α 公式。

- 配置：nanoGPT 124M 作为目标模型，更小的模型作为草稿；生成 512 个 token，记录每步时间、α 和每轮产出；γ ∈ {2, 4, 8}。
- 预期：不开 cache 时每步时间随 T 线性增长，开 cache 后基本平坦；γ 增大时每轮产出增加，但逐渐饱和。

## 高频问题

- 问：KV cache 公式？
  答：2·L·n_kv·d_head·T·B·bytes，前面的 2 代表 K 和 V 各一份。
- 问：TTFT 和 TPOT 分别由什么决定？
  答：TTFT 由排队和 prefill 的算力决定；TPOT 由每步要读的权重与 KV 字节除以带宽决定。
- 问：贪心解码下推测解码怎么验证？
  答：草稿 token 等于目标模型在该位置的 argmax 就接受，否则用 argmax 替换并停止。
- 问：开了 temperature 或 top-p，推测解码还无损吗？
  答：无损，但前提是 p 和 q 都是施加了同样温度和截断之后的分布。
- 问：示例里用 torch.cat 追加 KV cache，有什么问题？
  答：每步都重新分配整块 cache，累计是 O(T²) 的拷贝。真实实现预分配 [B, H, T_max, d] 或用分页块。

## 面试追问链

1. 问：解释 prefill 和 decode 的区别。
   答：prefill 的 GEMM，M 维是 prompt 长度，所以算力受限；decode 的 M 维只是 batch，每步要读全部权重和 KV，所以访存受限。
2. 问：推一下 decode 的算术强度和临界 batch。
   答：权重部分是 2NB/2N = B，所以临界 batch 就是拐点，H100 bf16 约 295。attention 部分的强度是 n_heads/n_kv，和 batch 无关。
3. 问：推测解码的接受规则是什么？证明它无损。
   答：以 min(1, p/q) 接受，拒绝后从 norm(max(0, p − q)) 重采样。P(x) = q·min(1, p/q) + β·max(0, p−q)/β = min(p, q) + max(0, p − q) = p。
4. 问：α=0.8、c=0.05，γ 取多少？
   答：γ=4 时 2.80 倍，γ=8 时 3.09 倍，γ=10 反而降到 3.05，所以取 8 左右。实际还要看验证 γ+1 个 token 是否仍然免费。
5. 问：GQA、MLA、KV int8 量化，各自的代价是什么？
   答：GQA 改了结构，需要从训练开始用或者做 uptrain；MLA 也改结构，实现更复杂，RoPE 还要解耦；KV int8 不用重训，KV 减半，代价是轻微的精度损失和反量化开销。
6. 问：70B 模型上线，用 4 卡还是 8 卡 H100？
   答：4 卡时每卡权重 35 GB，KV 余量 147 GB，能放 54 条 8K 序列，单流上界 95 token/s；8 卡余量 435 GB、162 条、190 token/s。按 SLO 选：TPOT 要求严或并发高就选 8 卡，成本优先就选 4 卡。另外还要算上 TP 每层 2 次小 all-reduce 的延迟，8 卡时更明显。
7. 问：服务负载很高时，还要不要开推测解码？
   答：低负载时 batch 小、访存受限，开了收益接近公式；高负载时 batch 已经接近临界值，应该关掉或调低 γ，把算力留给更多请求。可以按当前 batch 动态切换。

## 自测题

1. Llama-2-7B 一条 4K 序列的 KV 有多大？
   要点：512 KB × 4096 = 2 GiB（2.15 GB）。
2. 8B 模型、H100、上下文 4K、batch=32，decode 一步的下界是多少？
   要点：权重 16.06 GB + KV 32×4096×128 KB = 17.2 GB，共 33.3 GB / 3.35 TB/s ≈ 9.9 ms。
3. 证明 α = 1 − TV(p, q)。
   要点：TV = ½Σ|p − q| = Σ max(0, p − q) = 1 − Σ min(p, q)。
4. α=0.9、γ=4，期望产出是多少？
   要点：(1 − 0.9⁵)/0.1 = (1 − 0.59)/0.1 = 4.10。
5. 设计题：为什么草稿模型要和目标模型共享 tokenizer？
   要点：接受检验是按 token 比较 p(x) 和 q(x) 的，词表不同，x 就对不上。

## 关联笔记

- [M08 FlashAttention](module-08-attention-flashattention.md)：prefill 的 attention kernel
- [M16 LLM 推理服务](module-16-llm-serving.md)：KV 显存的分页管理与调度
- [M17 量化](module-17-compression-quantization.md)：weight-only 量化与 KV 量化减少每步字节
- [M01 性能模型](module-01-performance-models.md)：roofline 与拐点
- [edge-deploy M08 块量化解码](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-08-block-quantized-decode.md)：端侧 batch 1 解码：Q4 × Q8 GEMV 与 tok/s ≤ 带宽 / 权重字节

## 本讲小结

推理的核心事实是：decode 的强度约等于 batch，远小于拐点，所以 decode 的速度由要读的字节决定。
KV cache 用 2·L·n_kv·d_head·T·B·bytes 的显存换掉了 O(T²) 的重算，GQA、MLA 和 KV 量化都在压这一项。
推测解码用闲置的算力换步数，min(1, p/q) 加残差采样保证无损，加速比是 (1−α^{γ+1})/((1−α)(γc+1))。
分析任何推理优化时，都先问一句：这一步读了多少字节。

## 延伸阅读

- Pope et al., Efficiently Scaling Transformer Inference, 2022, https://arxiv.org/abs/2211.05102
- Ainslie et al., GQA, 2023, https://arxiv.org/abs/2305.13245
- DeepSeek-AI, DeepSeek-V2, 2024, https://arxiv.org/abs/2405.04434
- Leviathan et al., Fast Inference from Transformers via Speculative Decoding, 2023, https://arxiv.org/abs/2211.17192
- Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling, 2023, https://arxiv.org/abs/2302.01318
- Cai et al., Medusa, 2024, https://arxiv.org/abs/2401.10774
- Li et al., EAGLE, 2024, https://arxiv.org/abs/2401.15077
- Fu et al., Lookahead Decoding, 2024, https://arxiv.org/abs/2402.02057
