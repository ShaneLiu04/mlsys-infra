# 公式卡

面试官说「写一下」或者「估一下」的时候，翻这里。每张卡依次是：什么时候会用到它、公式、推导、口算检查、常见错答。
硬件数字统一取自 [HARDWARE-NUMBERS.md](HARDWARE-NUMBERS.md)。每条都能用 `python tools/mlsys_calc.py` 复算，
对应的测试在 `tools/tests/test_mlsys_calc.py`。

## 目录

| # | 公式 | 模块 |
| ---: | --- | --- |
| F1 | Amdahl 与 Gustafson | [M01](../notes/module-01-performance-models.md) |
| F2 | Roofline 与拐点 | [M01](../notes/module-01-performance-models.md) |
| F3 | GEMM 算术强度 | [M07](../notes/module-07-matmul-hardware-aware.md) |
| F4 | 分块后的访存量 | [M07](../notes/module-07-matmul-hardware-aware.md) |
| F5 | Work–span 与贪心调度上界 | [M03](../notes/module-03-parallel-programming-scheduling.md) |
| F6 | Scan 的 work 和 span | [M06](../notes/module-06-data-parallel-primitives.md) |
| F7 | Attention 的 FLOPs 和显存 | [M08](../notes/module-08-attention-flashattention.md) |
| F8 | Online softmax 合并 | [M08](../notes/module-08-attention-flashattention.md) |
| F9 | 参数量与训练 FLOPs（6ND） | [M11](../notes/module-11-frameworks-autodiff.md) |
| F10 | 训练静态显存 16N 与 ZeRO | [M12](../notes/module-12-data-parallel-zero.md) |
| F11 | Ring all-reduce 的 α-β 模型 | [M12](../notes/module-12-data-parallel-zero.md) |
| F12 | 张量并行的通信量 | [M13](../notes/module-13-tensor-pipeline-parallel.md) |
| F13 | 流水线气泡 | [M13](../notes/module-13-tensor-pipeline-parallel.md) |
| F14 | 重计算的 √n 方案 | [M14](../notes/module-14-memory-optimization.md) |
| F15 | KV cache 大小 | [M15](../notes/module-15-inference-kv-speculative.md) |
| F16 | Decode 的 roofline 与临界 batch | [M15](../notes/module-15-inference-kv-speculative.md) |
| F17 | 推测解码的期望长度与加速比 | [M15](../notes/module-15-inference-kv-speculative.md) |
| F18 | Little 定律与服务容量 | [M16](../notes/module-16-llm-serving.md) |
| F19 | 线性量化 | [M17](../notes/module-17-compression-quantization.md) |
| F20 | LoRA 参数量 | [M17](../notes/module-17-compression-quantization.md) |
| F21 | MoE 的 all-to-all 通信量 | [M18](../notes/module-18-architectures-moe-clusters.md) |
| F22 | MFU | [M07](../notes/module-07-matmul-hardware-aware.md) |

---

## F1. Amdahl 与 Gustafson

什么时候用到它：有人说「卡数翻倍，速度就翻倍」，或者问你加并行度到底能快多少的时候。

```text
Amdahl（问题规模固定）:   S(n) = 1 / ((1−p) + p/n)   ，n→∞ 时 S → 1/(1−p)
Gustafson（规模随 n 增长）: S(n) = (1−p) + p·n
```

- 推导：串行部分的时间 `1−p` 不随 n 变化，并行部分 `p` 变成 `p/n`，两种总时间之比就是加速比。Gustafson 换了个前提：并行部分的工作量随 n 放大，于是单核要花的时间变成 `(1−p) + p·n`。
- 口算：p=0.95 时，核再多也只有 20 倍；8 核时是 `1/(0.05+0.11875) = 5.9` 倍。
- 常见错答：「加 GPU 就线性加速」。在大规模下，数据加载、优化器步进、日志这些串行部分决定了上限。

## F2. Roofline 与拐点

什么时候用到它：判断一个 kernel 慢是因为算不过来还是搬不过来。几乎所有性能题的第一步都是它。

```text
t ≥ max( F / P_peak ,  B / BW )
算术强度 I = F / B（FLOP/byte）
拐点 I* = P_peak / BW     I < I* 受访存限制，I > I* 受算力限制
可达性能 = min(P_peak, I · BW)
```

- 推导：计算和访存可以重叠，所以总时间由两者中较慢的那个决定。令两者相等，就得到拐点。
- 口算：H100 是 989e12 / 3.35e12 ≈ 295 FLOP/B，A100 是 312/2.04 ≈ 153。逐元素加法 `y=a+b`（bf16）每个元素做 1 FLOP、搬 6 字节，I≈0.17，离拐点差了三个数量级。
- 常见错答：「GPU 利用率 100% 就是算力打满了」。`nvidia-smi` 的 utilization 只说明有 kernel 在跑，访存受限的 kernel 也能显示 100%。

## F3. GEMM 算术强度

什么时候用到它：解释为什么大矩阵乘能跑满 tensor core，而 batch=1 的 decode 只能跑出峰值的零头。

```text
C[M,N] = A[M,K] · B[K,N]
F = 2MNK
B = s·(MK + KN + MN)          s = 每元素字节数
方阵 n: I = 2n³ / (3s·n²) = 2n / (3s)      bf16 时 I = n/3
GEMV (M=1): I ≈ 2NK / (s·NK) = 2/s = 1 FLOP/B（bf16）
```

- 推导：每个输出元素做 K 次乘加，也就是 2K FLOP。理想情况下，三个矩阵各从 HBM 过一次。
- 口算：bf16 方阵在 H100 上要满足 n/3 ≥ 295，也就是 n ≳ 900，才受算力限制。batch=1 的 decode 本质上是 GEMV，强度只有 1，所以只能用到峰值的 1/295。
- 常见错答：把一次乘加（MAC）当成 1 FLOP。约定是一次乘加记 2 FLOP，厂商的峰值也是这么算的。

## F4. 分块后的访存量

什么时候用到它：被问「分块为什么能让矩阵乘变快」或者「块大小怎么选」的时候。

```text
快速存储容量 M（元素），方块 b×b，三块同时驻留：3b² ≤ M → b ≈ √(M/3)
每加载 2 个 b×b 块做 2b³ FLOP → I ≈ b / s
总访存 ≈ 2n³ / b + n²  =  Θ(n³ / √M)     （Hong–Kung 下界同阶）
```

- 推导：分块乘法一共做 `(n/b)³` 次块乘，每次读 `2b²` 个元素；C 块一直留在快存里累加，所以只算一次。
- 口算：H100 一个 SM 有 228 KB 共享内存。bf16 下要求 `3b² × 2 ≤ 228 KB`，得到 b ≤ 197，实际取 128。这时 I = b/s = 64 FLOP/B，单看一个 block 内部的复用还够不到拐点 295。真实的 kernel 之所以能越过拐点，是因为多个 block 在 L2 里共享 A、B 的同一块，再加上寄存器这一层的再分块。
- 常见错答：「块越大越好」。块大到共享内存放不下，或者 occupancy 掉到每个 SM 只有 1 个 block 时，就没有别的 block 来隐藏延迟了。

## F5. Work–span 与贪心调度上界

什么时候用到它：判断一个并行算法在 P 个核上能不能接近线性加速，或者解释任务切得太粗、太细各有什么问题。

```text
T1 = 总工作量，T∞ = 关键路径长度（span），并行度 = T1 / T∞
贪心调度（含 work stealing 的期望）:  T_P ≤ T1/P + T∞
线性加速条件：P ≪ T1 / T∞
```

- 推导：把执行过程按步分成两类。一类是 P 个处理器都在忙，这样的步最多有 T1/P 步。另一类是有处理器空闲，这时关键路径上的就绪任务一定在被执行，所以这样的步最多 T∞ 步。
- 口算：对 n=10⁶ 个数做归约，T1 ≈ 10⁶，T∞ ≈ log₂10⁶ = 20，并行度是 5×10⁴。8 核远低于这个数，可以线性加速。
- 常见错答：只看 T1/P。递归切分太粗时 T∞ 会变大；切得太细时，调度开销会进到 T1 里。

## F6. Scan 的 work 和 span

什么时候用到它：被问 GPU 上的 prefix sum 该用哪种算法，或者「步数少的并行算法是不是一定更快」。

```text
顺序:            work n−1,        span n−1
Hillis–Steele:   work n·log₂n,    span log₂n
Blelloch:        work 2(n−1),     span 2·log₂n   （up-sweep + down-sweep）
```

- 推导：Hillis–Steele 每一层让所有元素都加上偏移 `2^d` 处的值。Blelloch 的 up-sweep 是一棵做 n−1 次加法的归约树，down-sweep 和它对称。
- 口算：n=2²⁰ 时，Hillis–Steele 做 2000 万次加法，Blelloch 约 200 万次。所以 GPU 上的大数组用 work-efficient 的 Blelloch；warp 内只有 32 个元素，反而用 Hillis–Steele，因为步数少，多出来的线程本来也闲着。
- 常见错答：「log n 步的算法一定更快」。它的工作量多了 log n 倍，在带宽受限时反而更慢。

## F7. Attention 的 FLOPs 和显存

什么时候用到它：估算长上下文下 attention 占多少算力和显存，或者解释 FlashAttention 到底省了什么。

```text
每层每个头，前向：  QKᵀ 2T²d  +  PV 2T²d  =  4T²d
整层（h 头，d_model = h·d）: 4T²·d_model
朴素实现的分数矩阵: B·h·T²·s 字节（每层，反传要存）
FlashAttention 额外显存: O(B·h·T)（只存 logsumexp）
```

- 推导：`S = QKᵀ` 是 `[T,d]×[d,T]` 的矩阵乘，`O = PV` 是 `[T,T]×[T,d]`，各 2T²d。
- 口算：T=8192、d_model=4096 时，attention 是 4·8192²·4096 ≈ 1.1 TFLOP/层。同一层的投影和 MLP 约 `24·T·d_model²` ≈ 3.3 TFLOP，所以 attention 已经占到 1/4。分数矩阵按 bf16、32 头算，是 32·8192²·2 ≈ 4.3 GB/层/样本。
- 常见错答：「FlashAttention 减少了 FLOPs」。它没有减少 FLOPs，反传时还要重算一遍；它减少的是 HBM 读写。

## F8. Online softmax 合并

什么时候用到它：手写 FlashAttention，或者被问「softmax 需要整行的最大值，怎么能分块算」的时候。

```text
两段的统计量 (m₁, l₁, o₁)、(m₂, l₂, o₂)，其中 l = Σ e^{s−m}，o = Σ e^{s−m}·v
m = max(m₁, m₂)
l = e^{m₁−m}·l₁ + e^{m₂−m}·l₂
o = e^{m₁−m}·o₁ + e^{m₂−m}·o₂
最终输出 O = o / l，反传保存 L = m + log l
```

- 推导：softmax 的分母可以按块累加。只要每来一个新块，就把旧块的指数项乘上 `e^{m_old − m_new}`，所有项就统一到了新的最大值下。
- 口算：s=[1,3] 分成两块。块一 m=1、l=1；块二 m=3、l=1。合并后 m=3，l = e^{−2}+1 ≈ 1.135，和一次性算 `e^{1−3}+e^0` 的结果相同。
- 常见错答：不减最大值，直接累加 `e^s`。在 bf16/fp16 下，`e^{12}` 就已经超过了 fp16 的上限 65504。

## F9. 参数量与训练 FLOPs（6ND）

什么时候用到它：估算训练一个模型要多少卡、多少天。设计题里的总算力几乎都从这里开始。

```text
每层（MHA，非门控 MLP，d_ff = 4d）: 4d² + 8d² = 12d²
每层（GQA + SwiGLU）: 2d² + 2d·d_kv + 3d·d_ff
N ≈ 12·L·d²（+ 词表 V·d）
训练 C ≈ 6·N·D：前向 2ND，反向 4ND（对激活和对权重各一次 GEMM）
attention 每 token 额外前向 4·L·T·d_model，与 2N ≈ 24·L·d² 之比为 T/(6d)，T ≪ 6d 时可忽略
```

- 推导：每个参数在前向里对每个 token 参与一次乘加，即 2 FLOP/token。反向要分别算对输入的梯度和对权重的梯度，各是一次同样大小的 GEMM，所以是前向的两倍。
- 口算：Llama-3-8B 训练 15T token，总算力是 6 × 8×10⁹ × 1.5×10¹³ = 7.2×10²³ FLOP。用 1 万张 H100、MFU 40%，需要 7.2e23 / (1e4 × 989e12 × 0.4) ≈ 1.8×10⁵ s，约 2 天。
- 常见错答：「推理每个 token 也是 6N」。推理只有前向，是 2N。

## F10. 训练静态显存 16N 与 ZeRO

什么时候用到它：判断一个模型在几张卡上放不放得下，以及该用 ZeRO 的哪个 stage。

```text
bf16 权重 2N + bf16 梯度 2N + fp32 主权重 4N + Adam m 4N + v 4N = 16N
ZeRO-1: 4N + 12N/d       ZeRO-2: 2N + 14N/d       ZeRO-3: 16N/d
通信量（每步，每卡）: DP / ZeRO-1 / ZeRO-2 ≈ 2N·s，ZeRO-3 ≈ 3N·s（多一次前向 all-gather）
```

- 推导：ZeRO-1 把优化器状态切到 d 张卡上。ZeRO-2 再切梯度，用 reduce-scatter 代替 all-reduce。ZeRO-3 连权重也切，所以前向和反向各要 all-gather 一次权重。
- 口算：7B 模型、8 卡。不切是 112 GB；ZeRO-1 是 28 + 10.5 = 38.5 GB；ZeRO-3 是 14 GB，再加上激活。
- 常见错答：「ZeRO-3 的通信是 DP 的 3 倍」。实际是 1.5 倍：DP 的 all-reduce 相当于 RS 加 AG 两次传输，ZeRO-3 是前向 AG、反向 AG、再加 RS，一共三次。

## F11. Ring all-reduce 的 α-β 模型

什么时候用到它：估算梯度同步要多久，判断通信能不能藏进反向里，或者解释为什么跨节点要分桶 overlap。

```text
reduce-scatter: p−1 步，每步发 S/p
all-gather:     p−1 步，每步发 S/p
all-reduce:     t = 2(p−1)·α + 2·(p−1)/p · S/β
```

- 推导：把数据切成 p 份，环上每一步每张卡收发一份。带宽项 `2(p−1)/p · S` 随 p 增大趋近 2S，几乎和卡数无关；延迟项则随 p 线性增长。
- 口算：7B 模型的 bf16 梯度是 14 GB。8×H100 走 NVLink，单向 450 GB/s，耗时 `2·7/8·14e9/450e9 ≈ 54 ms`。如果跨节点走 50 GB/s 的 IB，就要约 0.5 s，所以必须做 overlap 和分桶。
- 常见错答：「环越大越慢」。带宽项不变，变慢的只是延迟项。所以小消息用 tree，大消息用 ring。

## F12. 张量并行的通信量

什么时候用到它：解释张量并行为什么只能放在节点内，或者估算 TP 每一步要传多少数据。

```text
Megatron MLP：W1 按列切（无通信），W2 按行切 → 前向 1 次 all-reduce
Attention：按头切，W_O 按行切 → 前向 1 次 all-reduce
每层前向 2 次、反向 2 次 all-reduce，每次大小 B·T·d_model·s
序列并行：all-reduce → reduce-scatter + all-gather，通信量相同，激活按 t 切
```

- 推导：MLP 是 `Y = GeLU(X W1) W2`。W1 按列切以后，每张卡拿到 GeLU 输出的一部分列，GeLU 是逐元素的，不需要同步。W2 按行切以后，每张卡算出的是部分和，必须求和一次。
- 口算：B·T = 8192 个 token、d=8192、bf16 时，每次 all-reduce 是 134 MB。每层 4 次、80 层，每步每卡约 43 GB。走 NVLink 是 0.1 s 量级，跨 IB 就是 1 s 量级，所以 TP 留在节点内。
- 常见错答：「两层都按列切更省通信」。第二层也按列切的话，得先把第一层的输出 all-gather 起来，通信反而更多。

## F13. 流水线气泡

什么时候用到它：决定流水线要切几级、一个 batch 要拆成多少个 micro-batch，或者比较 GPipe 和 1F1B。

```text
GPipe / 1F1B:   bubble = (p−1) / (m + p − 1)
交错（每卡 v 个虚拟 stage）: bubble = (p−1)/v / (m + (p−1)/v)
在途激活：GPipe m 个 micro-batch，1F1B 最多 p 个
```

- 推导：流水线填满和排空各需要 p−1 个时隙，m 个 micro-batch 占 m 个时隙。1F1B 并不减少气泡，它的好处是一个 micro-batch 的反向一做完，就能释放它的激活。
- 口算：p=8、m=32 时气泡是 7/39 = 18%；m=8 时是 47%。所以 m 至少要取 4p。
- 常见错答：「1F1B 的气泡比 GPipe 小」。两者气泡相同，1F1B 省的是激活显存。

## F14. 重计算的 √n 方案

什么时候用到它：激活显存放不下，需要权衡「多算一点」和「少存一点」的时候。

```text
n 层，每 k 层存一个检查点：显存 ≈ n/k + k，k = √n 时最小 ≈ 2√n
额外计算：约一次前向（总 FLOPs 从 3F 变成 4F，+33%）
```

- 推导：前向时只保留检查点处的激活。反向走到某一段时，从这段的检查点出发把 k 层重算一遍，临时存下来用完就丢。
- 口算：n=100 层时，激活从存 100 份降到约 20 份，代价是多 33% 的计算。选择性重计算只重算 attention 里的 softmax 和 dropout，开销小于 5%。
- 常见错答：「重计算让训练慢一倍」。它只多了一次前向，而反向本来就是前向的 2 倍。

## F15. KV cache 大小

什么时候用到它：估算一张卡能同时服务多少条请求、多长的上下文。推理设计题的容量都从这里算。

```text
KV 字节 = 2 · L · n_kv · d_head · T · B · s
每 token：2 · L · n_kv · d_head · s
```

- 推导：每一层对每个 token 存一个 K 向量和一个 V 向量，每个向量长 `n_kv · d_head`。
- 口算：Llama-3-8B 是 2·32·8·128·2 = 128 KB/token。8K 上下文、batch 32 时是 128 KB × 8192 × 32 = 32 GiB（34 GB）。Llama-2-7B 没有 GQA，同样条件下是 128 GiB。
- 常见错答：用 `n_heads` 代替 `n_kv_heads`。在 GQA 模型上会算大 4–8 倍。

## F16. Decode 的 roofline 与临界 batch

什么时候用到它：估算 decode 的速度上限，或者回答「batch 开多大才划算」。

```text
单步 decode 时间 ≈ max( 2N·B / P_peak ,  (s·N + KV字节) / BW )
只看权重: I ≈ 2N·B / (s·N) = B（bf16）
临界 batch ≈ I* = P_peak / BW（bf16）
上限: tokens/s ≤ B · BW / (s·N + KV)
```

- 推导：每一步都要把全部权重从 HBM 读一遍，读一次可以给 B 个 token 用。所以权重的读取被 B 个请求分摊了，但每条请求自己的 KV 不能分摊。
- 口算：H100、8B 模型、B=1 时，16 GB / 3.35 TB/s ≈ 4.8 ms，约 200 token/s。B 要到约 300 才开始受算力限制；如果上下文很长，KV 读取会先成为瓶颈。
- 常见错答：「decode 慢是因为计算量大」。B=1 时计算只用到峰值的 0.3%。

## F17. 推测解码的期望长度与加速比

什么时候用到它：判断推测解码在某个场景下值不值得上，或者挑 draft 模型和猜测长度 γ。

```text
每次目标模型前向接受的期望 token 数: E = (1 − α^{γ+1}) / (1 − α)
加速比（草稿一次前向成本是目标的 c 倍）: S = E / (γ·c + 1)
接受规则: 以概率 min(1, p(x)/q(x)) 接受；拒绝时从 norm(max(0, p − q)) 重采样
```

- 推导：假设各位置独立，第 i 个草稿 token 被接受的概率是 α^i。被接受的前缀长度，再加上最后由目标模型给出的那一个 token，期望就是 `Σ_{i=0}^{γ} α^i`。接受和重采样这两步合起来，恰好还原出分布 p。
- 口算：α=0.8、γ=4 时，E = (1−0.8⁵)/0.2 = 3.36；c=0.05 时，S = 3.36/1.2 = 2.8。
- 常见错答：「推测解码会改变输出分布」。按上面的规则采样，输出分布和目标模型完全相同。只有贪心版本才能说「输出逐字一样」。

## F18. Little 定律与服务容量

什么时候用到它：把「要扛多少 req/s」换算成「要支持多少并发」，再对照 KV 预算看机器够不够。

```text
并发数 L = 到达率 λ × 平均停留时间 W
单请求时延 ≈ TTFT + 输出长度 × TPOT
容量：λ_max ≈ KV 预算下的最大并发 / W
```

- 推导：系统处于稳态时，里面的平均请求数，等于单位时间进来的请求数乘以每个请求待的时间。
- 口算：平均输出 500 token、TPOT 30 ms、TTFT 0.5 s，所以 W ≈ 15.5 s。如果 KV 预算支持 64 个并发，λ_max ≈ 4.1 req/s。
- 常见错答：只看吞吐，不看 SLO。goodput 只统计满足 TTFT 和 TPOT 约束的那些请求。

## F19. 线性量化

什么时候用到它：解释量化误差从哪来、为什么要分组，或者为什么激活比权重难量化。

```text
对称: scale = max|x| / (2^{b−1} − 1)，  q = round(x / scale)，  x̂ = q·scale
非对称: scale = (max − min)/(2^b − 1)，zero_point = round(−min/scale)
误差上界: |x − x̂| ≤ scale / 2
粒度: per-tensor < per-channel < per-group（g=64/128）
```

- 推导：把 `[−max|x|, max|x|]` 均匀映射到整数网格上，四舍五入的误差最多是半个格子。
- 口算：一行权重里有一个 100 的离群值，其余都在 ±1 以内。int8 对称量化的 scale = 100/127 ≈ 0.79，于是绝大多数权重被量化成 0 或 ±1。per-group 的作用是把这个离群值关在它自己的 group 里。
- 常见错答：「int8 激活和 int8 权重一样好做」。LLM 的激活在固定的通道上有数十到上百倍的离群值，per-tensor 量化激活会把其他通道压没。LLM.int8 和 SmoothQuant 就是为了解决这个问题。

## F20. LoRA 参数量

什么时候用到它：估算微调要多少显存，或者设计多 LoRA 服务时算每个 adapter 有多大。

```text
W' = W + (α/r)·B·A，  A: r×k，B: d×r，B 初始化为 0
可训练参数: r·(d + k)，相对全量 r(d+k)/(dk)
合并推理: W ← W + (α/r)·BA，无额外延迟
```

- 口算：d=k=4096、r=16 时，LoRA 是 131K 个参数，全量是 16.8M，占 0.78%。7B 模型只在 q、v 上加 r=16 的 LoRA，约 8M 参数，Adam 状态从 84 GB 降到 0.1 GB。
- 常见错答：「LoRA 省了前向显存」。基座权重和激活都还在，省下的是梯度和优化器状态。

## F21. MoE 的 all-to-all 通信量

什么时候用到它：估算专家并行的通信开销，或者回答「MoE 省了算力，那它贵在哪」。

```text
每层两次 all-to-all（dispatch + combine）
每卡发送 ≈ B·T·k·d·s · (E_p − 1)/E_p      （k = top-k，E_p = 专家并行度）
参数量随专家数 E 线性增长，每 token FLOPs 只随 k 增长
```

- 口算：每卡 8192 个 token、d=4096、top-2、bf16、EP=8 时，每次是 8192·2·4096·2·7/8 ≈ 117 MB。每层前向 2 次、反向 2 次。
- 常见错答：「MoE 省算力，所以训练便宜」。FLOPs 确实省了，但全部专家的显存和 all-to-all 通信一点没省。

## F22. MFU

什么时候用到它：汇报训练效率，或者判断一个训练任务离硬件上限还有多远。

```text
MFU = 实测 tokens/s × 6N / (卡数 × P_peak)
HFU 把重计算的 FLOPs 也算进去，所以 HFU ≥ MFU
```

- 口算：8×H100 训练 7B 模型，每秒处理 1.0×10⁵ 个 token，MFU 是 6·7e9·1e5 / (8·989e12) = 53%。
- 常见错答：拿 HFU 当 MFU 报。开了全量重计算时，HFU 会比 MFU 高出约 1/3。
