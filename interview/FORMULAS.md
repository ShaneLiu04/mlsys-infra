# 公式卡

对方说「写一下」或「估一下」时用。每条四段：公式、推导、口算检查、常见错答。
硬件数字统一取自 [HARDWARE-NUMBERS.md](HARDWARE-NUMBERS.md)，每条都能用
`python tools/mlsys_calc.py` 复算，测试在 `tools/tests/test_mlsys_calc.py`。

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

```text
Amdahl（问题规模固定）:   S(n) = 1 / ((1−p) + p/n)   ，n→∞ 时 S → 1/(1−p)
Gustafson（规模随 n 增长）: S(n) = (1−p) + p·n
```

- **推导**：串行部分时间 `1−p` 不随 n 变，并行部分 `p` 变成 `p/n`，总时间之比即加速比。Gustafson 把并行部分的工作量放大 n 倍，单核时间变成 `(1−p) + p·n`。
- **口算**：p=0.95，无穷多核也只有 20 倍；8 核是 `1/(0.05+0.11875) = 5.9` 倍。
- **常见错答**：「加 GPU 就线性加速」。数据加载、优化器步进、日志这类串行部分在大规模下决定上限。

## F2. Roofline 与拐点

```text
t ≥ max( F / P_peak ,  B / BW )
算术强度 I = F / B（FLOP/byte）
拐点 I* = P_peak / BW     I < I* 受访存限制，I > I* 受算力限制
可达性能 = min(P_peak, I · BW)
```

- **推导**：计算和访存可以重叠时，时间由较慢的那个决定。令两者相等得到拐点。
- **口算**：H100 989e12 / 3.35e12 ≈ 295 FLOP/B；A100 312/2.04 ≈ 153。逐元素加法 `y=a+b`（bf16）每个元素 1 FLOP、6 字节，I≈0.17，离拐点差三个数量级。
- **常见错答**：「GPU 利用率 100% 就是算力打满」。`nvidia-smi` 的 utilization 只表示有 kernel 在跑，访存受限的 kernel 也能显示 100%。

## F3. GEMM 算术强度

```text
C[M,N] = A[M,K] · B[K,N]
F = 2MNK
B = s·(MK + KN + MN)          s = 每元素字节数
方阵 n: I = 2n³ / (3s·n²) = 2n / (3s)      bf16 时 I = n/3
GEMV (M=1): I ≈ 2NK / (s·NK) = 2/s = 1 FLOP/B（bf16）
```

- **推导**：每个输出元素做 K 次乘加（2K FLOP）。理想情况下三个矩阵各过一次 HBM。
- **口算**：bf16 方阵在 H100 上要 n/3 ≥ 295，即 n ≳ 900 才能受算力限制。batch=1 的 decode 就是 GEMV，强度 1，所以只能用到峰值的 1/295。
- **常见错答**：把 MAC 当成 1 FLOP。约定是一次乘加记 2 FLOP，厂商峰值也按这个算。

## F4. 分块后的访存量

```text
快速存储容量 M（元素），方块 b×b，三块同时驻留：3b² ≤ M → b ≈ √(M/3)
每加载 2 个 b×b 块做 2b³ FLOP → I ≈ b / s
总访存 ≈ 2n³ / b + n²  =  Θ(n³ / √M)     （Hong–Kung 下界同阶）
```

- **推导**：分块乘法共 `(n/b)³` 次块乘，每次读 `2b²` 元素；C 块留在快存里累加。
- **口算**：H100 一个 SM 228 KB 共享内存，bf16 下 `3b² × 2 ≤ 228 KB` → b ≤ 197，实际取 128。此时 I = b/s = 64 FLOP/B，只算一个 block 的复用还不到拐点 295；真实 kernel 靠多个 block 在 L2 里共享 A、B 的同一块，加上寄存器层的再分块，才越过拐点。
- **常见错答**：「块越大越好」。块大到 SMEM 放不下或 occupancy 掉到 1 个 block/SM 时，延迟隐藏消失。

## F5. Work–span 与贪心调度上界

```text
T1 = 总工作量，T∞ = 关键路径长度（span），并行度 = T1 / T∞
贪心调度（含 work stealing 的期望）:  T_P ≤ T1/P + T∞
线性加速条件：P ≪ T1 / T∞
```

- **推导**：每一步要么 P 个处理器都忙（这类步骤最多 T1/P 步），要么有处理器空闲，此时关键路径上的就绪任务一定被执行，这类步骤最多 T∞ 步。
- **口算**：归约 n=10⁶ 个数，T1 ≈ 10⁶，T∞ ≈ log₂10⁶ = 20，并行度 5×10⁴，8 核远低于这个数，可以线性加速。
- **常见错答**：只看 T1/P。递归切分太粗时 T∞ 大，切分太细时调度开销进 T1。

## F6. Scan 的 work 和 span

```text
顺序:            work n−1,        span n−1
Hillis–Steele:   work n·log₂n,    span log₂n
Blelloch:        work 2(n−1),     span 2·log₂n   （up-sweep + down-sweep）
```

- **推导**：Hillis–Steele 每一层所有元素都加一次偏移 `2^d` 处的值；Blelloch up-sweep 是一棵 n−1 次加法的归约树，down-sweep 对称。
- **口算**：n=2²⁰，Hillis–Steele 做 2000 万次加法，Blelloch 约 200 万次。GPU 上大数组用 work-efficient，warp 内（32 个元素）用 Hillis–Steele，因为步数少、线程本来就空着。
- **常见错答**：「log n 步的算法一定更快」。工作量多了 log n 倍，带宽受限时更慢。

## F7. Attention 的 FLOPs 和显存

```text
每层每个头，前向：  QKᵀ 2T²d  +  PV 2T²d  =  4T²d
整层（h 头，d_model = h·d）: 4T²·d_model
朴素实现的分数矩阵: B·h·T²·s 字节（每层，反传要存）
FlashAttention 额外显存: O(B·h·T)（只存 logsumexp）
```

- **推导**：`S = QKᵀ` 是 `[T,d]×[d,T]`，`O = PV` 是 `[T,T]×[T,d]`，各 2T²d。
- **口算**：T=8192、d_model=4096：4·8192²·4096 ≈ 1.1 TFLOP/层；同层投影和 MLP 约 `24·T·d_model²` ≈ 3.3 TFLOP，attention 已经占到 1/4。分数矩阵 bf16、32 头：32·8192²·2 ≈ 4.3 GB/层/样本。
- **常见错答**：「FlashAttention 减少了 FLOPs」。它不减 FLOPs（反传还要重算），减的是 HBM 读写。

## F8. Online softmax 合并

```text
两段的统计量 (m₁, l₁, o₁)、(m₂, l₂, o₂)，其中 l = Σ e^{s−m}，o = Σ e^{s−m}·v
m = max(m₁, m₂)
l = e^{m₁−m}·l₁ + e^{m₂−m}·l₂
o = e^{m₁−m}·o₁ + e^{m₂−m}·o₂
最终输出 O = o / l，反传保存 L = m + log l
```

- **推导**：`softmax` 的分母可以按块累加，只要把旧块的指数项乘上 `e^{m_old − m_new}` 统一到新的最大值。
- **口算**：s=[1,3] 分两块。块一 m=1,l=1；块二 m=3,l=1；合并 m=3，l = e^{−2}+1 ≈ 1.135，和一次算 `e^{1−3}+e^0` 相同。
- **常见错答**：不减最大值直接累加 `e^s`。bf16/fp16 下 `e^{12}` 已经溢出 fp16 的 65504。

## F9. 参数量与训练 FLOPs（6ND）

```text
每层（MHA，非门控 MLP，d_ff = 4d）: 4d² + 8d² = 12d²
每层（GQA + SwiGLU）: 2d² + 2d·d_kv + 3d·d_ff
N ≈ 12·L·d²（+ 词表 V·d）
训练 C ≈ 6·N·D：前向 2ND，反向 4ND（对激活和对权重各一次 GEMM）
attention 每 token 额外前向 4·L·T·d_model，与 2N ≈ 24·L·d² 之比为 T/(6d)，T ≪ 6d 时可忽略
```

- **推导**：每个参数在前向参与一次乘加（2 FLOP/token）；反向要算输入梯度和权重梯度，各一次同样大小的 GEMM。
- **口算**：Llama-3-8B 训练 15T token：6 × 8×10⁹ × 1.5×10¹³ = 7.2×10²³ FLOP。1 万张 H100、MFU 40%：7.2e23 / (1e4 × 989e12 × 0.4) ≈ 1.8×10⁵ s ≈ 2 天。
- **常见错答**：「推理每 token 6N」。推理只有前向，2N。

## F10. 训练静态显存 16N 与 ZeRO

```text
bf16 权重 2N + bf16 梯度 2N + fp32 主权重 4N + Adam m 4N + v 4N = 16N
ZeRO-1: 4N + 12N/d       ZeRO-2: 2N + 14N/d       ZeRO-3: 16N/d
通信量（每步，每卡）: DP / ZeRO-1 / ZeRO-2 ≈ 2N·s，ZeRO-3 ≈ 3N·s（多一次前向 all-gather）
```

- **推导**：ZeRO-1 切优化器状态，ZeRO-2 再切梯度（reduce-scatter 代替 all-reduce），ZeRO-3 再切权重（前向、反向各 all-gather 一次）。
- **口算**：7B、8 卡：不切 112 GB；ZeRO-1 是 28 + 10.5 = 38.5 GB；ZeRO-3 是 14 GB，再加激活。
- **常见错答**：「ZeRO-3 的通信是 DP 的 3 倍」。是 1.5 倍：DP 的 all-reduce 等于 RS+AG 两次，ZeRO-3 是 AG(前向)+AG(反向)+RS 三次。

## F11. Ring all-reduce 的 α-β 模型

```text
reduce-scatter: p−1 步，每步发 S/p
all-gather:     p−1 步，每步发 S/p
all-reduce:     t = 2(p−1)·α + 2·(p−1)/p · S/β
```

- **推导**：数据切成 p 份，环上每步每卡收发一份。带宽项 `2(p−1)/p · S` 随 p 趋近 2S，和卡数几乎无关；延迟项随 p 线性增长。
- **口算**：7B 的 bf16 梯度 14 GB，8×H100 NVLink 单向 450 GB/s：`2·7/8·14e9/450e9 ≈ 54 ms`。跨节点走 50 GB/s IB 就是约 0.5 s，所以要 overlap 和分桶。
- **常见错答**：「环越大越慢」。带宽项不变，变慢的是延迟项，所以小消息用 tree，大消息用 ring。

## F12. 张量并行的通信量

```text
Megatron MLP：W1 按列切（无通信），W2 按行切 → 前向 1 次 all-reduce
Attention：按头切，W_O 按行切 → 前向 1 次 all-reduce
每层前向 2 次、反向 2 次 all-reduce，每次大小 B·T·d_model·s
序列并行：all-reduce → reduce-scatter + all-gather，通信量相同，激活按 t 切
```

- **推导**：`Y = GeLU(X W1) W2`。W1 列切后每卡得到 GeLU 的一部分列，不需要同步；W2 行切后每卡得到部分和，必须求和。
- **口算**：B·T = 8192 token、d=8192、bf16：每次 134 MB；每层 4 次，80 层约 43 GB/步/卡。NVLink 上约 0.1 s 量级，跨 IB 是 1 s 量级——所以 TP 留在节点内。
- **常见错答**：「两层都列切更省通信」。第二层列切需要先 all-gather 第一层的输出，通信反而更多。

## F13. 流水线气泡

```text
GPipe / 1F1B:   bubble = (p−1) / (m + p − 1)
交错（每卡 v 个虚拟 stage）: bubble = (p−1)/v / (m + (p−1)/v)
在途激活：GPipe m 个 micro-batch，1F1B 最多 p 个
```

- **推导**：填充和排空各需要 p−1 个时隙，m 个 micro-batch 占 m 个时隙。1F1B 不减少气泡，只是一个 micro-batch 的反向做完就释放它的激活。
- **口算**：p=8、m=32：7/39 = 18%。m=8 时是 47%。所以 m 至少取 4p。
- **常见错答**：「1F1B 比 GPipe 气泡小」。气泡相同，省的是激活显存。

## F14. 重计算的 √n 方案

```text
n 层，每 k 层存一个检查点：显存 ≈ n/k + k，k = √n 时最小 ≈ 2√n
额外计算：约一次前向（总 FLOPs 从 3F 变成 4F，+33%）
```

- **推导**：只保留检查点激活；反向到某段时，从检查点重算该段 k 层并临时保存。
- **口算**：n=100 层：从存 100 份降到约 20 份，代价是多 33% 计算。选择性重计算只重算 attention 的 softmax/dropout，开销小于 5%。
- **常见错答**：「重计算让训练慢一倍」。只多一次前向，反向本来就是前向的 2 倍。

## F15. KV cache 大小

```text
KV 字节 = 2 · L · n_kv · d_head · T · B · s
每 token：2 · L · n_kv · d_head · s
```

- **推导**：每层每个 token 存一个 K 向量和一个 V 向量，每个长 `n_kv · d_head`。
- **口算**：Llama-3-8B：2·32·8·128·2 = 128 KB/token。8K 上下文、batch 32：128 KB × 8192 × 32 = 32 GiB（34 GB）。Llama-2-7B 没有 GQA，同条件 128 GiB。
- **常见错答**：用 `n_heads` 而不是 `n_kv_heads`。GQA 下差 4–8 倍。

## F16. Decode 的 roofline 与临界 batch

```text
单步 decode 时间 ≈ max( 2N·B / P_peak ,  (s·N + KV字节) / BW )
只看权重: I ≈ 2N·B / (s·N) = B（bf16）
临界 batch ≈ I* = P_peak / BW（bf16）
上限: tokens/s ≤ B · BW / (s·N + KV)
```

- **推导**：每步把全部权重从 HBM 读一遍，参与 B 个 token 的计算。权重读取被 B 个请求分摊，KV 不能分摊。
- **口算**：H100、8B、B=1：16 GB / 3.35 TB/s ≈ 4.8 ms，约 200 token/s。B 到约 300 时才受算力限制；长上下文下 KV 读取先成为瓶颈。
- **常见错答**：「decode 慢是因为计算量大」。B=1 时计算只用到峰值的 0.3%。

## F17. 推测解码的期望长度与加速比

```text
每次目标模型前向接受的期望 token 数: E = (1 − α^{γ+1}) / (1 − α)
加速比（草稿一次前向成本是目标的 c 倍）: S = E / (γ·c + 1)
接受规则: 以概率 min(1, p(x)/q(x)) 接受；拒绝时从 norm(max(0, p − q)) 重采样
```

- **推导**：第 i 个草稿 token 被接受的概率是 α^i（独立假设）。被接受的前缀长度加上最后一个由目标模型给出的 token，期望 `Σ_{i=0}^{γ} α^i`。接受-重采样的组合恰好还原分布 p。
- **口算**：α=0.8、γ=4：E = (1−0.8⁵)/0.2 = 3.36；c=0.05 时 S = 3.36/1.2 = 2.8。
- **常见错答**：「推测解码会改变输出分布」。按上面的规则采样，输出分布和目标模型完全相同；只有贪心版才是「输出完全一样」。

## F18. Little 定律与服务容量

```text
并发数 L = 到达率 λ × 平均停留时间 W
单请求时延 ≈ TTFT + 输出长度 × TPOT
容量：λ_max ≈ KV 预算下的最大并发 / W
```

- **推导**：稳态下系统里的平均请求数等于单位时间进入的请求数乘以每个请求停留的时间。
- **口算**：平均输出 500 token、TPOT 30 ms、TTFT 0.5 s：W ≈ 15.5 s。KV 预算支持 64 个并发：λ_max ≈ 4.1 req/s。
- **常见错答**：只看吞吐不看 SLO。goodput 只统计满足 TTFT/TPOT 约束的请求。

## F19. 线性量化

```text
对称: scale = max|x| / (2^{b−1} − 1)，  q = round(x / scale)，  x̂ = q·scale
非对称: scale = (max − min)/(2^b − 1)，zero_point = round(−min/scale)
误差上界: |x − x̂| ≤ scale / 2
粒度: per-tensor < per-channel < per-group（g=64/128）
```

- **推导**：把 `[−max|x|, max|x|]` 均匀映射到整数网格，舍入误差最多半个格子。
- **口算**：一行权重里有一个 100 的离群值、其余在 ±1，int8 对称 scale = 100/127 ≈ 0.79，绝大多数权重被量化成 0 或 ±1。per-group 把离群值关在一个 group 里。
- **常见错答**：「int8 激活和 int8 权重一样好做」。LLM 激活有固定通道的离群值（数十到上百倍），per-tensor 激活量化会把其它通道压没，这是 LLM.int8 和 SmoothQuant 要解决的问题。

## F20. LoRA 参数量

```text
W' = W + (α/r)·B·A，  A: r×k，B: d×r，B 初始化为 0
可训练参数: r·(d + k)，相对全量 r(d+k)/(dk)
合并推理: W ← W + (α/r)·BA，无额外延迟
```

- **口算**：d=k=4096、r=16：131K 对 16.8M，0.78%。7B 模型只在 q,v 上加 r=16：约 8M 参数，Adam 状态从 84 GB 降到 0.1 GB。
- **常见错答**：「LoRA 省了前向显存」。基座权重和激活仍在，省的是梯度和优化器状态。

## F21. MoE 的 all-to-all 通信量

```text
每层两次 all-to-all（dispatch + combine）
每卡发送 ≈ B·T·k·d·s · (E_p − 1)/E_p      （k = top-k，E_p = 专家并行度）
参数量随专家数 E 线性增长，每 token FLOPs 只随 k 增长
```

- **口算**：每卡 8192 token、d=4096、top-2、bf16、EP=8：8192·2·4096·2·7/8 ≈ 117 MB/次，每层前向 2 次、反向 2 次。
- **常见错答**：「MoE 省算力所以训练便宜」。FLOPs 省了，但显存（全部专家）和 all-to-all 通信没省。

## F22. MFU

```text
MFU = 实测 tokens/s × 6N / (卡数 × P_peak)
HFU 把重计算的 FLOPs 也算进去，所以 HFU ≥ MFU
```

- **口算**：8×H100 训练 7B 模型，每秒 1.0×10⁵ token：6·7e9·1e5 / (8·989e12) = 53%。
- **常见错答**：拿 HFU 当 MFU 报。开了全量重计算的 HFU 会高出约 1/3。
