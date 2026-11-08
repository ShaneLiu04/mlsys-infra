# 项目作品集：面试用 STAR 故事

被问到「讲一个你做过的项目」时，就用这里的故事。每个故事的顺序是：先一段 20 秒能讲完的版本，
然后是完整的 STAR（背景、任务、做法、结果），再是白板上要写的推导、这个实验不能推出的结论，最后是可能的追问。
数字都来自各实验 `artifacts/raw/` 下的原始文件，分属两类平台：

- 实测：RTX 5090（sm_120，32 GB GDDR7）和 2 × Xeon Gold 6459C（固定 12 个物理核），代码是 C++、CUDA 和 Triton。
  本仓库自测的硬件基线（copy 带宽、GEMM 峰值等）统一放在 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md#本仓库实测rtx-5090--xeon-6459c)。
- 模型：用 H100 参数算的 roofline、α-β 模型和事件模型，以及在 4 核 CPU 上的 numpy 复现。这一类用来讲机制和相对关系，
  每个故事里都标明了哪些数字属于它。

完整的推导和全部图在 [`reports/`](../reports/) 的 10 份实验报告里，公式在 [FORMULAS](FORMULAS.md)，排查方法在 [ENGINEERING](ENGINEERING.md)。

怎么讲：只准备两个故事就够了。推理岗讲 2 和 3，训练岗讲 4 和 6，算子或芯片岗讲 1 和 7，CPU 或系统岗讲 8 和 9。
开口先说结果数字，讲到 90 秒左右就停下来，让对方打断提问。有一点要提前想好怎么说：容器不开放 GPU 性能计数器
（ncu 报 `ERR_NVGPUCTRPERM`），所以被问到 occupancy 或 DRAM 字节数时，要讲清楚你说的是理论值还是推算值。

## 索引

| # | 20 秒结论 | 考察点 | 实验 · 报告 |
| ---: | --- | --- | --- |
| 1 | Triton FlashAttention 前向 224 TFLOP/s，与 FA2 持平；比朴素快 7.15 倍，而「流量模型」只预测 2 倍 | IO 复杂度、roofline | [P1](../solutions/cs229s/project1-transformer-kernels/README.md) · [PDF](../reports/cs229s-project1/cs229s_project1_report.pdf) |
| 2 | gpt2→gpt2-xl 推测解码贪心 1.63 倍：GPU 上验证 4 个 token 只比 1 个贵 11%；CPU 上贵 41–61%，加速消失 | 推理加速、口径 | [P2](../solutions/cs229s/project2-kv-speculative/README.md) · [PDF](../reports/cs229s-project2/cs229s_project2_report.pdf) |
| 3 | 真实引擎上分页 KV 浪费 86.8% → 1.26%、并发 7.0 倍，间接寻址开销 <1%；小模型 decode 先卡在 CPU 派发 | serving | [15-849 服务](../solutions/cmu15849/serving-memory/README.md) · [PDF](../reports/cmu15849-serving-memory/serving_memory_report.pdf) |
| 4 | 16 B/参数实测 16.00–16.32；ZeRO 分片显存与公式差 0.099%，分片 AdamW 逐位相等；1F1B 气泡停在 0.23 而不是公式的 0.086 | 分布式训练 | [15-849 分布式](../solutions/cmu15849/distributed-training/README.md) · [PDF](../reports/cmu15849-distributed-training/distributed_training_report.pdf) |
| 5 | `torch.compile` 的加速就是字节比（RMSNorm 9.86 倍 / 字节比 10.0）；MLP 块融合只省 15% | 编译器、roofline | [15-849 编译](../solutions/cmu15849/compiler-compression/README.md) · [PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf) |
| 6 | GPT-2 W3 量化 RTN ppl 65.9 → GPTQ 37.2；SmoothQuant 的 α 选错（0.5）ppl 爆到 407 | 量化 | [15-849 编译](../solutions/cmu15849/compiler-compression/README.md) · [PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf) |
| 7 | 乱序画圆 93% 像素出错；分块 + scan 保序后逐位正确，CUDA 上 rand1M 从 844 ms 到 0.538 ms | 并行正确性 | [A3](../solutions/cs149/asst3-scan-render/README.md) · [PDF](../reports/cs149-asst3/cs149_asst3_report.pdf) |
| 8 | sqrt 每 8 个元素混 1 个慢的，AVX-512 耗时与全慢完全相同；AVX2 只有 0.72 倍，比标量还慢 | 发散、负载均衡 | [A1](../solutions/cs149/asst1-cpu-parallel/README.md) · [PDF](../reports/cs149-asst1/cs149_asst1_report.pdf) |
| 9 | spin 线程池 launch 0.78 µs、sleep 10.8 µs，但空闲时 spin 烧 11.96 个核，2 倍超订时慢 766 倍 | 同步、调度 | [A2](../solutions/cs149/asst2-task-scheduler/README.md) · [PDF](../reports/cs149-asst2/cs149_asst2_report.pdf) |
| 10 | SGEMM 从 0.94 调到 49.4 TFLOP/s（cuBLAS 的 78%）；最大的一跳不是消 bank conflict，而是 LDS.32 换 LDS.128 | GEMM、分块 | [A4](../solutions/cs149/asst4-dnn-accelerator/README.md) · [PDF](../reports/cs149-asst4/cs149_asst4_report.pdf) |
| 11 | RMAT 上 direction-optimizing BFS 8 线程 6.58 倍（三个种子复现），图放大到 6619 万边时 12.2 倍；网格上并行反而慢 100 倍 | 图计算、算法 × 系统 | [A5](../solutions/cs149/asst5-big-graphs/README.md) · [PDF](../reports/cs149-asst5/cs149_asst5_report.pdf) |
| 12 | 第一版测出 8 进程 21 倍加速，是块大小改变了缓存行为 | 实验可信度 | [A1](../solutions/cs149/asst1-cpu-parallel/README.md) |

---

## 1. FlashAttention：模型说 2 倍，实测 7 倍

20 秒版本：我先在 numpy 版的 FlashAttention 里加了一个 HBM 读写计数器。按论文的块大小（Br 被 d 截住），
它的流量只比一个「S、P 各读写一次」的朴素模型少 2 倍。然后我在 RTX 5090 上用 Triton 写了 FA2，
前向跑到 224 TFLOP/s，和 PyTorch 自带的 FA2（217.9）持平；T=8192 时比朴素实现快 7.15 倍。
模型说 2 倍、实测 7 倍，差距出在那个朴素模型低估了朴素实现：eager 版本实际有 4Td+6T² 次读写，
它本身就是一个跑到带宽 93% 的访存程序。

背景和任务：这是 CS229S 的 Project 1。论文说 FlashAttention 把 HBM 访问降到 `Θ(T²d²/M)`，
我想弄清楚两件事：在具体参数下它到底省了多少，以及在 GPU 上实际能快多少。

做法：先写 numpy 版，前向分块，反向用 logsumexp 重算，再加上 IO 计数器，数清楚每种实现读写了多少字节。
然后写 Triton 版：前向是 online softmax，用 exp2，因果的情况分两段处理；反向拆成 dK/dV 和 dQ 两个 kernel。
最后对块大小做了一遍扫描。

结果：

- 正确性：numpy 版在 float64 下和 torch 差 4.9e-15；Triton 版 fp16 误差 6.6e-5，和 SDPA 的 flash 后端一样。
- 速度：加速比可以写成 `3·ridge/d`。d=64 时预测 7.3、实测 7.15；d=128 时预测 3.6、实测 3.8。
- 显存：朴素实现在 T=16384 时 OOM，Triton 版到 T=32768 额外显存也只有 138 MB。
- 训练：GPT-2 small 用 bf16 加 flash，MFU 是 0.447。
- 短板：反向只有 FA2 的 80%。我的两个 kernel 各自重算了 QKᵀ 和 dO·Vᵀ，一共 9 个矩阵乘，FA2 只要 7 个。

白板：朴素 eager 的流量约 `4Td + 6T²`（fp16），flash 约 `Td·(2 + 2T/Br)`。按这个 IO 模型反推，
flash 的「隐含带宽」最高到 6.57 TB/s，是 DRAM 的 4.3 倍。这说明 K、V 的重复读取大多命中了 96 MB 的 L2。

不能推出：「隐含带宽」和 L2 命中都是推断，没有 ncu 计数器直接证明。也不能说那个 2 倍的流量模型错了，它只是只数了 S、P 两个张量。

追问：反向为什么慢？我把 dK/dV 和 dQ 拆成两个 kernel，好处是避开了 atomic，代价是两个 kernel 都要重算一遍 S。
FA2 用一个 kernel 加上 dQ 的 atomic 累加，少做两次矩阵乘。

## 2. 推测解码：验证免费与否决定一切

20 秒版本：我把同一套接受-拒绝采样分别在 CPU 小模型和 GPU 上的真实 GPT-2 上跑了一遍。
两边每次目标前向产出的 token 数都和 `(1−α^{γ+1})/(1−α)` 吻合，差距不超过 4%。但加速比完全不同，差别全在验证的代价上。
CPU 上验证 γ+1 个 token 要花单 token 前向的 1.41–1.61 倍，加速只有 0.98–1.99 倍。
RTX 5090 上 gpt2-xl 验证 4 个 token 只贵 11%（v=1.106），gpt2→xl 贪心实测 1.631 倍，和 `E/(γc+v)` 的预测只差 3%。

背景和任务：这是 CS229S 的 Project 2。GPU 部分用的是我自己写的 GPT-2 forward，直接读 HF 的权重。
任务是实现 Leviathan/Chen 的接受-拒绝采样，验证它的输出分布无偏，测出加速，再解释加速是从哪来的。

做法：接受概率取 `min(1, p/q)`，拒绝时从 `norm(max(0, p−q))` 重采样。KV 的回滚很简单，把 `length` 调小就行。
为了不让 CPU 和 GPU 之间的同步吃掉收益，接受-拒绝全部在 GPU 上算，decode 用 CUDA Graph，每一轮只同步一次。

结果：

- 正确性：fp32 下贪心解码 120/120 条都和 target 单独解码一致。采样模式下，对第一个 token 做卡方检验，对 target 分布 p=0.657，对 draft 分布 p=8.9e-168，所以输出服从的是 target 的分布。
- 选 draft：更大的 draft（gpt2-medium）α 更高，0.77 对 0.71；但它的 c 也翻倍到 0.308，最后加速只有 1.283。
- α 的口径：我最初用「接受数 / 提议数」来算 α，γ=8 时把 0.80 算成了 0.41。后来改成「接受数 /（接受数 + 拒绝数）」才对。

白板：F17，`S = E / (γc + v)`。GPU 上 decode 受带宽限制，所以 v≈1。但 c 远大于两个模型的参数量之比（0.137 对 0.08），
因为每个 kernel 都有固定开销，小模型摊不掉。

不能推出：大 batch 下的结论，是把实测的 c(B)、v(B) 代入公式拼出来的预测（B=128 时只剩 1.005 倍）。`SpecEngine` 只实现了 batch 1。

追问：为什么最优的 γ 只有 3–4？E 随 γ 增大会饱和到 1/(1−α)，而分母里的 γc 是线性增长的。α≈0.7 时，γ 超过 4 收益就是负的。

## 3. 分页 KV 与 chunked prefill：真实引擎上的账

20 秒版本：我先用 H100 的 roofline 模型做了一个调度模拟，再在 RTX 5090 上用 Qwen2.5-0.5B 写了一个真实的推理引擎，
包括分页 KV、iteration-level 调度、chunked prefill 和一个 Triton 写的分页 decode kernel。在真实引擎上算出来的账有三条：

- 分页：4 GiB 的 KV 预算下，按 4096 个 token 预留会浪费 86.8%，只能并发 85 条；分页、block=16 时浪费 1.26%，并发 593 条，是 7.0 倍。这和模型的预测相差不到 1.2 个百分点。
- 间接寻址：物理块随机分布和按顺序排列，耗时相差不到 1%，所以块表带来的间接寻址几乎不花钱。
- 模型漏掉的东西：小模型的 decode 先卡在了 CPU 派发上。eager 模式每步 8.2–9.0 ms，用 CUDA graph 是 2.16 ms，而带宽下限只有 0.65 ms。

背景和任务：这是 15-849 的服务方向，对应 vLLM、Orca、Sarathi 这几篇论文。我想分别弄清楚分页和调度各自带来了什么，以及模型在哪里不准。

做法：实现了块表、引用计数、copy-on-write fork，以及 swap 和 recompute 两种抢占方式。分页 decode kernel 比用 torch gather 拼 KV 快 2.2–10.7 倍。
然后把 static、continuous、chunked 三种调度策略在一组到达率下扫到期望负载。

结果：

- static 在 5.36 req/s 就饱和了；continuous 在 λ=40 时做到 23.2 req/s，TTFT P99 是 38 ms。
- 负载里混入 10% 的长 prompt（3072 token）时，chunked prefill 把逐请求最大卡顿的 P99 从 33.5 ms 降到 17.6 ms。代价是长 prompt 自己的 TTFT 从 34 ms 涨到 110 ms，吞吐下降 2.6%。
- 拟合出来的引擎迭代时间是 9.15 ms + 1.03 × roofline。也就是说模型的斜率是对的，但它漏掉了约 9 ms 的固定开销。

白板：每条序列浪费不超过 block−1 个 token，平均是 (b−1)/2。decode 每一步的时间约等于 `max((权重 + KV) / 带宽, 派发开销)`。

不能推出：0.5B 模型上的派发瓶颈不能代表 8B 或 70B。lab 自带的是短 prompt 分布，在那上面 chunked prefill 没有收益；这个结论只在混有长 prompt 的负载下成立。

追问：chunked prefill 为什么依赖 attention 的实现？带前缀的 prefill 块需要一个非方阵的因果 mask。如果用布尔掩码传进去，
SDPA 会退回 math 路径，每块多花 5.8 ms，收益直接被抵消。

## 4. 训练显存的账：逐字节对上

20 秒版本：我在单张 RTX 5090 上把训练显存的公式逐项和实测对上了。

- 每参数 16 字节：fp32、AMP、bf16 加 fp32 master 三种布局，实测都是 16.00–16.32；纯 bf16 是 8.0–8.3。
- 激活：要按 flash attention 实际保存下来的张量来数（AMP 下是 36 字节/sbh），误差才降到 1% 以内。
- ZeRO 0–3：用真实的扁平分片测每个 rank 的显存，和公式最多差 0.099%；分片后的 AdamW 和不分片的结果逐位相等。

背景和任务：这是 15-849 的分布式训练方向，涉及 ZeRO、Megatron、GPipe/1F1B 和重计算。我要搞清楚哪些显存公式可以直接拿来用，哪些需要修正。

做法：自己写了一个带 checkpoint 模式的 GPT-2，对 b×s 扫了一遍激活显存。流水线部分只有一张卡，
所以我把 4 个 stage 放在同一张卡上，按 1F1B 或 GPipe 的顺序执行真实的 kernel，用 CUDA event 计时，再按依赖关系重放出时间线。
CPU 部分用集合通信模拟器验证了 ZeRO-3 的通信量恰好是 DP 的 1.5 倍。

结果：

- MFU：774M 的模型纯 bf16 训练达到 70.2%。
- 重计算：全量重计算让每步时间变成原来的 1.189 倍（理论是 4/3），换来最大 batch 从 8 涨到 24。
- 1F1B：m=8/16/32 时气泡分别是 0.343/0.245/0.233，峰值显存固定在 7.64 GB；GPipe 在 m=16 时就 OOM 了。

白板：F10、F13、F14。气泡公式 (p−1)/(m+p−1) 假设每一级一样长。

不能推出：多卡通信没有测，通信时间是用实测的 PCIe/D2D 带宽参数化的 α-β 模型算出来的。单卡上的「流水线」是串行执行后重放出来的。

追问：1F1B 的气泡为什么停在 0.23，而不是公式给的 0.086？因为最后一级多了 LM head，各级并不均衡。
用实测的各级时间建一个不均衡模型，它给出 0.354/0.254/0.203，和实测吻合。

## 5. 融合的收益就是字节比

20 秒版本：在访存受限的算子链上，`torch.compile` 的加速比和「eager 的字节数 / 融合后的字节数」相差不到 7%。
RMSNorm 快 9.86 倍，字节比是 10.0；softmax 快 5.05 倍，残差链快 2.68 倍。融合之后，它们都顶在约 1.53 TB/s 的带宽墙上。
MLP 块就不一样了，里面两个大 GEMM 受算力限制，max-autotune 只省了 15%，从 419 µs 降到 357 µs。
这和我自写的图 IR 用 roofline 给出的预测一致（从 435.7 µs 到 338.1 µs）。

做法：CPU 部分是我自己写的图 IR，包括常量折叠、CSE、TASO 式的代数替换，以及逐元素融合和 epilogue 融合。
GPU 部分拿 Inductor 做对照，又跑了 Triton matmul 的 192 个配置组成的网格，用代价模型给这些配置排序。

结果：

- 8192³ 时，网格里最好的配置达到 221.5 TFLOP/s，超过了 cuBLAS 的 214.3。
- 代价模型：复用型模型在方阵上的 Spearman 相关是 0.65，在 M=256 时只有 0.10；波次 roofline 模型正好反过来。没有哪个解析模型在所有形状上都准，这也是 Ansor 要用学习型模型加实测的原因。

白板：k 个逐元素算子融合以后，流量从 `2kS` 降到 `2S`；时间 = `max(F/P, B/BW)`。

不能推出：小尺寸时算出来的「带宽」超过了 GDDR7 的峰值，那是 L2 命中，不能当成 DRAM 带宽来引用。

追问：工程上有什么代价？max-autotune 用 CUDA graph 时要拷贝输入，这让 RMSNorm 慢了 2.08 倍。
另外，10 个序列长度如果静态编译，要编出 8 个图，花 4.46 s；用 `mark_dynamic` 只要 1 个图，0.54 s。

## 6. 量化：位宽越低越需要数据

20 秒版本：我在 GPT-2 124M 上比较了几种量化方法，fp16 基线的困惑度（ppl，越低越好）是 29.96。
W4 时 RTN 和 GPTQ 差得不多；到了 W3，RTN 的 ppl 涨到 65.92，用校准数据的 GPTQ 还能守在 37.24。
W8A8 那一组里，SmoothQuant 的 α 取 0.5 时 ppl 爆到 407.25，换成 0.8 就回到 30.49。

| 设置 | 方法 | ppl |
| --- | --- | ---: |
| W4 g128 | RTN | 33.27 |
| W4 g128 | GPTQ | 31.20 |
| W3 | RTN | 65.92 |
| W3 | AWQ 式 | 53.35 |
| W3 | GPTQ | 37.24 |
| W8A8 | per-tensor | 49.94 |
| W8A8 | per-token | 30.18 |
| W8A8 | LLM.int8 | 30.06 |
| W8A8 | SmoothQuant α=0.5 | 407.25 |
| W8A8 | SmoothQuant α=0.8 | 30.49 |

速度上，batch 1 的 weight-only GEMV 是按字节数加速的：bf16 要 163.0 µs，W8 是 82.5 µs，W4 是 62.2 µs。
如果先把权重反量化回来再做 matmul，要 3688 µs。

做法：实现了 absmax、per-channel、per-group 三种粒度的量化，LLM.int8 的混合分解，SmoothQuant、GPTQ 和 AWQ 式缩放。
CPU 部分用合成的离群通道来讲机制：1024 个通道里只放 8 个 50 倍的离群值，per-tensor 的误差就从 2.3% 涨到 10.2%。

结果：大 GEMM 上 bf16 是 226.9、FP8 是 456.2、int8 是 679.4 TOP/s。剪掉 50% 的权重时，幅值剪枝的 ppl 是 538，Wanda 是 47.2。
2:4 稀疏在大 GEMM 上快 1.54 倍。

白板：F19，`scale = max|x|/127`。GPTQ 用校准数据的 Hessian `XᵀX` 逐列补偿量化误差。

不能推出：W8A8、SmoothQuant 和 LLM.int8 都是精度模拟，实际跑的是 fp16 GEMM，所以没有端到端的 int8 速度。另外，GPT-2 的离群值远比 6.7B 以上的模型弱。

追问：SmoothQuant 的 α 为什么这么敏感？它按 `s = max|X|^α / max|W|^{1−α}` 把激活的量化难度挪一部分给权重。
α 太小，激活的离群值没被压下去，per-tensor int8 的 scale 仍然被它主导。

## 7. 保序渲染

20 秒版本：半透明圆的混合不满足交换律，先画哪个圆，结果就不一样。所以按圆并行、无序写入像素时，
2000 个圆的场景里有 93% 的像素是错的。我改成按 16×16 的像素块并行：每块先用 exclusive scan 压缩出一个保持原始顺序的圆列表，
再按这个顺序着色，结果和顺序参考逐位相等。CUDA 版在 rand1M 场景上，「每个像素遍历所有圆」的正确版本要 844 ms；
分桶之后要做的相交测试少了 1640 倍，只要 0.538 ms。

做法：

- scan 写了三种：Blelloch、Hillis–Steele 和 cub。n=2²⁰ 时，Blelloch 做 2.10M 次加法，Hillis–Steele 做 19.9M 次。
- CUDA 上对 2²⁸ 个元素做 scan：朴素版 16.75 ms，自写版 2.82 ms，cub 1.41 ms。
- 渲染器关闭 FMA 编译，这样才能和 CPU 参考逐位一致。

结果：

- naive 版本有两种错法。rgb 场景的三个圆落在同一个 warp 里，结果可以复现，但顺序是错的，24.3% 的像素出错；其余场景存在读写竞争，两次运行的结果都不一样。
- 用 shared memory 批量取圆，带来 1.7–2.1 倍的加速。
- micro2M 场景（圆的半径约 1 像素）用 T=8 最快。

不能推出：occupancy 是按寄存器数算出来的理论值（寄存器从 52 个降到 34 个，occupancy 从 0.67 变成 1.0），没有计数器验证。

追问：用 atomic 行不行？不行。原子操作只能解决写冲突，不保证顺序，结果还是错的。

## 8. 一条慢 lane 拖垮 SIMD

20 秒版本：我在 sqrt 的输入里每 8 个元素放 1 个慢的，AVX2 版本就只剩标量的 0.72 倍。
AVX-512 上，all_slow、one_slow_8、one_slow_16 三种输入的耗时完全相同（96.8 / 96.8 / 97.3 ms）。
原因是一个向量的代价由它最慢的那条 lane 决定。GPU 上也一样，每 32 个元素混 1 个慢的，和全部都慢一样贵。

背景：这是 CS149 的 A1。我把六个 program 用 C++17 和 AVX2/AVX-512 intrinsics 重写了一遍，另外用 CUDA 回答同样的问题。

做法：Mandelbrot 比较了三种划分方式（连续块、隔行、动态）；做了 SMT 和「单线程交错两组向量」的对照；
测了 saxpy 的带宽和 streaming store 的效果；对 K-means 做了 Amdahl 分析。

结果：

- 负载均衡：12 线程时，连续块划分只有 5.89 倍，动态分块是 11.1 倍。3 线程甚至比 2 线程还慢，因为中间那一块独揽了集合的主体。
- 延迟隐藏：单线程交错两组 16 lane 的向量，加速从 11.2 倍提到 18.3 倍，超过了向量宽度。这说明内层循环是一条依赖链。
- 带宽：saxpy 单线程的 DRAM 带宽是 23 GB/s，12 线程是 173 GB/s。
- GPU：Mandelbrot 比 12 核版本再快 60–100 倍；但 saxpy 一旦要过 PCIe，端到端只有 52 GB/s，比 CPU 还慢。

白板：利用率 = 有效的 lane·迭代数 / (W × 组内最大迭代数)。numpy 模型给出，8 宽的向量里有一条慢 lane 时，利用率是 0.163。

不能推出：服务器是共享的 cgroup 环境，结论来自 3 轮独立实验的中位数。刚唤醒的核上计时会慢 18%–23%，这个现象还没有完全解释（见报告的局限一节）。

追问：GPU 上怎么办？按工作量排序或分桶，让同一个 warp 里的线程工作量接近；或者用 persistent kernel 持续拉取工作。

## 9. spin 还是 sleep：线程池的账

20 秒版本：同一个任务系统 API，我写了四种 C++ 实现，在 12 个物理核上比较。只看单个空任务的 launch 开销，spin 线程池最快：
spin 线程池 0.78 µs，sleep 线程池 10.8 µs，每次新建线程是 53.3 µs。但 spin 的代价藏在别的地方：

- 空闲的时候，spin 池也烧掉 11.96 个核。
- 任务一多，12 个线程抢同一个 CAS，每个空任务要 494 ns，是 sleep 的 7 倍。
- 2 倍超订时，带锁的 spin 调度器慢了 766 倍。

做法：实现了 Part B 的 `runAsyncWithDeps` 和 `sync`；把 20 个官方测试按原来的结构重写，并校验结果；做 3 轮独立实验，每轮 3 次，取中位数。

结果：

- 用 async 依赖图以后，相互独立的 launch 能互相填满对方的尾部，快 1.64 倍（理想是 1.83 倍）。
- 原子计数器放在同一条 cache line 上，比各占一行慢 19 倍。
- 12 个 worker 绑满 12 个核时，自旋等待的调用者要和 worker 分时，出现了 32 ms 的时间片平台；减到 11 个 worker 以后变成 18.7 ms。

白板：任务间隔比一次唤醒还短，就用 spin；否则用 sleep。spin 的前提是自旋的线程不和干活的线程抢核。

不能推出：Python 版（`threading`）里 spin 反而最慢（31–50 ms），那是 GIL 带来的结论，不是线程池本身的结论。

追问：生产环境里怎么选？先 spin 一小段时间再 park（自适应自旋），并且让 worker 数比可用核数少一个，留给调用者。

## 10. SGEMM 阶梯：每一步都能算出来

20 秒版本：我在 RTX 5090 上把 4096³ 的 SGEMM 一级一级往上调，每一级为什么快、快多少，都能用模型解释。
最后的 float4 / LDS.128 版本到了 49.4 TFLOP/s，是 cuBLAS fp32（63.1）的 78%；WMMA fp16 版本是 177.8 TFLOP/s，是 cuBLAS fp16 的 86%。
中间有一步和直觉相反：2D 寄存器分块比 1D 还慢。

| 版本 | TFLOP/s |
| --- | ---: |
| naive | 0.94 |
| coalesced | 6.43 |
| smem | 9.82 |
| 1D 寄存器分块 | 25.7 |
| 2D 寄存器分块 | 21.5（反而变慢） |
| float4 / LDS.128 | 49.4 |

做法：CPU 部分是加速器上的分块 traffic/cycle 模型（带 SBUF/PSUM 的容量约束）和寄存器级的脉动阵列。
GPU 部分包括 SGEMM 阶梯、tile 热力图、im2col 卷积、skinny GEMM 和 roofline。

结果：

- 分块模型：GPU 上用的是和 CPU 部分完全相同的流量公式。8192³ 时工作集远超 L2，按公式反推出的 DRAM 带宽是 1585 GB/s，和实测 triad 的 1549 GB/s 只差 2.3%。
- 2D 分块为什么变慢：解析的 bank 模型显示有 4-way / 8-way 冲突，但把冲突消掉只快了 2.5%。真正的跳变来自把 16 条 LDS.32 换成 4 条 LDS.128，快了 2.3 倍。
- 双缓冲：第一次编译用了 129 个寄存器，每个 SM 只放得下 1 个 block，反而慢了 6.6%；压到 128 个寄存器后回到 49.44。

白板：bf16 方阵 GEMM 的强度是 `n/3`；分块后的流量约为 `b·(MK·⌈N/BN⌉ + KN·⌈M/BM⌉ + MN)`；强度约等于 tile 边长的一半，直到被 PSUM 或寄存器容量卡住。

不能推出：bank conflict 的次数来自解析模型，没有 ncu 计数器；Trainium 的参数是模型假设，不是规格表上的值。

追问：剩下的 22% 为什么追不上 cuBLAS？cuBLAS 用了更深的软件流水线，还会按形状选择 kernel。
我的双缓冲只多了 0.1%，说明瓶颈已经不在访存延迟上，而在指令发射和寄存器压力上。

## 11. 大图 BFS：方向比线程数重要

20 秒版本：RMAT 图用的是 Graph500 的参数，100 万个顶点、800 万条边。Beamer 的 direction-optimizing BFS
会在 frontier（当前这一层待扩展的顶点集合）变大时切到 bottom-up。这样检查的边只有 top-down 的 9.1%，
8 线程下快 6.58 倍（7.47 对 49.18 ms）。frontier 最大的那一层，bottom-up 只看 8.6 万条边，top-down 要看 389 万条。
图放大到 6619 万条边时，加速到 12.2 倍。反过来，网格和链状的图每层 frontier 只有几十个顶点，4 线程反而比串行慢约 100 倍。

做法：top-down 写了三种（critical、CAS 加共享下标、CAS 加线程私有 frontier）；bottom-up 不需要原子操作；
hybrid 用 α 阈值决定何时切换，并且只在 frontier 增长时切。另外实现了 PageRank 的 pull 和 push 两种方式。

结果：

- 共享的原子下标比线程私有 frontier 慢 1.90 倍。
- PageRank 的 pull 快 7.03 倍，push（用 atomic add）快 4.22 倍。
- α 从 2 扫到 64，时间都在 9.21–9.85 ms 之间，切换发生的层没有变。
- 规模：RMAT 从 2^16 增长到 2^23，hybrid 8 线程的加速从 1.33 倍涨到 12.2 倍。串行版每条边的时间从 2.6 ns 涨到 10.0 ns，因为图出了缓存，而多线程能同时挂起多个缺失。超过 8 倍的那部分来自算法少看的边：同样工作量的 top-down 只从 1.02 倍涨到 4.78 倍。
- 重复性：三个种子各跑 7 次，中位加速在 6.56–6.94 倍之间。

白板：切换条件是 `E_td = Σ outdeg(frontier) > E_bu ≈ unvisited · m/n / α`。

不能推出：这是 4 核 8 线程的本机结果，没有 NUMA，也没有在十亿边的规模上验证过。

追问：星图怎么办？我实现了按边区间切分，4 线程从 33.0 ms 降到 28.3 ms，负载是均衡了，但串行只要 10.4 ms。
原因是并行版本用 1 个线程跑就已经是串行的 3.3 倍（每个顶点一次 CAS、私有数组、层末拼接）。这张图每个顶点的工作太少，不值得并行。

## 12. 伪加速：21 倍

第一版 Mandelbrot 测出 8 个 worker 有 21 倍加速，超过了核数。问题出在我的切分方式上：我按 worker 数切块，
worker 越多，块就越小，块小到一定程度，numpy 的中间数组就落进了缓存。同一张图串行算要 3.65 s，按小块算只要 1.29 s，
也就是说「加速」里有一大块来自块大小的变化，而不是并行。把所有调度方式的块大小统一以后，加速回到了合理的范围。

白板：看到超线性加速，先怀疑基线，检查基线和并行版是不是做了同样的工作、用了同样的访存模式。

不能推出：这不能说明「多进程有超线性加速」，只说明基线不公平。

追问：真实的超线性加速存在吗？存在，比如切分后每个核的数据刚好放进它自己的缓存。但要先在单核上用同样的块大小复现基线，才能这么说。
