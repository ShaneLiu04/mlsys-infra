# 项目作品集：面试用 STAR 故事

每条按「20 秒结论 → STAR → 白板 → 不能推出的结论 → 追问」组织。数字来自各实验 `artifacts/raw/` 的原始文件，
两类平台：

- **实测**：RTX 5090（sm_120，32 GB GDDR7）与 2 × Xeon Gold 6459C（固定 12 个物理核）；C++ / CUDA / Triton。
  本仓库自测的硬件基线（copy 带宽、GEMM 峰值等）统一在 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md#本仓库实测rtx-5090--xeon-6459c)。
- **模型**：H100 参数的 roofline / α-β / 事件模型，以及 4 核 CPU 上的 numpy 复现。用来讲机制和相对关系，
  每个故事里都标明哪些数字属于这一类。

完整推导和全部图在 [`reports/`](../reports/) 的 10 份实验报告里，公式在 [FORMULAS](FORMULAS.md)，排查方法在 [ENGINEERING](ENGINEERING.md)。

讲法：只准备两个故事。推理岗用 2 和 3，训练岗用 4 和 6，算子/芯片岗用 1 和 7，CPU / 系统岗用 8 和 9。
先说结果数字，90 秒停下来让对方打断。容器不开放 GPU 性能计数器（ncu 报 `ERR_NVGPUCTRPERM`），
被问到 occupancy、DRAM 字节数时要说清是理论值还是推算。

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

**20 秒**：我先在 numpy 版里加了 HBM 读写计数器，按论文块大小（Br 被 d 截住）流量只比「S、P 各读写一次」
的朴素模型少 2 倍。然后在 RTX 5090 上写了 Triton FA2：前向 224 TFLOP/s，与 PyTorch 的 FA2（217.9）持平，
T=8192 时比朴素实现快 7.15 倍。差距来自朴素模型低估了朴素实现：eager 版有 4Td+6T² 次读写，本身就是一个
跑满带宽 93% 的访存程序。

- **S**：CS229S Project 1。
- **T**：验证「HBM 访问降到 `Θ(T²d²/M)`」在具体参数下是多少，再看 GPU 上实际能快多少。
- **A**：numpy 版分块前向 / logsumexp 重算反向 + IO 计数器；Triton 版 online softmax（exp2、因果分两段）、反向 dK/dV 与 dQ 两个 kernel；块大小扫描。
- **R**：
  - 正确性：numpy 版 float64 下与 torch 差 4.9e-15；Triton fp16 误差 6.6e-5，与 SDPA flash 相同。
  - 速度：加速比可以写成 `3·ridge/d`，d=64 预测 7.3、实测 7.15，d=128 预测 3.6、实测 3.8。
  - 显存：朴素实现 T=16384 OOM，Triton 在 T=32768 时额外显存只有 138 MB。
  - 训练：GPT-2 small bf16 + flash 的 MFU 是 0.447。
  - 短板：反向只有 FA2 的 80%（两个 kernel 各自重算 QKᵀ 和 dO·Vᵀ，矩阵乘 9 个对 7 个）。
- **白板**：朴素 eager 流量 ≈ `4Td + 6T²`（fp16），flash ≈ `Td·(2 + 2T/Br)`；按 IO 模型算的 flash「隐含带宽」最高 6.57 TB/s，是 DRAM 的 4.3 倍，说明 K/V 重读大多命中 96 MB L2。
- **不能推出**：「隐含带宽」和 L2 命中是推断，没有 ncu 计数器直接证明；也不能说 2 倍流量模型错了，它只是只数了 S、P 两个张量。
- **追问**：为什么反向慢？dK/dV 与 dQ 拆成两个 kernel 避免了 atomic，但各自重算 S；FA2 用一个 kernel 加 dQ 的 atomic 累加，少两次矩阵乘。

## 2. 推测解码：验证免费与否决定一切

**20 秒**：同一套接受-拒绝采样，在 CPU 小模型和 GPU 真实 GPT-2 上各做一遍。每次目标前向产出的 token 数
都和 `(1−α^{γ+1})/(1−α)` 吻合（差 ≤4%）。CPU 上验证 γ+1 个 token 要单 token 前向的 1.41–1.61 倍，加速只有
0.98–1.99 倍；RTX 5090 上 gpt2-xl 验证 4 个 token 只贵 11%（v=1.106），gpt2→xl 贪心实测 1.631 倍，与
`E/(γc+v)` 的预测差 3%。

- **S**：CS229S Project 2；GPU 部分是自写的 GPT-2 forward，直接读 HF 权重。
- **T**：实现 Leviathan/Chen 的接受-拒绝采样，验证分布无偏，测加速，并解释加速从哪来。
- **A**：
  - 接受概率 `min(1, p/q)`，拒绝时从 `norm(max(0, p−q))` 重采样。
  - KV 回滚就是把 `length` 调小。
  - 接受-拒绝全在 GPU 上算，CUDA Graph decode，每轮只同步一次。
- **R**：
  - 正确性：fp32 下贪心 120/120 与 target 单独解码一致；采样第一个 token 对 target 分布的卡方 p=0.657，对 draft 分布 p=8.9e-168。
  - 选 draft：更大的 draft（gpt2-medium）α 更高（0.77 对 0.71），但 c 翻倍到 0.308，加速只有 1.283。
  - α 的口径：最初用「接受数 / 提议数」，γ=8 时把 0.80 算成 0.41，要改成「接受数 /（接受数 + 拒绝数）」。
- **白板**：F17，`S = E / (γc + v)`；GPU decode 受带宽限制所以 v≈1，c 却远大于参数量之比（0.137 对 0.08），因为 per-kernel 固定开销。
- **不能推出**：大 batch 的结论是用实测 c(B)、v(B) 代入公式拼出来的预测（B=128 时只剩 1.005 倍），`SpecEngine` 只实现了 batch 1。
- **追问**：为什么最优 γ 只有 3–4？E 随 γ 饱和于 1/(1−α)，分母 γc 线性增长；α≈0.7 时 γ>4 收益为负。

## 3. 分页 KV 与 chunked prefill：真实引擎上的账

**20 秒**：先用 H100 roofline 模型做调度模拟，再在 RTX 5090 上用 Qwen2.5-0.5B 写了一个真实引擎（分页 KV +
iteration-level 调度 + chunked prefill + Triton 分页 decode kernel）。

- 分页：4 GiB 预算下按 4096 预留浪费 86.8%（并发 85），分页 block=16 浪费 1.26%（并发 593，7.0 倍），与模型差不到 1.2 个百分点。
- 间接寻址：随机物理块和按序块表的耗时差 <1%。
- 模型漏掉的东西：小模型 decode 先卡在 CPU 派发。eager 每步 8.2–9.0 ms，CUDA graph 2.16 ms，带宽下限 0.65 ms。

- **S**：15-849 服务方向（vLLM、Orca、Sarathi）。
- **T**：分页和调度各自带来什么，模型哪里不准。
- **A**：块表、引用计数、copy-on-write fork、swap/recompute 抢占；分页 decode kernel 比 torch gather 快 2.2–10.7 倍；static / continuous / chunked 三种策略扫到期望负载。
- **R**：
  - static 饱和于 5.36 req/s；continuous 在 λ=40 时 23.2 req/s，TTFT P99 38 ms。
  - 10% 长 prompt（3072 token）时，chunked prefill 把逐请求最大卡顿的 P99 从 33.5 ms 降到 17.6 ms，代价是长 prompt 的 TTFT 从 34 ms 变成 110 ms，吞吐 −2.6%。
  - 拟合得到引擎迭代时间 = 9.15 ms + 1.03 × roofline：模型的斜率是对的，但漏了约 9 ms 的固定开销。
- **白板**：每条序列浪费 ≤ block−1 个 token（平均 (b−1)/2）；decode 步长 ≈ `max((权重 + KV) / 带宽, 派发开销)`。
- **不能推出**：0.5B 模型的派发瓶颈不代表 8B/70B；lab 的短 prompt 分布下 chunked prefill 没有收益，结论只在长 prompt 混合负载下成立。
- **追问**：chunked prefill 为什么依赖 attention 实现？带前缀的 prefill 块需要非方阵因果 mask；布尔掩码让 SDPA 退回 math 路径，每块多 5.8 ms，收益直接抵消。

## 4. 训练显存的账：逐字节对上

**20 秒**：单卡 RTX 5090 上把训练显存模型逐项对上实测。

- 16 B/参数：fp32、AMP、bf16+fp32 master 三种布局都是 16.00–16.32；纯 bf16 是 8.0–8.3。
- 激活：要按 flash attention 真实保存的张量计数（AMP 36 字节/sbh），误差才降到 1% 以内。
- ZeRO 0–3：用真实扁平分片测每 rank 显存，与公式最大差 0.099%，分片 AdamW 与不分片逐位相等。

- **S**：15-849 分布式训练（ZeRO、Megatron、GPipe/1F1B、重计算）。
- **T**：哪些显存公式能直接用，哪些需要修正。
- **A**：自写 GPT-2 + checkpoint 模式；b×s 激活扫描；4 个 stage 在同一张卡上按 1F1B/GPipe 的顺序执行真实 kernel，用 CUDA event 计时后按依赖重放。CPU 部分用集合通信模拟器验证了 ZeRO-3 通信恰好是 DP 的 1.5 倍。
- **R**：
  - MFU：774M 纯 bf16 70.2%。
  - 重计算：全量重计算步时间 ×1.189（理论 4/3），最大 batch 从 8 到 24。
  - 1F1B：m=8/16/32 时气泡 0.343/0.245/0.233，峰值显存固定在 7.64 GB；GPipe m=16 OOM。
- **白板**：F10、F13、F14；气泡 (p−1)/(m+p−1) 假设各级等长。
- **不能推出**：多卡通信没测，通信时间是用实测 PCIe/D2D 带宽参数化的 α-β 模型；单卡上的「流水线」是串行重放。
- **追问**：1F1B 气泡为什么停在 0.23 而不是 0.086？最后一级多了 LM head，各级不均衡；按实测各级时间的不均衡模型给出 0.354/0.254/0.203，和实测吻合。

## 5. 融合的收益就是字节比

**20 秒**：访存受限链上 `torch.compile` 的加速与 eager / 融合后的字节比相差不到 7%：RMSNorm 9.86 倍（字节比 10.0），
softmax 5.05 倍，残差链 2.68 倍，融合后都顶在约 1.53 TB/s 的带宽墙上。MLP 块里两个大 GEMM 受算力限制，
max-autotune 只省 15%（419 → 357 µs），和我自写图 IR 的 roofline 预测（435.7 → 338.1 µs）一致。

- **A**：CPU 部分是自写的图 IR，包括常量折叠、CSE、TASO 式替换、逐元素与 epilogue 融合；GPU 部分是 Inductor 对照、Triton matmul 的 192 个配置网格和代价模型排序。
- **R**：
  - 8192³ 时网格最优 221.5 TFLOP/s，超过 cuBLAS 的 214.3。
  - 代价模型：复用型模型在方阵上 Spearman 0.65，在 M=256 时只有 0.10；波次 roofline 正好相反。这也是 Ansor 用学习型模型加实测的原因。
- **白板**：k 个逐元素算子融合后流量从 `2kS` 到 `2S`；时间 = `max(F/P, B/BW)`。
- **不能推出**：小尺寸时「带宽」超过 GDDR7 峰值，是 L2 命中，不能当 DRAM 带宽引用。
- **追问**：工程代价？max-autotune 的 CUDA graph 输入拷贝让 RMSNorm 慢 2.08 倍；10 个序列长度静态编译 8 个图（4.46 s），`mark_dynamic` 只要 1 个图（0.54 s）。

## 6. 量化：位宽越低越需要数据

**20 秒**：GPT-2 124M（fp16 ppl 29.96）上：

- W4 g128：RTN 33.27 → GPTQ 31.20。
- W3：RTN 65.92 → AWQ 式 53.35 → GPTQ 37.24。
- W8A8：per-tensor 49.94，per-token 30.18，LLM.int8 30.06；SmoothQuant α=0.5 爆到 407.25，α=0.8 是 30.49。

速度上，batch 1 的 weight-only GEMV 按字节加速：W8 82.5 µs、W4 62.2 µs（bf16 163.0 µs）；先反量化再 matmul 要 3688 µs。

- **A**：absmax / per-channel / per-group 量化，LLM.int8 混合分解，SmoothQuant，GPTQ，AWQ 式缩放；CPU 部分用合成离群通道讲机制（8/1024 个 50 倍离群让 per-tensor 误差 2.3% → 10.2%）。
- **R**：大 GEMM 上 bf16 226.9、FP8 456.2、int8 679.4 TOP/s；剪枝 50% 时幅值剪枝 ppl 538，Wanda 47.2；2:4 稀疏大 GEMM 1.54 倍。
- **白板**：F19，`scale = max|x|/127`；GPTQ 用校准数据的 Hessian `XᵀX` 逐列补偿误差。
- **不能推出**：W8A8/SmoothQuant/LLM.int8 是精度模拟（fp16 GEMM），没有端到端 int8 速度；GPT-2 的离群远弱于 6.7B 以上的模型。
- **追问**：为什么 SmoothQuant 的 α 这么敏感？它把激活的难度按 `s = max|X|^α / max|W|^{1−α}` 挪给权重；α 太小，激活的离群值没被压下去，per-tensor int8 仍被它主导。

## 7. 保序渲染

**20 秒**：半透明圆的混合不满足交换律。按圆并行、无序写入时 2000 个圆的场景 93% 像素出错。按 16×16 块
并行，每块用 exclusive scan 压缩出保序的圆列表再按顺序着色，与顺序参考逐位相等。CUDA 版 rand1M 上，
「每像素遍历所有圆」的正确版本 844 ms，分桶后测试少 1640 倍，0.538 ms。

- **A**：
  - 三种 scan：Blelloch、Hillis–Steele、cub。n=2²⁰ 时 Blelloch 2.10M 次加法，Hillis–Steele 19.9M 次。
  - CUDA scan 2²⁸：朴素 16.75 ms、自写 2.82 ms、cub 1.41 ms。
  - 渲染器关闭 FMA 编译，与 CPU 参考逐位一致。
- **R**：
  - naive 版本有两种错：rgb 的三个圆在同一个 warp 里，结果可复现但顺序错（24.3% 像素错）；其余场景有读写竞争，两次运行结果不同。
  - shared memory 批量取圆带来 1.7–2.1 倍。
  - micro2M（半径约 1 像素）用 T=8 最快。
- **不能推出**：occupancy 是按寄存器数算的理论值（52 → 34 个寄存器，0.67 → 1.0），没有计数器。
- **追问**：用 atomic 行不行？原子操作解决写冲突，不保证顺序，结果仍然错。

## 8. 一条慢 lane 拖垮 SIMD

**20 秒**：sqrt 每 8 个元素放 1 个慢的，AVX2 版只有标量的 0.72 倍。AVX-512 上 all_slow、one_slow_8、
one_slow_16 的耗时完全相同（96.8 / 96.8 / 97.3 ms）：一个向量的代价由最慢的 lane 决定。GPU 上也一样，
每 32 个混 1 个慢的和全慢一样贵。

- **S**：CS149 A1，六个 program 用 C++17 + AVX2/AVX-512 intrinsics 重写，另用 CUDA 回答同样的问题。
- **A**：Mandelbrot 三种划分（连续块、隔行、动态）；SMT 与单线程交错两组向量的对照；saxpy 带宽与 streaming store；K-means 的 Amdahl 分析。
- **R**：
  - 负载均衡：12 线程连续块 5.89 倍，动态分块 11.1 倍；3 线程比 2 线程还慢，因为中间那块独揽集合主体。
  - 延迟隐藏：单线程交错两组 16 lane 向量，从 11.2 倍提到 18.3 倍，超过向量宽度，说明内层是依赖链。
  - 带宽：saxpy 单线程 DRAM 23 GB/s、12 线程 173 GB/s。
  - GPU：Mandelbrot 比 12 核版本再快 60–100 倍；但 saxpy 一旦过 PCIe，端到端只有 52 GB/s，比 CPU 还慢。
- **白板**：利用率 = 有效 lane·迭代 / (W × 组内最大迭代数)；numpy 模型给出 8 宽一条慢 lane 的利用率 0.163。
- **不能推出**：服务器是共享的 cgroup 环境，结论来自 3 轮独立实验的中位数；刚唤醒的核上计时慢 18%–23% 的现象还没有完全解释（见报告局限）。
- **追问**：GPU 上怎么办？按工作量排序或分桶，让同一个 warp 里的线程工作量接近；或者用 persistent kernel 持续拉取工作。

## 9. spin 还是 sleep：线程池的账

**20 秒**：同一个任务系统 API 的四种 C++ 实现，12 个物理核。单个空任务的 launch 开销：

- spin 线程池 0.78 µs，sleep 线程池 10.8 µs，每次新建线程 53.3 µs。
- 但空闲的 spin 池烧掉 11.96 个核。
- 任务一多，12 个线程抢同一个 CAS，每个空任务 494 ns，是 sleep 的 7 倍。
- 2 倍超订时，带锁的 spin 调度器慢 766 倍。

- **A**：Part B 的 `runAsyncWithDeps` + `sync`；20 个官方测试按结构重写并校验结果；3 轮独立实验 × 每轮 3 次，取中位数。
- **R**：
  - async 依赖图让独立 launch 互相填满尾部，快 1.64 倍（理想 1.83 倍）。
  - 同一 cache line 上的原子计数器比各占一行慢 19 倍。
  - 12 个 worker 绑满 12 个核时，自旋等待的调用者要和 worker 分时，出现 32 ms 的时间片平台；减到 11 个 worker 后是 18.7 ms。
- **白板**：任务间隔短于一次唤醒用 spin，否则 sleep；spin 的前提是自旋线程不和干活的线程抢核。
- **不能推出**：Python 版（`threading`）里 spin 反而最慢（31–50 ms），那是 GIL 的结论，不是线程池的结论。
- **追问**：生产里怎么选？先 spin 一小段再 park（自适应自旋），并让 worker 数比可用核少一个给调用者。

## 10. SGEMM 阶梯：每一步都能算出来

**20 秒**：RTX 5090 上 4096³ SGEMM 的阶梯：naive 0.94 → coalesced 6.43 → smem 9.82 → 1D 寄存器分块 25.7 →
2D 寄存器分块 21.5（反而变慢）→ float4 / LDS.128 49.4 TFLOP/s（cuBLAS fp32 63.1 的 78%）。WMMA fp16 177.8 TFLOP/s（cuBLAS fp16 的 86%）。

- **A**：CPU 部分是加速器上的分块 traffic/cycle 模型（SBUF/PSUM 容量约束）和寄存器级脉动阵列；GPU 部分是 SGEMM 阶梯、tile 热力图、im2col 卷积、skinny GEMM 和 roofline。
- **R**：
  - 分块模型：GPU 上用和 CPU 部分完全相同的流量公式。8192³ 时工作集远超 L2，按公式反推的 DRAM 带宽 1585 GB/s，与实测 triad 1549 GB/s 只差 2.3%。
  - 2D 分块为什么变慢：解析 bank 模型显示有 4-way / 8-way 冲突，但消掉冲突只快 2.5%；真正的跳变来自把 16 条 LDS.32 换成 4 条 LDS.128（2.3 倍）。
  - 双缓冲：第一次编译用了 129 个寄存器，每 SM 只放得下 1 个 block，反而慢 6.6%；压到 128 个寄存器后回到 49.44。
- **白板**：方阵 GEMM 强度 `n/3`（bf16）；分块后流量 ≈ `b·(MK·⌈N/BN⌉ + KN·⌈M/BM⌉ + MN)`；强度 ≈ tile 边长的一半，直到 PSUM / 寄存器卡住。
- **不能推出**：bank conflict 次数来自解析模型，没有 ncu 计数器；Trainium 参数是模型假设，不是规格表。
- **追问**：为什么剩下 22% 追不上 cuBLAS？cuBLAS 用更深的软件流水线和按形状选择的 kernel；我的双缓冲只多了 0.1%，说明瓶颈已经不是访存延迟，而是指令发射与寄存器压力。

## 11. 大图 BFS：方向比线程数重要

**20 秒**：RMAT（Graph500 参数，100 万顶点、800 万边）上，Beamer 的 direction-optimizing BFS 在 frontier 变大时
切到 bottom-up：检查的边只有 top-down 的 9.1%，8 线程 6.58 倍（7.47 对 49.18 ms）。frontier 最大的那一层
bottom-up 只看 8.6 万条边，top-down 要看 389 万条。图放大到 6619 万边时 12.2 倍。网格和链上每层 frontier
只有几十个顶点，4 线程比串行慢约 100 倍。

- **A**：top-down（critical / CAS + 共享下标 / CAS + 线程私有 frontier）、bottom-up（无原子）、hybrid（α 阈值、只在 frontier 增长时切换）；PageRank pull 与 push。
- **R**：
  - 共享原子下标比线程私有 frontier 慢 1.90 倍。
  - PageRank pull 7.03 倍，push（atomic add）4.22 倍。
  - α 从 2 扫到 64，时间都在 9.21–9.85 ms，切换层不变。
  - 规模：RMAT 从 2^16 到 2^23，hybrid 8 线程从 1.33 倍涨到 12.2 倍；串行每条边从 2.6 ns 涨到 10.0 ns（出了缓存，多线程能同时挂起多个缺失）。超过 8 倍的部分是算法省的边，同样工作量的 top-down 是 1.02 → 4.78 倍。
  - 重复性：三个种子各 7 次，中位加速 6.56–6.94 倍。
- **白板**：切换条件 `E_td = Σ outdeg(frontier) > E_bu ≈ unvisited · m/n / α`。
- **不能推出**：4 核 8 线程的本机结果，没有 NUMA，也没有在十亿边规模上验证。
- **追问**：星图怎么办？我实现了按边区间切分：4 线程 33.0 → 28.3 ms，负载均衡了，但串行只要 10.4 ms。1 线程的并行版本已经是串行的 3.3 倍（每个顶点一次 CAS、私有数组、层末拼接），这张图每个顶点的工作太少，不值得并行。

## 12. 伪加速：21 倍

**20 秒**：第一版 Mandelbrot 测出 8 个 worker 21 倍加速，超过核数。原因是我按 worker 数切块，块变小后
numpy 的中间数组落进了缓存，同一张图串行算要 3.65 s、按小块算只要 1.29 s。把所有调度的块大小统一后，
加速回到合理范围。

- **白板**：超线性加速先怀疑基线：基线和并行版是否做了同样的工作、同样的访存模式。
- **不能推出**：不是「多进程有超线性加速」，是基线不公平。
- **追问**：真实的超线性存在吗？存在，例如切分后每个核的数据刚好放进自己的缓存；但要在单核上用同样的块大小复现基线后才能这么说。
