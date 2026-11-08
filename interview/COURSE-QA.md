# 十八讲一句话答案

每讲三个问题：一个定义题、一个计算题、一个取舍题。答案都控制在一口气能说完的长度。
面试官顺着问下去的时候，展开的内容在对应笔记的「面试追问链」里。

## 硬件与并行

**M01 性能模型** · [笔记](../notes/module-01-performance-models.md)
- 什么是 roofline？可达性能取峰值算力和「算术强度 × 带宽」中较小的那个。两者相等的点叫拐点，等于峰值除以带宽，H100 约 295 FLOP/B。
- 5% 串行的程序最多加速多少？核再多也只有 1/0.05 = 20 倍；8 核时是 5.9 倍。
- 一个 kernel 慢，先做什么？先算它的算术强度，判断它是受算力、访存还是开销限制，再决定往哪个方向优化。直接上手调参数是常见的错误做法。

**M02 多核、SIMD、多线程** · [笔记](../notes/module-02-multicore-simd-threads.md)
- 多核、SIMD、硬件多线程分别利用什么并行？多核跑独立的指令流；SIMD 让同一条指令处理多个数据；硬件多线程在一个线程等内存时切到另一个线程，把延迟藏起来。
- 8 核 AVX2 每周期峰值多少 FLOP？8 核 × 8 lane × 2 个 FMA 单元 × 2 = 256。
- saxpy 为什么加核不变快？因为每个元素只做 2 FLOP，却要搬 12–16 字节，瓶颈在内存带宽，加核不加带宽。

**M03 并行编程与调度** · [笔记](../notes/module-03-parallel-programming-scheduling.md)
- 并行化分哪四步？分解、分配、编排、映射。
- 贪心调度的时间上界是什么？`T_P ≤ T1/P + T∞`，也就是总工作量平摊到 P 个处理器上，再加一条关键路径。
- 静态分配还是动态分配？任务代价可预测就用静态；不可预测用动态，但任务粒度要远大于调度开销。work stealing 兼顾了两者。

**M04 局部性、一致性与同步** · [笔记](../notes/module-04-locality-coherence-synchronization.md)
- coherence 和 consistency 有什么区别？coherence 管同一个地址的写什么时候对别人可见；consistency 管不同地址之间读写的先后顺序。
- false sharing 怎么修？按 64 B 的 cache line 填充，让每个变量独占一行；或者每个线程写私有副本，最后再归约。
- TTAS 比 TAS 好在哪？等锁时只读本地缓存里的副本，不产生总线上的写流量；只有锁释放时才有一轮失效。

**M05 GPU 与 CUDA** · [笔记](../notes/module-05-gpu-architecture-cuda.md)
- warp 是什么？32 个线程同时执行同一条指令的调度单位。分支发散时，两个分支要串行地各执行一遍。
- 每线程 64 个寄存器、每 block 256 线程，H100 的 occupancy 是多少？寄存器文件只够放 65536 / (64 × 256) = 4 个 block，也就是 32 个 warp，占上限 64 的 50%。
- 按列读 32×32 的共享内存为什么慢？因为 32 个线程落在同一个 bank 上，形成 32 路冲突。把数组声明成 32×33，就能把它们错开。

**M06 数据并行原语** · [笔记](../notes/module-06-data-parallel-primitives.md)
- Blelloch scan 的 work 和 span 是多少？work 是 2(n−1)，span 是 2 log n。
- stream compaction 怎么做？先给要保留的元素打标记，对标记做 exclusive scan 得到每个元素的写入位置，再按位置 scatter。
- `reduceByKey` 为什么比 `groupByKey` 好？因为它先在 map 端把同一个 key 合并掉，shuffle 时要传的数据量小得多。

## 单卡算子与编译

**M07 矩阵乘与硬件感知** · [笔记](../notes/module-07-matmul-hardware-aware.md)
- 方阵 bf16 GEMM 的算术强度是多少？n/3。所以在 H100 上 n 要 ≳ 900 才会受算力限制。
- 分块为什么有效？块大小为 b 时，每读一个元素能做约 b 次 FLOP，总访存降到 Θ(n³/√M)。
- MFU 怎么算？6N × tokens/s，除以卡数 × 单卡峰值。

**M08 Attention 与 FlashAttention** · [笔记](../notes/module-08-attention-flashattention.md)
- 单头 attention 前向多少 FLOPs？4T²d。
- FlashAttention 省了什么？它不把 T×T 矩阵写回 HBM，所以显存降到 O(T)。FLOPs 没变，反向还多一次重算；HBM 访问量在 T 上仍然是二次的。说它「减少了计算」是常见的错答。
- GQA 改变了什么？FLOPs 不变。变的是 KV cache 和 decode 时的 KV 读取量，都按 `n_heads / n_kv` 缩小。

**M09 专用硬件** · [笔记](../notes/module-09-accelerators.md)
- 为什么要专用加速器？因为能耗主要花在搬数据上，而不是运算本身。专用的数据流让每次取来的数被复用更多次。
- N×N 的 systolic array 处理长度为 K 的输入要多少周期？约 K + 2N − 2，利用率约 K/(K + 2N)。
- Trainium 编程和 CUDA 最大的不同是什么？片上的 SBUF 和 PSUM 由软件显式管理，数据靠 DMA 搬运，没有硬件缓存兜底。

**M10 ML 编译器** · [笔记](../notes/module-10-ml-compilers.md)
- 算子融合省什么？省中间结果的 HBM 读写和 kernel 发射。k 个逐元素算子融合后，流量从 2kS 降到 2S。融合不会减少 FLOPs。
- 为什么调度要靠搜索？一个矩阵乘的分块和循环顺序组合起来有 10⁵–10⁶ 种，人挑不过来，所以要用代价模型加搜索，Ansor 就是这么做的。
- `torch.compile` 分哪三段？Dynamo 抓图并加上 guard；AOTAutograd 生成前向和反向的联合图；Inductor 生成 Triton 代码。

**M11 框架与自动微分** · [笔记](../notes/module-11-frameworks-autodiff.md)
- 为什么用反向模式？损失是一个标量，参数有 n 个，反向模式一次反传就能拿到全部梯度，代价约是前向的 2–3 倍。
- 训练 FLOPs 为什么是 6ND？前向 2ND；反向要分别算对输入的梯度和对权重的梯度，各 2ND。
- BSP、ASP、SSP 分别是什么？BSP 每步全体同步；ASP 完全异步；SSP 允许最快的 worker 最多领先 s 步。

## 训练系统

**M12 数据并行与 ZeRO** · [笔记](../notes/module-12-data-parallel-zero.md)
- ring all-reduce 每卡收发多少数据？2(p−1)/p × S，几乎与卡数无关。
- 7B 模型、8 卡，ZeRO 0/1/2/3 每卡的静态显存？依次是 112 / 38.5 / 26.25 / 14 GB。
- ZeRO-3 的通信是 DP 的几倍？1.5 倍，因为多了一次前向的 all-gather。常见的错答是 3 倍。

**M13 张量并行与流水线并行** · [笔记](../notes/module-13-tensor-pipeline-parallel.md)
- Megatron 每层几次 all-reduce？前向 2 次，反向 2 次。
- 8 级流水、32 个 micro-batch，气泡多大？7/39 ≈ 18%。
- 3D 并行怎么排？TP 放在节点内走 NVLink，PP 跨节点，DP 在最外层。

**M14 显存优化** · [笔记](../notes/module-14-memory-optimization.md)
- 全量重计算多花多少算力？多一次前向，约 +33%，而不是慢一倍。
- 为什么 bf16 不需要 loss scaling？它的指数位和 fp32 一样多，梯度不会下溢。
- 选择性重计算重算什么？attention 里规模是 T²、但计算很便宜的 softmax 和 dropout 部分。

## 推理系统

**M15 KV cache 与推测解码** · [笔记](../notes/module-15-inference-kv-speculative.md)
- Llama-3-8B 每个 token 的 KV 多大？2 × 32 × 8 × 128 × 2 = 128 KB。
- batch=1 的 decode 速度上限是多少？每步至少要把权重读一遍：16 GB / 3.35 TB/s ≈ 4.8 ms，约 200 token/s。
- 推测解码 α=0.8、γ=4 时，每次目标模型前向产出几个 token？(1 − 0.8⁵)/0.2 = 3.36。

**M16 LLM 服务** · [笔记](../notes/module-16-llm-serving.md)
- PagedAttention 的浪费上限是多少？每条序列最多浪费 block_size − 1 个 token，而且没有外部碎片。
- continuous batching 改了什么？调度的粒度从「一批请求」变成了「一个 decode 步」。
- 为什么要 PD 分离？prefill 吃算力，decode 吃带宽，混在一起跑会互相干扰 TTFT 和 TPOT。

## 模型效率与集群

**M17 压缩与量化** · [笔记](../notes/module-17-compression-quantization.md)
- 对称 int8 的 scale 怎么取？max|x| / 127，量化误差不超过 scale/2。
- weight-only 量化为什么加速 decode 却不加速 prefill？decode 受带宽限制，省下的字节直接变成省下的时间；prefill 受算力限制，权重还得先反量化，反而多了一步。
- LoRA 有多少参数？r(d + k)。d = k = 4096、r = 16 时是全量的 0.78%。

**M18 架构、MoE 与集群** · [笔记](../notes/module-18-architectures-moe-clusters.md)
- MoE 为什么参数多但每个 token 不贵？每个 token 只经过 top-k 个专家。代价是全部专家都要占显存，每层还有两次 all-to-all。
- Chinchilla 的配比是多少？约 20 个 token 对应 1 个参数，总算力 C ≈ 6ND。
- 分布式训练调度为什么要 gang scheduling？所有 worker 必须同时在线，否则先启动的只能空等。
