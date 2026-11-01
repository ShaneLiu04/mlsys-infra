# 十八讲一句话答案

每讲三问：定义题、计算题、取舍题。答案控制在一口气说完的长度，展开在对应笔记的「面试追问链」。

## 硬件与并行

**M01 性能模型** · [笔记](../notes/module-01-performance-models.md)
- 什么是 roofline？可达性能 = min(峰值算力, 算术强度 × 带宽)，拐点 = 峰值 / 带宽，H100 约 295 FLOP/B。
- 5% 串行的程序最多加速多少？1/0.05 = 20 倍，8 核时 5.9 倍。
- 一个 kernel 慢，先做什么？算它的算术强度，判断是受算力、访存还是开销限制，再决定优化方向。

**M02 多核、SIMD、多线程** · [笔记](../notes/module-02-multicore-simd-threads.md)
- 多核、SIMD、硬件多线程分别利用什么并行？独立指令流、同一指令的多个数据、等待内存时切换线程隐藏延迟。
- 8 核 AVX2 每周期峰值多少 FLOP？8 核 × 8 lane × 2 FMA 单元 × 2 = 256。
- saxpy 加核为什么不变快？每元素 2 FLOP 搬 12–16 字节，受内存带宽限制。

**M03 并行编程与调度** · [笔记](../notes/module-03-parallel-programming-scheduling.md)
- 并行化的四步？分解、分配、编排、映射。
- 贪心调度的时间上界？`T_P ≤ T1/P + T∞`。
- 静态还是动态分配？代价可预测用静态；不可预测用动态，任务粒度要远大于调度开销；work stealing 兼顾两者。

**M04 局部性、一致性与同步** · [笔记](../notes/module-04-locality-coherence-synchronization.md)
- coherence 和 consistency 的区别？同一地址的写何时可见 vs 不同地址的读写顺序。
- false sharing 怎么修？按 64 B cache line 填充，或每线程写私有副本最后归约。
- TTAS 比 TAS 好在哪？等锁时只读本地缓存，不产生总线写流量；释放时才有一轮失效。

**M05 GPU 与 CUDA** · [笔记](../notes/module-05-gpu-architecture-cuda.md)
- warp 是什么？32 个线程同时执行同一条指令的调度单位，分支发散时串行执行两个分支。
- 每线程 64 个寄存器、256 线程/block，H100 occupancy 多少？65536 / (64 × 256) = 4 个 block，32 warps / 64 = 50%。
- 按列读 32×32 共享内存为什么慢？32 个线程落在同一个 bank，32 路冲突；声明成 32×33 错开。

**M06 数据并行原语** · [笔记](../notes/module-06-data-parallel-primitives.md)
- Blelloch scan 的 work 和 span？work 2(n−1)，span 2 log n。
- stream compaction 怎么做？标记 → exclusive scan 得到写入位置 → scatter。
- `reduceByKey` 为什么比 `groupByKey` 好？先在 map 端合并，shuffle 的数据量小得多。

## 单卡算子与编译

**M07 矩阵乘与硬件感知** · [笔记](../notes/module-07-matmul-hardware-aware.md)
- 方阵 bf16 GEMM 的算术强度？n/3；H100 上 n ≳ 900 才受算力限制。
- 分块为什么有效？块大小 b 时每读一个元素做约 b 次 FLOP，总访存 Θ(n³/√M)。
- MFU 怎么算？6N × tokens/s / (卡数 × 峰值)。

**M08 Attention 与 FlashAttention** · [笔记](../notes/module-08-attention-flashattention.md)
- 单头 attention 前向 FLOPs？4T²d。
- FlashAttention 省了什么？不把 T×T 矩阵写回 HBM，显存 O(T)；FLOPs 不变，反向多一次重算；HBM 访问在 T 上仍是二次。
- GQA 改变了什么？不改 FLOPs，KV cache 和 decode 的 KV 读取按 `n_heads / n_kv` 缩小。

**M09 专用硬件** · [笔记](../notes/module-09-accelerators.md)
- 为什么要专用加速器？能耗主要花在搬数据而不是运算上，专用数据流让每次取数被复用更多次。
- N×N systolic array 处理 K 长的输入要多少周期？约 K + 2N − 2，利用率约 K/(K + 2N)。
- Trainium 编程和 CUDA 最大的不同？片上 SBUF/PSUM 由软件显式管理，用 DMA 搬运，没有硬件缓存兜底。

**M10 ML 编译器** · [笔记](../notes/module-10-ml-compilers.md)
- 算子融合省什么？中间结果的 HBM 读写和 kernel 发射；k 个逐元素算子流量从 2kS 降到 2S。
- 为什么调度要靠搜索？一个矩阵乘的分块、循环顺序组合有 10⁵–10⁶ 种，靠代价模型 + 搜索（Ansor）。
- `torch.compile` 的三段？Dynamo 抓图（带 guard），AOTAutograd 生成联合前反向图，Inductor 生成 Triton。

**M11 框架与自动微分** · [笔记](../notes/module-11-frameworks-autodiff.md)
- 为什么用反向模式？标量损失对 n 个参数，一次反传得到全部梯度，代价约 2–3 倍前向。
- 训练 FLOPs 为什么是 6ND？前向 2ND，反向要算输入梯度和权重梯度，各 2ND。
- BSP、ASP、SSP？同步、完全异步、允许最多落后 s 步。

## 训练系统

**M12 数据并行与 ZeRO** · [笔记](../notes/module-12-data-parallel-zero.md)
- ring all-reduce 每卡收发多少？2(p−1)/p × S，几乎与卡数无关。
- 7B、8 卡，ZeRO 0/1/2/3 每卡静态显存？112 / 38.5 / 26.25 / 14 GB。
- ZeRO-3 通信是 DP 的几倍？1.5 倍。

**M13 张量并行与流水线并行** · [笔记](../notes/module-13-tensor-pipeline-parallel.md)
- Megatron 每层几次 all-reduce？前向 2 次、反向 2 次。
- 8 级流水、32 个 micro-batch 的气泡？7/39 ≈ 18%。
- 3D 并行怎么排？TP 在节点内（NVLink），PP 跨节点，DP 在最外层。

**M14 显存优化** · [笔记](../notes/module-14-memory-optimization.md)
- 全量重计算多多少算力？多一次前向，约 +33%。
- 为什么 bf16 不需要 loss scaling？指数位和 fp32 一样多，梯度不会下溢。
- 选择性重计算重算什么？attention 里 T² 规模、计算便宜的 softmax 和 dropout 部分。

## 推理系统

**M15 KV cache 与推测解码** · [笔记](../notes/module-15-inference-kv-speculative.md)
- Llama-3-8B 每 token 的 KV？2 × 32 × 8 × 128 × 2 = 128 KB。
- batch=1 decode 的上限？读一遍权重：16 GB / 3.35 TB/s ≈ 4.8 ms，约 200 token/s。
- 推测解码 α=0.8、γ=4 每次目标前向几个 token？(1 − 0.8⁵)/0.2 = 3.36。

**M16 LLM 服务** · [笔记](../notes/module-16-llm-serving.md)
- PagedAttention 的浪费上限？每条序列最多 block_size − 1 个 token，没有外部碎片。
- continuous batching 改了什么？调度粒度从「一批请求」变成「一个 decode 步」。
- 为什么要 PD 分离？prefill 吃算力、decode 吃带宽，混跑互相干扰 TTFT 和 TPOT。

## 模型效率与集群

**M17 压缩与量化** · [笔记](../notes/module-17-compression-quantization.md)
- 对称 int8 的 scale？max|x| / 127，误差不超过 scale/2。
- 为什么 weight-only 量化加速 decode 不加速 prefill？decode 受带宽限制，省字节就省时间；prefill 受算力限制，还多了反量化。
- LoRA 的参数量？r(d + k)；d = k = 4096、r = 16 时是全量的 0.78%。

**M18 架构、MoE 与集群** · [笔记](../notes/module-18-architectures-moe-clusters.md)
- MoE 为什么参数多但每 token 不贵？每 token 只过 top-k 个专家；代价是全部专家的显存和两次 all-to-all。
- Chinchilla 的配比？约 20 token / 参数，C ≈ 6ND。
- 分布式训练调度为什么要 gang scheduling？所有 worker 必须同时在线，否则先到的空等。
