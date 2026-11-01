# 术语与符号表

## 符号

| 符号 | 含义 | 典型值 |
| --- | --- | --- |
| `N` | 模型参数量 | 8×10⁹ |
| `D` | 训练 token 数 | 1.5×10¹³ |
| `L` | 层数 | 32 |
| `d` / `d_model` | 隐藏维度 | 4096 |
| `d_head` | 每个注意力头的维度 | 128 |
| `h` / `n_kv` | 注意力头数 / KV 头数 | 32 / 8 |
| `T` | 序列长度 | 8192 |
| `B` | batch（序列数） | 32 |
| `V` | 词表大小 | 128256 |
| `s` | 每元素字节数 | bf16 为 2 |
| `p` | 并行度（卡数、stage 数）或可并行比例（Amdahl） | 8 |
| `m` | 流水线 micro-batch 数 | 32 |
| `M` | 快速存储容量（SRAM/共享内存） | 228 KB/SM |
| `α`, `β` | 通信启动延迟、带宽（α-β 模型）；推测解码里 α 是接受率 | — |
| `γ` | 推测解码每轮草稿 token 数 | 4 |
| `I` | 算术强度（FLOP/byte） | — |

## 术语

| 术语 | 一句话 | 讲 |
| --- | --- | --- |
| Amdahl 定律 | 串行比例 `1−p` 决定加速上限 `1/(1−p)` | [M01](module-01-performance-models.md) |
| Gustafson 定律 | 问题规模随资源增长时的加速比 `(1−p)+p·n` | [M01](module-01-performance-models.md) |
| roofline | 性能上限 = min(峰值算力, 算术强度 × 带宽) | [M01](module-01-performance-models.md) |
| 算术强度（arithmetic intensity） | 每字节访存对应的 FLOP 数；低于拐点受访存限制 | [M01](module-01-performance-models.md) |
| 拐点（ridge point） | 峰值算力 / 带宽；H100 约 295 FLOP/B | [M01](module-01-performance-models.md) |
| ILP | 指令级并行，同一线程内相互独立的指令同时发射 | [M02](module-02-multicore-simd-threads.md) |
| SIMD | 一条指令作用于多个数据 lane；AVX2 为 8 个 fp32 | [M02](module-02-multicore-simd-threads.md) |
| 发散（divergence） | 同一 SIMD 组/warp 的 lane 走不同分支或迭代次数不同，利用率下降 | [M02](module-02-multicore-simd-threads.md) |
| 硬件多线程 | 用多份线程上下文在等待内存时切换，隐藏延迟 | [M02](module-02-multicore-simd-threads.md) |
| ISPC gang | ISPC 中以 SIMD 宽度一起执行的一组程序实例 | [M02](module-02-multicore-simd-threads.md) |
| work–span | 总工作量 T1 与关键路径 T∞；并行度 = T1/T∞ | [M03](module-03-parallel-programming-scheduling.md) |
| work stealing | 空闲线程从别的线程队列另一端偷任务的动态调度 | [M03](module-03-parallel-programming-scheduling.md) |
| 线程池 spin / sleep | 空闲 worker 忙等或在条件变量上休眠 | [M03](module-03-parallel-programming-scheduling.md) |
| cache coherence | 保证多核缓存中同一地址的写对其它核可见的协议（MSI/MESI） | [M04](module-04-locality-coherence-synchronization.md) |
| false sharing | 不同核写同一 cache line 的不同变量，导致该行来回失效 | [M04](module-04-locality-coherence-synchronization.md) |
| memory consistency | 不同地址的读写在其它核看来以什么顺序发生（SC、TSO、relaxed） | [M04](module-04-locality-coherence-synchronization.md) |
| CAS / ABA | 比较并交换；值被改回原值导致 CAS 误判成功 | [M04](module-04-locality-coherence-synchronization.md) |
| SM | GPU 的流式多处理器，含 warp 调度器、寄存器、共享内存、tensor core | [M05](module-05-gpu-architecture-cuda.md) |
| warp | 32 个线程一起执行同一条指令的调度单位 | [M05](module-05-gpu-architecture-cuda.md) |
| occupancy | 每个 SM 上活跃 warp 数与上限之比，受寄存器和共享内存限制 | [M05](module-05-gpu-architecture-cuda.md) |
| memory coalescing | warp 内连续对齐的访问合并成少量内存事务 | [M05](module-05-gpu-architecture-cuda.md) |
| bank conflict | 同一 warp 访问共享内存同一 bank 的不同地址而被串行化 | [M05](module-05-gpu-architecture-cuda.md) |
| tensor core | 执行小矩阵乘加（MMA）的专用单元 | [M05](module-05-gpu-architecture-cuda.md) |
| scan（前缀和） | 输出每个位置之前（exclusive）或含自身（inclusive）的累计结果 | [M06](module-06-data-parallel-primitives.md) |
| segmented scan | 按段边界重置的前缀和，用于不规则数据 | [M06](module-06-data-parallel-primitives.md) |
| stream compaction | 用 scan 计算输出位置，把满足条件的元素紧凑写出 | [M06](module-06-data-parallel-primitives.md) |
| RDD / lineage | Spark 的不可变分布式数据集；靠记录变换历史重算丢失分区 | [M06](module-06-data-parallel-primitives.md) |
| tiling / blocking | 把计算切成能放进快存的块以提高数据复用 | [M07](module-07-matmul-hardware-aware.md) |
| im2col | 把卷积展开成矩阵乘，代价是输入被复制 K² 倍 | [M07](module-07-matmul-hardware-aware.md) |
| MFU | 模型 FLOPs 利用率 = 6N × tokens/s / 峰值 | [M07](module-07-matmul-hardware-aware.md) |
| online softmax | 分块维护行最大值和分母，一遍算出 softmax | [M08](module-08-attention-flashattention.md) |
| FlashAttention | 分块 + online softmax + 反向重算的精确 attention，不把 T² 矩阵写回 HBM | [M08](module-08-attention-flashattention.md) |
| MQA / GQA | 多个查询头共享一组（MQA）或若干组（GQA）K/V 头，缩小 KV cache | [M08](module-08-attention-flashattention.md) |
| systolic array | 数据在相邻 PE 间流动的阵列，TPU 的矩阵单元 | [M09](module-09-accelerators.md) |
| scratchpad | 软件显式管理的片上存储（Trainium 的 SBUF） | [M09](module-09-accelerators.md) |
| dataflow（weight/output stationary） | 加速器中让哪种数据驻留在 PE 里 | [M09](module-09-accelerators.md) |
| operator fusion | 把多个算子合成一个 kernel，省去中间结果的 HBM 读写 | [M10](module-10-ml-compilers.md) |
| schedule（Halide/TVM） | 与算法分离的循环变换描述：split、reorder、tile、vectorize | [M10](module-10-ml-compilers.md) |
| graph substitution（TASO） | 自动生成并验证的等价子图替换 | [M10](module-10-ml-compilers.md) |
| graph break | `torch.compile` 无法追踪而切分计算图的位置 | [M10](module-10-ml-compilers.md) |
| reverse-mode AD | 一次反传得到标量对全部输入的梯度 | [M11](module-11-frameworks-autodiff.md) |
| parameter server | 参数集中存放、worker push 梯度 pull 参数的架构 | [M11](module-11-frameworks-autodiff.md) |
| all-reduce | 所有 rank 得到全体输入之和；= reduce-scatter + all-gather | [M12](module-12-data-parallel-zero.md) |
| reduce-scatter / all-gather | 分片求和 / 分片收集 | [M12](module-12-data-parallel-zero.md) |
| ring all-reduce | 环形拓扑上的 all-reduce，每卡收发 2(p−1)/p·S | [M12](module-12-data-parallel-zero.md) |
| ZeRO | 在数据并行组内分片优化器状态（1）、梯度（2）、权重（3） | [M12](module-12-data-parallel-zero.md) |
| FSDP | PyTorch 的全分片数据并行，相当于 ZeRO-3 | [M12](module-12-data-parallel-zero.md) |
| tensor parallel | 把单层的矩阵按行/列切到多卡（Megatron） | [M13](module-13-tensor-pipeline-parallel.md) |
| sequence parallel | 在 LayerNorm/dropout 区域按序列切分激活 | [M13](module-13-tensor-pipeline-parallel.md) |
| pipeline parallel | 按层切成 stage，micro-batch 流水执行 | [M13](module-13-tensor-pipeline-parallel.md) |
| pipeline bubble | 流水线填充与排空的空闲比例 (p−1)/(m+p−1) | [M13](module-13-tensor-pipeline-parallel.md) |
| 1F1B | 稳态时交替执行一个前向一个反向的流水调度 | [M13](module-13-tensor-pipeline-parallel.md) |
| 3D 并行 | TP × PP × DP 组合 | [M13](module-13-tensor-pipeline-parallel.md) |
| rematerialization / 重计算 | 反向时重算前向激活以节省显存（gradient checkpointing） | [M14](module-14-memory-optimization.md) |
| loss scaling | fp16 训练时放大 loss 防止梯度下溢 | [M14](module-14-memory-optimization.md) |
| offload | 把优化器状态/参数放到 CPU 或 NVMe | [M14](module-14-memory-optimization.md) |
| prefill / decode | 处理整段 prompt / 逐 token 生成 | [M15](module-15-inference-kv-speculative.md) |
| KV cache | 缓存每层已生成 token 的 K、V，避免重算 | [M15](module-15-inference-kv-speculative.md) |
| TTFT / TPOT | 首 token 时延 / 每输出 token 时延 | [M15](module-15-inference-kv-speculative.md) |
| speculative decoding | 草稿模型提议、目标模型一次验证多个 token 的无损加速 | [M15](module-15-inference-kv-speculative.md) |
| continuous batching | 每个 decode 步都允许请求加入和离开（iteration-level scheduling） | [M16](module-16-llm-serving.md) |
| PagedAttention | 把 KV cache 按固定大小块分配并用块表寻址（vLLM） | [M16](module-16-llm-serving.md) |
| prefix caching | 复用相同前缀的 KV 块（RadixAttention） | [M16](module-16-llm-serving.md) |
| chunked prefill | 把长 prompt 切块与 decode 混合调度 | [M16](module-16-llm-serving.md) |
| PD 分离 | prefill 和 decode 放在不同实例上 | [M16](module-16-llm-serving.md) |
| goodput | 满足 SLO 的请求吞吐 | [M16](module-16-llm-serving.md) |
| quantization | 用低位整数或低精度浮点表示权重/激活 | [M17](module-17-compression-quantization.md) |
| per-channel / per-group | 每个输出通道 / 每 g 个元素一个 scale | [M17](module-17-compression-quantization.md) |
| SmoothQuant | 把激活的离群尺度按通道迁移到权重上 | [M17](module-17-compression-quantization.md) |
| GPTQ / AWQ | 基于二阶信息 / 激活重要性的权重量化 | [M17](module-17-compression-quantization.md) |
| 2:4 稀疏 | 每 4 个元素保留 2 个的结构化稀疏，可用 sparse tensor core | [M17](module-17-compression-quantization.md) |
| 知识蒸馏 | 学生模型拟合教师的软标签 | [M17](module-17-compression-quantization.md) |
| LoRA | 冻结 W，只训练低秩增量 BA | [M17](module-17-compression-quantization.md) |
| QLoRA | 4-bit NF4 基座 + LoRA | [M17](module-17-compression-quantization.md) |
| SSM / Mamba | 状态空间模型；递归推理每步 O(1) | [M18](module-18-architectures-moe-clusters.md) |
| MoE | 混合专家：每 token 只激活 top-k 个专家 | [M18](module-18-architectures-moe-clusters.md) |
| expert parallel / all-to-all | 专家分布在多卡上，token 通过 all-to-all 派发和收回 | [M18](module-18-architectures-moe-clusters.md) |
| gang scheduling | 分布式作业的所有 worker 同时调度 | [M18](module-18-architectures-moe-clusters.md) |
| scaling law | 损失随参数量、数据量、算力的幂律关系；Chinchilla 约 20 token/参数 | [M18](module-18-architectures-moe-clusters.md) |

## 易混对照

| A | B | 区别 |
| --- | --- | --- |
| 延迟（latency） | 吞吐（throughput） | 单个任务多快完成 vs 单位时间完成多少；batch 提高吞吐通常牺牲延迟 |
| 并发（concurrency） | 并行（parallelism） | 逻辑上同时存在 vs 物理上同时执行 |
| coherence | consistency | 同一地址的可见性 vs 不同地址间的顺序 |
| occupancy | 利用率 | 活跃 warp 比例 vs 实际吞吐占峰值比例 |
| ZeRO-3 | 张量并行 | 都切权重；ZeRO-3 计算前 all-gather 回完整权重，TP 始终只算本地分片 |
| 1F1B | 交错 1F1B | 前者省激活不减气泡；后者减气泡但多通信 |
| MFU | HFU | 是否把重计算的 FLOPs 算进去 |
| 推测解码（采样） | 推测解码（贪心） | 分布相同 vs 输出逐 token 相同 |
| weight-only 量化 | W8A8 | 只省带宽 vs 也能用整数 tensor core 省算力 |
