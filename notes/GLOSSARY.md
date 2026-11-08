# 术语与符号表

读笔记时遇到不认识的符号或术语，可以先在这里查一句话的解释，再顺着最后一列找到详细讲它的那一讲。

## 符号

各讲的公式尽量统一使用下面这些符号。「典型值」大多取自 Llama-3-8B 这一量级的模型，方便做白板估算时直接代入。

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

注意 `p` 和 `α` 各有两种用法，要看上下文：`p` 在 Amdahl 定律里是可并行的比例，在并行训练里是卡数或 stage 数；`α` 在通信模型里是启动延迟，在推测解码里是接受率。

## 术语

术语按它首次出现的那一讲排序，所以从上往下读，大致就是从硬件一路到模型的顺序。

| 术语 | 一句话 | 讲 |
| --- | --- | --- |
| Amdahl 定律 | 程序里只能串行的那部分（比例 `1−p`）决定了加速的上限：核再多，加速比也超不过 `1/(1−p)` | [M01](module-01-performance-models.md) |
| Gustafson 定律 | 如果资源变多时问题规模也跟着变大，加速比就是 `(1−p)+p·n`，不再被串行部分卡死 | [M01](module-01-performance-models.md) |
| roofline | 一个程序的性能上限是峰值算力和「算术强度 × 带宽」中较小的那个，即 min(峰值算力, 算术强度 × 带宽) | [M01](module-01-performance-models.md) |
| 算术强度（arithmetic intensity） | 每读写一个字节的内存，做多少次浮点运算；低于拐点时，程序被访存卡住 | [M01](module-01-performance-models.md) |
| 拐点（ridge point） | 峰值算力除以带宽，是「受带宽限制」和「受算力限制」的分界线；H100 约 295 FLOP/B | [M01](module-01-performance-models.md) |
| ILP | 指令级并行：同一个线程里互不依赖的指令，硬件可以同时发射 | [M02](module-02-multicore-simd-threads.md) |
| SIMD | 一条指令同时处理多个数据 lane；AVX2 一次处理 8 个 fp32 | [M02](module-02-multicore-simd-threads.md) |
| 发散（divergence） | 同一 SIMD 组或 warp 里的 lane 走了不同分支、或循环次数不同，一部分 lane 只能空等，利用率随之下降 | [M02](module-02-multicore-simd-threads.md) |
| 硬件多线程 | 核里保存多份线程上下文，一个线程等内存时就切到另一个，用这种方式隐藏访存延迟 | [M02](module-02-multicore-simd-threads.md) |
| ISPC gang | ISPC 里按 SIMD 宽度绑在一起执行的一组程序实例 | [M02](module-02-multicore-simd-threads.md) |
| work–span | 用总工作量 T1 和关键路径长度 T∞ 描述一个并行程序，两者之比 T1/T∞ 就是它的并行度 | [M03](module-03-parallel-programming-scheduling.md) |
| work stealing | 一种动态调度：线程闲下来时，从别的线程任务队列的另一端偷任务来做 | [M03](module-03-parallel-programming-scheduling.md) |
| 线程池 spin / sleep | 空闲 worker 的两种等法：spin 是忙等着不停检查有没有活，sleep 是在条件变量上睡下、等人唤醒 | [M03](module-03-parallel-programming-scheduling.md) |
| cache coherence | 多个核各有缓存时，保证一个核对某个地址的写能被其他核看到的协议（MSI/MESI） | [M04](module-04-locality-coherence-synchronization.md) |
| false sharing | 不同核写的是不同变量，但它们恰好落在同一个 cache line 上，于是这一行在核之间来回失效 | [M04](module-04-locality-coherence-synchronization.md) |
| memory consistency | 对不同地址的读写，在其他核看来按什么顺序发生（SC、TSO、relaxed） | [M04](module-04-locality-coherence-synchronization.md) |
| CAS / ABA | CAS 是「比较并交换」的原子操作；ABA 指值被改走又改回原值，CAS 看不出来，误以为没人动过 | [M04](module-04-locality-coherence-synchronization.md) |
| SM | GPU 的流式多处理器，大致相当于 GPU 的一个「核」，里面有 warp 调度器、寄存器、共享内存和 tensor core | [M05](module-05-gpu-architecture-cuda.md) |
| warp | GPU 的调度单位：32 个线程一起执行同一条指令 | [M05](module-05-gpu-architecture-cuda.md) |
| occupancy | 一个 SM 上同时活跃的 warp 数占上限的比例；每个线程用的寄存器和共享内存越多，它越低 | [M05](module-05-gpu-architecture-cuda.md) |
| memory coalescing | 一个 warp 的线程访问连续且对齐的地址时，硬件把这些访问合并成少数几次内存事务 | [M05](module-05-gpu-architecture-cuda.md) |
| bank conflict | 共享内存分成多个 bank，同一 warp 的线程访问同一 bank 的不同地址时，只能排队一个个来 | [M05](module-05-gpu-architecture-cuda.md) |
| tensor core | 专门做小矩阵乘加（MMA）的硬件单元 | [M05](module-05-gpu-architecture-cuda.md) |
| scan（前缀和） | 每个位置输出它之前所有元素的累计结果（exclusive），或者把自己也算进去（inclusive） | [M06](module-06-data-parallel-primitives.md) |
| segmented scan | 遇到段边界就重新开始累计的前缀和，用来处理长短不一的数据 | [M06](module-06-data-parallel-primitives.md) |
| stream compaction | 先用 scan 算出每个要保留的元素该写到哪里，再把它们紧凑地写出来 | [M06](module-06-data-parallel-primitives.md) |
| RDD / lineage | RDD 是 Spark 的不可变分布式数据集；lineage 是它记下的变换历史，分区丢了就按历史重算 | [M06](module-06-data-parallel-primitives.md) |
| tiling / blocking | 把计算切成能放进快速存储的小块，让同一份数据被多次复用 | [M07](module-07-matmul-hardware-aware.md) |
| im2col | 把卷积的输入展开成一个矩阵，这样就能用矩阵乘来算卷积，代价是输入被复制 K² 倍 | [M07](module-07-matmul-hardware-aware.md) |
| MFU | 模型 FLOPs 利用率：模型实际完成的有效计算（6N × tokens/s）占硬件峰值的比例 | [M07](module-07-matmul-hardware-aware.md) |
| online softmax | 分块扫过一行时，边走边更新这一行的最大值和分母，一遍就能算出 softmax | [M08](module-08-attention-flashattention.md) |
| FlashAttention | 一种精确的 attention 算法：分块计算加 online softmax，反向时重算，从不把 T² 的注意力矩阵写回 HBM | [M08](module-08-attention-flashattention.md) |
| MQA / GQA | 让多个查询头共享一组（MQA）或若干组（GQA）K/V 头，从而缩小 KV cache | [M08](module-08-attention-flashattention.md) |
| systolic array | 由处理单元（PE）排成的阵列，数据在相邻 PE 之间一拍一拍地流过去；TPU 的矩阵单元就是这种结构 | [M09](module-09-accelerators.md) |
| scratchpad | 由软件显式管理的片上存储，不像 cache 那样由硬件自动换入换出（例如 Trainium 的 SBUF） | [M09](module-09-accelerators.md) |
| dataflow（weight/output stationary） | 加速器让哪种数据一直留在 PE 里不动：权重（weight stationary）还是输出（output stationary） | [M09](module-09-accelerators.md) |
| operator fusion | 把多个算子合成一个 kernel，中间结果就不用写回 HBM 再读出来 | [M10](module-10-ml-compilers.md) |
| schedule（Halide/TVM） | 和算法本身分开写的循环变换描述，比如 split、reorder、tile、vectorize：算什么不变，只改怎么算 | [M10](module-10-ml-compilers.md) |
| graph substitution（TASO） | 把计算图里的一个子图换成等价的另一个；TASO 能自动生成这些替换规则并验证它们确实等价 | [M10](module-10-ml-compilers.md) |
| graph break | `torch.compile` 遇到无法追踪的代码时，在这里把计算图切断 | [M10](module-10-ml-compilers.md) |
| reverse-mode AD | 反向模式自动微分：一次反向传播就能得到一个标量对全部输入的梯度 | [M11](module-11-frameworks-autodiff.md) |
| parameter server | 参数集中放在服务器上，worker 把梯度 push 上去、再把新参数 pull 回来的架构 | [M11](module-11-frameworks-autodiff.md) |
| all-reduce | 每个 rank 最后都拿到所有 rank 输入之和；它可以拆成 reduce-scatter 加 all-gather | [M12](module-12-data-parallel-zero.md) |
| reduce-scatter / all-gather | reduce-scatter 求和后每个 rank 只留一片；all-gather 把各 rank 手里的分片收集齐 | [M12](module-12-data-parallel-zero.md) |
| ring all-reduce | 在环形拓扑上做 all-reduce，每张卡收发 2(p−1)/p·S 字节 | [M12](module-12-data-parallel-zero.md) |
| ZeRO | 在数据并行组内，把原本每张卡各存一份的东西分片：优化器状态（1）、梯度（2）、权重（3） | [M12](module-12-data-parallel-zero.md) |
| FSDP | PyTorch 的全分片数据并行，相当于 ZeRO-3 | [M12](module-12-data-parallel-zero.md) |
| tensor parallel | 把一层里的矩阵按行或按列切到多张卡上（Megatron） | [M13](module-13-tensor-pipeline-parallel.md) |
| sequence parallel | 在 LayerNorm、dropout 这些区域，把激活按序列维切开分到多张卡上 | [M13](module-13-tensor-pipeline-parallel.md) |
| pipeline parallel | 按层把模型切成若干 stage，micro-batch 像流水线一样依次流过 | [M13](module-13-tensor-pipeline-parallel.md) |
| pipeline bubble | 流水线在填充和排空时总有卡在空等，空闲的比例是 (p−1)/(m+p−1) | [M13](module-13-tensor-pipeline-parallel.md) |
| 1F1B | 一种流水调度：进入稳态后，每张卡交替做一个前向、一个反向 | [M13](module-13-tensor-pipeline-parallel.md) |
| 3D 并行 | 把张量并行、流水线并行和数据并行组合起来用（TP × PP × DP） | [M13](module-13-tensor-pipeline-parallel.md) |
| rematerialization / 重计算 | 前向时不保存某些激活，反向需要时再算一遍，用计算换显存（也叫 gradient checkpointing） | [M14](module-14-memory-optimization.md) |
| loss scaling | fp16 训练时先把 loss 乘上一个大数，防止很小的梯度在 fp16 里下溢 | [M14](module-14-memory-optimization.md) |
| offload | 把优化器状态或参数挪到 CPU 内存或 NVMe 上，给显存腾地方 | [M14](module-14-memory-optimization.md) |
| prefill / decode | prefill 一次处理完整段 prompt；decode 在那之后一个一个地生成 token | [M15](module-15-inference-kv-speculative.md) |
| KV cache | 把每层已经处理过的 token 的 K、V 存下来，生成新 token 时就不用重算整个前缀 | [M15](module-15-inference-kv-speculative.md) |
| TTFT / TPOT | TTFT 是请求到达后等到第一个 token 的时间；TPOT 是之后每多出一个 token 的时间 | [M15](module-15-inference-kv-speculative.md) |
| speculative decoding | 推测解码：草稿模型先猜几个 token，目标模型一次验证，能加速而且不改变输出分布 | [M15](module-15-inference-kv-speculative.md) |
| continuous batching | 每个 decode 步都允许请求加入和离开批次，不必等整批跑完（iteration-level scheduling） | [M16](module-16-llm-serving.md) |
| PagedAttention | 像操作系统的分页一样，把 KV cache 切成固定大小的块来分配，用块表找到它们（vLLM） | [M16](module-16-llm-serving.md) |
| prefix caching | 多个请求有相同前缀时，复用这段前缀已经算好的 KV 块（RadixAttention） | [M16](module-16-llm-serving.md) |
| chunked prefill | 把长 prompt 切成小块，和 decode 混在同一步里调度 | [M16](module-16-llm-serving.md) |
| PD 分离 | 把 prefill 和 decode 放到不同的实例上跑 | [M16](module-16-llm-serving.md) |
| goodput | 只把满足 SLO 的请求计入的吞吐 | [M16](module-16-llm-serving.md) |
| quantization | 用低位整数或低精度浮点数来表示权重或激活 | [M17](module-17-compression-quantization.md) |
| per-channel / per-group | 量化 scale 的粒度：每个输出通道一个 scale，或者每 g 个元素一个 scale | [M17](module-17-compression-quantization.md) |
| SmoothQuant | 激活里的离群值很难量化，于是按通道把这部分尺度挪到权重上去 | [M17](module-17-compression-quantization.md) |
| GPTQ / AWQ | 两种权重量化方法：GPTQ 用二阶信息来补偿舍入误差，AWQ 按激活的重要性来缩放通道 | [M17](module-17-compression-quantization.md) |
| 2:4 稀疏 | 每 4 个元素里只留 2 个的结构化稀疏，能用 sparse tensor core 加速 | [M17](module-17-compression-quantization.md) |
| 知识蒸馏 | 让小的学生模型去拟合大的教师模型给出的软标签 | [M17](module-17-compression-quantization.md) |
| LoRA | 冻结原权重 W，只训练一个低秩的增量 BA | [M17](module-17-compression-quantization.md) |
| QLoRA | 基座模型用 4-bit NF4 存，再在上面做 LoRA | [M17](module-17-compression-quantization.md) |
| SSM / Mamba | 状态空间模型；推理时可以写成递归形式，每生成一步的代价是 O(1) | [M18](module-18-architectures-moe-clusters.md) |
| MoE | 混合专家：模型里有很多个专家网络，每个 token 只激活其中 top-k 个 | [M18](module-18-architectures-moe-clusters.md) |
| expert parallel / all-to-all | 专家分布在多张卡上，token 通过 all-to-all 通信发给对应的专家，算完再收回来 | [M18](module-18-architectures-moe-clusters.md) |
| gang scheduling | 分布式作业的所有 worker 必须同时被调度起来，少一个都跑不了 | [M18](module-18-architectures-moe-clusters.md) |
| scaling law | 损失随参数量、数据量和算力按幂律下降；按 Chinchilla 的结论，每个参数约配 20 个 token | [M18](module-18-architectures-moe-clusters.md) |

## 易混对照

下面这些概念成对出现时最容易说混，面试里也常被拿来追问区别。

| A | B | 区别 |
| --- | --- | --- |
| 延迟（latency） | 吞吐（throughput） | 延迟是单个任务多快完成，吞吐是单位时间完成多少个；加大 batch 通常提高吞吐、牺牲延迟 |
| 并发（concurrency） | 并行（parallelism） | 并发是多个任务在逻辑上同时存在，并行是它们在物理上同时执行 |
| coherence | consistency | coherence 管同一个地址的写何时对别人可见，consistency 管不同地址之间读写的先后顺序 |
| occupancy | 利用率 | occupancy 是活跃 warp 占上限的比例，利用率是实际吞吐占峰值的比例 |
| ZeRO-3 | 张量并行 | 两者都切权重；但 ZeRO-3 在计算前用 all-gather 拼回完整权重，TP 从头到尾只算本地那一片 |
| 1F1B | 交错 1F1B | 1F1B 省激活但不减气泡；交错 1F1B 能减气泡，代价是更多通信 |
| MFU | HFU | 区别在于是否把重计算的 FLOPs 算进去 |
| 推测解码（采样） | 推测解码（贪心） | 采样版保证输出分布与目标模型相同，贪心版保证输出与目标模型逐 token 相同 |
| weight-only 量化 | W8A8 | weight-only 只省读权重的带宽；W8A8 还能用整数 tensor core 省算力 |
