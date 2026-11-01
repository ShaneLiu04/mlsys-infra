# 题库

按系统栈分组。每题三行：答案要点、常见错答、出处。先遮住答案自己讲 60 秒，再对照。
追问链在各讲笔记的「面试追问链」，公式在 [FORMULAS](FORMULAS.md)。

## 一、硬件与并行（M01–M06）

**Q1. 为什么说「GPU 利用率 100%」不代表算力打满？**
- 要点：`nvidia-smi` 的 utilization 是「采样窗口内有 kernel 在跑的时间比例」。访存受限的逐元素 kernel 也能 100%。要看 SM 吞吐和 DRAM 吞吐（Nsight Compute）或直接算 MFU。
- 错答：「100% 说明已经最优」。
- 出处：[M01](../notes/module-01-performance-models.md)、F2

**Q2. Amdahl 定律在大模型训练里体现在哪？**
- 要点：不能并行或不随卡数缩小的部分：dataloader 的串行段、优化器步进中的全局范数、日志与 checkpoint、all-reduce 的延迟项 `2(p−1)α`。卡数上去后它们决定上限。
- 错答：「训练是完全数据并行的，没有串行部分」。
- 出处：[M01](../notes/module-01-performance-models.md)、F1

**Q3. SIMD 发散（divergence）的代价怎么算？**
- 要点：一组 lane 执行到组内最慢的 lane 结束，利用率 = 有效 lane·迭代 / (宽度 × 组内最大迭代数)。CPU 的 SIMD 和 GPU 的 warp 都一样。把相似工作量的元素放在同一组（排序、分桶）能提高利用率。
- 错答：「if/else 两个分支只执行一个，所以没有代价」。
- 出处：[M02](../notes/module-02-multicore-simd-threads.md)、[CS149 A1](../solutions/cs149/asst1-cpu-parallel/README.md)

**Q4. 硬件多线程怎么隐藏延迟？代价是什么？**
- 要点：一个线程等内存时切到另一个线程执行，需要足够多的线程把延迟填满（所需线程数 ≈ 延迟 × 每周期发射数）。代价是每个线程都要有寄存器和上下文，片上存储被分摊，单线程的可用缓存变小。GPU 的 occupancy 就是这件事。
- 错答：「多线程让单个线程更快」。
- 出处：[M02](../notes/module-02-multicore-simd-threads.md)、[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q5. 静态分配和动态分配怎么选？work stealing 为什么好？**
- 要点：任务代价可预测且均匀用静态（零调度开销）；不可预测用动态。集中式队列会争抢，work stealing 让每个线程从自己队列尾部取，空闲时随机从别人队列头部偷，偷的是大块工作，争抢少，满足 `T_P ≤ T1/P + O(T∞)`。
- 错答：「动态一定比静态好」——细粒度任务的入队出队开销可能比任务本身还大。
- 出处：[M03](../notes/module-03-parallel-programming-scheduling.md)、F5

**Q6. 线程池里 spin 和 sleep 怎么选？**
- 要点：任务间隔短于一次线程唤醒（微秒级）时 spin，否则 sleep（条件变量）。spin 空闲时也烧满 CPU，会抢走同机其它进程（例如 dataloader）的核。
- 错答：「sleep 总是更省」——频繁唤醒的延迟会吃掉吞吐。
- 出处：[M03](../notes/module-03-parallel-programming-scheduling.md)、[CS149 A2](../solutions/cs149/asst2-task-scheduler/README.md)

**Q7. 什么是 false sharing？怎么发现、怎么修？**
- 要点：两个核写同一 cache line 上的不同变量，coherence 协议让这条线在两个核之间来回失效。表现为加线程反而变慢。修法是按 cache line（64 B）填充或让每个线程写私有副本最后归约。
- 错答：「没有共享变量就没有竞争」。
- 出处：[M04](../notes/module-04-locality-coherence-synchronization.md)

**Q8. CAS 实现的无锁栈有什么问题？**
- 要点：ABA：线程读到栈顶 A，被挂起；别人弹出 A、弹出 B、再压回 A；CAS 发现栈顶还是 A 于是成功，但 next 指针已经失效。解法：带版本号的指针、hazard pointer、epoch 回收。
- 错答：「CAS 是原子的所以无锁结构一定正确」。
- 出处：[M04](../notes/module-04-locality-coherence-synchronization.md)

**Q9. 什么是 memory coalescing？给一个不合并的例子。**
- 要点：一个 warp 的 32 个线程访问连续且对齐的地址时，硬件合并成少量 32 B/128 B 事务。按列访问行主序矩阵（stride = 行宽）时每个线程一个事务，有效带宽掉一个数量级。常用 shared memory 做转置中转。
- 错答：「同一个 block 内的访问会自动合并」——合并的单位是 warp，看的是地址模式。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q10. Shared memory bank conflict 怎么产生？怎么避免？**
- 要点：32 个 bank、每 bank 4 字节。同一 warp 的线程访问同一 bank 的不同地址会被串行化。典型是 32×32 float 矩阵按列读：所有线程落在同一 bank。把矩阵声明成 32×33（padding 一列）错开 bank。同一地址的广播不冲突。
- 错答：「bank conflict 发生在 global memory」。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q11. occupancy 越高越好吗？**
- 要点：不是。occupancy 只是隐藏延迟的手段。高 ILP、寄存器分块很重的 GEMM kernel 常常故意低 occupancy（每线程多寄存器），换取更高的数据复用。目标是吞吐，不是 occupancy。
- 错答：「把 occupancy 拉到 100% 性能就最好」。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q12. 为什么 GPU 上的 scan 用 Blelloch 而不是 Hillis–Steele？**
- 要点：Hillis–Steele 的 work 是 n·log n，Blelloch 是 2n。大数组受带宽限制，多出来的 log n 倍工作就是多出来的访存。warp 内（32 元素）反过来用 Hillis–Steele，因为步数少、线程空着也是空着。
- 错答：「log n 步的算法更快」。
- 出处：[M06](../notes/module-06-data-parallel-primitives.md)、F6、[CS149 A3](../solutions/cs149/asst3-scan-render/README.md)

**Q13. 圆形渲染器为什么不能按圆并行直接写像素？**
- 要点：半透明混合不满足交换律，像素结果依赖圆的顺序；按圆并行还会对同一像素产生写冲突。按图像块并行，每块先用 scan 压缩出按原顺序排列的相交圆列表，再在块内按顺序着色，既保序又无冲突。
- 错答：「用 atomicAdd 就行」——原子操作解决冲突，但不保证顺序。
- 出处：[M06](../notes/module-06-data-parallel-primitives.md)、[CS149 A3](../solutions/cs149/asst3-scan-render/README.md)

## 二、单卡算子与编译（M07–M11）

**Q14. batch=1 的矩阵-向量乘为什么在任何 GPU 上都慢？**
- 要点：bf16 的 GEMV 算术强度约 1 FLOP/B，H100 拐点 295，只能用到峰值的 0.3%。这就是 decode 的处境。提高强度只能靠 batch（把 GEMV 变 GEMM）或降低字节数（量化）。
- 错答：「换更快的卡」——H100 相对 A100 的带宽只多 1.6 倍。
- 出处：[M07](../notes/module-07-matmul-hardware-aware.md)、F3

**Q15. 矩阵乘分块时块大小怎么选？**
- 要点：三块要同时放进快存（`3b²·s ≤ M`）；算术强度约 `b/s`；还要让 block 数足够填满所有 SM，并照顾 occupancy 和 tensor core 的 MMA 形状。太大放不下或 SM 不够分，太小强度不够。
- 错答：「越大越好」。
- 出处：[M07](../notes/module-07-matmul-hardware-aware.md)、F4

**Q16. FlashAttention 快在哪？FLOPs 减少了吗？**
- 要点：没减少，反向还多了一次重算。快在不把 T×T 分数矩阵写回 HBM：按块计算，用 online softmax 维护每行的最大值和分母。显存从 O(T²) 降到 O(T)。HBM 访问是 Θ(T²d²/M)，d=128 时在 T 上仍是二次；收益来自省掉 eager 实现里 scale/mask/softmax/dropout 各自的 T² 读写，以及 K/V 重读命中 L2。
- 错答：「用了近似 attention」——它是精确的。「访存从二次降到线性」——只有显存是线性的。
- 实测：本仓库的 IO 计数器里，按论文的块大小，Flash 只比「S、P 各读写一次」的朴素模型少 2.00 倍（因果 3.7–4.0 倍）。
- 出处：[M08](../notes/module-08-attention-flashattention.md)、F7、F8、[CS229S P1](../solutions/cs229s/project1-transformer-kernels/README.md)

**Q17. FlashAttention-2 相对 v1 改了什么？**
- 要点：外层循环改成遍历 Q 块（每个 thread block 负责一段 Q，输出只写一次）；沿序列长度并行，长序列小 batch 时也能填满 SM；减少非矩阵乘的 FLOPs（rescale 推迟到最后）；warp 间按 Q 切分，减少 shared memory 通信。
- 错答：「v2 换了算法」——数学上同一个 online softmax。
- 出处：[M08](../notes/module-08-attention-flashattention.md)

**Q18. Systolic array 为什么适合矩阵乘？缺点是什么？**
- 要点：数据在相邻 PE 间流动，每个操作数从 SRAM 读一次后被复用 P 次，大幅减少存储访问，能效高。缺点是矩阵小于阵列或形状不匹配时利用率低（填充/排空开销），对非 GEMM 运算不友好。
- 错答：「TPU 快是因为频率高」。
- 出处：[M09](../notes/module-09-accelerators.md)、[CS149 A4](../solutions/cs149/asst4-dnn-accelerator/README.md)

**Q19. 算子融合省的是什么？什么样的算子不能随便融合？**
- 要点：省的是中间结果的 HBM 读写和 kernel 发射。逐元素链（bias+GeLU+dropout+残差）融合收益最大。reduction 后面接需要全局结果的操作（如 LayerNorm 要先算完整行的均值）融合要更小心；融合后寄存器压力变大可能降低 occupancy。
- 错答：「融合减少了 FLOPs」。
- 出处：[M10](../notes/module-10-ml-compilers.md)、[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)

**Q20. TVM/Ansor 和 Triton 的定位有什么不同？**
- 要点：TVM 把算法和调度分开，靠搜索（Ansor 生成 sketch 再随机注解 + 进化搜索 + 学习的代价模型）自动找调度。Triton 让人写 block 级程序，编译器负责 block 内的线程映射、合并访问和 shared memory。前者是自动调优，后者是降低手写门槛。
- 错答：「Triton 是 TVM 的前端」。
- 出处：[M10](../notes/module-10-ml-compilers.md)

**Q21. `torch.compile` 什么时候会 graph break？代价是什么？**
- 要点：遇到 Dynamo 不能追踪的 Python（数据相关的控制流、`.item()`、打印、某些第三方库）就切图。每个断点前后是独立的图，融合机会被切断，还要回到 Python 解释执行。
- 错答：「compile 之后整个模型都是一张图」。
- 出处：[M10](../notes/module-10-ml-compilers.md)

**Q22. 为什么深度学习用反向模式自动微分？**
- 要点：标量损失对 n 个参数求梯度，反向模式一次反传得到全部（代价约 2–3 倍前向），前向模式要 n 次。代价是必须保存前向的中间激活，这就是训练显存里激活那一块。
- 错答：「反向模式更精确」。
- 出处：[M11](../notes/module-11-frameworks-autodiff.md)、F9

## 三、训练系统（M12–M14）

**Q23. all-reduce 为什么可以拆成 reduce-scatter + all-gather？这有什么用？**
- 要点：先把和分片算好（每卡得到 1/p 的和），再把分片广播给所有人。ZeRO-2 只做 reduce-scatter（每卡只需要自己负责那份梯度），ZeRO-3 和 FSDP 的前向 all-gather 权重，序列并行用它替代 all-reduce。
- 错答：「all-reduce 是原子操作，不能拆」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、F11

**Q24. ring 和 tree all-reduce 怎么选？**
- 要点：ring 带宽项最优（2(p−1)/p·S），但延迟项随 p 线性增长；tree 延迟是 log p。小消息延迟主导用 tree，大消息带宽主导用 ring。NCCL 会按消息大小和拓扑自动选。
- 错答：「ring 总是最好」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)

**Q25. ZeRO 三个 stage 分别切了什么？通信量变不变？**
- 要点：1 切优化器状态，2 再切梯度，3 再切权重。1、2 的通信量与 DP 相同（all-reduce = RS + AG），3 多一次前向 all-gather，是 DP 的 1.5 倍。7B、8 卡的静态显存：112 → 38.5 → 26.3 → 14 GB。
- 错答：「ZeRO-3 通信量是 3 倍」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、F10

**Q26. DDP 怎么让通信和反向 overlap？bucket 大小怎么取？**
- 要点：梯度按反向顺序分桶，一个桶的梯度全部就绪就异步发起 all-reduce，后面的层继续反向。桶太小延迟项多、kernel 多；太大则最后一个桶暴露在反向之后。默认 25 MB 量级，按网络带宽调。
- 错答：「等反向全部结束再 all-reduce」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)

**Q27. Megatron 的 MLP 为什么第一层按列切、第二层按行切？**
- 要点：列切后每卡得到 `XW1` 的一部分列，GeLU 是逐元素的，可以直接在本地算；第二层行切正好消费这部分列，得到部分和，一次 all-reduce 求和。整个 MLP 前向只有一次通信。
- 错答：「两层都列切」——中间要先 all-gather。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)、F12

**Q28. 为什么张量并行不出节点，流水线并行可以跨节点？**
- 要点：TP 每层 4 次 all-reduce，每次大小 `B·T·d·s`，必须走 NVLink（450 GB/s）；PP 只在 stage 边界点对点传激活，量小且可以和计算重叠，走 IB（50 GB/s）也可以接受。
- 错答：「随便放」。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)

**Q29. 1F1B 比 GPipe 好在哪？**
- 要点：气泡一样，都是 (p−1)/(m+p−1)。1F1B 在稳态时每做一个前向就做一个反向，在途的 micro-batch 最多 p 个，激活显存从 m 份降到 p 份。交错调度（每卡 v 个虚拟 stage）才减小气泡，代价是 v 倍的点对点通信。
- 错答：「1F1B 减少了气泡」。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)、F13

**Q30. 激活重计算多花多少算力？**
- 要点：全量重计算多一次前向，总计算从 3F 变 4F（+33%）。按 √n 设检查点，激活显存从 O(n) 降到 O(√n)。选择性重计算只重算 attention 里占显存大但计算便宜的部分（softmax、dropout mask），开销几个百分点。
- 错答：「慢一倍」。
- 出处：[M14](../notes/module-14-memory-optimization.md)、F14

**Q31. bf16 和 fp16 训练的区别？**
- 要点：bf16 有 8 位指数，和 fp32 动态范围相同，不需要 loss scaling；只有 7 位尾数，精度低。fp16 有 5 位指数，梯度容易下溢，必须动态 loss scaling。两者都要 fp32 主权重和 fp32 累加。
- 错答：「bf16 精度更高」。
- 出处：[M14](../notes/module-14-memory-optimization.md)

## 四、推理系统（M15–M16）

**Q32. prefill 和 decode 的瓶颈分别是什么？**
- 要点：prefill 一次处理整段 prompt，是 GEMM，受算力限制；decode 每步一个 token，GEMV，受带宽限制（读全部权重和 KV）。所以 prefill 决定 TTFT 和算力需求，decode 决定 TPOT 和显存带宽需求，这是 PD 分离的出发点。
- 错答：「decode 计算量大」。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F16

**Q33. KV cache 为什么能省计算？它本身的代价是什么？**
- 要点：不缓存时每步要重算全部前缀的 K、V，总代价 O(T²)；缓存后每步 O(T)。代价是显存 `2·L·n_kv·d·T·B·s`，8B 模型 128 KB/token，它决定了最大并发。
- 错答：「KV cache 减少了 attention 的计算」——每步仍要读全部 KV 做点积。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F15、[CS229S P2](../solutions/cs229s/project2-kv-speculative/README.md)

**Q34. 推测解码为什么不改变输出分布？什么时候不划算？**
- 要点：接受概率 min(1, p/q)，拒绝时从 `norm(max(0, p−q))` 重采样，两者合起来每个 token 的边际分布恰好是 p。不划算：接受率低、草稿模型不够便宜（c 大）、大 batch 下 decode 已接近算力受限（多验证的 token 不再免费）。
- 错答：「草稿模型猜错的 token 会混进输出」。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F17

**Q35. PagedAttention 解决的是什么问题？代价呢？**
- 要点：按最大长度连续预分配 KV 会产生大量内部碎片和预留浪费。按固定大小的块分配、用块表映射，浪费只在最后一块；还能在并行采样和 beam search 间共享前缀块（copy-on-write）。代价是 attention kernel 要按块表间接寻址，块太小会降低访存效率。
- 错答：「PagedAttention 减少了 KV 的大小」。
- 出处：[M16](../notes/module-16-llm-serving.md)、[15-849 服务与显存](../solutions/cmu15849/serving-memory/README.md)

**Q36. continuous batching 和 static batching 的区别？**
- 要点：static batching 一批请求要等最长的那条生成完才换下一批；continuous（iteration-level）每个 decode 步都可以让完成的请求离开、新请求加入。吞吐和平均时延都更好。chunked prefill 进一步把长 prompt 切块混进 decode 步，减少 TPOT 抖动。
- 错答：「continuous batching 就是 batch 更大」。
- 出处：[M16](../notes/module-16-llm-serving.md)

## 五、模型效率与集群（M17–M18）

**Q37. 为什么 LLM 的激活难量化，权重容易？**
- 要点：权重分布接近零均值、离群值少；激活在少数固定通道上有比其它通道大几十到上百倍的离群值，per-tensor int8 的 scale 被它撑大，其余通道几乎都量化成 0。LLM.int8 把离群通道拆出来走 fp16；SmoothQuant 把激活的尺度按通道挪到权重上（`X·diag(s)⁻¹ · diag(s)·W`）。
- 错答：「int8 对所有张量都一样」。
- 出处：[M17](../notes/module-17-compression-quantization.md)、F19

**Q38. weight-only 量化为什么能加速 decode，却不一定加速 prefill？**
- 要点：decode 受带宽限制，读 int4 权重比 bf16 少 4 倍字节；反量化在寄存器里做，计算不是瓶颈。prefill 受算力限制，权重仍要反量化成 bf16 才能上 tensor core，没有省算力，反而多了反量化开销。
- 错答：「量化总是更快」。
- 出处：[M17](../notes/module-17-compression-quantization.md)

**Q39. 2:4 稀疏和非结构化稀疏怎么选？**
- 要点：非结构化稀疏同样稀疏度下精度更好，但硬件很难加速（不规则访存）。2:4 是每 4 个元素保留 2 个，Ampere 之后的 sparse tensor core 能直接跑出约 2 倍的算力上限。
- 错答：「剪掉 50% 就快 2 倍」——非结构化的 50% 几乎不会变快。
- 出处：[M17](../notes/module-17-compression-quantization.md)

**Q40. MoE 为什么「参数多但不贵」？它贵在哪？**
- 要点：每个 token 只经过 top-k 个专家，FLOPs 随 k 而不是专家总数增长。贵在：全部专家都要放进显存；专家并行的 all-to-all 每层前后各一次；负载不均时热门专家所在的卡拖慢整步（需要负载均衡损失和容量因子）。
- 错答：「MoE 省算力所以训练和部署都便宜」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)、F21

**Q41. Mamba/SSM 相对 Transformer 的推理优势是什么？**
- 要点：递归形式下每步状态大小固定，decode 不需要随长度增长的 KV cache，每 token 代价 O(1)；训练时用卷积或并行扫描形式并行。代价是固定大小的状态会压缩历史，精确检索（如长距离复制）比 attention 弱。
- 错答：「SSM 在所有任务上都比 attention 好」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)

**Q42. 深度学习集群调度和传统批处理调度有什么不同？**
- 要点：分布式训练需要 gang scheduling（所有 worker 同时启动，否则空等）；对拓扑敏感（同一节点/同一交换机）；作业可以弹性伸缩并能被抢占（依赖 checkpoint）；吞吐与分配的卡数非线性（Pollux 同时调 batch 和资源）。
- 错答：「按 GPU 数量排队就行」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)
