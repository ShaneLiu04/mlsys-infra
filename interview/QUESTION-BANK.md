# 题库

42 道概念题，按系统栈从下往上分组。每题给一段能直接说出口的回答，先结论后理由，再附上常见的错答和出处。
练的时候先把答案遮住，自己讲 60 秒，再来对照。追问链在各讲笔记的「面试追问链」里，公式在 [FORMULAS](FORMULAS.md)。

## 一、硬件与并行（M01–M06）

**Q1. 为什么说「GPU 利用率 100%」不代表算力打满？**
- 答：因为 `nvidia-smi` 的 utilization 只是「采样窗口里有 kernel 在跑的时间比例」，它不管 kernel 跑得满不满。访存受限的逐元素 kernel 也能显示 100%。要判断算力用了多少，得看 Nsight Compute 里的 SM 吞吐和 DRAM 吞吐，或者直接算 MFU。
- 错答：「100% 说明已经最优了」。
- 出处：[M01](../notes/module-01-performance-models.md)、F2

**Q2. Amdahl 定律在大模型训练里体现在哪？**
- 答：体现在那些不能并行、或者不随卡数缩小的部分上：dataloader 里的串行段、优化器步进里算全局范数、日志和 checkpoint，还有 all-reduce 的延迟项 `2(p−1)α`。卡数一多，决定上限的就是它们。
- 错答：「训练是完全数据并行的，没有串行部分」。
- 出处：[M01](../notes/module-01-performance-models.md)、F1

**Q3. SIMD 发散（divergence）的代价怎么算？**
- 答：一组 lane 要一直执行到组内最慢的那条结束，所以利用率 = 有效的 lane·迭代数 / (宽度 × 组内最大迭代数)。CPU 的 SIMD 和 GPU 的 warp 都是这样。提高利用率的办法是把工作量相近的元素放进同一组，比如先排序或分桶。
- 错答：「if/else 两个分支只执行一个，所以没有代价」。
- 出处：[M02](../notes/module-02-multicore-simd-threads.md)、[CS149 A1](../solutions/cs149/asst1-cpu-parallel/README.md)

**Q4. 硬件多线程怎么隐藏延迟？代价是什么？**
- 答：一个线程在等内存时，硬件切到另一个线程去执行。要把延迟完全填满，需要足够多的线程，所需线程数约等于延迟乘以每周期的发射数。代价是每个线程都要有自己的寄存器和上下文，片上存储被分摊掉，每个线程能用的缓存就变小了。GPU 的 occupancy 讲的就是这件事。
- 错答：「多线程能让单个线程跑得更快」。
- 出处：[M02](../notes/module-02-multicore-simd-threads.md)、[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q5. 静态分配和动态分配怎么选？work stealing 为什么好？**
- 答：任务代价可预测而且均匀，就用静态分配，因为它没有调度开销；不可预测时用动态。但集中式的任务队列会有争抢。work stealing 让每个线程从自己队列的尾部取任务，空闲时随机从别人队列的头部偷，偷到的是大块工作，所以争抢少，并且满足 `T_P ≤ T1/P + O(T∞)`。
- 错答：「动态一定比静态好」。任务粒度很细时，入队出队的开销可能比任务本身还大。
- 出处：[M03](../notes/module-03-parallel-programming-scheduling.md)、F5

**Q6. 线程池里 spin 和 sleep 怎么选？**
- 答：看任务间隔和一次线程唤醒（微秒级）哪个长。间隔更短就 spin，否则 sleep，也就是用条件变量等待。spin 的问题是空闲时也会把 CPU 烧满，抢走同一台机器上其他进程（比如 dataloader）的核。
- 错答：「sleep 总是更省」。唤醒很频繁时，唤醒延迟会吃掉吞吐。
- 出处：[M03](../notes/module-03-parallel-programming-scheduling.md)、[CS149 A2](../solutions/cs149/asst2-task-scheduler/README.md)

**Q7. 什么是 false sharing？怎么发现、怎么修？**
- 答：两个核写的是同一条 cache line 上的不同变量，coherence 协议会让这条 line 在两个核之间来回失效。它的表现是加线程反而变慢。修法是按 cache line（64 B）填充，让每个变量独占一行；或者每个线程写私有副本，最后再归约。
- 错答：「没有共享变量就没有竞争」。
- 出处：[M04](../notes/module-04-locality-coherence-synchronization.md)

**Q8. CAS 实现的无锁栈有什么问题？**
- 答：有 ABA 问题。一个线程读到栈顶是 A，然后被挂起；这期间别人弹出 A、弹出 B，又把 A 压了回去。等它恢复后做 CAS，发现栈顶还是 A，于是成功了，但它手里的 next 指针已经失效。解法有带版本号的指针、hazard pointer 和 epoch 回收。
- 错答：「CAS 是原子的，所以无锁结构一定正确」。
- 出处：[M04](../notes/module-04-locality-coherence-synchronization.md)

**Q9. 什么是 memory coalescing？给一个不合并的例子。**
- 答：一个 warp 的 32 个线程访问连续且对齐的地址时，硬件会把它们合并成少量 32 B 或 128 B 的事务。反例是按列访问行主序的矩阵：stride 等于行宽，每个线程各占一个事务，有效带宽会掉一个数量级。常见的做法是用 shared memory 做一次转置中转。
- 错答：「同一个 block 内的访问会自动合并」。合并的单位是 warp，看的是地址模式。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q10. Shared memory bank conflict 怎么产生？怎么避免？**
- 答：shared memory 分成 32 个 bank，每个 bank 4 字节宽。同一个 warp 里的线程访问同一个 bank 的不同地址时，这些访问会被串行化。典型例子是按列读 32×32 的 float 矩阵，所有线程都落在同一个 bank。把矩阵声明成 32×33，多填一列，就能把 bank 错开。多个线程读同一个地址是广播，不算冲突。
- 错答：「bank conflict 发生在 global memory 上」。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q11. occupancy 越高越好吗？**
- 答：不是。occupancy 只是隐藏延迟的一种手段。很多 ILP 高、寄存器分块很重的 GEMM kernel 故意把 occupancy 压低，让每个线程多用寄存器，换来更高的数据复用。要优化的目标是吞吐，不是 occupancy。
- 错答：「把 occupancy 拉到 100%，性能就最好」。
- 出处：[M05](../notes/module-05-gpu-architecture-cuda.md)

**Q12. 为什么 GPU 上的 scan 用 Blelloch 而不是 Hillis–Steele？**
- 答：因为 Hillis–Steele 的 work 是 n·log n，Blelloch 只有 2n。大数组的 scan 受带宽限制，多出来的 log n 倍工作就是多出来的访存。但在 warp 内部（32 个元素）反过来用 Hillis–Steele，因为它步数少，而那些线程闲着也是闲着。
- 错答：「log n 步的算法更快」。
- 出处：[M06](../notes/module-06-data-parallel-primitives.md)、F6、[CS149 A3](../solutions/cs149/asst3-scan-render/README.md)

**Q13. 圆形渲染器为什么不能按圆并行直接写像素？**
- 答：半透明混合不满足交换律，一个像素的结果取决于圆的绘制顺序；按圆并行还会让多个线程同时写同一个像素。正确的做法是按图像块并行：每块先用 scan 压缩出一个按原顺序排列的相交圆列表，再在块内按顺序着色。这样既保序，又没有写冲突。
- 错答：「用 atomicAdd 就行」。原子操作解决了冲突，但不保证顺序。
- 出处：[M06](../notes/module-06-data-parallel-primitives.md)、[CS149 A3](../solutions/cs149/asst3-scan-render/README.md)

## 二、单卡算子与编译（M07–M11）

**Q14. batch=1 的矩阵-向量乘为什么在任何 GPU 上都慢？**
- 答：bf16 GEMV 的算术强度只有约 1 FLOP/B，而 H100 的拐点是 295，所以只能用到峰值的 0.3%。decode 的处境就是这样。要提高强度，只能靠加 batch（把 GEMV 变成 GEMM），或者减少字节数（量化）。
- 错答：「换一张更快的卡」。H100 的带宽相比 A100 只多 1.6 倍。
- 出处：[M07](../notes/module-07-matmul-hardware-aware.md)、F3

**Q15. 矩阵乘分块时块大小怎么选？**
- 答：有几个约束要同时满足。三块要能同时放进快存（`3b²·s ≤ M`），算术强度约是 `b/s`；block 的数量要足够把所有 SM 填满；还要照顾 occupancy 和 tensor core 的 MMA 形状。块太大会放不下，或者 block 太少分不满 SM；块太小则强度不够。
- 错答：「越大越好」。
- 出处：[M07](../notes/module-07-matmul-hardware-aware.md)、F4

**Q16. FlashAttention 快在哪？FLOPs 减少了吗？**
- 答：FLOPs 没有减少，反向还多了一次重算。它快在不把 T×T 的分数矩阵写回 HBM：按块计算，用 online softmax 维护每一行的最大值和分母，所以显存从 O(T²) 降到 O(T)。HBM 访问量是 Θ(T²d²/M)，d=128 时在 T 上仍然是二次的。真正的收益来自两处：省掉了 eager 实现里 scale、mask、softmax、dropout 各自对 T² 矩阵的读写，以及 K/V 的重复读取命中了 L2。
- 错答：「它用了近似 attention」，其实它是精确的。还有「访存从二次降到了线性」，其实只有显存是线性的。
- 实测：在本仓库的 IO 计数器里，按论文的块大小，Flash 只比「S、P 各读写一次」的朴素模型少 2.00 倍（因果时 3.7–4.0 倍）。
- 出处：[M08](../notes/module-08-attention-flashattention.md)、F7、F8、[CS229S P1](../solutions/cs229s/project1-transformer-kernels/README.md)

**Q17. FlashAttention-2 相对 v1 改了什么？**
- 答：四处。外层循环改成遍历 Q 块，每个 thread block 负责一段 Q，输出只写一次；沿序列长度方向并行，长序列、小 batch 时也能填满 SM；减少非矩阵乘的 FLOPs，把 rescale 推迟到最后；warp 之间按 Q 切分，减少 shared memory 上的通信。
- 错答：「v2 换了算法」。数学上它们是同一个 online softmax。
- 出处：[M08](../notes/module-08-attention-flashattention.md)

**Q18. Systolic array 为什么适合矩阵乘？缺点是什么？**
- 答：数据在相邻的 PE（处理单元）之间流动，每个操作数从 SRAM 读出一次后被复用 P 次，存储访问少了很多，所以能效高。缺点是矩阵比阵列小或者形状不匹配时，填充和排空的开销让利用率变低；而且对非 GEMM 的运算不友好。
- 错答：「TPU 快是因为频率高」。
- 出处：[M09](../notes/module-09-accelerators.md)、[CS149 A4](../solutions/cs149/asst4-dnn-accelerator/README.md)

**Q19. 算子融合省的是什么？什么样的算子不能随便融合？**
- 答：省的是中间结果的 HBM 读写和 kernel 发射。逐元素链（bias、GeLU、dropout、残差）融合的收益最大。reduction 后面接着需要全局结果的操作时要小心，比如 LayerNorm 必须先算完一整行的均值。另外，融合后寄存器压力变大，可能会降低 occupancy。
- 错答：「融合减少了 FLOPs」。
- 出处：[M10](../notes/module-10-ml-compilers.md)、[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)

**Q20. TVM/Ansor 和 Triton 的定位有什么不同？**
- 答：TVM 把算法和调度分开，靠搜索自动找调度。Ansor 的做法是先生成 sketch，再随机注解，用进化搜索加学习出来的代价模型去挑。Triton 则让人写 block 级的程序，block 内部的线程映射、合并访问和 shared memory 交给编译器。前者是自动调优，后者是降低手写 kernel 的门槛。
- 错答：「Triton 是 TVM 的前端」。
- 出处：[M10](../notes/module-10-ml-compilers.md)

**Q21. `torch.compile` 什么时候会 graph break？代价是什么？**
- 答：遇到 Dynamo 追踪不了的 Python 代码就会切图，比如依赖数据的控制流、`.item()`、打印、某些第三方库。每个断点前后各是一张独立的图，跨断点的融合机会就没了，中间还要回到 Python 解释执行。
- 错答：「compile 之后整个模型就是一张图」。
- 出处：[M10](../notes/module-10-ml-compilers.md)

**Q22. 为什么深度学习用反向模式自动微分？**
- 答：损失是一个标量，要对 n 个参数求梯度。反向模式一次反传就能拿到全部梯度，代价约是前向的 2–3 倍；前向模式要跑 n 次。反向模式的代价是必须保存前向的中间激活，这就是训练显存里激活那一块的来源。
- 错答：「反向模式更精确」。
- 出处：[M11](../notes/module-11-frameworks-autodiff.md)、F9

## 三、训练系统（M12–M14）

**Q23. all-reduce 为什么可以拆成 reduce-scatter + all-gather？这有什么用？**
- 答：可以先把求和按分片做完，每张卡得到 1/p 的和，这是 reduce-scatter；再把各分片广播给所有卡，这是 all-gather。拆开以后能单独用其中一半：ZeRO-2 只做 reduce-scatter，因为每张卡只需要自己负责的那份梯度；ZeRO-3 和 FSDP 用 all-gather 在前向时收集权重；序列并行用这两步替代 all-reduce。
- 错答：「all-reduce 是原子操作，不能拆」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、F11

**Q24. ring 和 tree all-reduce 怎么选？**
- 答：按消息大小选。ring 的带宽项是最优的（2(p−1)/p·S），但延迟项随 p 线性增长；tree 的延迟是 log p。所以小消息由延迟主导，用 tree；大消息由带宽主导，用 ring。NCCL 会根据消息大小和拓扑自动选。
- 错答：「ring 总是最好」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)

**Q25. ZeRO 三个 stage 分别切了什么？通信量变不变？**
- 答：stage 1 切优化器状态，stage 2 再切梯度，stage 3 连权重也切。1 和 2 的通信量和 DP 一样，因为 all-reduce 本来就等于 RS 加 AG；3 多了一次前向的 all-gather，是 DP 的 1.5 倍。以 7B、8 卡为例，每卡静态显存从 112 降到 38.5、26.3，最后是 14 GB。
- 错答：「ZeRO-3 的通信量是 3 倍」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)、F10

**Q26. DDP 怎么让通信和反向 overlap？bucket 大小怎么取？**
- 答：梯度按反向的顺序分桶，一个桶里的梯度全部就绪，就异步发起它的 all-reduce，后面的层继续算反向。桶太小，延迟项和 kernel 数都多；桶太大，最后一个桶就暴露在反向结束之后。默认在 25 MB 量级，再按网络带宽调。
- 错答：「等反向全部结束再做 all-reduce」。
- 出处：[M12](../notes/module-12-data-parallel-zero.md)

**Q27. Megatron 的 MLP 为什么第一层按列切、第二层按行切？**
- 答：因为这样整个 MLP 前向只需要一次通信。第一层按列切，每张卡得到 `XW1` 的一部分列；GeLU 是逐元素的，可以直接在本地算。第二层按行切，刚好消费这部分列，每张卡得到一个部分和，最后一次 all-reduce 求和。
- 错答：「两层都按列切」。那样中间就得先做一次 all-gather。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)、F12

**Q28. 为什么张量并行不出节点，流水线并行可以跨节点？**
- 答：因为两者的通信量差得很远。TP 每层 4 次 all-reduce，每次的大小是 `B·T·d·s`，必须走 NVLink（450 GB/s）。PP 只在 stage 边界上点对点传激活，量小，还能和计算重叠，走 IB（50 GB/s）也能接受。
- 错答：「放哪都行」。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)

**Q29. 1F1B 比 GPipe 好在哪？**
- 答：好在激活显存，不在气泡。两者的气泡一样，都是 (p−1)/(m+p−1)。1F1B 在稳态时每做一个前向就做一个反向，同时在途的 micro-batch 最多 p 个，所以激活显存从 m 份降到 p 份。真正能减小气泡的是交错调度（每张卡放 v 个虚拟 stage），代价是 v 倍的点对点通信。
- 错答：「1F1B 减少了气泡」。
- 出处：[M13](../notes/module-13-tensor-pipeline-parallel.md)、F13

**Q30. 激活重计算多花多少算力？**
- 答：全量重计算多一次前向，总计算从 3F 变成 4F，也就是 +33%。按 √n 设检查点，激活显存从 O(n) 降到 O(√n)。选择性重计算只重算 attention 里占显存多、计算便宜的部分（softmax 和 dropout mask），开销只有几个百分点。
- 错答：「慢一倍」。
- 出处：[M14](../notes/module-14-memory-optimization.md)、F14

**Q31. bf16 和 fp16 训练的区别？**
- 答：bf16 有 8 位指数，动态范围和 fp32 一样，所以不需要 loss scaling；但它只有 7 位尾数，精度低。fp16 只有 5 位指数，梯度容易下溢，必须做动态 loss scaling。两者都需要 fp32 的主权重和 fp32 累加。
- 错答：「bf16 精度更高」。
- 出处：[M14](../notes/module-14-memory-optimization.md)

## 四、推理系统（M15–M16）

**Q32. prefill 和 decode 的瓶颈分别是什么？**
- 答：prefill 一次处理整段 prompt，做的是 GEMM，受算力限制；decode 每步只出一个 token，做的是 GEMV，要把全部权重和 KV 读一遍，受带宽限制。所以 prefill 决定 TTFT 和算力需求，decode 决定 TPOT 和显存带宽需求。这正是 PD 分离的出发点。
- 错答：「decode 的计算量大」。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F16

**Q33. KV cache 为什么能省计算？它本身的代价是什么？**
- 答：不缓存的话，每一步都要把整个前缀的 K、V 重算一遍，总代价是 O(T²)；缓存以后每步只需 O(T)。代价是显存，大小是 `2·L·n_kv·d·T·B·s`，8B 模型每个 token 128 KB，它决定了最多能并发多少条请求。
- 错答：「KV cache 减少了 attention 的计算」。每一步仍然要读全部 KV 来做点积。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F15、[CS229S P2](../solutions/cs229s/project2-kv-speculative/README.md)

**Q34. 推测解码为什么不改变输出分布？什么时候不划算？**
- 答：接受概率是 min(1, p/q)，拒绝时从 `norm(max(0, p−q))` 重采样，这两步合起来，每个 token 的边际分布恰好是 p。不划算的情况有三种：接受率低；草稿模型不够便宜（c 大）；batch 很大时 decode 已经接近算力受限，多验证的 token 不再是免费的。
- 错答：「草稿模型猜错的 token 会混进输出」。
- 出处：[M15](../notes/module-15-inference-kv-speculative.md)、F17

**Q35. PagedAttention 解决的是什么问题？代价呢？**
- 答：按最大长度连续预分配 KV，会产生大量内部碎片和预留浪费。PagedAttention 按固定大小的块分配，用块表做映射，浪费只出现在最后一块；还能在并行采样和 beam search 之间用 copy-on-write 共享前缀块。代价是 attention kernel 要按块表间接寻址，块太小时访存效率会下降。
- 错答：「PagedAttention 减小了 KV 的大小」。
- 出处：[M16](../notes/module-16-llm-serving.md)、[15-849 服务与显存](../solutions/cmu15849/serving-memory/README.md)

**Q36. continuous batching 和 static batching 的区别？**
- 答：static batching 下，一批请求要等最长的那条生成完，才能换下一批。continuous batching（也叫 iteration-level）在每个 decode 步都可以让已完成的请求离开、新请求加入，所以吞吐和平均时延都更好。chunked prefill 更进一步，把长 prompt 切成块混进 decode 步里，减少 TPOT 的抖动。
- 错答：「continuous batching 就是把 batch 开大」。
- 出处：[M16](../notes/module-16-llm-serving.md)

## 五、模型效率与集群（M17–M18）

**Q37. 为什么 LLM 的激活难量化，权重容易？**
- 答：权重的分布接近零均值，离群值少。激活则在少数固定的通道上有比其他通道大几十到上百倍的离群值，per-tensor int8 的 scale 被它们撑大，其余通道几乎都被量化成 0。LLM.int8 的做法是把离群通道拆出来走 fp16；SmoothQuant 则把激活的尺度按通道挪到权重上（`X·diag(s)⁻¹ · diag(s)·W`）。
- 错答：「int8 对所有张量都一样」。
- 出处：[M17](../notes/module-17-compression-quantization.md)、F19

**Q38. weight-only 量化为什么能加速 decode，却不一定加速 prefill？**
- 答：decode 受带宽限制，读 int4 权重比读 bf16 少 4 倍字节；反量化在寄存器里做，计算本来就不是瓶颈。prefill 受算力限制，权重仍然要反量化成 bf16 才能上 tensor core，算力一点没省，反而多了反量化的开销。
- 错答：「量化总是更快」。
- 出处：[M17](../notes/module-17-compression-quantization.md)

**Q39. 2:4 稀疏和非结构化稀疏怎么选？**
- 答：非结构化稀疏在同样的稀疏度下精度更好，但访存不规则，硬件很难加速。2:4 稀疏是每 4 个元素保留 2 个，Ampere 之后的 sparse tensor core 可以直接跑，算力上限约是 2 倍。
- 错答：「剪掉 50% 就快 2 倍」。非结构化的 50% 几乎不会变快。
- 出处：[M17](../notes/module-17-compression-quantization.md)

**Q40. MoE 为什么「参数多但不贵」？它贵在哪？**
- 答：每个 token 只经过 top-k 个专家，所以 FLOPs 随 k 增长，而不是随专家总数增长。贵的地方有三处：全部专家都要放进显存；专家并行时每层前后各有一次 all-to-all；负载不均时，热门专家所在的卡会拖慢整步，所以需要负载均衡损失和容量因子。
- 错答：「MoE 省算力，所以训练和部署都便宜」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)、F21

**Q41. Mamba/SSM 相对 Transformer 的推理优势是什么？**
- 答：在递归形式下，每一步的状态大小是固定的，decode 不需要随长度增长的 KV cache，每个 token 的代价是 O(1)；训练时则用卷积或并行扫描的形式来并行。代价是固定大小的状态会压缩历史，所以精确检索（比如长距离复制）不如 attention。
- 错答：「SSM 在所有任务上都比 attention 好」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)

**Q42. 深度学习集群调度和传统批处理调度有什么不同？**
- 答：主要有四点不同。分布式训练需要 gang scheduling，所有 worker 必须同时启动，否则先到的只能空等；它对拓扑敏感，希望放在同一个节点或同一个交换机下；作业可以弹性伸缩，也可以被抢占，前提是有 checkpoint；吞吐和分到的卡数不是线性关系，所以 Pollux 会同时调 batch 和资源。
- 错答：「按 GPU 数量排队就行」。
- 出处：[M18](../notes/module-18-architectures-moe-clusters.md)
