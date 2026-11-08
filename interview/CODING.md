# 手写题

infra 岗的手写题来来回回就是下面这些。每道题列了建议限时、本仓库的参考实现、能直接跑的测试，
以及最容易写错的地方。

练法很简单：不看参考实现，在一个空文件里从头写，写完跑对应的测试。测试没过再去翻参考，
因为自己先卡住一次，才记得住错在哪。

测试命令有两种。实验目录下的题，在实验目录里跑 `python -m pytest -q tests/<文件> -k <关键字>`；
`tools/` 下的题，在仓库根目录跑 `python -m pytest -q tools/tests/<文件>`。

## 一、Attention 与推理（推理岗必练）

推理岗几乎一定会让你写 attention 或者 decode 循环里的某一段。这组题的难点不在代码量，
而在数值稳定性和边界条件：分块时旧的累加量要重新缩放，自回归时位置要对上。

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C1 | 数值稳定的 softmax / online softmax | 10 min | [`attention.py`](../solutions/cs229s/project1-transformer-kernels/code/tfkernels/attention.py) `online_softmax` | `test_kernels.py` | 不减最大值；合并两段时忘了缩放旧的分母 |
| C2 | FlashAttention 前向（含因果） | 30 min | 同上 `flash_forward` | `test_kernels.py -k flash` | 旧的输出累加器 `O` 也要乘 `exp(m_old − m_new)`；块大小不整除 T；因果时整块跳过的条件 |
| C3 | FlashAttention 反向 | 45 min | 同上 `flash_backward` | `test_kernels.py -k backward` | `D = rowsum(dO ∘ O)` 这一项；用保存的 logsumexp 重建 P |
| C4 | 带 KV cache 的自回归生成 | 25 min | [`model.py`](../solutions/cs229s/project2-kv-speculative/code/kvspec/model.py) `KVCache`、`generate_kv` | `test_kvspec.py -k "kv_cache or generate_kv"` | 位置编码要用绝对位置而不是 0；prefill 和 decode 走同一条路径 |
| C5 | 推测解码的接受-拒绝采样 | 25 min | [`speculative.py`](../solutions/cs229s/project2-kv-speculative/code/kvspec/speculative.py) `accept_reject_sample` | `test_kvspec.py -k accept` | 全部接受时要从目标分布再采一个 bonus token；残差分布 `max(0, p−q)` 要归一化 |
| C6 | temperature / top-k / top-p 采样 | 15 min | [`sampling.py`](../tools/sampling.py) | `tools/tests/test_sampling.py` | top-p 要保留越过阈值的那个 token；顺序是先 temperature 再截断 |
| C7 | 分页 KV 块管理器（含 fork 与 CoW） | 40 min | [`block_manager.py`](../solutions/cmu15849/serving-memory/code/servesim/block_manager.py) `BlockManager` | `test_block_manager.py` | 引用计数归零才回收；写共享块前先复制；最后一块未满时的追加位置 |

## 二、分布式训练（训练岗必练）

训练岗的题考的是你能不能把通信模式和切分方式写对。这几题都可以在单进程里模拟多个 rank，
所以最容易出错的地方是下标：哪一步发哪一块，切不整除时多出来的部分归谁。

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C8 | ring all-reduce（reduce-scatter + all-gather） | 25 min | [`collectives.py`](../solutions/cmu15849/distributed-training/code/disttrain/collectives.py) `ring_allreduce` | `test_collectives.py` | 第 s 步 rank r 发送的分片下标是 `(r − s) mod p`；长度不能被 p 整除时的切分 |
| C9 | Megatron 列并行 + 行并行 MLP | 25 min | [`tensor_parallel.py`](../solutions/cmu15849/distributed-training/code/disttrain/tensor_parallel.py) `ParallelMLP` | `test_tensor_parallel.py` | bias 在 all-reduce 之后只加一次；反向时 f 和 g 两个算子的通信互换 |
| C10 | ZeRO 的一步更新（stage 1/2/3） | 40 min | [`zero.py`](../solutions/cmu15849/distributed-training/code/disttrain/zero.py) `train_zero` | `test_zero.py` | stage 2 用 reduce-scatter 代替 all-reduce；更新后要 all-gather 参数 |
| C11 | 流水线调度模拟（GPipe / 1F1B） | 30 min | [`pipeline.py`](../solutions/cmu15849/distributed-training/code/disttrain/pipeline.py) `simulate` | `test_pipeline.py` | 1F1B 的 warmup 前向数是 `p − stage − 1`；反向依赖下一级的反向 |
| C12 | √n 激活检查点计划 | 20 min | [`checkpointing.py`](../solutions/cmu15849/distributed-training/code/disttrain/checkpointing.py) `sqrt_checkpoint` | `test_checkpointing.py` | 峰值显存 = 检查点 + 当前段重算的激活，不是只有检查点 |

## 三、并行原语与算子（算子/芯片岗必练）

算子和芯片岗更关心你对并行原语和片上存储的理解。scan 和 compaction 是很多 GPU 算法的积木，
分块矩阵乘和 systolic array 则考你能不能把容量约束和数据流动画清楚。

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C13 | Blelloch exclusive scan | 25 min | [`scan.py`](../solutions/cs149/asst3-scan-render/code/scanrender/scan.py) `blelloch_exclusive` | `test_scan_render.py -k blelloch` | 非 2 的幂要补齐；down-sweep 前把根置 0；整数溢出 |
| C14 | find_repeats / stream compaction | 15 min | [`primitives.py`](../solutions/cs149/asst3-scan-render/code/scanrender/primitives.py) | `test_scan_render.py -k repeats` | 用 exclusive scan 的结果当写入下标；最后一个元素的边界 |
| C15 | 带依赖的线程池（条件变量） | 40 min | [`systems.py`](../solutions/cs149/asst2-task-scheduler/code/tasksys/systems.py) `TaskSystemParallelThreadPoolSleeping` | `test_tasksys.py` | 等待条件要放在 while 循环里（虚假唤醒）；`sync` 要等所有 launch 而不是当前一个 |
| C16 | 分块矩阵乘（满足片上容量约束） | 20 min | [`tiling.py`](../solutions/cs149/asst4-dnn-accelerator/code/dnnaccel/tiling.py) `tiled_matmul` | `test_dnnaccel.py -k tiled` | 边缘块不整除；K 方向累加放在最内层还是最外层决定了 C 块要不要反复读写 |
| C17 | im2col 卷积 | 20 min | [`im2col.py`](../solutions/cs149/asst4-dnn-accelerator/code/dnnaccel/im2col.py) | `test_dnnaccel.py -k im2col` | 输出尺寸 `(H + 2p − k)/s + 1`；通道和核位置的展开顺序要与权重 reshape 一致 |
| C18 | systolic array 矩阵乘模拟 | 30 min | [`systolic.py`](../solutions/cs149/asst4-dnn-accelerator/code/dnnaccel/systolic.py) `systolic_matmul` | `test_dnnaccel.py -k systolic` | 输入要按行错开一个周期进入；填充和排空周期 |

## 四、框架与压缩

这组题代码短，但每题都有一个一旦写错结果就整体不对的细节，比如梯度没累加、scale 除了零。
限时短，适合作为面试里的热身题来练。

| # | 题目 | 限时 | 参考实现 | 测试 | 最容易错 |
| ---: | --- | --- | --- | --- | --- |
| C19 | 反向模式自动微分（拓扑序） | 30 min | [`autodiff.py`](../solutions/cmu15849/compiler-compression/code/compkit/autodiff.py) `Tensor` | `test_autodiff.py` | 一个节点被多次使用时梯度要累加；广播的反向要按维度求和 |
| C20 | 逐元素算子融合 pass | 30 min | [`graph_ir.py`](../solutions/cmu15849/compiler-compression/code/compkit/graph_ir.py) `fuse_elementwise` | `test_graph_ir.py` | 中间结果被融合组外的节点使用时不能融合 |
| C21 | absmax 量化（per-tensor / per-channel / per-group） | 15 min | [`quantization.py`](../solutions/cmu15849/compiler-compression/code/compkit/quantization.py) `quantize_weight` | `test_compression.py -k quant` | int4 的对称范围是 ±7；全零行的 scale 为 0 要防除零 |
| C22 | LoRA 线性层 | 15 min | [`lora.py`](../solutions/cmu15849/compiler-compression/code/compkit/lora.py) `LoRALinear` | `test_compression.py -k lora` | B 初始化为 0 才能保证初始输出等于基座；缩放是 `α/r` |
| C23 | 2:4 结构化稀疏掩码 | 10 min | [`pruning.py`](../solutions/cmu15849/compiler-compression/code/compkit/pruning.py) `nm_mask` | `test_compression.py -k pruning` | 按输入维度每 4 个一组，不是按输出维度 |

## 面试现场的顺序

现场写题时，面试官看的不只是最后的代码，还有你怎么组织问题。下面这个顺序可以照着走：

1. 动笔之前先说接口和 shape，比如「Q 是 `[T, d]`，块大小是 `Br × Bc`」。这样双方对题目的理解先对齐了。
2. 先写一个朴素版本当对照，再写优化版本。然后说你打算怎么测：和朴素版逐元素比较，覆盖边界长度和因果的情况，必要时在 fp64 下比。
3. 写完主动讲复杂度和瓶颈，也就是 FLOPs、访存量和显存各是多少。能把这些说清楚，比多写一个优化更加分。
