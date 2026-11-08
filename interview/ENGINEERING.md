# 工程手册：怎么定位、怎么证明、怎么提速

面试官问「如果训练慢了、OOM 了、挂住了，你怎么查」，想听的不是一串工具名，而是两样东西：
你按什么顺序查，以及哪一个观测能把几个假设区分开。每节先给一段能直接复述的结论，再给检查清单。
用到的公式在 [FORMULAS](FORMULAS.md)，设计题在 [SYSTEM-DESIGN](SYSTEM-DESIGN.md)。

## 1. 性能排查顺序

一句话版本：先算出理论上能有多快，再看时间线上时间花在哪，最后才看单个 kernel 和调超参。
顺序反过来的话，很容易花一天调 block size，最后发现 GPU 有一半时间在等 dataloader。

1. 先算上界。用 roofline 算出这个 kernel 或这一步的理论时间（F2、F16）。如果实测已经在上界的 1.2 倍以内，就别再优化 kernel 了，该去改算法或 batch。
2. 看时间线，不看平均值。用 `torch.profiler` 或 Nsight Systems 看 GPU 上有没有空洞。有空洞，说明 CPU 发射跟不上，或者有同步点、dataloader 拖后腿、通信没有和计算 overlap。
3. 再看单个 kernel。用 Nsight Compute 看 achieved occupancy、DRAM throughput 占峰值的比例和 SM throughput。DRAM 吞吐接近峰值就是访存受限，下一步该做融合或换低精度。
4. 最后才调超参，比如 block size、num_warps、bucket size。每次只动一个。

下面这张表是时间线上常见的几种形状，以及怎么确认是哪一种原因。

| 时间线上看到的 | 最可能的原因 | 区分它的观测 |
| --- | --- | --- |
| kernel 之间大量小空洞 | CPU 发射开销（小 kernel 太多、Python 开销） | 空洞宽度和 kernel 数成正比；CUDA Graph 或 `torch.compile` 后消失 |
| 每步开头一段空白 | dataloader 跟不上 | `num_workers` 加倍空白变短；GPU 空闲时 CPU 满载 |
| 反向结束后一大段 NCCL | 梯度 all-reduce 没和反向 overlap | DDP bucket 太大或 `find_unused_parameters=True` |
| 周期性的长尖峰 | `cudaMalloc` / 分配器碎片整理 / checkpoint | `torch.cuda.memory_stats` 里 `num_alloc_retries` 增长 |
| GPU 利用率 100% 但很慢 | 访存受限 kernel（逐元素、LayerNorm、softmax） | Nsight Compute 里 DRAM 吞吐接近峰值、SM 吞吐很低 |
| 多卡里一张卡总是最后到 | 负载不均（变长序列、MoE 热门专家）或慢卡 | 每 rank 的 step 时间分布；换卡后是否跟着走 |

## 2. 显存排查

OOM 时第一件事是分清是哪一块显存超了。静态部分可以直接算出来：训练按 16N 字节，推理按 2N 字节（N 是参数量）。
总占用减掉静态部分，剩下的才是激活、KV、临时 buffer 和碎片，再按下表对症状。

| 症状 | 常见原因 | 检查 |
| --- | --- | --- |
| 第一步就 OOM | 静态显存本来就放不下；logits `B·T·V·4` 字节太大 | 按 F10 算；`T=4096, V=128256, B=4` 的 fp32 logits 是 8.4 GB |
| 跑几百步后 OOM | 碎片；变长序列让缓存块对不上；把 loss 张量（带计算图）存进列表 | `memory_summary()` 里 reserved ≫ allocated；`.item()` 再存 |
| 评测时 OOM，训练不 OOM | 评测没包 `torch.no_grad()`；评测 batch 更大或生成更长 | 检查激活是否被保存 |
| 同卡起第二个进程失败 | 第一个进程的缓存不会还给别人；vLLM 按整卡比例预留 | `nvidia-smi` 看进程占用；串行或同进程切换 |
| 长序列 OOM 而短序列不会 | attention 分数 `B·h·T²` 或 logits 随 T 增长 | 换 SDPA/FlashAttention，分块算 log-softmax |

变长负载下碎片多的话，可以设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，它能明显减少碎片。
另一个常见误解是 `torch.cuda.empty_cache()` 能腾出显存。它只会把本进程缓存着、没在用的块还给驱动，已经分配出去的张量一个字节都不会少。

## 3. 分布式挂住与变慢

多卡训练挂住，几乎都是集合通信对不上：某个 rank 少调或多调了一次 collective，或者各 rank 调用的顺序不一样。
所以排查的思路是先让「挂住」变成「报错」，再找出哪个 rank 走了不同的路。

1. 打开 `NCCL_DEBUG=INFO` 和 `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，再设一个合理的超时。这样挂住会变成带信息的报错。
2. 用 `py-spy dump` 看每个 rank 停在哪一行。如果它们停在不同的 collective 上，就是调用顺序不一致。
3. 常见的根因有三种：在只有 rank 0 进入的 `if` 分支里调了 collective；变长输入让某个 rank 的循环多跑了一轮；某些参数在部分 rank 上没有梯度。
4. 如果不是挂住而是变慢，先测带宽。`nccl-tests` 的 `all_reduce_perf` 会报 bus bandwidth，拿它和理论值（F11）比。NVLink 上只跑到理论值的一小部分时，去查拓扑（`nvidia-smi topo -m`）、是不是走了 PCIe、IB 有没有被识别（`NCCL_IB_HCA`）。

## 4. 数值正确性

被问「你怎么知道你的实现是对的」时，最有说服力的回答是一个不依赖训练结果的恒等式。
loss 曲线看起来正常不能证明实现正确，逐元素相等才能。

1. 找一个恒等式来测。比如 TP 切分前后的前向输出和梯度逐元素相等；FlashAttention 和朴素实现在 fp64 下相差 1e-10 以内；带 KV cache 和不带 cache 的贪心生成输出完全一致；推测解码的贪心输出和目标模型一致。本仓库每个实验都有这类测试。
2. 低精度要盯住累加。bf16 只有 8 位有效位，所以求和、softmax、LayerNorm 的统计量和 loss 都必须在 fp32 里算。fp16 训练还要做 loss scaling，出现 `inf` 时跳过这一步并调低 scale。
3. 并行会改变求和顺序。浮点加法不满足结合律，所以 DP 卡数不同、all-reduce 算法不同时，结果的最后几位会不一样。要求逐位复现，就得固定算法和顺序（`torch.use_deterministic_algorithms`）。
4. 量化之前先看分布。先画出每个通道的 absmax，再决定用 per-channel 还是 per-group，或者先做 SmoothQuant 那样的缩放。

## 5. 提速手段按收益排序

被问「怎么让它更快」时，按下表从左往右说：左边的手段改动小、收益大，右边的投入大、只在前面做完后才值得。

| 场景 | 先做 | 再做 | 最后 |
| --- | --- | --- | --- |
| 训练，单卡 | bf16 + fused AdamW + FlashAttention | `torch.compile` 融合逐元素算子 | 手写 Triton kernel |
| 训练，多卡 | 梯度分桶 overlap；TP 留节点内 | ZeRO/FSDP 选 stage；PP 的 m ≥ 4p | 通信压缩、拓扑感知 |
| 推理，吞吐 | continuous batching + PagedAttention | 量化（FP8/INT8 权重、KV） | 推测解码（对吞吐帮助有限） |
| 推理，时延 | 减小 TP 内的 all-reduce（更少卡、FP8） | 推测解码、CUDA Graph | 自定义 attention kernel |
| 数据管线 | 预分词、内存映射、多 worker | 按长度分桶减少 padding | GPU 解码（图像/视频） |

## 6. 汇报数字的口径

同一个数字，口径没说清楚就没法比较。报数时把下面几项说出来：

- 吞吐：token/s 指的是输入还是输出，每卡还是每节点，含不含 padding。
- 时延：报 P50 和 P99，并说明负载（到达率和长度分布）。脱离负载谈时延没有意义。
- MFU：按 6N 计算（F22）。开了重计算时，HFU 要单独报。
- 加速比：说清楚基线，要求同硬件、同精度、同 batch。
- 本仓库的实验：GPU 数字是在 RTX 5090 等消费级或工作站卡上实测的，引用时带上卡型；用 H100 参数算的结果和 4 核 CPU 上的 numpy 复现属于模型，只能讲相对关系和机制。各实验 README 的「不能推出的结论」写明了边界。
