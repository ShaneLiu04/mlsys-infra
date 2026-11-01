# 工程手册：怎么定位、怎么证明、怎么提速

面试官问「如果慢/OOM/挂住了你怎么查」，要的是**顺序**和**能区分假设的那一个观测**。
每节先给可复述的结论，再给检查清单。公式在 [FORMULAS](FORMULAS.md)，设计题在
[SYSTEM-DESIGN](SYSTEM-DESIGN.md)。

## 1. 性能排查顺序

1. **先算上界**。用 roofline 算这个 kernel 或这一步的理论时间（F2、F16）。实测离上界 1.2 倍以内就别优化 kernel 了，去改算法或 batch。
2. **看时间线，不看平均值**。`torch.profiler` 或 Nsight Systems 看 GPU 上有没有空洞。空洞意味着 CPU 发射跟不上、同步点、dataloader 或通信没 overlap。
3. **再看单个 kernel**。Nsight Compute 看 achieved occupancy、DRAM throughput 占峰值比例、SM throughput。DRAM 接近峰值就是访存受限，下一步是融合或换低精度。
4. **最后才调超参**（block size、num_warps、bucket size），每次只动一个。

| 时间线上看到的 | 最可能的原因 | 区分它的观测 |
| --- | --- | --- |
| kernel 之间大量小空洞 | CPU 发射开销（小 kernel 太多、Python 开销） | 空洞宽度和 kernel 数成正比；CUDA Graph 或 `torch.compile` 后消失 |
| 每步开头一段空白 | dataloader 跟不上 | `num_workers` 加倍空白变短；GPU 空闲时 CPU 满载 |
| 反向结束后一大段 NCCL | 梯度 all-reduce 没和反向 overlap | DDP bucket 太大或 `find_unused_parameters=True` |
| 周期性的长尖峰 | `cudaMalloc` / 分配器碎片整理 / checkpoint | `torch.cuda.memory_stats` 里 `num_alloc_retries` 增长 |
| GPU 利用率 100% 但很慢 | 访存受限 kernel（逐元素、LayerNorm、softmax） | Nsight Compute 里 DRAM 吞吐接近峰值、SM 吞吐很低 |
| 多卡里一张卡总是最后到 | 负载不均（变长序列、MoE 热门专家）或慢卡 | 每 rank 的 step 时间分布；换卡后是否跟着走 |

## 2. 显存排查

**先分清是哪一块**。静态部分按 16N（训练）或 2N（推理）算出来，剩下的才是激活、KV、
临时 buffer 和碎片。

| 症状 | 常见原因 | 检查 |
| --- | --- | --- |
| 第一步就 OOM | 静态显存本来就放不下；logits `B·T·V·4` 字节太大 | 按 F10 算；`T=4096, V=128256, B=4` 的 fp32 logits 是 8.4 GB |
| 跑几百步后 OOM | 碎片；变长序列让缓存块对不上；把 loss 张量（带计算图）存进列表 | `memory_summary()` 里 reserved ≫ allocated；`.item()` 再存 |
| 评测时 OOM，训练不 OOM | 评测没包 `torch.no_grad()`；评测 batch 更大或生成更长 | 检查激活是否被保存 |
| 同卡起第二个进程失败 | 第一个进程的缓存不会还给别人；vLLM 按整卡比例预留 | `nvidia-smi` 看进程占用；串行或同进程切换 |
| 长序列 OOM 而短序列不会 | attention 分数 `B·h·T²` 或 logits 随 T 增长 | 换 SDPA/FlashAttention，分块算 log-softmax |

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 能显著减少变长负载下的碎片。
`torch.cuda.empty_cache()` 只把**本进程**未使用的缓存还给驱动，不会减少已分配的张量。

## 3. 分布式挂住与变慢

**挂住几乎都是集合通信不匹配**：某个 rank 少调或多调了一次 collective，或者调用顺序不同。

1. 打开 `NCCL_DEBUG=INFO`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`，设置合理的超时，让挂住变成报错。
2. 用 `py-spy dump` 看每个 rank 停在哪一行。停在不同的 collective 上就是顺序不一致。
3. 常见根因：只在 rank 0 上做的 `if` 分支里调了 collective；变长输入导致某个 rank 的循环多跑一轮；某些参数在部分 rank 上没有梯度。
4. 变慢先测带宽：`nccl-tests` 的 `all_reduce_perf` 报 bus bandwidth，和理论值（F11）比。NVLink 上只有理论值的一小部分时，查拓扑（`nvidia-smi topo -m`）、是否走了 PCIe、IB 是否被识别（`NCCL_IB_HCA`）。

## 4. 数值正确性

1. **找一个不依赖训练的恒等式**：TP 切分前后前向和梯度逐元素相等；FlashAttention 与朴素实现在 fp64 下差 1e-10 以内；KV cache 生成与无 cache 生成的贪心输出完全一致；推测解码的贪心输出与目标模型一致。本仓库每个实验都有这类测试。
2. **低精度要看累加**：bf16 只有 8 位有效位，求和、softmax、LayerNorm 统计量、loss 必须在 fp32 里做。fp16 训练要 loss scaling，`inf` 时跳过该步并降低 scale。
3. **并行改变求和顺序**：浮点加法不满足结合律，DP 卡数不同、all-reduce 算法不同，结果在最后几位不同。要求逐位复现时固定算法和顺序（`torch.use_deterministic_algorithms`）。
4. **量化先看分布**：先画每个通道的 absmax，再选 per-channel / per-group，或先做 SmoothQuant 式的缩放。

## 5. 提速手段按收益排序

| 场景 | 先做 | 再做 | 最后 |
| --- | --- | --- | --- |
| 训练，单卡 | bf16 + fused AdamW + FlashAttention | `torch.compile` 融合逐元素算子 | 手写 Triton kernel |
| 训练，多卡 | 梯度分桶 overlap；TP 留节点内 | ZeRO/FSDP 选 stage；PP 的 m ≥ 4p | 通信压缩、拓扑感知 |
| 推理，吞吐 | continuous batching + PagedAttention | 量化（FP8/INT8 权重、KV） | 推测解码（对吞吐帮助有限） |
| 推理，时延 | 减小 TP 内的 all-reduce（更少卡、FP8） | 推测解码、CUDA Graph | 自定义 attention kernel |
| 数据管线 | 预分词、内存映射、多 worker | 按长度分桶减少 padding | GPU 解码（图像/视频） |

## 6. 汇报数字的口径

- **吞吐**写清楚 token/s 是输入还是输出、每卡还是每节点、是否含 padding。
- **时延**报 P50 和 P99，写清楚负载（到达率、长度分布）。脱离负载的时延没有意义。
- **MFU** 用 6N 计（F22）；开了重计算的 HFU 要单独报。
- **加速比**写清楚基线：同硬件、同精度、同 batch。
- **本仓库的实验**都在 4 物理核（8 逻辑核）的 CPU 上跑，绝对时间不代表 GPU；能引用的是相对比较、正确性和由公式推出的模型输出。各实验 README 的「不能推出的结论」写明了边界。
