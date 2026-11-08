# 系统设计题

infra 岗的设计题考的不是框图画得漂不漂亮，而是你能不能先算账、再画图。很多人一上来就画网关和负载均衡，
却说不出需要多少张卡、瓶颈在哪一步。每道题都按五步走：澄清需求、估算、架构、瓶颈与取舍、追问。
数字都用 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md) 里的参数现场算，可以用 `python tools/mlsys_calc.py` 复核。
讲的时候先报估算结果，再画图。

| # | 题目 | 主要模块 | 一句话答案 |
| ---: | --- | --- | --- |
| D1 | 70B 模型在线推理，100 req/s | [M15](../notes/module-15-inference-kv-speculative.md) [M16](../notes/module-16-llm-serving.md) | prefill 吃算力、decode 吃带宽，算下来约 12 台 8×H100，其中一半多是给 prefill 的 |
| D2 | 70B 预训练，15T token | [M12](../notes/module-12-data-parallel-zero.md) [M13](../notes/module-13-tensor-pipeline-parallel.md) | 6ND = 6.3×10²⁴，1024 卡要半年，8192 卡约 23 天；TP8 × PP4 × DP |
| D3 | 7B 模型的 GRPO 训练系统 | [M15](../notes/module-15-inference-kv-speculative.md) [M16](../notes/module-16-llm-serving.md) | 时间花在 rollout；训练和推理共卡要 offload + sleep，权重同步 31 ms 不是瓶颈 |
| D4 | 一张卡服务 1000 个 LoRA | [M16](../notes/module-16-llm-serving.md) [M17](../notes/module-17-compression-quantization.md) | 不合并权重，按 adapter 分组做 batched gather-matmul，adapter 放 CPU 按需换入 |
| D5 | 万卡训练的容错 | [M12](../notes/module-12-data-parallel-zero.md) [M18](../notes/module-18-architectures-moe-clusters.md) | 故障间隔按小时计，checkpoint 要异步、分片、写内存再落盘 |
| D6 | 128K 上下文推理 | [M08](../notes/module-08-attention-flashattention.md) [M15](../notes/module-15-inference-kv-speculative.md) | 一条 8B 序列的 KV 16 GiB；瓶颈从权重变成 KV，靠 GQA、KV 量化、context parallel |

---

## D1. 70B 模型在线推理，100 req/s

结论先行：prefill 和 decode 要分开算账，因为前者受算力限制，后者受带宽限制。算下来大约要 12 个 8×H100 节点，其中一半多是给 prefill 的。

澄清：模型是 Llama-3-70B（bf16）。平均输入 2K token，输出 500 token。SLO 是 TTFT P99 < 2 s，TPOT P99 < 50 ms。硬件是 8×H100 的节点。

估算：先看一个节点能放多少并发，再分别算 decode 和 prefill 各需要几个节点。

```text
权重     70.6B × 2 B = 141 GB → 至少 2 张卡；取 TP=8，每卡 17.6 GB
KV/token 2·80·8·128·2 = 320 KB；TP=8 按 KV 头切，每卡 40 KB/token
KV 预算  每卡 80 × 0.9 − 17.6 − 4（激活与临时） ≈ 50 GB → 节点合计约 123 万 token
并发     每请求 2.5K token → 一个节点约 490 条并发序列

decode（batch 256，平均上下文 2.5K）
  读权重   141 GB / (8 × 3.35 TB/s) = 5.3 ms
  读 KV    256 × 2500 × 320 KB / 26.8 TB/s = 7.8 ms
  计算     2 × 70.6e9 × 256 / (8 × 989e12) = 4.6 ms   → 仍受访存限制
  单步下界 ≈ 13 ms → 理论 19.5K token/s/节点，按 50% 效率约 10K token/s

需求
  输出   100 × 500 = 50K token/s → decode 约 5 个节点
  prefill 100 × 2K = 200K token/s → 2 × 70.6e9 × 2e5 = 28 PFLOP/s
         单节点 7.9 PFLOP/s × 50% MFU ≈ 4 PFLOP/s → 约 7 个节点
  合计   约 12 个节点（96 张 H100）
```

注意 decode 这一段：batch 开到 256 时，读 KV 的时间已经超过了读权重的时间，而计算仍然是最短的一项。所以 decode 节点要按带宽和 KV 容量来配。

架构：请求先到网关，再经过一层前缀感知的路由，让带同一个 system prompt 的请求落到同一个实例上，命中 prefix cache。
推理实例用 vLLM 或 SGLang，TP=8，开 continuous batching、PagedAttention 和 chunked prefill。
规模上来以后做 prefill/decode 分离：prefill 池按算力配，decode 池按显存带宽和 KV 容量配，KV 通过节点间的 RDMA 传过去。

瓶颈与取舍：

- prefill 和 decode 混跑时，一个 2K 的 prefill 会让同一批里 decode 的 TPOT 抖一下。chunked prefill 把它切成每块 512 token，和 decode 拼在同一步里跑。
- 分离以后要传 KV。2K token × 320 KB = 640 MB，在 50 GB/s 的 IB 上约 13 ms，远小于 TTFT 的预算，所以分离是可行的。
- FP8 权重能把读权重的时间减半，TP 也可以降到 4，省掉一半的 all-reduce。代价是要做精度验证。
- 如果 system prompt 的共享率高，prefix cache 能省掉大部分 prefill 算力，这是最便宜的优化。

追问：

1. 为什么不用 TP=2？那样每张卡要放 70 GB 权重，KV 只剩几 GB，并发上不去，decode 的吞吐反而更低。
2. TTFT 的 P99 超了，先查什么？依次看排队时间（调度问题）、prefill 是否被 decode 挤占（chunk 大小）、有没有抢占后的重算（KV 不够）。
3. 流量涨 10 倍怎么办？先开 prefix cache 和 FP8，再水平扩实例。不要先上更大的 TP。

## D2. 70B 预训练，15T token

结论先行：总算力 6ND = 6.3×10²⁴ FLOP，1024 张卡要半年，8192 张卡约 23 天。并行方案是 TP8 × PP4 × DP，原则是让通信量最大的维度留在 NVLink 上。

澄清：序列长度 8K，global batch 16M token，bf16 混合精度加 AdamW。H100 集群，节点内是 NVLink，节点间是 400 Gb/s 的 IB（每卡一个端口）。

估算：先用 6ND 定卡数和天数，再用 16N 看静态显存怎么切。

```text
总计算  C = 6 × 70.6e9 × 15e12 = 6.35e24 FLOP
1024 卡、MFU 40%: 6.35e24 / (1024 × 989e12 × 0.4) = 1.57e7 s ≈ 181 天
8192 卡、MFU 40%: ≈ 23 天；每步 6 × 70.6e9 × 16e6 / (8192 × 989e12 × 0.4) ≈ 2.1 s

静态显存 16N = 1.13 TB
TP=8（节点内）× PP=4: 每卡 16N/32 = 35 GB
再加 ZeRO-1（DP=256）: 4N/32 + 12N/(32×256) ≈ 8.9 GB
DP 梯度：每卡分片 2N/32 = 4.4 GB，ring all-reduce 走 IB ≈ 2 × 4.4 / 50 = 0.18 s，可藏在反向里
```

架构：TP=8 放在节点内，因为它每层要做 4 次 all-reduce，只能走 NVLink。PP=4 跨节点，它只在 stage 之间点对点传激活，
每次 `micro_batch × T × d × 2` 字节。DP 放在最外层，配上 ZeRO-1 和梯度分桶 overlap。
调度用交错的 1F1B，每个 DP 组的 micro-batch 数取 m ≥ 4·PP，让气泡不超过 16%。重计算用选择性的，只重算 attention 里的 softmax 和 dropout。

瓶颈与取舍：

- 为什么不只用 ZeRO-3/FSDP？因为它每层的前向和反向都要 all-gather 权重，跨 IB 的通信量是 DP 的 1.5 倍。70B 在 8K 张卡上这样做是可行的，但 TP+PP 能把大部分通信留在 NVLink 上。
- PP 的级数越多，气泡越大；级数越少，每张卡的显存越多。所以先定 TP=8，再取能放得下的最小 PP。
- global batch 16M token 已经接近这个规模下的临界 batch，DP 不能无限加。再加卡就要靠 context parallel 或更大的 PP。

追问：

1. MFU 只有 25%，怎么查？先看 profile 里通信有没有暴露出来（没有 overlap），再看气泡、dataloader 和 kernel 有没有融合；
   最后看 micro-batch 是不是太小，让 GEMM 吃不饱。
2. 换成 MoE（总参数 400B，激活参数 40B）会怎么变？要加上 EP，all-to-all 放在节点内；显存按总参数算，
   FLOPs 按激活参数算。

## D3. 7B 模型的 GRPO 训练系统

结论先行：一步的时间主要花在 rollout 上。训练和推理共用一组卡时，要靠 offload 和 sleep 来回腾显存；权重同步只要 31 ms，不是瓶颈。

澄清：每步 512 个 prompt，每个 prompt 采样 8 条回复，最长 4K token。硬件是单节点 8×H100。

估算：分别算 rollout、训练和权重同步三段各要多久。

```text
rollout token  512 × 8 × ~2K（平均） ≈ 8.4M token/步
decode 吞吐     7B、batch 512、H100：读权重 14 GB / 3.35 TB/s ≈ 4.2 ms/步
               计算 2 × 7e9 × 512 / 989e12 ≈ 7.2 ms/步 → batch 512 已越过临界 batch（≈295），受算力限制
               → 单卡上限约 7 万 token/s，8 卡约 57 万 token/s → rollout 下界约 15 s
训练            6 × 7e9 × 8.4e6 = 3.5e17 FLOP / (8 × 989e12 × 0.4) ≈ 110 s
权重同步        14 GB / 450 GB/s ≈ 31 ms
```

上面的 rollout 时间是理想下界。实际上长尾样本会把它拉到和训练时间同一个量级，因为同步 RL 的一步要等最长的那条回复生成完。

架构：有两种做法。

第一种是共卡，也就是 veRL 和 OpenRLHF 的 hybrid engine。训练时让推理引擎 sleep，把它的 KV 和权重 offload 出去；
推理时再把优化器状态 offload 出去。两者在同一个进程里切换，避免两个进程各占一份显存，因为 `empty_cache` 只能释放自己进程的缓存。

第二种是分卡：rollout 卡和训练卡分开，异步流水，允许策略滞后一步，这种 off-policy 的偏差用重要性比值来修正。

瓶颈与取舍：

- 长尾：按长度分桶、限制最大生成长度，或者用部分回放（partial rollout）。
- 训练引擎和推理引擎算出来的 log-prob 数值不同，所以旧策略的 log-prob 要用训练引擎重算。
- 在同一张卡上同时起训练进程和 vLLM 会失败，因为 vLLM 按 `gpu_memory_utilization × 整卡显存` 的方式预留显存。
  这一条在 [cs229-r 的作品集](https://gitee.com/liu-xingyan04/cs229-r/blob/main/interview/PORTFOLIO.md) 里有实测。

## D4. 一张卡服务 1000 个 LoRA

结论先行：不要把 LoRA 合并进权重。同一个 batch 里的请求共享基座的 GEMM，LoRA 部分按 adapter 分组做 batched gather-matmul；adapter 平时放在 CPU 内存里，按需换入。

估算：基座是 8B 模型（16 GB）。r=16 的 LoRA 加在 q、k、v、o 上，每层约 4 × 16 × (4096 + 4096) ≈ 0.5M 个参数
（k、v 的输出维度因为 GQA 更小，这里按上限算），32 层约 17M 参数、34 MB。1000 个 adapter 就是 34 GB，
所以放在 CPU 内存里，GPU 上只缓存热门的几十个。

架构：不合并权重，因为合并之后每个 adapter 都变成一份 16 GB 的完整模型。同一个 batch 里的不同请求可以用不同的 adapter：
基座的 GEMM 是共享的，LoRA 部分用分段的 gather-matmul 来算，比如 Punica 的 SGMV；S-LoRA 则用统一分页，把 adapter 权重和 KV 放进同一个内存池。
调度器按 adapter 的亲和性组 batch，冷门的 adapter 从 CPU 异步换入。

追问：各个 adapter 的 rank 不一样怎么办？可以按 rank 分组，也可以都 pad 到最大的 rank。后者浪费一些算力，但 kernel 写起来简单。

## D5. 万卡训练的容错

结论先行：一万张卡的规模下，故障间隔是按小时计的。所以 checkpoint 必须异步、分片，先写内存再落盘，不能让训练停下来等它。

估算：单卡的年故障率按 2% 估，1 万张卡每年就是 200 次，平均约 44 小时一次；再算上网络和主机的故障，
实际是十几个小时就有一次。70B 模型的 checkpoint（权重加 fp32 优化器状态）约 16 × 70.6e9 ≈ 1.1 TB。

架构：用异步 checkpoint。先把状态拷到主机内存，每张卡几 GB，几秒钟就完成，训练接着跑；后台再分片写到分布式存储。
每个 DP 组只需要存一份。故障检测靠 NCCL 超时加心跳，发现后自动摘掉坏节点，从最近的 checkpoint 重启。
checkpoint 的间隔 T 可以用经验公式 `T ≈ √(2 × 保存开销 × 平均故障间隔)` 来定，这是 Young 公式。

追问：保存一次要 30 s、故障间隔 12 h 时，`√(2 × 30 × 43200) ≈ 1610 s`，也就是大约 27 分钟存一次。

## D6. 128K 上下文推理

结论先行：上下文到 128K 时，一条 8B 序列的 KV 就有 16 GiB，瓶颈从读权重变成了读 KV。应对手段是 GQA、KV 量化和 context parallel。

估算：Llama-3-8B 在 128K token 下，KV 是 128 KB × 131072 = 16 GiB，和权重一样大。
prefill 的 attention 每层是 `4T²d / 2`（因果只算一半）= 2 × 131072² × 4096 ≈ 1.4×10¹⁴ FLOP，32 层就是
4.5×10¹⁵。在 H100 上，单是 attention 就要约 5 s，这还是按 MFU 100% 算的下界。

架构：prefill 用 context parallel 或 ring attention 把序列切到多张卡上。decode 阶段的主要瓶颈是读 KV，
所以用 FP8/INT8 的 KV cache、选 GQA 模型、开 prefix cache。如果上下文超长、但模型只需要局部信息，可以考虑滑窗或稀疏 attention，
但这会改变模型行为，需要重新评测。

追问：为什么 decode 时 FlashAttention 帮不上大忙？因为 decode 时 Q 只有一行，瓶颈是把 KV 读一遍，
而不是 T² 的中间矩阵。这时需要的是沿 KV 长度切分来并行，也就是 Flash-Decoding 或 split-K。
