# 系统设计题

infra 岗的设计题不考框图，考的是**先算账再画图**。每题按五步走：澄清 → 估算 → 架构 →
瓶颈与取舍 → 追问。数字都用 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md) 里的参数现算，
可以用 `python tools/mlsys_calc.py` 复核。讲的时候先报估算结果，再画图。

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

**澄清**：模型 Llama-3-70B（bf16）。平均输入 2K token、输出 500 token。SLO：TTFT P99 < 2 s，
TPOT P99 < 50 ms。硬件 8×H100 节点。

**估算**

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

**架构**：网关 → 前缀感知的路由（同一个 system prompt 落到同一实例，命中 prefix cache）→
推理实例（vLLM/SGLang，TP=8，continuous batching + PagedAttention + chunked prefill）。
规模上来后做 prefill/decode 分离：prefill 池按算力配，decode 池按显存带宽和 KV 容量配，
KV 通过节点间 RDMA 传过去。

**瓶颈与取舍**

- prefill 和 decode 混跑时，一个 2K 的 prefill 会让同批 decode 的 TPOT 抖一下。chunked prefill 把它切成 512 token 一块，和 decode 拼在同一步里。
- KV 传输代价：2K token × 320 KB = 640 MB，50 GB/s IB 上约 13 ms，远小于 TTFT 预算，所以分离可行。
- FP8 权重把读权重时间减半，TP 可以降到 4，省掉一半 all-reduce。代价是精度验证。
- 系统提示词共享率高时，prefix cache 能省掉大部分 prefill 算力，这是最便宜的优化。

**追问**

1. 为什么不用 TP=2？每卡 70 GB 权重，KV 只剩几 GB，并发上不去，decode 吞吐反而低。
2. P99 TTFT 超了先查什么？排队时间（调度）、prefill 被 decode 挤占（chunk 大小）、抢占重算（KV 不够）。
3. 流量涨 10 倍？先开 prefix cache 和 FP8，再水平扩实例；不要先上更大 TP。

## D2. 70B 预训练，15T token

**澄清**：序列长度 8K，global batch 16M token，bf16 混合精度 + AdamW，H100 集群，节点内
NVLink、节点间 400 Gb/s IB（每卡一端口）。

**估算**

```text
总计算  C = 6 × 70.6e9 × 15e12 = 6.35e24 FLOP
1024 卡、MFU 40%: 6.35e24 / (1024 × 989e12 × 0.4) = 1.57e7 s ≈ 181 天
8192 卡、MFU 40%: ≈ 23 天；每步 6 × 70.6e9 × 16e6 / (8192 × 989e12 × 0.4) ≈ 2.1 s

静态显存 16N = 1.13 TB
TP=8（节点内）× PP=4: 每卡 16N/32 = 35 GB
再加 ZeRO-1（DP=256）: 4N/32 + 12N/(32×256) ≈ 8.9 GB
DP 梯度：每卡分片 2N/32 = 4.4 GB，ring all-reduce 走 IB ≈ 2 × 4.4 / 50 = 0.18 s，可藏在反向里
```

**架构**：TP=8 放节点内（每层 4 次 all-reduce，只能走 NVLink）；PP=4 跨节点（只传
激活，`micro_batch × T × d × 2` 字节，点对点）；DP 在最外层，配 ZeRO-1 和梯度分桶 overlap。
1F1B 交错调度，每个 DP 组的 micro-batch 数 m ≥ 4·PP 让气泡 ≤ 16%。选择性重计算只重算
attention 里的 softmax/dropout。

**瓶颈与取舍**

- 为什么不只用 ZeRO-3/FSDP：每层前向和反向都要 all-gather 权重，跨 IB 的通信量是 DP 的 1.5 倍；70B 在 8K 卡上能做，但 TP+PP 能把大部分通信留在 NVLink。
- PP 级数越多气泡越大，PP 越少每卡显存越多；先定 TP=8，再取能放下的最小 PP。
- global batch 16M token 已接近这个规模的临界 batch，DP 不能无限加，再加卡要靠 context parallel 或更大 PP。

**追问**

1. MFU 只有 25%，怎么查？先看 profile 里通信是否暴露（没 overlap）、气泡、dataloader、
   kernel 是否融合；再看 micro-batch 是否太小导致 GEMM 不饱和。
2. 换成 MoE（总参数 400B，激活 40B）怎么变？加 EP，all-to-all 放节点内；显存按总参数算，
   FLOPs 按激活参数算。

## D3. 7B 模型的 GRPO 训练系统

**澄清**：每步 512 个 prompt，每个采样 8 条，最长 4K token；单节点 8×H100。

**估算**

```text
rollout token  512 × 8 × ~2K（平均） ≈ 8.4M token/步
decode 吞吐     7B、batch 512、H100：读权重 14 GB / 3.35 TB/s ≈ 4.2 ms/步
               计算 2 × 7e9 × 512 / 989e12 ≈ 7.2 ms/步 → batch 512 已越过临界 batch（≈295），受算力限制
               → 单卡上限约 7 万 token/s，8 卡约 57 万 token/s → rollout 下界约 15 s
训练            6 × 7e9 × 8.4e6 = 3.5e17 FLOP / (8 × 989e12 × 0.4) ≈ 110 s
权重同步        14 GB / 450 GB/s ≈ 31 ms
```

上面的 rollout 是理想下界，长尾样本会把它拉到训练时间的量级——同步 RL 的一步由最长的那条
回复决定。

**架构**：两种。共卡（veRL/OpenRLHF 的 hybrid engine）：训练时把推理引擎的 KV 和权重
sleep/offload，推理时把优化器状态 offload；同一进程内切换，避免两个进程各占显存
（`empty_cache` 只释放自己进程的缓存）。分卡：rollout 卡和训练卡分开，异步流水，
允许策略滞后一步（off-policy 用重要性比值修正）。

**瓶颈与取舍**

- 长尾：按长度分桶、限制最大生成长度、部分回放（partial rollout）。
- 训练引擎与推理引擎的 log-prob 数值不同，旧策略 log-prob 要用训练引擎重算。
- 同卡同时起训练和 vLLM 会因为 `gpu_memory_utilization × 整卡显存` 的预留方式失败。
  这条在 [cs229-r 的作品集](https://gitee.com/liu-xingyan04/cs229-r/blob/main/interview/PORTFOLIO.md) 里有实测。

## D4. 一张卡服务 1000 个 LoRA

**估算**：基座 8B（16 GB）。r=16 的 LoRA 加在 q,k,v,o 上：每层约 4 × 16 × (4096 + 4096)
≈ 0.5M 参数（k,v 的输出维度因 GQA 更小，这里按上限算），32 层约 17M 参数、34 MB。1000 个
是 34 GB，放 CPU 内存，GPU 上缓存热门的几十个。

**架构**：不合并权重（合并后每个 adapter 都是一份 16 GB 的完整模型）。同一 batch 里不同
请求用不同 adapter：基座 GEMM 共享，LoRA 部分用分段 gather-matmul（Punica 的 SGMV、
S-LoRA 的统一分页把 adapter 权重和 KV 放在同一个内存池）。调度器按 adapter 亲和性批处理，
冷 adapter 异步从 CPU 换入。

**追问**：rank 不一样怎么办？按 rank 分组，或 pad 到最大 rank；后者浪费算力但 kernel 简单。

## D5. 万卡训练的容错

**估算**：单卡年故障率按 2% 估，1 万卡每年 200 次，平均约 44 小时一次；加上网络和主机故障，
实际是十几个小时一次。70B 的 checkpoint（权重 + fp32 优化器状态）约 16 × 70.6e9 ≈ 1.1 TB。

**架构**：异步 checkpoint——先拷到主机内存（每卡几 GB，几秒），训练继续，后台分片写分布式
存储；每 DP 组只需要一份。故障检测靠 NCCL 超时 + 心跳；自动摘除坏节点，从最近 checkpoint
重启。checkpoint 间隔 T 的经验公式：`T ≈ √(2 × 保存开销 × 平均故障间隔)`（Young 公式）。

**追问**：保存 30 s、故障间隔 12 h：`√(2 × 30 × 43200) ≈ 1610 s`，约 27 分钟存一次。

## D6. 128K 上下文推理

**估算**：Llama-3-8B，128K token 的 KV 是 128 KB × 131072 = 16 GiB，和权重一样大。
prefill attention：每层 `4T²d / 2`（因果）= 2 × 131072² × 4096 ≈ 1.4×10¹⁴ FLOP，32 层
4.5×10¹⁵，H100 上单是 attention 就要约 5 s（MFU 100% 的下界）。

**架构**：prefill 用 context parallel / ring attention 把序列切到多卡；decode 阶段 KV 读取
是主瓶颈，用 FP8/INT8 KV cache、GQA 模型、prefix cache；超长且只用到局部信息时考虑
滑窗或稀疏 attention（会改变模型行为，需要评测）。

**追问**：为什么 decode 时 FlashAttention 帮不上大忙？decode 的 Q 只有一行，瓶颈是把 KV 读
一遍；需要的是沿 KV 长度切分并行（Flash-Decoding / split-K），而不是减少 T² 的中间矩阵。
