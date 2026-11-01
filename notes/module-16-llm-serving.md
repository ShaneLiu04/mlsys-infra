# Module 16 — LLM 推理服务系统

- 课程出处：CS229S Fall 2023《Scheduling ML Clusters》 · CMU 15-849《Library to Accelerate Training and Inference（客座）》 · 课程之后的系统：vLLM、SGLang、Sarathi-Serve、DistServe、S-LoRA
- 必读论文：Yu et al., Orca, 2022；Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention, 2023；Zheng et al., SGLang (RadixAttention), 2024；Agrawal et al., Sarathi-Serve, 2024；Zhong et al., DistServe, 2024；Patel et al., Splitwise, 2024；Sheng et al., S-LoRA, 2024；Chen et al., Punica, 2024
- 对应实验：[15-849 服务与显存](../solutions/cmu15849/serving-memory/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-serving-memory/serving_memory_report.pdf)（实测：真实引擎分页 KV 浪费 86.8% → 1.26%，并发 7.0 倍）
- 状态：初稿

## 摘要

M15 说明 decode 由字节决定、必须把 batch 做大；服务系统的任务就是在延迟 SLO 下把 batch 做大。
三个约束互相牵制：KV 显存决定能并发多少请求，调度决定 prefill 和 decode 怎么共享 GPU，SLO 决定 batch 能大到什么程度。
Orca 的 iteration-level 调度和 vLLM 的 PagedAttention 解决前两个问题：每步都可以加入或移出请求，KV 按固定大小的块分配，碎片接近零。
之后的系统在同一框架上继续优化：前缀共享（RadixAttention）、chunked prefill、prefill-decode 分离、多 LoRA 共享底座。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 指标：TTFT、TPOT（ITL）、吞吐、P99，以及 goodput = 同时满足 TTFT 与 TPOT SLO 的请求速率。
2. 静态批要等最长请求结束；continuous batching 每个 decode 步都能换入换出请求。
3. PagedAttention：KV 按块（vLLM 默认 16 token）分配，block table 把逻辑块映射到物理块；内部碎片只在最后一块，无外部碎片。
4. 前缀缓存把共同前缀的 KV 只算一次、存一份；chunked prefill 把长 prompt 切块和 decode 混跑，抑制 TPOT 尖峰。
5. Little's law：并发 = 到达率 × 平均停留时间，是容量规划的第一步。

**必须能默写的公式**

```text
端到端延迟 = TTFT + (n_out − 1) × TPOT
Little's law：L = λ · W
每块 KV 字节 = block_size × 2 · L_layers · n_kv · d_head · bytes
内部碎片 ≤ (block_size − 1) 个 token / 序列
可并发序列 ≈ (HBM × 利用率 − 权重) / (每 token KV × 平均上下文)
decode 步时间 ≈ (权重字节 + Σ_i T_i × 每 token KV) / HBM 带宽
静态批利用率 ≈ mean(n_out) / max(n_out)
```

**一句话区分**

- 静态批 vs continuous batching：按请求成批 vs 按迭代成批。
- swap vs recompute 抢占：KV 搬到 CPU 再搬回 vs 丢掉 KV、恢复时重跑 prefill。
- chunked prefill vs PD 分离：同一 GPU 上切块混跑 vs prefill 和 decode 放不同 GPU，传 KV。
- S-LoRA vs Punica：统一分页管理适配器与 KV vs 用 SGMV kernel 让不同 LoRA 的请求同批计算。

## 学习目标

- 定义并区分 TTFT、TPOT、吞吐与 goodput；
- 解释 PagedAttention 的 block table、copy-on-write 和抢占机制；
- 说明前缀缓存、chunked prefill、PD 分离各自解决什么瓶颈；
- 用 Little's law 和 KV 公式做一次单卡容量规划；
- 比较 FCFS、SJF 近似与公平调度的取舍。

## 先修知识

- M15：KV cache 公式、decode 步时间下界、TTFT/TPOT；
- M04：内存分配与碎片；
- M13：张量并行。

## 核心概念

### 1. 指标

TTFT 是用户看到第一个字的时间，包含排队与 prefill。TPOT（ITL）是相邻两个 token 的间隔。
吞吐（token/s 或 req/s）是运营指标；goodput 是满足 SLO 的吞吐，DistServe 以它为优化目标。
P99 比均值重要：一个长 prompt 的 prefill 会让同批所有请求的这一步变慢，平均看不出来。

### 2. 从静态批到 continuous batching

静态批：凑齐 B 个请求一起跑，直到最长的结束。输出长度 50–500 时，平均利用率约 mean/max。
Orca：调度粒度从请求降到迭代。每步结束后，完成的请求立即离开，新请求立即加入。
不同请求长度不一，attention 无法直接拼成一个张量。Orca 的 selective batching：线性层把所有 token 拼成一个 GEMM，attention 按请求分别做。

### 3. PagedAttention

连续预分配 KV（按最大长度）造成三种浪费：为未来 token 预留、内部碎片、外部碎片。vLLM 论文报告此前系统只有约 20%–40% 的 KV 显存存的是真实 token 状态。
vLLM 把 KV 切成固定大小的物理块，每个序列维护 block table。新 token 填满一块才申请下一块，所以每个序列最多浪费 block_size − 1 个 token。
块大小相同，空闲块可以分给任何序列，没有外部碎片。代价：attention kernel 要按 block table 间接寻址。

### 4. 共享与抢占

并行采样与 beam search：多个输出共享 prompt 的物理块，块上有引用计数；某个分支要写入一个被共享的块时才复制（copy-on-write）。
显存不足时抢占：swap 把被抢占序列的块拷到 CPU，恢复时拷回，代价是 PCIe 带宽；recompute 直接丢弃，恢复时把已生成的 token 拼到 prompt 后重新 prefill，代价是算力。短序列 recompute 更便宜，长序列 swap 更便宜。

### 5. 前缀缓存与 RadixAttention

多个请求共享系统提示、few-shot 示例、多轮对话历史。SGLang 用 radix tree 以 token 序列为键索引 KV 块，新请求做最长前缀匹配，命中部分跳过 prefill。
叶子按 LRU 驱逐，正在被请求引用的节点不驱逐。调度时优先排命中率高的请求（cache-aware scheduling）进一步提高复用。

### 6. Chunked prefill

一个 8K 的 prefill 插进 decode 批，会让这一步时间从十几毫秒涨到几百毫秒，所有在跑请求的 TPOT 出现尖峰。
Sarathi-Serve 把 prefill 切成固定大小的块（如 512 token），每步的 token 预算 = decode token 数 + 一个 prefill 块，decode 不停顿。
代价：长 prompt 的 TTFT 略增，因为它被分成多步，且每块都要读一遍已有前缀的 KV。

### 7. Prefill-decode 分离

prefill 算力受限，decode 访存受限，混在一起互相干扰，且两者最优并行度不同。
DistServe / Splitwise 把它们放在不同 GPU 池，prefill 完成后把 KV 传给 decode 实例。两个池可独立扩缩、独立选 TP 度。
代价是 KV 传输：需要 NVLink 或 RDMA，并和计算重叠。

### 8. 多 LoRA 服务与调度

S-LoRA：底座权重共享一份，数千个适配器放在主存，按需换入；适配器与 KV 用统一分页池管理。
Punica：SGMV kernel 让一个批里不同请求乘不同的 B·A，底座 GEMM 仍合批。
调度：FCFS 简单但有队头阻塞；SJF 能降平均延迟但输出长度未知，只能用预测器近似，且会饿死长请求；公平调度（如按每个租户已服务的 token 计数）防止单个用户占满。

## 关键推导

### 碎片上界

块大小 k，序列长度 T，占用 ⌈T/k⌉ 块，浪费 ⌈T/k⌉·k − T ≤ k − 1 个 token。
平均浪费约 k/2。T=1500、k=16 时浪费约 8/1500 = 0.5%。
连续预分配按 T_max=8192 时，T=1500 的序列浪费 (8192 − 1500)/8192 = 82%。

### Little's law

稳态下系统内平均请求数 L = 到达率 λ × 平均停留时间 W，与分布无关。
用法：给定目标 λ 和 SLO 推出的 W，得到必须支持的并发 L，再用 KV 公式检查显存放不放得下。

## 深入推导

### 单卡容量规划：Llama-3-8B，一张 H100

工作负载：prompt 1000、输出 500，平均上下文按 1250 计、峰值 1500。可用 HBM 按 90%（72 GB）。
KV 预算 72 − 16.06 = 55.9 GB，每 token 131072 B → 约 42.7 万 token；每条峰值 1500 → 最多 284 条并发。
并发 L 时 decode 步时间 t(L) = (16.06 GB + L × 1250 × 131072 B) / 3.35 TB/s = 4.79 ms + L × 0.049 ms。
L = 256：t = 4.79 + 12.5 = 17.3 ms，decode 吞吐 256/17.3 ms ≈ 14800 token/s，只看 decode 可服务 29.6 req/s。
prefill 成本：每请求 2 × 8.03e9 × 1000 = 1.6e13 FLOP，MFU 50% 时 32.5 ms。
若不重叠，每秒 GPU 时间：R × (500/256 × 17.3 ms + 32.5 ms) = R × 66.3 ms ≤ 1 s → R ≈ 15 req/s。
此时 decode 只拿到约一半时间，有效 TPOT ≈ 34 ms；Little 校验：W ≈ 500 × 34 ms = 17 s，L = 15 × 17 ≈ 256，自洽。
结论：若 SLO 是 TPOT ≤ 25 ms，要么降并发，要么用 chunked prefill 或 PD 分离。分离后一张 decode 卡约 29.6 req/s，一张 prefill 卡 1/32.5 ms ≈ 30.8 req/s，这个负载下约 1:1 配比。

### PD 分离的 KV 传输

8B、prompt 2048：KV 2048 × 128 KB = 268 MB；IB 50 GB/s 上 5.4 ms，相对 66 ms 的 prefill 可忽略且可按层流水传输。
70B、prompt 2048：671 MB，IB 上 13.4 ms；NVLink 450 GB/s 上 1.5 ms。

```python
class BlockAllocator:
    """vLLM 风格：固定大小物理块 + 引用计数 + copy-on-write。"""
    def __init__(self, num_blocks, block_size=16):
        self.free = list(range(num_blocks)); self.ref = [0] * num_blocks
        self.block_size = block_size
        self.tables: dict[int, list[int]] = {}      # seq_id -> 物理块号列表
        self.lens: dict[int, int] = {}

    def _alloc(self):
        if not self.free:
            raise MemoryError("preempt: swap or recompute")
        b = self.free.pop(); self.ref[b] = 1; return b

    def append_token(self, seq):
        table, n = self.tables.setdefault(seq, []), self.lens.get(seq, 0)
        if n % self.block_size == 0:
            table.append(self._alloc())              # 填满才申请下一块
        elif self.ref[table[-1]] > 1:                # 写共享块：copy-on-write
            self.ref[table[-1]] -= 1
            table[-1] = self._alloc()                # 调用方负责拷贝块内容
        self.lens[seq] = n + 1

    def fork(self, parent, child):                   # 并行采样 / beam search
        self.tables[child] = list(self.tables[parent]); self.lens[child] = self.lens[parent]
        for b in self.tables[child]:
            self.ref[b] += 1

    def release(self, seq):
        for b in self.tables.pop(seq, []):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)
        self.lens.pop(seq, None)
```

## 白板估算

用 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 80 GB、3.35 TB/s、989 TFLOP/s；Llama-3-8B KV 128 KB/token，70B 320 KB/token。

```text
块大小：Llama-3-8B 一块 16 token × 128 KB = 2 MB；70B 一块 16 × 320 KB = 5 MB（全层合计）
静态批浪费：输出长度均匀分布在 50–500，mean/max = 275/500 = 55% 利用率
前缀缓存：2000 token 系统提示，8B 每请求省 KV 2000 × 128 KB = 262 MB
  省 prefill 2 × 8.03e9 × 2000 = 3.2e13 FLOP → 满峰 32 ms
  100 个并发请求共享：省 26 GB KV，约为 55.9 GB 预算的一半
70B，8×H100 TP（M15）：KV 余量 434.8 GB / (320 KB × 1500) = 885 条平均 1500 token 的序列
Little's law：目标 20 req/s，每请求停留 10 s → 并发 200
多 LoRA（8B，r=16，作用于 q/k/v/o）：
  q、o：4096×16 + 16×4096 = 131072；k、v：4096×16 + 16×1024 = 81920
  每层 2×131072 + 2×81920 = 425984；32 层 13.6 M 参数 = 27 MB（bf16）
  底座 16 GB，600 个适配器才等于一份底座
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 块分配器 | [服务与显存实验](../solutions/cmu15849/serving-memory/README.md) 的 block allocator | fork 后共享块引用计数正确，写入触发复制，释放后块全部归还 |
| continuous batching | 同上，调度模拟器 | 相同到达序列下吞吐高于静态批，P99 TPOT 有界 |
| 碎片统计 | 同上，连续分配 vs 分页分配 | 分页方案浪费小于 block_size 每序列 |
| Little's law | [mlsys_calc.py](../tools/mlsys_calc.py) `littles_law_concurrency` | 20 req/s × 10 s = 200 |

## 易错点与反例

- 「吞吐高就是服务好」：吞吐可以靠无限加大 batch 刷高，同时 TPOT 违反 SLO；应看 goodput。
- 「PagedAttention 让 attention 更快」：它主要省显存、提高并发；kernel 本身因间接寻址略慢，吞吐提升来自更大的 batch。
- 「continuous batching 就是动态 batching」：动态 batching 是按请求凑批（等一个时间窗），仍要等批内最长请求；continuous 是每个迭代重组。
- 「前缀缓存对所有负载有效」：对无共享前缀的负载命中率为零，还占用显存。
- 「chunked prefill 没有代价」：TTFT 增加，每块要重读前缀 KV；块太小时 GEMM 效率下降。
- 「PD 分离总是更好」：短 prompt、低负载时 KV 传输与额外实例的开销不值；它针对的是 TTFT 与 TPOT 同时有严格 SLO 的场景。
- 「SJF 一定更好」：输出长度未知，预测错会更糟；纯 SJF 饿死长请求。
- recompute 抢占会改变已生成内容吗？不会，已生成 token 作为 prompt 的一部分重跑，只是付出算力。

## 实验或练习

- 假设：continuous batching + 分页 KV 在相同显存下的并发数接近 KV 预算 / 平均上下文，而连续预分配只能按最大长度计。
- 配置：泊松到达，prompt 与输出长度服从对数正态；KV 预算固定；比较静态批、continuous + 连续分配、continuous + 分页三种方案的吞吐与 P99。
- 预期：分页方案并发最高；静态批 P99 最差；到达率接近容量时排队时间急剧上升（Little's law 下 W 爆炸）。

## 高频问题

- 问：TTFT 和 TPOT 分别受什么影响？
  答：TTFT 受排队与 prefill 算力影响；TPOT 受 batch 内总 KV 与权重字节、以及插入的 prefill 影响。
- 问：PagedAttention 解决了什么？
  答：KV 的预留浪费和内外碎片，使并发数从按最大长度计变为按实际长度计。
- 问：copy-on-write 用在哪里？
  答：并行采样和 beam search 共享 prompt 块，分支写入共享块时才复制。
- 问：抢占用 swap 还是 recompute？
  答：短序列 recompute（重算便宜），长序列 swap（PCIe 搬运比重算便宜）。
- 问：为什么要 chunked prefill？
  答：避免长 prefill 让同批 decode 的这一步变长，平滑 TPOT。

## 面试追问链

1. 问：LLM 服务看哪些指标？
   答：TTFT、TPOT、吞吐、P99，以及满足双 SLO 的 goodput。
2. 问：为什么 Orca 要按迭代调度？
   答：输出长度差异大，按请求调度要等最长的；按迭代调度完成即离开、新请求即加入，GPU 不空转。
3. 问：PagedAttention 的碎片上界是多少？代价是什么？
   答：每序列至多 block_size − 1 个 token；代价是 attention 要查 block table 间接寻址，块太小访存效率差，太大内部碎片多。
4. 问：长 prompt 进来，正在 decode 的用户卡了一下，为什么，怎么修？
   答：prefill 和 decode 在同一步，步时间被 prefill 撑大。修法：chunked prefill 按 token 预算切块；或 PD 分离；调度上限制每步 prefill token 数。
5. 问：PD 分离要传多少 KV？值不值？
   答：8B 2048 token 268 MB，IB 上 5.4 ms，远小于 66 ms 的 prefill，可按层流水；70B 671 MB、13 ms。值不值看 SLO：TTFT 与 TPOT 都严格时值。
6. 问：给一张 H100 跑 Llama-3-8B，prompt 1000、输出 500，能扛多少 req/s？
   答：KV 预算 55.9 GB，最多 284 条并发；L=256 时 decode 步 17.3 ms，只看 decode 29.6 req/s；加上每请求 32.5 ms 的 prefill，不重叠约 15 req/s，有效 TPOT 34 ms。SLO 更严就降并发或分离。
7. 问：要做一个服务 1000 个租户各自 LoRA 的平台，怎么设计？
   答：底座一份；适配器每个 27 MB（r=16），常驻热门的、其余放主存按需换入，与 KV 共用分页池；批内用 SGMV 类 kernel 混合不同适配器；调度按租户公平分配 token，防止热门租户独占；按适配器亲和性路由到实例以提高命中。

## 自测题

1. block_size=32，序列长 1000，浪费多少 token？
   要点：⌈1000/32⌉ = 32 块 = 1024，浪费 24。
2. 到达率 5 req/s，平均 TTFT 0.5 s、输出 400、TPOT 30 ms，平均并发？
   要点：W = 0.5 + 399 × 0.03 ≈ 12.5 s，L ≈ 62。
3. 为什么 recompute 抢占对短序列更划算？
   要点：重算 T 个 token 的 prefill 是算力受限、很快；swap 两次 PCIe 搬运，与长度成正比但带宽只有 32–64 GB/s。
4. 两条请求共享 3000 token 前缀、各自再有 200 token，前缀缓存省多少 KV（8B）？
   要点：一份 3000 × 128 KB = 393 MB。
5. 设计题：TTFT SLO 200 ms、TPOT SLO 30 ms，prompt 平均 6000，选混跑还是分离？
   要点：8B 6000 token prefill MFU 50% 约 195 ms，混跑会让 decode 步严重超标；分离并给 prefill 池更高 TP 降 TTFT。

## 关联笔记

- [M15 KV cache 与推测解码](module-15-inference-kv-speculative.md)：本讲的所有显存与时间公式来源
- [M17 量化](module-17-compression-quantization.md)：KV 量化与 weight-only 量化增加并发
- [M13 张量并行](module-13-tensor-pipeline-parallel.md)：大模型服务的 TP 切法
- [M18 集群调度](module-18-architectures-moe-clusters.md)：集群级调度与 MoE 服务

## 本讲小结

服务系统在 SLO 下把 decode batch 做大：iteration-level 调度让 batch 不空转，PagedAttention 让 KV 显存按实际长度计。
前缀缓存省重复 prefill，chunked prefill 和 PD 分离处理 prefill 对 decode 的干扰。
容量规划三步：KV 预算算并发上限，步时间公式算 TPOT，Little's law 校验到达率与并发是否自洽。

## 延伸阅读

- Yu et al., Orca, OSDI 2022, https://www.usenix.org/conference/osdi22/presentation/yu
- Kwon et al., vLLM / PagedAttention, SOSP 2023, https://arxiv.org/abs/2309.06180
- Zheng et al., SGLang, 2024, https://arxiv.org/abs/2312.07104
- Agrawal et al., Sarathi-Serve, OSDI 2024, https://arxiv.org/abs/2403.02310
- Zhong et al., DistServe, OSDI 2024, https://arxiv.org/abs/2401.09670
- Patel et al., Splitwise, ISCA 2024, https://arxiv.org/abs/2311.18677
- Sheng et al., S-LoRA, MLSys 2024, https://arxiv.org/abs/2311.03285
- Chen et al., Punica, MLSys 2024, https://arxiv.org/abs/2310.18547
