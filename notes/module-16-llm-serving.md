# Module 16 — LLM 推理服务系统

- 课程出处：CS229S Fall 2023《Scheduling ML Clusters》 · CMU 15-849《Library to Accelerate Training and Inference（客座）》 · 课程之后的系统：vLLM、SGLang、Sarathi-Serve、DistServe、S-LoRA
- 必读论文：Yu et al., Orca, 2022；Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention, 2023；Zheng et al., SGLang (RadixAttention), 2024；Agrawal et al., Sarathi-Serve, 2024；Zhong et al., DistServe, 2024；Patel et al., Splitwise, 2024；Sheng et al., S-LoRA, 2024；Chen et al., Punica, 2024
- 对应实验：[15-849 服务与显存](../solutions/cmu15849/serving-memory/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-serving-memory/serving_memory_report.pdf)（实测：真实引擎分页 KV 浪费 86.8% → 1.26%，并发 7.0 倍）
- 状态：初稿

## 摘要

M15 的结论是：decode 由字节决定，想提高效率就得把 batch 做大。可是一个线上服务里，请求是陆续到达的，长短不一，用户还在等着看字。服务系统要解决的，就是在延迟 SLO 下把 batch 做大。

这件事受三个互相牵制的约束：KV 显存决定能同时服务多少请求；调度决定 prefill 和 decode 怎么分享 GPU；SLO 决定 batch 最多能做多大。

Orca 和 vLLM 解决了前两个问题。Orca 做 iteration-level 调度：每个 decode 步都可以加入或移出请求。vLLM 的 PagedAttention 把 KV 按固定大小的块分配，碎片接近零。
之后的系统都在这个框架上继续优化：前缀共享（RadixAttention）、chunked prefill、prefill-decode 分离，以及让多个 LoRA 共享同一个底座。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 指标有 TTFT、TPOT（也叫 ITL）、吞吐、P99，以及 goodput，即同时满足 TTFT 与 TPOT SLO 的请求速率。
2. 静态批要等批里最长的请求结束；continuous batching 每个 decode 步都能换入换出请求。
3. PagedAttention 把 KV 按块分配（vLLM 默认 16 token 一块），用 block table 把逻辑块映射到物理块。内部碎片只出现在最后一块，没有外部碎片。
4. 前缀缓存让共同前缀的 KV 只算一次、只存一份；chunked prefill 把长 prompt 切块和 decode 混着跑，压住 TPOT 的尖峰。
5. Little's law：并发 = 到达率 × 平均停留时间。这是容量规划的第一步。

必须能默写的公式：

```text
端到端延迟 = TTFT + (n_out − 1) × TPOT
Little's law：L = λ · W
每块 KV 字节 = block_size × 2 · L_layers · n_kv · d_head · bytes
内部碎片 ≤ (block_size − 1) 个 token / 序列
可并发序列 ≈ (HBM × 利用率 − 权重) / (每 token KV × 平均上下文)
decode 步时间 ≈ (权重字节 + Σ_i T_i × 每 token KV) / HBM 带宽
静态批利用率 ≈ mean(n_out) / max(n_out)
```

一句话区分：

- 静态批与 continuous batching：前者按请求成批，后者按迭代成批。
- swap 与 recompute 两种抢占：前者把 KV 搬到 CPU 再搬回来，后者丢掉 KV、恢复时重跑 prefill。
- chunked prefill 与 PD 分离：前者在同一块 GPU 上切块混跑，后者把 prefill 和 decode 放到不同的 GPU 上，中间传 KV。
- S-LoRA 与 Punica：前者统一分页管理适配器和 KV，后者用 SGMV kernel 让用不同 LoRA 的请求同批计算。

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

在动手优化之前，先要说清楚「服务得好」是什么意思，因为不同的指标会把系统往不同的方向推。

TTFT 是用户看到第一个字的时间，包含排队和 prefill。TPOT（ITL）是相邻两个 token 之间的间隔。
吞吐（token/s 或 req/s）是运营方关心的指标；goodput 是满足 SLO 的那部分吞吐，DistServe 就以它为优化目标。

P99 比均值更重要。一个长 prompt 的 prefill 会让同批所有请求的这一步都变慢，这种尖峰在均值里是看不出来的。

### 2. 从静态批到 continuous batching

LLM 请求的输出长度差别很大，有的几十个 token 就结束，有的要好几百。这让传统的成批方式很吃亏。

静态批的做法是凑齐 B 个请求一起跑，直到最长的那个结束。早结束的请求只能占着位置空等，所以输出长度在 50–500 之间时，平均利用率只有约 mean/max。
Orca 把调度粒度从请求降到了迭代：每一步结束后，完成的请求立刻离开，新请求立刻加入。

这样做有一个实现上的麻烦：批里各请求长度不一，attention 没法直接拼成一个张量。Orca 的办法是 selective batching：线性层把所有 token 拼成一个 GEMM 一起算，attention 则按请求分别做。

### 3. PagedAttention

continuous batching 让请求能随时进出，但 KV 显存怎么分配仍是问题。早期系统按最大长度连续预分配，浪费有三种：为还没生成的 token 预留的空间、内部碎片、外部碎片。vLLM 论文报告，此前的系统只有约 20%–40% 的 KV 显存存的是真实的 token 状态。

vLLM 的思路和操作系统的分页类似。它把 KV 切成固定大小的物理块，每个序列维护一张 block table。新 token 填满一块才申请下一块，所以每个序列最多浪费 block_size − 1 个 token。
因为块大小都一样，任何空闲块都能分给任何序列，也就没有外部碎片。代价是 attention kernel 要按 block table 间接寻址。

### 4. 共享与抢占

分页之后，有两件事变得容易了：多个序列共享同一份 KV，以及显存不够时把某个序列请出去。

共享用在并行采样和 beam search 上。多个输出共享 prompt 的物理块，块上带引用计数；某个分支要往一个被共享的块里写时，才复制一份（copy-on-write）。

显存不足时要抢占，有两种做法。swap 把被抢占序列的块拷到 CPU，恢复时再拷回来，代价是 PCIe 带宽。recompute 直接丢掉这些块，恢复时把已生成的 token 拼到 prompt 后面重新 prefill，代价是算力。
短序列用 recompute 更便宜，长序列用 swap 更便宜。

### 5. 前缀缓存与 RadixAttention

很多请求的开头其实是一样的：同一段系统提示、同一组 few-shot 示例、同一段多轮对话历史。每次都重新 prefill 这一段是浪费。

SGLang 用一棵 radix tree，以 token 序列为键来索引 KV 块。新请求进来时做最长前缀匹配，命中的部分直接跳过 prefill。
叶子节点按 LRU 驱逐，正在被请求引用的节点不会被驱逐。调度时再优先排命中率高的请求（cache-aware scheduling），复用率还能进一步提高。

### 6. Chunked prefill

continuous batching 带来一个副作用：新请求的 prefill 和老请求的 decode 会挤在同一步里。

一个 8K 的 prefill 插进 decode 批，会让这一步的时间从十几毫秒涨到几百毫秒，所有正在生成的请求都会看到 TPOT 的尖峰。
Sarathi-Serve 的做法是把 prefill 切成固定大小的块（比如 512 token）。每一步的 token 预算 = decode 的 token 数 + 一个 prefill 块，这样 decode 不会停顿。

代价是长 prompt 的 TTFT 会略增：它被分成了多步，而且每一块都要把已有前缀的 KV 读一遍。

### 7. Prefill-decode 分离

chunked prefill 是在同一块 GPU 上协调两种负载。另一种思路是干脆把它们分开。

prefill 算力受限，decode 访存受限，混在一起会互相干扰，而且两者的最优并行度也不一样。
DistServe 和 Splitwise 把它们放到不同的 GPU 池里，prefill 完成后把 KV 传给 decode 实例。两个池可以独立扩缩容，也可以各自选 TP 度。

代价是 KV 传输：需要 NVLink 或 RDMA，并且要和计算重叠起来。

### 8. 多 LoRA 服务与调度

当平台上有很多租户、每家都有自己的 LoRA 适配器时，为每家单独部署一份底座太浪费。

S-LoRA 让底座权重只存一份，数千个适配器放在主存里按需换入，适配器和 KV 用同一个分页池管理。
Punica 写了 SGMV kernel，让一个批里不同的请求各乘各的 B·A，底座的 GEMM 仍然合批计算。

调度策略也有取舍：

- FCFS 简单，但会有队头阻塞。
- SJF 能降低平均延迟，但输出长度事先未知，只能用预测器近似，而且会饿死长请求。
- 公平调度（比如按每个租户已经服务的 token 数计数）能防止单个用户占满资源。

## 关键推导

### 碎片上界

这一节量化分页到底比连续预分配省多少。

设块大小为 k，序列长度为 T，占用 ⌈T/k⌉ 块，浪费 ⌈T/k⌉·k − T ≤ k − 1 个 token，平均约 k/2。
T=1500、k=16 时，浪费约 8/1500 = 0.5%。
换成按 T_max=8192 连续预分配，同样 T=1500 的序列要浪费 (8192 − 1500)/8192 = 82%。

### Little's law

容量规划要回答的第一个问题是：在给定的到达率下，系统里同时会有多少个请求？

Little's law 说，稳态下系统内的平均请求数 L = 到达率 λ × 平均停留时间 W，与分布无关。
用法是：给定目标 λ，和由 SLO 推出的 W，得到必须支持的并发 L；再用 KV 公式检查显存放不放得下。

## 深入推导

### 单卡容量规划：Llama-3-8B，一张 H100

这一节把前面的公式串起来，回答一个具体问题：一张 H100 跑 Llama-3-8B，到底能扛多少 req/s，瓶颈在哪里。

工作负载设为 prompt 1000、输出 500，平均上下文按 1250 计，峰值 1500。可用 HBM 按 90%（72 GB）算。

第一步算并发上限。KV 预算 72 − 16.06 = 55.9 GB，每 token 131072 B，约 42.7 万 token；每条序列峰值 1500，所以最多 284 条并发。

第二步算 decode 步时间。并发为 L 时，t(L) = (16.06 GB + L × 1250 × 131072 B) / 3.35 TB/s = 4.79 ms + L × 0.049 ms。
取 L = 256，t = 4.79 + 12.5 = 17.3 ms，decode 吞吐为 256/17.3 ms ≈ 14800 token/s。只看 decode 的话，可以服务 29.6 req/s。

第三步加上 prefill。每个请求的 prefill 是 2 × 8.03e9 × 1000 = 1.6e13 FLOP，MFU 50% 时要 32.5 ms。
如果 prefill 和 decode 不重叠，每秒的 GPU 时间是 R × (500/256 × 17.3 ms + 32.5 ms) = R × 66.3 ms ≤ 1 s，得到 R ≈ 15 req/s。
这时 decode 只拿到约一半的时间，有效 TPOT ≈ 34 ms。用 Little's law 校验一下：W ≈ 500 × 34 ms = 17 s，L = 15 × 17 ≈ 256，前后自洽。

结论：如果 SLO 要求 TPOT ≤ 25 ms，要么降低并发，要么用 chunked prefill 或 PD 分离。分离之后，一张 decode 卡约 29.6 req/s，一张 prefill 卡约 1/32.5 ms ≈ 30.8 req/s，所以在这个负载下大约按 1:1 配比。

### PD 分离的 KV 传输

PD 分离的代价是传 KV。算一下它是否真的值得担心：

8B、prompt 2048：KV 是 2048 × 128 KB = 268 MB，在 IB 50 GB/s 上要 5.4 ms。相对 66 ms 的 prefill 可以忽略，而且可以按层流水传输。
70B、prompt 2048：671 MB，IB 上 13.4 ms；NVLink 450 GB/s 上 1.5 ms。

下面是一个 vLLM 风格的块分配器，展示了分页、引用计数和 copy-on-write 在代码里是怎么落地的：

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

下面这组估算覆盖本讲的几个机制：块有多大、静态批浪费多少、前缀缓存省多少、70B 能放多少条序列、Little's law 怎么用，以及多 LoRA 的适配器有多小。
硬件和模型数字取自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：H100 80 GB、3.35 TB/s、989 TFLOP/s；Llama-3-8B KV 128 KB/token，70B 320 KB/token。

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

其中前缀缓存那行很能说明问题：100 个请求共享一段 2000 token 的系统提示，省下的 KV 约占整个预算的一半。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 块分配器 | [服务与显存实验](../solutions/cmu15849/serving-memory/README.md) 的 block allocator | fork 后共享块引用计数正确，写入触发复制，释放后块全部归还 |
| continuous batching | 同上，调度模拟器 | 相同到达序列下吞吐高于静态批，P99 TPOT 有界 |
| 碎片统计 | 同上，连续分配 vs 分页分配 | 分页方案浪费小于 block_size 每序列 |
| Little's law | [mlsys_calc.py](../tools/mlsys_calc.py) `littles_law_concurrency` | 20 req/s × 10 s = 200 |

## 易错点与反例

- 「吞吐高就是服务好」：吞吐可以靠无限加大 batch 刷上去，但同时 TPOT 会违反 SLO。应该看 goodput。
- 「PagedAttention 让 attention 更快」：它主要是省显存、提高并发。kernel 本身因为间接寻址还会略慢，吞吐的提升来自更大的 batch。
- 「continuous batching 就是动态 batching」：动态 batching 是按请求凑批（等一个时间窗口），仍然要等批内最长的请求；continuous batching 是每个迭代都重新组批。
- 「前缀缓存对所有负载都有效」：没有共享前缀的负载命中率为零，缓存还白白占着显存。
- 「chunked prefill 没有代价」：TTFT 会增加，每块都要重读前缀的 KV；块切得太小，GEMM 效率也会下降。
- 「PD 分离总是更好」：短 prompt、低负载时，KV 传输和额外实例的开销不值得。它针对的是 TTFT 和 TPOT 同时有严格 SLO 的场景。
- 「SJF 一定更好」：输出长度未知，预测错了反而更糟；纯 SJF 还会饿死长请求。

## 实验或练习

这个练习验证一个假设：在相同显存下，continuous batching 加分页 KV 的并发数接近「KV 预算 / 平均上下文」，而连续预分配只能按最大长度来算。

- 配置：泊松到达，prompt 和输出长度服从对数正态分布，KV 预算固定。比较三种方案的吞吐与 P99：静态批、continuous 加连续分配、continuous 加分页。
- 预期：分页方案并发最高，静态批的 P99 最差；到达率接近容量时，排队时间急剧上升（按 Little's law，W 会爆炸）。

## 高频问题

- 问：TTFT 和 TPOT 分别受什么影响？
  答：TTFT 受排队和 prefill 算力影响；TPOT 受 batch 内总的 KV 与权重字节影响，也受插进来的 prefill 影响。
- 问：PagedAttention 解决了什么？
  答：KV 的预留浪费和内外碎片。并发数从按最大长度计，变成按实际长度计。
- 问：copy-on-write 用在哪里？
  答：并行采样和 beam search 共享 prompt 的块，某个分支写入共享块时才复制。
- 问：continuous batching 和动态 batching 有什么区别？
  答：动态 batching 在一个时间窗口内按请求凑批，仍要等批内最长的请求；continuous batching 每个迭代都重新组批。
- 问：recompute 抢占会改变已生成的内容吗？
  答：不会。已生成的 token 作为 prompt 的一部分重跑，只是要付出算力。

## 面试追问链

1. 问：LLM 服务看哪些指标？
   答：TTFT、TPOT、吞吐、P99，以及同时满足两个 SLO 的 goodput。
2. 问：为什么 Orca 要按迭代调度？
   答：输出长度差异很大，按请求调度就得等最长的那个。按迭代调度时，完成的请求立刻离开、新请求立刻加入，GPU 不会空转。
3. 问：PagedAttention 的碎片上界是多少？代价是什么？
   答：每个序列最多浪费 block_size − 1 个 token。代价是 attention 要查 block table 间接寻址；块太小访存效率差，块太大内部碎片多。
4. 问：一个长 prompt 进来，正在 decode 的用户卡了一下。为什么？怎么修？
   答：prefill 和 decode 挤在同一步里，步时间被 prefill 撑大了。修法有三种：chunked prefill 按 token 预算切块；或者做 PD 分离；或者在调度上限制每步的 prefill token 数。
5. 问：PD 分离要传多少 KV？值不值？
   答：8B、2048 token 是 268 MB，IB 上 5.4 ms，远小于 66 ms 的 prefill，还可以按层流水；70B 是 671 MB、13 ms。值不值看 SLO：TTFT 和 TPOT 都严格时才值得。
6. 问：给一张 H100 跑 Llama-3-8B，prompt 1000、输出 500，能扛多少 req/s？
   答：KV 预算 55.9 GB，最多 284 条并发。L=256 时 decode 步 17.3 ms，只看 decode 能到 29.6 req/s；加上每个请求 32.5 ms 的 prefill，不重叠时约 15 req/s，有效 TPOT 34 ms。SLO 更严的话，就降并发或者做分离。
7. 问：要做一个服务 1000 个租户、每家有自己 LoRA 的平台，怎么设计？
   答：底座只放一份；每个适配器 27 MB（r=16），热门的常驻，其余放在主存按需换入，和 KV 共用分页池；批内用 SGMV 类 kernel 混合不同的适配器；调度上按租户公平分配 token，防止热门租户独占；路由时按适配器亲和性把请求送到对应实例，提高命中率。

## 自测题

1. block_size=32，序列长 1000，浪费多少 token？
   要点：⌈1000/32⌉ = 32 块 = 1024，浪费 24。
2. 到达率 5 req/s，平均 TTFT 0.5 s、输出 400、TPOT 30 ms，平均并发是多少？
   要点：W = 0.5 + 399 × 0.03 ≈ 12.5 s，L ≈ 62。
3. 为什么 recompute 抢占对短序列更划算？
   要点：重算 T 个 token 的 prefill 是算力受限的，很快；swap 要两次 PCIe 搬运，耗时和长度成正比，而带宽只有 32–64 GB/s。
4. 两条请求共享 3000 token 前缀，各自再有 200 token，前缀缓存省多少 KV（8B）？
   要点：一份 3000 × 128 KB = 393 MB。
5. 设计题：TTFT SLO 200 ms、TPOT SLO 30 ms，prompt 平均 6000，选混跑还是分离？
   要点：8B 模型 6000 token 的 prefill 在 MFU 50% 时约 195 ms，混跑会让 decode 步严重超标；应该分离，并给 prefill 池更高的 TP 来降 TTFT。

## 关联笔记

- [M15 KV cache 与推测解码](module-15-inference-kv-speculative.md)：本讲的所有显存与时间公式来源
- [M17 量化](module-17-compression-quantization.md)：KV 量化与 weight-only 量化增加并发
- [M13 张量并行](module-13-tensor-pipeline-parallel.md)：大模型服务的 TP 切法
- [M18 集群调度](module-18-architectures-moe-clusters.md)：集群级调度与 MoE 服务

## 本讲小结

服务系统的目标是在 SLO 下把 decode 的 batch 做大。iteration-level 调度让 batch 不空转，PagedAttention 让 KV 显存按实际长度计。
前缀缓存省掉重复的 prefill；chunked prefill 和 PD 分离处理 prefill 对 decode 的干扰。
容量规划分三步：用 KV 预算算并发上限，用步时间公式算 TPOT，再用 Little's law 校验到达率和并发是否自洽。

## 延伸阅读

- Yu et al., Orca, OSDI 2022, https://www.usenix.org/conference/osdi22/presentation/yu
- Kwon et al., vLLM / PagedAttention, SOSP 2023, https://arxiv.org/abs/2309.06180
- Zheng et al., SGLang, 2024, https://arxiv.org/abs/2312.07104
- Agrawal et al., Sarathi-Serve, OSDI 2024, https://arxiv.org/abs/2403.02310
- Zhong et al., DistServe, OSDI 2024, https://arxiv.org/abs/2401.09670
- Patel et al., Splitwise, ISCA 2024, https://arxiv.org/abs/2311.18677
- Sheng et al., S-LoRA, MLSys 2024, https://arxiv.org/abs/2311.03285
- Chen et al., Punica, MLSys 2024, https://arxiv.org/abs/2310.18547
