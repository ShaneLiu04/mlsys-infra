# Module 06 — 数据并行原语：scan、reduce、Spark

- 课程出处：CS149 Fall 2024《Data-Parallel Thinking》《Distributed Data-Parallel Computing Using Spark》 · CMU 15-849《Parameter Server》（对照）
- 必读论文：Blelloch, Prefix Sums and Their Applications, 1990；Zaharia et al., Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing, 2012；Li et al., Scaling Distributed Machine Learning with the Parameter Server, 2014
- 对应实验：[CS149 A3 scan 与 circle renderer](../solutions/cs149/asst3-scan-render/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst3/cs149_asst3_report.pdf)（实测：scan 2²⁸ 朴素 16.75 ms → 自写 2.82 ms → cub 1.41 ms）
- 状态：初稿

## 摘要

数据并行编程的思路是：不写线程，写对整个集合的操作。map、reduce、scan、gather/scatter、sort、groupByKey 这几个原语
组合起来能表达绝大多数并行算法。scan 是其中最有用也最常考的：Hillis-Steele 版本 span 短但工作量 O(n log n)，
Blelloch 版本工作量 O(n)，span 2 log n。stream compaction、radix sort、按 tile 收集圆都建立在 scan 上。
Spark 把同一套原语放到集群上，用 lineage 重算代替副本做容错，宽依赖（shuffle）是代价和故障恢复的分界线。
分布式训练的 reduce-scatter、all-gather 和 parameter server，就是分布式内存上的 reduce 与 gather。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. reduce 和 scan 只要求运算满足结合律，不要求交换律。浮点加法不严格结合，并行归约结果与串行不逐位相同。
2. Hillis-Steele：log n 步，每步约 n 次加法，work O(n log n)，span O(log n)。Blelloch：up-sweep + down-sweep，work 约 2n，span 2 log n。
3. stream compaction = 标志 → exclusive scan 得目标下标 → scatter。A3 的 find_repeats 和渲染器的圆列表都是它。
4. Spark 窄依赖（map、filter）可在一个 stage 内流水执行、丢分区只需重算一个父分区；宽依赖（groupByKey、未共分区的 join）要 shuffle，是 stage 边界。
5. all-reduce = reduce-scatter + all-gather，ring 实现每卡收发 `2(p−1)/p·S`；分片 parameter server 的通信量相同，单点 PS 才有热点。

**必须能默写的公式**

```text
inclusive scan:  y_i = x_0 ⊕ x_1 ⊕ … ⊕ x_i
exclusive scan:  y_0 = id,  y_i = x_0 ⊕ … ⊕ x_{i−1}
Hillis-Steele:   W = Σ_{d=0}^{log n − 1} (n − 2^d) = n·log n − (n − 1),   S = log n
Blelloch:        W = 2(n − 1) 次 ⊕,   S = 2·log n
分块 scan:       W ≈ 2n,   S ≈ 2n/P + log P              P 个 worker 各扫 n/P
segmented op:    (f1, x1) ⊕ (f2, x2) = (f1 | f2,  f2 ? x2 : x1 ⊕ x2)
ring 集合通信:   reduce-scatter = all-gather = (p−1)/p · S；   all-reduce = 2(p−1)/p · S
```

**一句话区分**

- inclusive vs exclusive scan：包含自身 vs 不含自身，exclusive 的结果正好是 compaction 的写入下标。
- gather vs scatter：按下标读 `out[i] = in[idx[i]]` vs 按下标写 `out[idx[i]] = in[i]`，后者有写冲突问题。
- groupByKey vs reduceByKey：把所有值 shuffle 后再聚合 vs 先在 map 端局部聚合再 shuffle。
- parameter server vs all-reduce：中心化（可分片）的推拉、支持异步与稀疏 vs 对称的集合通信、同步稠密。

## 学习目标

- 能手推 8 个元素的 inclusive/exclusive scan，画出 Hillis-Steele 和 Blelloch 的过程并算出 work 与 span；
- 能用 scan 实现 stream compaction、segmented scan、radix sort 的一轮 split；
- 能解释 Spark RDD 的 lineage、窄/宽依赖、persist 和容错代价，并把 reduce-scatter、all-gather 与 parameter server 放进同一个通信量模型。

## 先修知识

- [M01](module-01-performance-models.md) 的 work/span，[M05](module-05-gpu-architecture-cuda.md) 的 block、shared memory 与 `__syncthreads`；
- Python 列表与 lambda。

## 核心概念

### 1. map 与 reduce

map 对每个元素独立应用函数，没有依赖，完美并行。reduce 用二元运算 ⊕ 把集合归成一个值。
只要 ⊕ 满足结合律，就能任意加括号，组成深度 log n 的树。不要求交换律：矩阵乘法也能并行归约，只要保持左右顺序。
浮点加法不严格满足结合律，所以 GPU 上 `torch.sum` 的结果可能和 CPU 不逐位相同，原子加法的归约甚至每次运行都不同。

### 2. scan

`x = [3, 1, 7, 0, 4, 1, 6, 3]` 的 inclusive scan 是 `[3, 4, 11, 11, 15, 16, 22, 25]`，exclusive 是 `[0, 3, 4, 11, 11, 15, 16, 22]`。
scan 看上去是严格串行的（每个输出依赖前一个），但结合律让它也能在 O(log n) 的 span 内完成。

### 3. gather、scatter、sort、groupByKey

gather 按下标读，scatter 按下标写。scatter 若多个元素写同一位置，需要规定冲突规则（任取一个、求和、原子操作）。
radix sort 的每一轮按一个 bit 把数组稳定地分成两半：对「该 bit 为 0」的标志做 exclusive scan 得到 0 组的位置，
1 组的位置 = 0 组总数 + 1 组内部的 scan。groupByKey 可以由「按 key 排序 + 标记段首 + segmented 操作」实现。

### 4. stream compaction 与 find_repeats

A3 的 `find_repeats` 要求输出所有满足 `a[i] == a[i+1]` 的下标 i：

```python
def find_repeats(a):
    n = len(a)
    flags = [1 if i + 1 < n and a[i] == a[i + 1] else 0 for i in range(n)]
    idx = exclusive_scan(flags)                  # GPU 上用自己写的 scan
    out = [0] * (idx[-1] + flags[-1])
    for i in range(n):                           # GPU 上每个线程处理一个 i
        if flags[i]:
            out[idx[i]] = i
    return out
```

每一步都是 map、scan 或 scatter，没有锁，写入位置互不冲突，输出保持原顺序。

### 5. segmented scan

很多问题是「若干段，各段内部求前缀和」，例如稀疏矩阵 CSR 格式下每行独立求和。
用标志数组标出段首，把元素换成对 (flag, value)，定义新运算：若右边是段首则丢弃左边的累积。
这个运算仍满足结合律，所以任何 scan 实现都能直接用于分段问题，一次 kernel 处理所有段，不怕段长不均。

### 6. Spark 与 RDD

RDD 是不可变、分区的分布式集合。transformation（map、filter、reduceByKey、join）惰性执行，只记录 lineage；action（count、collect、save）触发执行。
调度器按宽依赖把 lineage 切成 stage：stage 内的窄依赖算子流水执行、不落盘，stage 之间通过 shuffle 交换数据。

容错：某个分区丢失时，按 lineage 从输入或最近的 checkpoint 重算。窄依赖只需重算一个父分区；
宽依赖的一个子分区依赖所有父分区，可能牵动大量重算，所以长 lineage、迭代多轮的作业要定期 checkpoint。
`persist()` 把 RDD 留在内存，迭代算法（逻辑回归多轮读同一份数据）不必每轮读 HDFS，这是 Spark 相对 MapReduce 的主要优势。

```python
lines = sc.textFile("hdfs://logs/*")
errors = lines.filter(lambda l: "ERROR" in l).persist()      # 窄依赖，缓存在内存
by_host = errors.map(lambda l: (l.split()[0], 1)).reduceByKey(lambda a, b: a + b)  # 宽依赖，shuffle
print(by_host.take(10))                                       # action，触发两个 stage
```

## 关键推导

### 推导 1：Hillis-Steele 的 work 与 span

```text
for d = 0 .. log n − 1:            # 每步并行，需要双缓冲
    for all i ≥ 2^d:  x'[i] = x[i − 2^d] ⊕ x[i]
第 d 步做 n − 2^d 次 ⊕
W = Σ (n − 2^d) = n·log n − (n − 1)
S = log n
n = 8: 7 + 6 + 4 = 17 次；串行只需 7 次
```

它适合宽度足够、n 不大的场景：一个 warp 32 个元素、5 步完成，warp shuffle 实现，多出来的加法在空闲 lane 上是免费的。

### 推导 2：Blelloch 的 work 与 span

```text
up-sweep（建归约树）: 第 d 层把步长 2^{d+1} 的每段右端加上左半段的和
  共 n/2 + n/4 + … + 1 = n − 1 次 ⊕，结束时 x[n − 1] 是总和
置零: x[n − 1] = 0（⊕ 的单位元）
down-sweep: 从顶层往下，每个节点把自己的值给左孩子，把「自己 ⊕ 原左孩子」给右孩子
  再 n − 1 次 ⊕
W = 2(n − 1)，S = 2 log n，结果是 exclusive scan；完整代码见推导 3
```

## 深入推导

### 推导 3：手推 Blelloch（n = 8）

```text
输入             [3, 1, 7, 0, 4, 1, 6, 3]
up d=0           [3, 4, 7, 7, 4, 5, 6, 9]
up d=1           [3, 4, 7, 11, 4, 5, 6, 14]
up d=2           [3, 4, 7, 11, 4, 5, 6, 25]
置零              [3, 4, 7, 11, 4, 5, 6, 0]
down d=2         [3, 4, 7, 0, 4, 5, 6, 11]
down d=1         [3, 0, 7, 4, 4, 11, 6, 16]
down d=0         [0, 3, 4, 11, 11, 15, 16, 22]   = exclusive scan
```

```python
def blelloch_exclusive_scan(x):
    a, n = list(x), len(x)                     # n 为 2 的幂；否则补齐到下一个 2 的幂
    d = 1
    while d < n:                               # up-sweep
        for k in range(0, n, 2 * d):
            a[k + 2 * d - 1] += a[k + d - 1]
        d *= 2
    a[n - 1] = 0
    d = n // 2
    while d >= 1:                              # down-sweep
        for k in range(0, n, 2 * d):
            t = a[k + d - 1]
            a[k + d - 1] = a[k + 2 * d - 1]
            a[k + 2 * d - 1] += t
        d //= 2
    return a

assert blelloch_exclusive_scan([3, 1, 7, 0, 4, 1, 6, 3]) == [0, 3, 4, 11, 11, 15, 16, 22]
```

A3 要求支持任意长度：把长度向上取到 2 的幂，多出来的位置填 0（⊕ 的单位元），不影响前 n 个结果。

### 推导 4：GPU 上 scan 的实际形态

纯 Blelloch 在 GPU 上每一层一个 kernel，上层只有很少的活跃线程，且访问步长越来越大、合并性变差。实际做法分层：

```text
1. 每个 block 把一段数据读进 shared memory，在 block 内做 scan（warp 内用 Hillis-Steele + shuffle）
2. 每个 block 的总和写到辅助数组，对辅助数组递归 scan
3. 每个 block 给自己的段加上对应的前缀
```

```cuda
__global__ void block_inclusive_scan(int* data, int n) {
    extern __shared__ int s[];
    int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? data[i] : 0;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1) {     // Hillis-Steele，block 内
        int v = (tid >= off) ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }
    if (i < n) data[i] = s[tid];
}
```

两个 `__syncthreads()` 分别保证「先读完再写」和「写完再进入下一轮」，是双缓冲的另一种写法。CUB 的 decoupled look-back 把三步合成一次遍历，访存接近 `2n` 个元素的下限。

### 推导 5：分布式 reduce：reduce-scatter 与 all-gather

p 个节点各有长度 S 字节的向量。reduce-scatter 后，节点 j 拿到第 j 份（S/p）的全局和；all-gather 后，每个节点拿到全部 p 份。
ring 实现：每步每个节点把一份 S/p 发给右邻居，p − 1 步。

```text
reduce-scatter: (p − 1) · S/p
all-gather:     (p − 1) · S/p
all-reduce = reduce-scatter + all-gather = 2(p − 1)/p · S
```

parameter server 的 push 是把梯度 reduce 到服务器，pull 是把参数 broadcast 回 worker。按参数分片到 p 个节点后，push 就是 reduce-scatter，
pull 就是 all-gather。PS 的价值在于异步更新（不等最慢的 worker）和稀疏更新（只推拉被访问的 embedding 行）。ZeRO 的梯度分片用 reduce-scatter，参数重组用 all-gather（M12）。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)。

```text
1) work 对比，n = 2^20: Hillis-Steele ≈ n·log n ≈ 2.1e7 次加法，Blelloch ≈ 2n ≈ 2.1e6 次 → 少 10 倍

2) H100 上 scan n = 2^28 个 int32（1.07 GB）
   下限: 读 + 写 = 2.15 GB → 2.15e9 / 3.35e12 = 0.64 ms
   三遍分层方案（先求块和读一遍，再读 + 写一遍）: 3 × 1.07 GB = 3.2 GB → 0.96 ms
   每步一遍全局内存的 Hillis-Steele: 28 × 2.15 GB = 60 GB → 18 ms

3) ring all-reduce Llama-3-8B 的 bf16 梯度 S = 8.03e9 × 2 = 16.06 GB，8 卡
   每卡收发 2 × 7/8 × 16.06 = 28.1 GB
   节点内 NVLink4 450 GB/s → 62 ms
   8 个节点各一张卡，IB NDR 50 GB/s → 562 ms
   只做 reduce-scatter（ZeRO 梯度分片）: 14.05 GB → NVLink 上 31 ms

4) Spark shuffle: 1000 个 map 分区，每个 1e6 条 (key, value)，每条 16 B，全局只有 1e4 个不同 key
   groupByKey: shuffle 全部 1e9 × 16 B = 16 GB
   reduceByKey: 每个 map 分区先合并到最多 1e4 条 → 1000 × 1e4 × 16 B = 160 MB，少 100 倍
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| exclusive scan | [A3](../solutions/cs149/asst3-scan-render/README.md) 第二部分 `exclusive_scan` | 任意 n（含非 2 的幂）与 CPU 参考逐元素一致；大 n 时耗时接近参考实现 |
| stream compaction | A3 第二部分 `find_repeats` | 下标集合与顺序和参考一致，只用 map、scan、scatter 三类 kernel |
| block 内 scan | A3 第三部分渲染器中每个 tile 的相交圆列表 | 列表顺序等于输入顺序，渲染结果通过正确性检查 |
| 分布式 reduce | 与 [M12](module-12-data-parallel-zero.md) 的 all-reduce 计算对照 | 手算通信量与 `mlsys_calc.py allreduce` 输出一致 |

## 易错点与反例

- 「Hillis-Steele 比 Blelloch 差」：n 很大时 work 多 log n 倍；但在一个 warp 内，Hillis-Steele 步数少、无需下扫，更快。
- 「reduce 需要交换律」：只需结合律；交换律只在想乱序合并时才需要。
- 「并行求和结果和串行一样」：浮点不结合，结果有舍入差异；需要逐位可复现时要固定归约顺序。
- 「Blelloch 的结果是 inclusive」：down-sweep 产生 exclusive，要 inclusive 需再加一次原数组。
- 「Spark 的容错靠多副本」：RDD 靠 lineage 重算；只有 checkpoint 和 shuffle 文件落盘。
- 「Spark 里 groupByKey 和 reduceByKey 一样」：前者 shuffle 所有值，后者先 map 端合并，数据量可差数量级。

## 实验或练习

- 假设：work-efficient scan 在大 n 时比 Hillis-Steele 快；耗时随 n 线性增长，换算带宽接近 HBM 峰值的某个比例。
- 配置：在 A3 框架里实现两种 scan，n 取 `2^10` 到 `2^26`，记录耗时并与 Thrust 的 `exclusive_scan` 比较；用 Python 版覆盖 n = 1、非 2 的幂、全零做单元测试。
- 预期：小 n 时两者都受 kernel 启动开销主导；大 n 时 Blelloch 的优势随 log n 增大；Thrust 因单遍算法更快。find_repeats 约为一次 scan 加两次 map。

## 高频问题

- 问：怎么用 scan 做 stream compaction？
  答：满足条件处标 1，exclusive scan 得到每个保留元素的目标下标，再 scatter。
- 问：Spark 的窄依赖和宽依赖？
  答：窄依赖子分区只依赖少量父分区，可流水、易恢复；宽依赖依赖全部父分区，要 shuffle，是 stage 边界。
- 问：all-reduce 和 reduce-scatter 的关系？
  答：all-reduce = reduce-scatter + all-gather，ring 实现各占一半通信量。

## 面试追问链

1. 问：给 `[3, 1, 7, 0, 4, 1, 6, 3]` 写出 inclusive 和 exclusive scan。
   答：inclusive `[3, 4, 11, 11, 15, 16, 22, 25]`；exclusive `[0, 3, 4, 11, 11, 15, 16, 22]`。
2. 问：Hillis-Steele 和 Blelloch 的 work、span 分别是多少？
   答：前者 `n log n − (n−1)` 与 `log n`；后者 `2(n−1)` 与 `2 log n`。前者 span 短一半但工作多 log n 倍。
3. 问：GPU 上实际怎么实现一个 1 亿元素的 scan？
   答：分层：block 内 shared memory scan（warp 内用 shuffle 的 Hillis-Steele），block 和再 scan，最后加偏移；
   或单遍 decoupled look-back。目标是总访存接近读写各一遍的下限。
4. 问：用 scan 实现一轮 radix sort 的 split。
   答：对 bit 为 0 的标志做 exclusive scan 得 0 组位置；1 组位置 = 0 组总数 + 对 1 标志的 exclusive scan；scatter，稳定。
5. 问：Spark 一个 shuffle 之后的分区丢了，恢复代价是什么，怎么降低？
   答：宽依赖的子分区依赖所有父分区；若 shuffle 文件还在，只需重读；若 map 端也丢了，要重算多个父分区。
   降低方式：checkpoint、对迭代算法 persist、用 reduceByKey 减少 shuffle 量。
6. 问：MoE 层的 token dispatch 怎么用本讲的原语实现？
   答：每个 token 有专家号，按专家号做计数（直方图）和 exclusive scan 得到每个专家的输出起始位置，
   再 scatter 到按专家连续排列的缓冲区（等价于一次按 key 的稳定排序）；计算后按原下标 gather 回来。跨卡时 scatter 变成 all-to-all。
7. 问：ZeRO 为什么用 reduce-scatter 而不是 all-reduce？
   答：每张卡只负责更新自己那 1/p 的参数，只需要那一份的梯度和。reduce-scatter 通信量 `(p−1)/p·S`，是 all-reduce 的一半，
   更新后再用 all-gather 拼回完整参数。
8. 问：推荐系统的大 embedding 表训练，为什么常用 parameter server 而不是 all-reduce？
   答：embedding 表可达数百 GB 以上，每步只访问极少的行，梯度高度稀疏。all-reduce 要同步稠密的全表梯度，浪费带宽；
   分片 PS 只推拉被访问的行，并允许异步更新。稠密的 MLP 部分仍用 all-reduce。

## 自测题

1. 对 n = 16 计算 Hillis-Steele 和 Blelloch 的加法次数。
   要点：Hillis-Steele `15 + 14 + 12 + 8 = 49`；Blelloch `2 × 15 = 30`；串行 15。
2. 证明 segmented scan 的运算 `(f1, x1) ⊕ (f2, x2) = (f1 | f2, f2 ? x2 : x1 + x2)` 满足结合律。
   要点：分 `f3` 为真、`f3` 为假且 `f2` 为真、二者都假三种情况，两边加括号结果相同。
3. 只用 map、exclusive scan、scatter，写出删除数组中所有负数的算法。
   要点：`flag = x ≥ 0`；`idx = scan(flag)`；`flag[i]` 为 1 时 `out[idx[i]] = x[i]`；长度 = 最后的 `idx + flag`。
4. 设计题：64 卡（8 节点 × 8 卡）all-reduce 16 GB 梯度，ring 跨全部 64 卡和「节点内 reduce-scatter → 节点间 all-reduce → 节点内 all-gather」哪个快？
   要点：64 卡大 ring 受最慢的 IB 链路限制，每卡约 `2 × 63/64 × 16 GB ≈ 31.5 GB` 走 IB；分层方案跨节点只传 `S/8` 的分片，
   每卡走 IB 的量约 `2 × 7/8 × 2 GB = 3.5 GB`，IB 流量少约 9 倍。

## 关联笔记

- [M05](module-05-gpu-architecture-cuda.md)：scan 与 compaction 的 CUDA 实现、circle renderer。
- [M12](module-12-data-parallel-zero.md)：all-reduce、reduce-scatter 与 ZeRO。
- [M18](module-18-architectures-moe-clusters.md)：MoE 的 token dispatch 与集群数据管线。

## 本讲小结

结合律让 reduce 和 scan 有 O(log n) 的 span。Blelloch 用 2n 的工作量，Hillis-Steele 用更多工作换更短的 span，GPU 实现分层组合二者。stream compaction、radix sort、MoE dispatch 都是 scan 加 scatter。
Spark 把这些原语放到集群上，用 lineage 做容错，宽依赖是代价的分界线。分布式训练的集合通信是同一组原语在多卡内存上的版本。

## 延伸阅读

- Blelloch, Prefix Sums and Their Applications, CMU-CS-90-190, 1990.
- Hillis, Steele, Data Parallel Algorithms, CACM 1986.
- Merrill, Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back, NVIDIA Tech Report 2016.
- Zaharia et al., Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing, NSDI 2012.
- Li et al., Scaling Distributed Machine Learning with the Parameter Server, OSDI 2014.
