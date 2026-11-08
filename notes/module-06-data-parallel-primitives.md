# Module 06 — 数据并行原语：scan、reduce、Spark

- 课程出处：CS149 Fall 2024《Data-Parallel Thinking》《Distributed Data-Parallel Computing Using Spark》 · CMU 15-849《Parameter Server》（对照）
- 必读论文：Blelloch, Prefix Sums and Their Applications, 1990；Zaharia et al., Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing, 2012；Li et al., Scaling Distributed Machine Learning with the Parameter Server, 2014
- 对应实验：[CS149 A3 scan 与 circle renderer](../solutions/cs149/asst3-scan-render/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst3/cs149_asst3_report.pdf)（实测：scan 2²⁸ 朴素 16.75 ms → 自写 2.82 ms → cub 1.41 ms）
- 状态：初稿

## 摘要

前几讲一直在和线程、锁、warp 打交道。这一讲换一个角度：能不能不写线程，只描述「对整个集合做什么操作」，把怎么并行交给库和运行时？
数据并行编程就是这个思路。map、reduce、scan、gather/scatter、sort、groupByKey 这几个原语组合起来，能表达绝大多数并行算法。

其中最有用、也最常考的是 scan（前缀和）。它看起来每一步都依赖上一步，却能在 O(log n) 的深度内并行完成。
Hillis-Steele 版本步数少，但总工作量是 O(n log n)；Blelloch 版本把工作量压到 O(n)，代价是步数变成 2 log n。
stream compaction、radix sort、A3 里按 tile 收集圆，都建立在 scan 之上。

同一套原语放大到集群上就是 Spark。它不靠多副本，而是靠 lineage（记录每份数据是怎么算出来的）在出错时重算，
而宽依赖（需要 shuffle 的操作）是代价和故障恢复难度的分界线。
分布式训练里的 reduce-scatter、all-gather 和 parameter server，说到底就是分布式内存上的 reduce 与 gather。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. reduce 和 scan 只要求运算满足结合律，不要求交换律。浮点加法不严格满足结合律，所以并行归约的结果和串行不逐位相同。
2. Hillis-Steele 走 log n 步，每步约 n 次加法，work 是 O(n log n)，span 是 O(log n)。Blelloch 分 up-sweep 和 down-sweep 两趟，work 约 2n，span 是 2 log n。
3. stream compaction 分三步：打标志，对标志做 exclusive scan 得到目标下标，再 scatter。A3 的 find_repeats 和渲染器里的圆列表都是它。
4. Spark 的窄依赖（map、filter）可以在一个 stage 内流水执行，丢了分区只需重算一个父分区；宽依赖（groupByKey、未共分区的 join）要 shuffle，是 stage 的边界。
5. all-reduce 等于 reduce-scatter 加 all-gather，ring 实现下每卡收发 `2(p−1)/p·S`；分片 parameter server 的通信量与它相同，只有单点 PS 才有热点。

必须能默写的公式：

```text
inclusive scan:  y_i = x_0 ⊕ x_1 ⊕ … ⊕ x_i
exclusive scan:  y_0 = id,  y_i = x_0 ⊕ … ⊕ x_{i−1}
Hillis-Steele:   W = Σ_{d=0}^{log n − 1} (n − 2^d) = n·log n − (n − 1),   S = log n
Blelloch:        W = 2(n − 1) 次 ⊕,   S = 2·log n
分块 scan:       W ≈ 2n,   S ≈ 2n/P + log P              P 个 worker 各扫 n/P
segmented op:    (f1, x1) ⊕ (f2, x2) = (f1 | f2,  f2 ? x2 : x1 ⊕ x2)
ring 集合通信:   reduce-scatter = all-gather = (p−1)/p · S；   all-reduce = 2(p−1)/p · S
```

容易混的几对概念：

- inclusive scan 包含当前元素，exclusive scan 不包含。exclusive 的结果正好是 compaction 的写入下标。
- gather 按下标读，`out[i] = in[idx[i]]`；scatter 按下标写，`out[idx[i]] = in[i]`，后者要处理写冲突。
- groupByKey 把所有值 shuffle 过去再聚合；reduceByKey 先在 map 端局部聚合，再 shuffle。
- parameter server 是中心化（可以分片）的推拉模式，支持异步和稀疏更新；all-reduce 是对称的集合通信，适合同步的稠密梯度。

## 学习目标

- 能手推 8 个元素的 inclusive 和 exclusive scan，画出 Hillis-Steele 和 Blelloch 的过程，并算出 work 与 span；
- 能用 scan 实现 stream compaction、segmented scan 和 radix sort 的一轮 split；
- 能解释 Spark RDD 的 lineage、窄依赖与宽依赖、persist 和容错代价，并把 reduce-scatter、all-gather 与 parameter server 放进同一个通信量模型。

## 先修知识

- [M01](module-01-performance-models.md) 的 work/span，[M05](module-05-gpu-architecture-cuda.md) 的 block、shared memory 与 `__syncthreads`；
- Python 列表与 lambda。

## 核心概念

### 1. map 与 reduce

先从最简单的两个原语说起。map 对每个元素独立地应用一个函数，元素之间没有依赖，所以可以完美并行。

reduce 用一个二元运算 ⊕ 把整个集合归成一个值，比如求和、求最大值。它能并行的原因是结合律：
只要 ⊕ 满足结合律，括号就可以任意加，于是能组织成一棵深度为 log n 的树，每一层的运算同时进行。
注意这里不需要交换律。矩阵乘法不满足交换律，照样可以并行归约，只要保持左右顺序不变。

结合律在浮点数上并不严格成立，这会带来一个实际的后果：GPU 上 `torch.sum` 的结果可能和 CPU 不逐位相同，
而用原子加法做的归约，每次运行的结果都可能不一样。

### 2. scan

scan（前缀和）要求输出每个位置之前所有元素的累积。例如 `x = [3, 1, 7, 0, 4, 1, 6, 3]` 的 inclusive scan 是 `[3, 4, 11, 11, 15, 16, 22, 25]`，
exclusive scan 是 `[0, 3, 4, 11, 11, 15, 16, 22]`。

scan 看上去是严格串行的，因为每个输出都依赖前一个输出。但和 reduce 一样，结合律允许我们重新组织计算顺序，
所以 scan 也能在 O(log n) 的 span 内完成。后面的推导 1 到 3 会给出两种具体做法。

### 3. gather、scatter、sort、groupByKey

很多算法需要把数据挪到新的位置上，这就需要 gather 和 scatter。gather 按下标读，scatter 按下标写。
scatter 有一个 gather 没有的问题：如果多个元素要写到同一个位置，必须规定冲突怎么处理，是任取一个、求和，还是用原子操作。

有了 scan 和 scatter，排序也能做。radix sort 的每一轮按一个 bit 把数组稳定地分成两半：
对「该 bit 为 0」的标志做 exclusive scan，就得到 0 组每个元素的位置；1 组的位置等于 0 组的总数加上 1 组内部的 scan。
groupByKey 则可以由「按 key 排序、标记每段的段首、再做 segmented 操作」实现。

### 4. stream compaction 与 find_repeats

stream compaction 是 scan 最常见的用途：从一个数组里挑出满足条件的元素，紧凑地写到输出里，并保持原来的顺序。
难点在于每个线程都不知道自己该写到哪里，而 exclusive scan 正好算出了这个位置。

A3 的 `find_repeats` 要求输出所有满足 `a[i] == a[i+1]` 的下标 i，就是一个标准的 compaction：

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

每一步都是 map、scan 或 scatter 中的一种。整个过程不需要锁，写入位置互不冲突，输出也保持了原来的顺序。

### 5. segmented scan

很多问题是「数据分成若干段，每段内部各自求前缀和」，例如 CSR 格式的稀疏矩阵里每一行独立求和。
如果每段单独启动一次 scan，段长不均时既浪费启动开销，又难以均衡负载。

segmented scan 的做法是：用一个标志数组标出每段的段首，把每个元素换成一对 (flag, value)，
再定义一个新的运算：如果右边的元素是段首，就丢掉左边的累积。这个新运算仍然满足结合律，
所以任何现成的 scan 实现都能直接拿来处理分段问题，一次 kernel 就处理完所有段，也不怕段长不均。

### 6. Spark 与 RDD

把这些原语放到一个集群上，就要面对新问题：数据放不进一台机器，机器还会坏。Spark 的核心抽象是 RDD，
一种不可变、分区存放的分布式集合。

RDD 上的操作分两类。transformation（map、filter、reduceByKey、join）是惰性的，只记录 lineage，也就是这份数据由哪些输入经过哪些操作得到；
action（count、collect、save）才真正触发执行。调度器按宽依赖把 lineage 切成多个 stage：
stage 内的窄依赖算子流水执行，不落盘；stage 之间通过 shuffle 交换数据。

容错也靠 lineage。某个分区丢失时，Spark 从输入或最近的 checkpoint 出发，按 lineage 把它重算出来。
窄依赖下只需重算一个父分区；宽依赖下一个子分区依赖所有父分区，可能牵动大量重算。
所以 lineage 很长、要迭代很多轮的作业，需要定期 checkpoint。

Spark 相对 MapReduce 的主要优势来自 `persist()`：它把 RDD 留在内存里，于是迭代算法（比如逻辑回归多轮读同一份数据）
不必每一轮都重新读 HDFS。下面的例子里，filter 之后的结果被缓存，reduceByKey 引入了一次 shuffle：

```python
lines = sc.textFile("hdfs://logs/*")
errors = lines.filter(lambda l: "ERROR" in l).persist()      # 窄依赖，缓存在内存
by_host = errors.map(lambda l: (l.split()[0], 1)).reduceByKey(lambda a, b: a + b)  # 宽依赖，shuffle
print(by_host.take(10))                                       # action，触发两个 stage
```

## 关键推导

### 推导 1：Hillis-Steele 的 work 与 span

Hillis-Steele 的思路最直接：第 d 步让每个元素加上它左边距离 `2^d` 的元素，log n 步之后每个位置就累积了它之前的全部元素。

```text
for d = 0 .. log n − 1:            # 每步并行，需要双缓冲
    for all i ≥ 2^d:  x'[i] = x[i − 2^d] ⊕ x[i]
第 d 步做 n − 2^d 次 ⊕
W = Σ (n − 2^d) = n·log n − (n − 1)
S = log n
n = 8: 7 + 6 + 4 = 17 次；串行只需 7 次
```

它做了很多多余的加法，n = 8 时已经是串行的两倍多。但在宽度足够、n 又不大的场景下，这些多余的加法几乎是免费的。
例如一个 warp 有 32 个元素，5 步就能用 warp shuffle 完成，多出来的加法落在本来就空闲的 lane 上。

### 推导 2：Blelloch 的 work 与 span

Blelloch 想要的是 work-efficient（总工作量和串行同阶）的 scan。它先自底向上建一棵归约树，再自顶向下把前缀分发回去：

```text
up-sweep（建归约树）: 第 d 层把步长 2^{d+1} 的每段右端加上左半段的和
  共 n/2 + n/4 + … + 1 = n − 1 次 ⊕，结束时 x[n − 1] 是总和
置零: x[n − 1] = 0（⊕ 的单位元）
down-sweep: 从顶层往下，每个节点把自己的值给左孩子，把「自己 ⊕ 原左孩子」给右孩子
  再 n − 1 次 ⊕
W = 2(n − 1)，S = 2 log n，结果是 exclusive scan；完整代码见推导 3
```

和 Hillis-Steele 相比，Blelloch 的总工作量少了 log n 倍，代价是步数多了一倍。n 很大时，工作量更重要。

## 深入推导

### 推导 3：手推 Blelloch（n = 8）

面试里常要求在白板上手推一遍。用前面的例子，每一行是一步之后的数组状态：

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

对应的 Python 实现如下，最后一行的断言就是上面这张表的结果：

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

A3 要求支持任意长度。做法是把长度向上取到 2 的幂，多出来的位置填 0（⊕ 的单位元），这样不影响前 n 个结果。

### 推导 4：GPU 上 scan 的实际形态

纯 Blelloch 直接搬到 GPU 上并不好。每一层要一个 kernel，越往上活跃的线程越少，访问步长也越来越大，合并性越来越差。
所以实际实现是分层的：

```text
1. 每个 block 把一段数据读进 shared memory，在 block 内做 scan（warp 内用 Hillis-Steele + shuffle）
2. 每个 block 的总和写到辅助数组，对辅助数组递归 scan
3. 每个 block 给自己的段加上对应的前缀
```

第 1 步里 block 内的 scan 可以写成下面这样：

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

循环里的两个 `__syncthreads()` 分别保证「所有线程先读完再写」和「全部写完再进入下一轮」，相当于双缓冲的另一种写法。
CUB 的 decoupled look-back 更进一步，把三步合成一次遍历，访存量接近读写各 n 个元素（共 `2n`）的下限。

### 推导 5：分布式 reduce：reduce-scatter 与 all-gather

把 reduce 和 gather 放到多台机器上，就得到了分布式训练里的集合通信。
设 p 个节点各有一个长度为 S 字节的向量。reduce-scatter 之后，节点 j 拿到第 j 份（大小 S/p）的全局和；all-gather 之后，每个节点拿到全部 p 份。
ring 实现中，每一步每个节点把一份 S/p 发给右邻居，共 p − 1 步：

```text
reduce-scatter: (p − 1) · S/p
all-gather:     (p − 1) · S/p
all-reduce = reduce-scatter + all-gather = 2(p − 1)/p · S
```

parameter server 也能放进同一个模型。push 是把梯度 reduce 到服务器上，pull 是把参数 broadcast 回 worker。
把参数分片到 p 个节点上之后，push 就是 reduce-scatter，pull 就是 all-gather，通信量也一样。

PS 的价值因此不在通信量，而在两点：可以异步更新（不必等最慢的 worker），可以稀疏更新（只推拉被访问到的 embedding 行）。
ZeRO 的梯度分片用的是 reduce-scatter，参数重组用的是 all-gather（M12）。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)。第 2 题把 scan 的几种实现换算成 H100 上的时间下限，能直观看出 work-efficient 的价值。

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

- 「Hillis-Steele 比 Blelloch 差」：n 很大时它的工作量多 log n 倍；但在一个 warp 内，Hillis-Steele 步数少、不需要下扫，反而更快。
- 「reduce 需要交换律」：只需要结合律；只有想乱序合并时才需要交换律。
- 「并行求和的结果和串行一样」：浮点不满足结合律，结果有舍入差异；需要逐位可复现时，要固定归约顺序。
- 「Blelloch 的结果是 inclusive」：down-sweep 产生的是 exclusive，要 inclusive 需要再逐元素加一次原数组。
- 「Spark 的容错靠多副本」：RDD 靠 lineage 重算；只有 checkpoint 和 shuffle 文件会落盘。
- 「Spark 里 groupByKey 和 reduceByKey 一样」：前者 shuffle 所有值，后者先在 map 端合并，数据量可能差出数量级。

## 实验或练习

这个练习比较两种 scan 在不同规模下的表现，并和 Thrust 的实现对照，看工作量差别在什么规模下开始起作用。

- 假设：work-efficient 的 scan 在大 n 时比 Hillis-Steele 快；耗时随 n 线性增长，换算出的带宽达到 HBM 峰值的某个比例。
- 配置：在 A3 框架里实现两种 scan，n 取 `2^10` 到 `2^26`，记录耗时并与 Thrust 的 `exclusive_scan` 比较；
  用 Python 版覆盖 n = 1、非 2 的幂、全零等情况做单元测试。
- 预期：小 n 时两者都受 kernel 启动开销主导；大 n 时 Blelloch 的优势随 log n 增大；Thrust 因为用单遍算法而更快。
  find_repeats 的耗时约为一次 scan 加两次 map。

## 高频问题

- 问：reduce 和 scan 对运算有什么要求？
  答：只要求结合律，不要求交换律。
- 问：为什么 GPU 上 `torch.sum` 的结果可能和 CPU 不逐位相同？
  答：浮点加法不严格满足结合律，并行归约改变了加法顺序，舍入误差随之不同。
- 问：Spark 的窄依赖和宽依赖是什么？
  答：窄依赖的子分区只依赖少量父分区，可以流水执行、容易恢复；宽依赖依赖全部父分区，要 shuffle，是 stage 的边界。
- 问：groupByKey 和 reduceByKey 有什么区别？
  答：groupByKey 把所有值 shuffle 过去再聚合；reduceByKey 先在 map 端局部合并，shuffle 的数据量可能少出数量级。

## 面试追问链

1. 问：给 `[3, 1, 7, 0, 4, 1, 6, 3]` 写出 inclusive 和 exclusive scan。
   答：inclusive 是 `[3, 4, 11, 11, 15, 16, 22, 25]`；exclusive 是 `[0, 3, 4, 11, 11, 15, 16, 22]`。
2. 问：Hillis-Steele 和 Blelloch 的 work、span 分别是多少？
   答：前者 work 为 `n log n − (n−1)`、span 为 `log n`；后者 work 为 `2(n−1)`、span 为 `2 log n`。前者 span 短一半，但工作量多 log n 倍。
3. 问：在 GPU 上实际怎么实现一个 1 亿元素的 scan？
   答：分层做：block 内用 shared memory scan（warp 内用 shuffle 实现 Hillis-Steele），再对各 block 的总和做 scan，最后给每段加偏移；
   或者用单遍的 decoupled look-back。目标是让总访存接近读写各一遍的下限。
4. 问：用 scan 实现一轮 radix sort 的 split。
   答：对 bit 为 0 的标志做 exclusive scan，得到 0 组的位置；1 组的位置等于 0 组总数加上对 1 标志的 exclusive scan；最后 scatter，结果是稳定的。
5. 问：Spark 中一个 shuffle 之后的分区丢了，恢复代价是什么，怎么降低？
   答：宽依赖的子分区依赖所有父分区。如果 shuffle 文件还在，只需重读；如果 map 端也丢了，就要重算多个父分区。
   降低代价的办法有 checkpoint、对迭代算法做 persist，以及用 reduceByKey 减少 shuffle 量。
6. 问：MoE 层的 token dispatch 怎么用本讲的原语实现？
   答：每个 token 有一个专家号。先按专家号计数（直方图），做 exclusive scan 得到每个专家在输出中的起始位置，
   再把 token scatter 到按专家连续排列的缓冲区里，这等价于一次按 key 的稳定排序。计算完成后，按原下标 gather 回来。跨卡时 scatter 就变成了 all-to-all。
7. 问：ZeRO 为什么用 reduce-scatter 而不是 all-reduce？
   答：每张卡只负责更新自己那 1/p 的参数，只需要那一份梯度的和。reduce-scatter 的通信量是 `(p−1)/p·S`，只有 all-reduce 的一半；
   更新完再用 all-gather 拼回完整的参数。
8. 问：推荐系统训练大 embedding 表时，为什么常用 parameter server 而不是 all-reduce？
   答：embedding 表可达数百 GB 以上，每一步只访问极少的行，梯度高度稀疏。all-reduce 要同步整张表的稠密梯度，浪费带宽；
   分片 PS 只推拉被访问到的行，还允许异步更新。稠密的 MLP 部分仍然用 all-reduce。

## 自测题

1. 对 n = 16 计算 Hillis-Steele 和 Blelloch 的加法次数。
   要点：Hillis-Steele `15 + 14 + 12 + 8 = 49`；Blelloch `2 × 15 = 30`；串行 15。
2. 证明 segmented scan 的运算 `(f1, x1) ⊕ (f2, x2) = (f1 | f2, f2 ? x2 : x1 + x2)` 满足结合律。
   要点：分三种情况：`f3` 为真；`f3` 为假且 `f2` 为真；二者都为假。每种情况下两边加括号的结果都相同。
3. 只用 map、exclusive scan、scatter，写出删除数组中所有负数的算法。
   要点：`flag = x ≥ 0`；`idx = scan(flag)`；`flag[i]` 为 1 时 `out[idx[i]] = x[i]`；输出长度是最后一个 `idx + flag`。
4. 设计题：64 卡（8 节点 × 8 卡）all-reduce 16 GB 梯度。跨全部 64 卡的大 ring，和「节点内 reduce-scatter，节点间 all-reduce，节点内 all-gather」的分层方案，哪个快？
   要点：64 卡大 ring 受最慢的 IB 链路限制，每卡约 `2 × 63/64 × 16 GB ≈ 31.5 GB` 要走 IB；分层方案跨节点只传 `S/8` 的分片，
   每卡走 IB 的量约 `2 × 7/8 × 2 GB = 3.5 GB`，IB 流量少约 9 倍。

## 关联笔记

- [M05](module-05-gpu-architecture-cuda.md)：scan 与 compaction 的 CUDA 实现、circle renderer。
- [M12](module-12-data-parallel-zero.md)：all-reduce、reduce-scatter 与 ZeRO。
- [M18](module-18-architectures-moe-clusters.md)：MoE 的 token dispatch 与集群数据管线。

## 本讲小结

结合律让 reduce 和 scan 都能在 O(log n) 的 span 内完成。Blelloch 只用 2n 的工作量，Hillis-Steele 用更多的工作换更短的 span，
GPU 上的实现把两者分层组合起来。stream compaction、radix sort 和 MoE dispatch，都是 scan 加 scatter。
Spark 把这些原语放到集群上，用 lineage 做容错，宽依赖是代价的分界线；分布式训练的集合通信，则是同一组原语在多卡内存上的版本。

## 延伸阅读

- Blelloch, Prefix Sums and Their Applications, CMU-CS-90-190, 1990.
- Hillis, Steele, Data Parallel Algorithms, CACM 1986.
- Merrill, Garland, Single-pass Parallel Prefix Scan with Decoupled Look-back, NVIDIA Tech Report 2016.
- Zaharia et al., Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing, NSDI 2012.
- Li et al., Scaling Distributed Machine Learning with the Parameter Server, OSDI 2014.
