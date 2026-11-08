# Module 04 — 局部性、通信、一致性与同步

- 课程出处：CS149 Fall 2024《Locality, Communication, and Contention》《Cache Coherence》《Implementing Synchronization / Fine-Grained Locking and Lock-Free Programming》《Relaxed Memory Consistency》《Transactional Memory》
- 必读论文：Adve, Gharachorloo, Shared Memory Consistency Models: A Tutorial, 1996；Mellor-Crummey, Scott, Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors, 1991；Herlihy, Moss, Transactional Memory: Architectural Support for Lock-Free Data Structures, 1993
- 对应实验：[CS149 A2 任务调度器](../solutions/cs149/asst2-task-scheduler/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst2/cs149_asst2_report.pdf)（实测：同一 cache line 上的原子计数器 12 线程慢 19 倍；2 倍超订时带锁自旋慢 766 倍）
- 状态：初稿

## 摘要

一个并行程序加了线程反而变慢，原因多半不是算得慢，而是在搬数据和互相等待。这一讲就是要把这些看不见的成本讲清楚。

通信有两种来源。一种是算法固有的，比如网格边界上必须交换的邻居值；另一种是硬件粒度带来的，
比如明明只要 8 个字节，却得搬一整条 cache line。共享内存的机器用 cache coherence（缓存一致性）协议
保证对同一地址的写能被所有核看到，代价是 cache line 在核之间来回迁移；false sharing（伪共享）就是这种代价的无谓版本。

另一个容易混的概念是 memory consistency（内存一致性模型），它规定对不同地址的读写可以以什么顺序被别的核看到。
x86 是 TSO，ARM 更弱，所以无锁代码需要 fence 或 acquire/release 语义。
各种锁、CAS 和无锁结构的性能差异，都可以用「每次获取产生多少 coherence 流量」来解释。
同一套逻辑还能延伸到 GPU 原子操作，以及分布式训练里 parameter server 的热点问题。

## 一页速览卡（面试前 5 分钟）

必须记住的五件事：

1. 通信计算比：N×N 网格分给 P 个处理器，块划分是 `4√P/N`，条带划分是 `2P/N`，所以 P > 4 时块划分更好。
2. MSI 有三个状态。MESI 多一个 E 态，让「先读后写」的私有数据不再发总线事务。directory 用点对点消息代替广播，代价是每条 line 要 P 位存储。
3. false sharing：不同线程写同一条 64 B cache line 里的不同变量，line 在核之间来回迁移，每次数十到上百 ns。解法是按 64 B 对齐并填充。
4. TSO 只放松 W→R 这一种顺序（因为有 store buffer），ARM 和 POWER 四种全部放松。发布数据要用 release store 配 acquire load。
5. TAS 锁在持锁期间也一直制造写流量；TTAS 只在释放时爆发一次；ticket 锁公平，但释放时仍有 O(P) 次重读；MCS 每次释放只有 O(1)。

必须能默写的公式：

```text
通信计算比（2D 网格）  块: 4(N/√P) / (N²/P) = 4√P / N      条带: 2N / (N²/P) = 2P / N
directory 开销       P bits / (64 B × 8) = P / 512           P = 64 → 12.5%，P = 256 → 50%
流水化通信           非流水: n·(o + L + s/β)    流水: n·(o + s/β) + L
PS 单点              T ≈ W·S / β_PS                           W 个 worker 都推给一个 PS
分片 PS / ring       每卡收发 2(p−1)/p · S                     与 reduce-scatter + all-gather 相同
```

容易混的几对概念：

- coherence 管的是同一地址的写如何传播；consistency 管的是不同地址的读写之间允许怎样重排。
- snooping 靠广播，规模小时延迟低；directory 靠点对点查表，可扩展，但多一跳。
- true sharing 是确实在共享同一份数据；false sharing 只是不同数据碰巧落在同一条 cache line 上。
- lock-free 保证总有某个线程在前进；wait-free 保证每个线程都能在有限步内完成。

## 学习目标

- 能计算不同划分方式的通信计算比，画出 MSI 和 MESI 状态机，识别并修复 false sharing；
- 能写出 TSO 下会失败的 litmus test，并用 fence 或 acquire/release 修复；
- 能比较 TAS、TTAS、ticket、MCS 锁的流量，写出 CAS 无锁栈并解释 ABA 问题；
- 能把这些概念迁移到 CUDA 原子操作，以及 parameter server 与 all-reduce 的比较上。

## 先修知识

- [M02](module-02-multicore-simd-threads.md) 的多核与 cache 层次，[M03](module-03-parallel-programming-scheduling.md) 的线程池与条件变量；
- C++ `std::atomic` 的基本用法。

## 核心概念

### 1. 局部性与通信计算比

怎么判断一种划分方式好不好？一个简单的指标是通信计算比：需要和别的处理器交换的数据量，除以本地的计算量。越低越好。

通信有两类。固有通信由算法决定，比如网格边界上的邻居值，换什么实现都省不掉。
人为通信由实现细节造成：cache line 的粒度比实际需要的数据大，数据散落在很多条 line 上，或者 cache 容量不够、只好反复去取。
后一类才是优化的空间。

### 2. 消息传递、流水化与争用

在分布式内存上，数据靠显式的消息传递。这里要注意死锁、延迟和争用三件事。

先说死锁。两个进程都先 `send` 再 `recv`，如果 send 是同步的（要等对方接收才返回），双方就都卡在 send 上。
解法是让奇数号和偶数号进程交替顺序、用 `sendrecv`，或者用非阻塞的 `isend`、`irecv` 再 `wait`。
NCCL 的 ring 让每张卡同时向右发、从左收，靠异步和分块避开了这个问题。

延迟可以靠流水化来摊薄。一条大消息如果拆成 n 块发，前一块还在网络上飞时，下一块已经发出去了，于是延迟 L 只需要付一次。

争用是指很多请求同时打到同一个资源上，比如一个全局计数器、一个 memory bank 或一个 parameter server。
缓解的办法都是去中心化：先私有化再合并，分层归约，或者分片。

### 3. Cache coherence：snooping 与 MSI/MESI

每个核都有自己的 cache，同一个地址就可能在几个 cache 里各有一份副本。一个核写了它，别的核怎么知道？
snooping（总线监听）协议的做法是：每条 cache line 带一个状态，每个 cache 都监听总线上别人发出的请求。最基本的 MSI 协议有三个状态：

- I（invalid，无效）：处理器读（PrRd）时发 BusRd，转到 S；处理器写（PrWr）时发 BusRdX，转到 M。
- S（shared，共享）：读直接命中；写时发 BusRdX 或 Upgrade，转到 M；看到别人的 BusRdX，就转到 I。
- M（modified，已修改）：读写都直接命中；看到别人的 BusRd，先写回再转 S；看到别人的 BusRdX，先写回再转 I。

MSI 有一个浪费：一个核读入一条没人共享的数据、紧接着又写它，也要先进 S 再发一次总线事务升级到 M。
MESI 多加了一个 E 态（exclusive clean，独占且干净）：读 miss 时如果没有其他核持有这条 line，就进入 E；
之后写入时静默转为 M，不发总线事务。私有数据的「读-改-写」于是省掉了一次总线事务。

### 4. Directory

核数一多，每次都广播就不可扩展了。directory（目录）协议的做法是：每条 line 在它的 home 节点上有一个目录项，
记录 P 个 presence bit（哪些核持有副本）和一个 dirty bit。读 miss 时先问 home，写时只给真正持有副本的核发作废消息。

代价是存储。P = 256 时，目录的大小有数据本身的一半，所以实际系统会用 limited-pointer、sparse directory 等方式压缩。

### 5. False sharing

coherence 协议以 cache line 为单位工作，这带来一个反直觉的现象：两个线程各写各的变量，彼此毫无关系，却会互相拖慢。

```cpp
struct Counter { long v; };
Counter bad[8];                              // 8 × 8 B = 64 B，对齐时全在一条 line 上
struct alignas(64) PaddedCounter { long v; };
PaddedCounter good[8];                       // 每个计数器独占一条 line

void work(int tid, long iters) {
    for (long i = 0; i < iters; ++i) good[tid].v++;   // 换成 bad[tid] 会慢两到三个数量级
}
```

逻辑上每个线程只写自己的计数器，没有任何共享；但物理上它们在同一条 line 里，每次写都要先把这条 line 拿到 M 态，
同时把其他核的副本作废。A2 的线程池里，每个 worker 的状态和完成计数如果紧挨着放，也会出现同样的问题。

### 6. Memory consistency

coherence 只保证同一个地址上的写有统一的顺序。但无锁代码往往依赖不同地址之间的顺序，
比如「先写数据、再写标志」，这就要看 consistency 模型允许什么。

两个操作之间的顺序有四种：W→R、W→W、R→R、R→W，指前一个操作和后一个对不同地址的操作被别人看到的先后。
SC（顺序一致性）四种都保持，所有核看到同一个全局交错顺序。
TSO（x86）放松 W→R：写先进 store buffer，后面的读可以先完成。PSO 进一步放松 W→W。ARM 和 POWER 四种全部放松。

程序员用 fence 强制前后的操作有序。更细的工具是 acquire 和 release：acquire 保证后续的读写不会被提前到它之前，
release 保证之前的读写不会被推迟到它之后。下面是用它们发布数据的标准写法：

```cpp
std::atomic<int> flag{0};
int data = 0;
// 线程 1
data = 42;
flag.store(1, std::memory_order_release);
// 线程 2
while (flag.load(std::memory_order_acquire) == 0) {}
assert(data == 42);                          // 有 acquire/release 保证；在 ARM 上用 relaxed 可能失败
```

## 关键推导

### 推导 1：网格划分的通信计算比

同一个网格，切成方块和切成条带，哪个通信少？算一下边界就知道：

```text
N×N 网格，P 个处理器，每点更新读四邻居
块划分: 每块 (N/√P)×(N/√P)，边界 4·N/√P 个点
  ratio_block = (4N/√P) / (N²/P) = 4√P / N
条带划分: 每条 N/P 行，上下两行要交换 → 2N 个点
  ratio_strip = 2N / (N²/P) = 2P / N
块更优 ⇔ 4√P < 2P ⇔ P > 4
例: N = 1024, P = 64 → 块 0.031，条带 0.125（条带的好处是交换整行、内存连续、消息少）
```

P 大于 4 以后，块划分的通信量更少；但条带也有它的好处：交换的是整行，内存连续，消息数也少。

### 推导 2：TSO 下的 litmus test

「x86 是强内存模型」是一个常见的说法，但 TSO 仍然允许一种重排。下面这个 litmus test（用来检验内存模型的小程序）能把它暴露出来：

```text
初始 x = y = 0
T1: x = 1; r1 = y        T2: y = 1; r2 = x
SC:  不可能 r1 = r2 = 0（总有一个写先发生）
TSO: 可能 r1 = r2 = 0：两个写都还在各自 store buffer 里，读先去内存拿到了 0
修复: 在写和读之间加 mfence（或用 seq_cst 原子操作）
```

反过来，消息传递模式（一边写 data 再写 flag，另一边读 flag 再读 data）在 TSO 下是安全的，因为 TSO 不放松 W→W 和 R→R；
但到了 ARM 上就不安全了。

## 深入推导

### 推导 3：几种锁的 coherence 流量（P 个线程争一把锁）

为什么同样是自旋锁，有的随线程数增加急剧变慢，有的几乎不受影响？看每种锁在等待和释放时各产生多少 coherence 流量：

```text
TAS:    while (test_and_set(&l)) {}
        每次尝试都是写 → BusRdX，持锁期间所有等待者不断互相作废，流量持续
TTAS:   while (true) { while (l) {}; if (!test_and_set(&l)) break; }
        等待时在本地 S 态副本上读，零流量；释放时 1 次作废 + P 次重读 + 最多 P 次 TAS，爆发 O(P)
Ticket: my = fetch_add(&next, 1); while (serving != my) {}; ...; serving++
        获取只有 1 次原子操作；FIFO 公平；释放时一次写作废 P 个副本，P 次重读，O(P)
MCS:    每个等待者在自己的队列节点上自旋，释放者只写后继的节点 → O(1)
```

ticket 锁实现时有一个细节：`next` 和 `serving` 要放在不同的 cache line 上，否则会自己制造 false sharing。

```cpp
struct TicketLock {
    std::atomic<int> next{0};
    alignas(64) std::atomic<int> serving{0};   // 与 next 分开放，新来者的 fetch_add 不会作废自旋者的副本
    void lock() { int my = next.fetch_add(1); while (serving.load(std::memory_order_acquire) != my) {} }
    void unlock() { serving.store(serving.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
};
```

### 推导 4：CAS 无锁栈与 ABA

无锁结构用 CAS（compare-and-swap，比较并交换）代替锁：读出旧值，算出新值，只有当前值仍等于旧值时才写入，否则重试。
最简单的例子是无锁栈：

```cpp
struct Node { int val; Node* next; };
std::atomic<Node*> top{nullptr};

void push(Node* n) {
    n->next = top.load();
    while (!top.compare_exchange_weak(n->next, n)) {}   // 失败时 n->next 被更新为当前 top
}
Node* pop() {
    Node* t = top.load();
    while (t && !top.compare_exchange_weak(t, t->next)) {}
    return t;
}
```

这段代码有一个隐蔽的 bug，叫 ABA 问题：CAS 只比较值，看不出这个值中间被改过又改了回来。时间线如下：

```text
T1: pop 读到 t = A, t->next = B，然后被抢占
T2: pop A；pop B；push A（A 的内存被复用回来）       栈: A → C
T1: CAS(top, A, B) 成功，因为 top 仍等于 A         栈顶变成已弹出的 B，结构损坏
```

修复方法有几种。可以给指针加一个版本号，用双字 CAS 同时比较指针和计数；
也可以用 hazard pointer 或 epoch 回收，保证 T1 还拿着 A 的时候 A 不会被复用。
在支持 LL/SC 的平台上，中间任何一次写入都会让 SC 失败，所以天然没有 ABA。

还要记住：lock-free 保证的是系统整体一直在前进，不保证更快。高争用下 CAS 会不断重试，同样产生 cache line 迁移。

### 推导 5：事务内存的要点

锁难以组合，无锁结构又难写，事务内存（transactional memory）想换一种思路。
程序员用 `atomic { ... }` 声明一段代码要原子执行，系统乐观地执行它，同时记录读集和写集，发现冲突就回滚重做。

实现上有两个维度的选择。冲突检测可以是 eager（访问时就检查）或 lazy（提交时才检查）；
版本管理可以是 eager（原地写，同时记 undo log）或 lazy（写进缓冲区，提交时才生效）。

它的优点是可组合、不会死锁；代价是回滚会浪费工作，而且 I/O 这类无法回滚的操作不能放进事务。
硬件事务内存在主流 CPU 上的支持并不稳定，面试中知道这些取舍就够了。

### 推导 6：GPU 原子操作与私有化

GPU 上的争用问题更突出，因为同时在跑的线程多得多。所有线程对同一个全局地址做 `atomicAdd`，会在 L2 上被串行化。
直方图的标准写法是先在 shared memory 里私有化：

```cuda
__global__ void hist(const int* in, int n, int* out, int nbins) {
    extern __shared__ int local[];
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) local[b] = 0;
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[in[i]], 1);                 // block 内争用，shared memory 原子
    __syncthreads();
    for (int b = threadIdx.x; b < nbins; b += blockDim.x)
        atomicAdd(&out[b], local[b]);                // 每个 block 每个 bin 只做一次全局原子
}
```

这样，全局原子操作的次数从 n 降到了 `gridDim.x × nbins`。
CPU 上「每线程局部计数再合并」，分布式训练里「先本地累加梯度再通信」，用的都是同一个思想。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)。第 3 题是面试里最常见的一道：为什么大规模训练不用单点 parameter server。

```text
1) false sharing: 8 线程各自自增，计数器挤在同一条 line
   每次自增要迁移 line，约 100 ns → 全系统约 1e7 次/s
   填充后每次自增在 L1 内完成（~1 ns 量级）→ 每线程约 1e9 次/s，8 线程约 8e9 次/s
   差距约 800 倍，即两到三个数量级

2) directory 存储: 64 B line = 512 bit
   P = 64 → 64/512 = 12.5%；P = 256 → 50%；P = 1024 → 200%

3) 单点 parameter server vs ring all-reduce: 7B 模型 bf16 梯度 S = 14 GB，8 个 worker，IB HDR 25 GB/s
   单 PS 接收: 8 × 14 GB = 112 GB → 112 / 25 = 4.48 s（回传参数再来一遍）
   ring all-reduce: 2 × 7/8 × 14 = 24.5 GB/卡 → 24.5 / 25 = 0.98 s
   8 分片 PS: 每节点收发 7/8 × S（push）+ 7/8 × S（pull），与 ring 相同 0.98 s
   换到节点内 NVLink4 450 GB/s: 24.5 / 450 = 54 ms

4) 流水化通信: 1 GB 拆 64 块经 4 跳转发，每跳 25 GB/s
   不流水（存储转发整条消息）: 4 × 1/25 = 160 ms
   流水: 1/25 + 3 × (1/64)/25 ≈ 40 + 1.9 = 41.9 ms
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 互斥与条件变量 | [A2](../solutions/cs149/asst2-task-scheduler/README.md) Part A 线程池的任务计数与完成计数 | 无数据竞争；ThreadSanitizer 无报告 |
| 原子计数器争用 | A2 中 worker 领取任务的 `next_` 计数 | 小任务测试中一次领多个任务能明显降低耗时 |
| false sharing | A2 中每个 worker 的私有状态布局 | 私有状态按 64 B 对齐后，多线程测试耗时不变差 |
| 内存序 | A2 spinning 版本中用原子变量发布新 launch | 用 acquire/release 后在任何平台都正确 |

## 易错点与反例

- 「coherence 保证了所有线程看到一致的执行顺序」：不对。coherence 只管同一个地址，跨地址的顺序由 consistency 模型决定。
- 「x86 是强内存模型，所以不用 fence」：TSO 仍然允许 W→R 重排，Dekker 式的互斥在 x86 上同样会出错。
- 「`volatile` 能保证线程间的可见性和顺序」：C++ 的 `volatile` 既不提供原子性，也不提供跨线程的顺序，要用 `std::atomic`。
- 「无锁一定比加锁快」：高争用下 CAS 重试和 line 迁移同样严重。无锁的优点是进展有保证，也不怕持锁线程被调度走。
- 「ticket 锁没有流量问题」：释放时所有等待者都要重读 `serving`，仍是 O(P)；而且几个变量挨着放时，还可能因 false sharing 争用。
- 「parameter server 一定比 all-reduce 慢」：分片 PS 的通信量与 ring 相同，只有单点 PS 才有热点。PS 的优势是支持异步更新和稀疏参数。

## 实验或练习

这个练习用最小的程序复现 false sharing，并用硬件计数器确认原因。

- 假设：8 个线程写同一条 cache line 上的不同计数器，耗时远大于按 64 B 填充后的版本。
- 配置：用上面的 `Counter` 与 `PaddedCounter` 各跑 8 个线程，每个线程自增 1e8 次，比较总耗时；再用 `perf stat` 观察 cache miss 数。
- 预期：未填充的版本慢一到三个数量级（取决于 CPU 拓扑），cache miss 明显更多；填充后的版本耗时接近单线程。
  在 A2 中改成一次领多个任务后，小任务测试应该变快。

## 高频问题

- 问：snooping 和 directory 协议怎么选？
  答：核数少时用 snooping，广播简单、延迟低；核数多时广播不可扩展，用 directory 点对点查表，代价是多一跳和目录存储。
- 问：两个进程都先 `send` 再 `recv`，为什么可能死锁？
  答：同步 send 要等对方接收才返回，双方都卡在 send 上。用奇偶交替、`sendrecv` 或非阻塞的 `isend`/`irecv` 解决。
- 问：C++ 里能用 `volatile` 做线程间同步吗？
  答：不能。它不提供原子性和跨线程顺序，要用 `std::atomic` 和合适的 memory order。
- 问：lock-free 和 wait-free 有什么区别？
  答：lock-free 保证总有某个线程在前进，个别线程可能一直重试；wait-free 更强，保证每个线程都在有限步内完成。

## 面试追问链

1. 问：cache coherence 和 memory consistency 有什么区别？
   答：前者保证对同一地址的写最终被所有核以同一顺序看到；后者规定对不同地址的操作之间允许哪些重排。
2. 问：画一下 MSI 的状态转移。E 态解决了什么问题？
   答：I、S、M 三个状态，读 miss 发 BusRd，写发 BusRdX。E 态让没人共享的 line 在写入时静默升级为 M，省掉一次总线事务。
3. 问：一个多线程计数程序，加线程反而变慢，怎么定位？
   答：先看是不是共享同一个计数器（真争用），还是不同计数器挤在同一条 line 上（false sharing）。
   用 perf 看 cache miss 和 HITM 事件来确认。修复办法是每个线程用私有计数器并做填充，最后再合并。
4. 问：写一个在 TSO 下会出错的例子，怎么修？
   答：Dekker 式的 litmus test：两边都先写自己的变量、再读对方的，可能两边都读到 0。在写和读之间加 mfence，或者用 seq_cst 原子操作。
5. 问：P 个线程争用时，TAS、TTAS、ticket、MCS 锁的流量分别如何？
   答：TAS 一直有写流量；TTAS 在释放时爆发 O(P)；ticket 公平，但释放时有 O(P) 次重读；MCS 让每个等待者在自己的节点上自旋，释放只要 O(1)。
6. 问：写一个 CAS 无锁栈的 pop。ABA 是怎么发生的，怎么解决？
   答：读完 top 和 next 后被抢占，其他线程 pop 两次再把原节点 push 回来，CAS 照样成功，但 next 已经失效。
   解决办法是带版本号的双字 CAS，或者 hazard pointer。
7. 问：CUDA 里大量线程对同一地址 `atomicAdd` 会怎样，怎么优化？
   答：这些原子操作在 L2 上串行化，吞吐随之退化。先做 warp 内归约或在 shared memory 里私有化，让每个 block 只做一次全局原子。
8. 问：设计一个 64 卡训练的梯度同步方案。为什么一般不用单点 parameter server？
   答：单点 PS 的入口带宽要承受 `W·S`，是热点。用 ring 或分层 all-reduce，每卡收发 `2(p−1)/p·S`，与卡数几乎无关；
   节点内走 NVLink、节点间走 IB，并把通信分桶、与反向计算重叠。稀疏的 embedding 参数仍然适合用分片 PS。

## 自测题

1. N = 4096、P = 16，计算块划分与条带划分的通信计算比。
   要点：块 `4×4/4096 ≈ 0.0039`；条带 `32/4096 ≈ 0.0078`。
2. 给出一个只在 ARM 上出错、在 x86 上正确的程序，并修复它。
   要点：消息传递模式（写 data 再写 flag）；flag 改用 release store 与 acquire load。
3. 设计题：一个推理服务的请求统计模块，32 个线程处理每个请求时都更新全局 QPS 计数，CPU 占用异常高。怎么改？读的一侧怎么办？
   要点：每线程一个私有计数器，按 64 B 填充，定期汇总；或者用 per-CPU 计数器，读的时候把各份加起来。
4. 证明分片 parameter server 的每节点通信量等于 ring all-reduce。
   要点：每个节点负责 `S/p`。作为 worker 推出 `(p−1)/p·S`、拉回 `(p−1)/p·S`；作为 server 收到 `(p−1)/p·S`、发出 `(p−1)/p·S`。每个方向合计 `2(p−1)/p·S`。

## 关联笔记

- [M03](module-03-parallel-programming-scheduling.md)：线程池和任务队列是本讲同步原语的使用者。
- [M05](module-05-gpu-architecture-cuda.md)：GPU 上的 shared memory、原子操作和 `__syncthreads`。
- [M06](module-06-data-parallel-primitives.md)：reduce-scatter 与 all-gather 作为分布式归约。
- [M12](module-12-data-parallel-zero.md)：all-reduce、梯度分桶与通信重叠。

## 本讲小结

并行程序的主要成本是搬数据和等待。通信计算比衡量划分的好坏；coherence 协议决定了共享写的代价，false sharing 是最常见的无谓开销；
consistency 模型决定了无锁代码要在哪里加 fence。评价一把锁，看它每次获取和释放产生多少流量；写无锁结构，要处理 ABA 和内存回收。
GPU 原子操作和 parameter server 热点都是争用问题，解法也都一样：私有化、分层和分片。

## 延伸阅读

- Adve, Gharachorloo, Shared Memory Consistency Models: A Tutorial, IEEE Computer 1996.
- Mellor-Crummey, Scott, Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors, TOCS 1991.
- Herlihy, Moss, Transactional Memory: Architectural Support for Lock-Free Data Structures, ISCA 1993.
- Li et al., Scaling Distributed Machine Learning with the Parameter Server, OSDI 2014.
