# Module 04 — 局部性、通信、一致性与同步

- 课程出处：CS149 Fall 2024《Locality, Communication, and Contention》《Cache Coherence》《Implementing Synchronization / Fine-Grained Locking and Lock-Free Programming》《Relaxed Memory Consistency》《Transactional Memory》
- 必读论文：Adve, Gharachorloo, Shared Memory Consistency Models: A Tutorial, 1996；Mellor-Crummey, Scott, Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors, 1991；Herlihy, Moss, Transactional Memory: Architectural Support for Lock-Free Data Structures, 1993
- 对应实验：[CS149 A2 任务调度器](../solutions/cs149/asst2-task-scheduler/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst2/cs149_asst2_report.pdf)（实测：同一 cache line 上的原子计数器 12 线程慢 19 倍；2 倍超订时带锁自旋慢 766 倍）
- 状态：初稿

## 摘要

并行程序变慢，多半不是算得慢，而是在搬数据和互相等待。通信分两种：算法固有的，和硬件粒度带来的（cache line、
消息头）。共享内存机器用 cache coherence 协议保证同一地址的写对所有核可见，代价是 cache line 在核间来回迁移，
false sharing 就是这种代价的无谓版本。memory consistency 规定不同地址的读写顺序，x86 是 TSO，ARM 更弱，
所以无锁代码需要 fence 或 acquire/release。锁、CAS、无锁结构的性能差异，都能用「每次获取产生多少 coherence 流量」解释。
同一套逻辑延伸到 GPU 原子操作和分布式训练里的 parameter server 热点。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 通信计算比：N×N 网格分给 P 个处理器，块划分 `4√P/N`，条带划分 `2P/N`，P > 4 时块划分更好。
2. MSI 三态；MESI 加 E 态，让「先读后写」的私有数据不再发 bus 事务。directory 用点对点消息替代广播，每行要 P 位存储。
3. false sharing：不同线程写同一 64 B cache line 里的不同变量，line 来回迁移，每次数十到上百 ns。解法是按 64 B 对齐和填充。
4. TSO 只放松 W→R（store buffer）；ARM/POWER 放松全部四种。发布数据用 release store + acquire load。
5. TAS 锁持锁期间也在制造写流量；TTAS 只在释放时爆发；ticket 锁公平但释放时仍 O(P) 次读；MCS 每次释放 O(1)。

**必须能默写的公式**

```text
通信计算比（2D 网格）  块: 4(N/√P) / (N²/P) = 4√P / N      条带: 2N / (N²/P) = 2P / N
directory 开销       P bits / (64 B × 8) = P / 512           P = 64 → 12.5%，P = 256 → 50%
流水化通信           非流水: n·(o + L + s/β)    流水: n·(o + s/β) + L
PS 单点              T ≈ W·S / β_PS                           W 个 worker 都推给一个 PS
分片 PS / ring       每卡收发 2(p−1)/p · S                     与 reduce-scatter + all-gather 相同
```

**一句话区分**

- coherence vs consistency：同一地址的写如何传播 vs 不同地址的读写之间允许怎样重排。
- snooping vs directory：广播，规模小、延迟低 vs 点对点查表，可扩展但多一跳。
- true sharing vs false sharing：确实共享同一数据 vs 只是落在同一 cache line。
- lock-free vs wait-free：总有某个线程在前进 vs 每个线程都在有限步内完成。

## 学习目标

- 能计算不同划分方式的通信计算比，画出 MSI/MESI 状态机，识别并修复 false sharing；
- 能写出 TSO 下会失败的 litmus test，并给出 fence 或 acquire/release 修复；
- 能比较 TAS、TTAS、ticket、MCS 锁的流量，写出 CAS 无锁栈并解释 ABA；
- 能把这些概念迁移到 CUDA 原子操作和 parameter server vs all-reduce。

## 先修知识

- [M02](module-02-multicore-simd-threads.md) 的多核与 cache 层次，[M03](module-03-parallel-programming-scheduling.md) 的线程池与条件变量；
- C++ `std::atomic` 基本用法。

## 核心概念

### 1. 局部性与通信计算比

通信计算比 = 需要和别的处理器交换的数据量 / 本地计算量，越低越好。固有通信由算法决定（网格边界的邻居值）；
人为通信由实现细节造成：cache line 粒度大于所需数据、数据散落在多条 line 上、容量不足导致反复取。

### 2. 消息传递、流水化与争用

两个进程都先 `send` 再 `recv`，若 send 是同步的（等对方接收才返回），双方都卡在 send，死锁。
解法：奇偶编号交替顺序、`sendrecv`、或非阻塞 `isend/irecv` 再 `wait`。NCCL 的 ring 每卡同时向右发、从左收，靠异步与分块避开它。
一条大消息拆成 n 块发，前一块在网络上飞时下一块已经发出，延迟 L 只付一次。
争用是很多请求同时打到一个资源：全局计数器、一个 memory bank、一个 parameter server。缓解都靠去中心化：私有化后合并、分层归约、分片。

### 3. Cache coherence：snooping 与 MSI/MESI

每条 cache line 有状态。MSI：

- I（invalid）：PrRd → 发 BusRd → S；PrWr → 发 BusRdX → M。
- S（shared）：PrRd 命中；PrWr → 发 BusRdX/Upgrade → M；观察到别人的 BusRdX → I。
- M（modified）：读写都命中；观察到 BusRd → 写回，转 S；观察到 BusRdX → 写回，转 I。

MESI 加 E（exclusive clean）：读 miss 时若没有其他核持有，进 E；之后写入静默转 M，私有数据的「读-改-写」省掉一次总线事务。

### 4. Directory

核数多时广播不可扩展。每条 line 在 home 节点有目录项：P 个 presence bit + dirty bit。读 miss 先问 home，
写时只给持有者发 invalidation。代价是存储：P = 256 时目录有数据的一半大，所以有 limited-pointer、sparse directory 等压缩。

### 5. False sharing

```cpp
struct Counter { long v; };
Counter bad[8];                              // 8 × 8 B = 64 B，对齐时全在一条 line 上
struct alignas(64) PaddedCounter { long v; };
PaddedCounter good[8];                       // 每个计数器独占一条 line

void work(int tid, long iters) {
    for (long i = 0; i < iters; ++i) good[tid].v++;   // 换成 bad[tid] 会慢两到三个数量级
}
```

逻辑上每个线程只写自己的计数器，没有共享；物理上它们在同一条 line，每次写都要先拿到 M 态，把其他核的副本作废。
A2 的线程池里，每个 worker 的状态、完成计数如果紧挨着放，也会出现同样问题。

### 6. Memory consistency

四种顺序：W→R、W→W、R→R、R→W（前一个操作对后一个不同地址操作的可见顺序）。
SC：全部保持，所有核看到一个全局交错序。TSO（x86）：放松 W→R，写先进 store buffer，后面的读可以先完成。
PSO 再放松 W→W。ARM/POWER 放松全部四种。fence 强制前后操作有序；acquire 保证后续读写不被提前到它之前，
release 保证之前的读写不被推迟到它之后。

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

```text
N×N 网格，P 个处理器，每点更新读四邻居
块划分: 每块 (N/√P)×(N/√P)，边界 4·N/√P 个点
  ratio_block = (4N/√P) / (N²/P) = 4√P / N
条带划分: 每条 N/P 行，上下两行要交换 → 2N 个点
  ratio_strip = 2N / (N²/P) = 2P / N
块更优 ⇔ 4√P < 2P ⇔ P > 4
例: N = 1024, P = 64 → 块 0.031，条带 0.125（条带的好处是交换整行、内存连续、消息少）
```

### 推导 2：TSO 下的 litmus test

```text
初始 x = y = 0
T1: x = 1; r1 = y        T2: y = 1; r2 = x
SC:  不可能 r1 = r2 = 0（总有一个写先发生）
TSO: 可能 r1 = r2 = 0：两个写都还在各自 store buffer 里，读先去内存拿到了 0
修复: 在写和读之间加 mfence（或用 seq_cst 原子操作）
```

消息传递模式（写 data 再写 flag，另一边读 flag 再读 data）在 TSO 下是安全的，因为 W→W、R→R 不放松；在 ARM 上不安全。

## 深入推导

### 推导 3：几种锁的 coherence 流量（P 个线程争一把锁）

```text
TAS:    while (test_and_set(&l)) {}
        每次尝试都是写 → BusRdX，持锁期间所有等待者不断互相作废，流量持续
TTAS:   while (true) { while (l) {}; if (!test_and_set(&l)) break; }
        等待时在本地 S 态副本上读，零流量；释放时 1 次作废 + P 次重读 + 最多 P 次 TAS，爆发 O(P)
Ticket: my = fetch_add(&next, 1); while (serving != my) {}; ...; serving++
        获取只有 1 次原子操作；FIFO 公平；释放时一次写作废 P 个副本，P 次重读，O(P)
MCS:    每个等待者在自己的队列节点上自旋，释放者只写后继的节点 → O(1)
```

```cpp
struct TicketLock {
    std::atomic<int> next{0};
    alignas(64) std::atomic<int> serving{0};   // 与 next 分开放，新来者的 fetch_add 不会作废自旋者的副本
    void lock() { int my = next.fetch_add(1); while (serving.load(std::memory_order_acquire) != my) {} }
    void unlock() { serving.store(serving.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
};
```

### 推导 4：CAS 无锁栈与 ABA

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

ABA 时间线：

```text
T1: pop 读到 t = A, t->next = B，然后被抢占
T2: pop A；pop B；push A（A 的内存被复用回来）       栈: A → C
T1: CAS(top, A, B) 成功，因为 top 仍等于 A         栈顶变成已弹出的 B，结构损坏
```

修复：指针加版本号，用双字 CAS 比较 (指针, 计数)；或 hazard pointer / epoch 回收，保证 T1 持有 A 时 A 不会被复用；
硬件支持 LL/SC 的平台上，任何写入都会让 SC 失败，天然没有 ABA。
lock-free 保证系统整体进展，不保证更快：高争用下 CAS 不断重试，同样产生 line 迁移。

### 推导 5：事务内存的要点

`atomic { ... }` 声明一段代码要原子执行，系统乐观执行并记录读集和写集。冲突检测分 eager（访问时）与 lazy（提交时），
版本管理分 eager（原地写 + undo log）与 lazy（写缓冲，提交时生效），冲突时回滚重做。优点是可组合、无死锁；
代价是回滚浪费，且 I/O 这类不可回滚的操作不能放进事务。硬件 TM 在主流 CPU 上支持不稳定，面试知道取舍即可。

### 推导 6：GPU 原子操作与私有化

所有线程对同一个全局地址 `atomicAdd`，在 L2 上被串行化。直方图的标准写法是先在 shared memory 里私有化：

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

全局原子次数从 n 降到 `gridDim.x × nbins`。这与 CPU 上「每线程局部计数再合并」、分布式训练里「先本地累加梯度再通信」是同一个思想。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md)。

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

- 「coherence 保证了所有线程看到一致的执行顺序」：不对。coherence 只管同一地址，跨地址顺序由 consistency 决定。
- 「x86 是强内存模型所以不用 fence」：TSO 仍允许 W→R 重排，Dekker 式互斥在 x86 上同样会坏。
- 「`volatile` 能保证线程间可见性和顺序」：C++ 的 `volatile` 不提供原子性和跨线程顺序，要用 `std::atomic`。
- 「无锁一定比加锁快」：高争用下 CAS 重试和 line 迁移同样严重；无锁的优点是进展保证和不怕持锁线程被调度走。
- 「ticket 锁没有流量问题」：释放时所有等待者都要重读 `serving`，仍是 O(P)；各线程写不同变量也可能因 false sharing 争用。
- 「parameter server 一定比 all-reduce 慢」：分片 PS 的通信量与 ring 相同；单点 PS 才有热点。PS 的优势是支持异步更新和稀疏参数。

## 实验或练习

- 假设：8 线程写同一 cache line 上的不同计数器，耗时远大于按 64 B 填充后的版本。
- 配置：用上面的 `Counter` 与 `PaddedCounter` 各跑 8 线程、每线程 1e8 次自增，比较总耗时；再用 `perf stat` 观察 cache miss 数。
- 预期：未填充版本慢一到三个数量级（取决于 CPU 拓扑），cache miss 显著更多；填充版本耗时接近单线程。A2 中一次领多个任务，小任务测试应变快。

## 高频问题

- 问：MESI 比 MSI 多了什么？
  答：E 态。独占且干净的 line 写入时不需要发总线事务。
- 问：什么是 false sharing，怎么修？
  答：不同线程写同一 cache line 的不同数据导致 line 迁移；按 64 B 对齐或填充。
- 问：x86 允许哪种重排？
  答：TSO 允许后面的读越过前面对不同地址的写（W→R）。
- 问：什么是 ABA 问题？
  答：CAS 只比较值，值从 A 变 B 又变回 A 时 CAS 误以为没变；用版本号或安全内存回收解决。

## 面试追问链

1. 问：cache coherence 和 memory consistency 有什么区别？
   答：前者保证同一地址的写最终被所有核以同一顺序看到；后者规定不同地址操作之间允许哪些重排。
2. 问：画 MSI 的状态转移，E 态解决了什么问题？
   答：I/S/M 三态，读 miss 发 BusRd，写发 BusRdX；E 态让无人共享的 line 写入时静默升级，省一次总线事务。
3. 问：一个多线程计数程序加线程反而变慢，怎么定位？
   答：先查是否共享计数器（真争用）或计数器挤在同一 line（false sharing）；用 perf 看 cache miss 和 HITM 事件；修复用每线程私有计数加填充，最后合并。
4. 问：写一个 TSO 下会出错的例子，怎么修？
   答：Dekker litmus：两边先写自己的变量再读对方的，可能都读到 0；在写读之间加 mfence 或用 seq_cst 原子。
5. 问：TAS、TTAS、ticket、MCS 锁在 P 个线程争用时的流量分别如何？
   答：TAS 持续写流量；TTAS 释放时 O(P) 爆发；ticket 公平但释放时 O(P) 重读；MCS 每个等待者自旋在自己的节点上，释放 O(1)。
6. 问：写一个 CAS 无锁栈的 pop，ABA 怎么发生、怎么解决？
   答：读 top 和 next 后被抢占，别人 pop 两次再 push 回原节点，CAS 成功但 next 已失效；用带版本号的双字 CAS 或 hazard pointer。
7. 问：CUDA 里大量线程 `atomicAdd` 同一地址会怎样，怎么优化？
   答：在 L2 串行化，吞吐退化。先 warp 内归约或 shared memory 私有化，每个 block 只做一次全局原子。
8. 问：设计一个 64 卡训练的梯度同步方案，为什么一般不用单点 parameter server？
   答：单点 PS 的入口带宽要承受 `W·S`，是热点。用 ring 或分层 all-reduce，每卡收发 `2(p−1)/p·S`，与卡数几乎无关；
   节点内走 NVLink、节点间走 IB，并把通信分桶与反向计算重叠。稀疏 embedding 这类参数仍适合分片 PS。

## 自测题

1. N = 4096、P = 16，计算块划分与条带划分的通信计算比。
   要点：块 `4×4/4096 ≈ 0.0039`；条带 `32/4096 ≈ 0.0078`。
2. 给出一个只在 ARM 上出错、在 x86 上正确的程序，并修复。
   要点：消息传递模式（写 data 再写 flag）；flag 用 release store 与 acquire load。
3. 设计题：一个推理服务的请求统计模块，32 个线程每个请求都更新全局 QPS 计数，CPU 占用异常高，怎么改？
   要点：每线程私有计数器按 64 B 填充，定期汇总；或使用 per-CPU 计数器，读的时候求和。
4. 证明分片 parameter server 的每节点通信量等于 ring all-reduce。
   要点：每节点负责 `S/p`。作为 worker 推出 `(p−1)/p·S`、拉回 `(p−1)/p·S`；作为 server 收到 `(p−1)/p·S`、发出 `(p−1)/p·S`。每个方向合计 `2(p−1)/p·S`。

## 关联笔记

- [M03](module-03-parallel-programming-scheduling.md)：线程池和任务队列是本讲同步原语的使用者。
- [M05](module-05-gpu-architecture-cuda.md)：GPU 上的 shared memory、原子操作和 `__syncthreads`。
- [M06](module-06-data-parallel-primitives.md)：reduce-scatter 与 all-gather 作为分布式归约。
- [M12](module-12-data-parallel-zero.md)：all-reduce、梯度分桶与通信重叠。

## 本讲小结

数据搬运和等待是并行程序的主要成本。通信计算比衡量划分好坏，coherence 协议决定共享写的代价，
false sharing 是最常见的无谓开销。consistency 模型决定无锁代码要加什么 fence。锁的好坏看每次获取和释放产生多少流量，
无锁结构要处理 ABA 和内存回收。GPU 原子与 parameter server 热点都是争用问题，解法都是私有化、分层和分片。

## 延伸阅读

- Adve, Gharachorloo, Shared Memory Consistency Models: A Tutorial, IEEE Computer 1996.
- Mellor-Crummey, Scott, Algorithms for Scalable Synchronization on Shared-Memory Multiprocessors, TOCS 1991.
- Herlihy, Moss, Transactional Memory: Architectural Support for Lock-Free Data Structures, ISCA 1993.
- Li et al., Scaling Distributed Machine Learning with the Parameter Server, OSDI 2014.
