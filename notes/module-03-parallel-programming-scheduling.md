# Module 03 — 并行编程与调度

- 课程出处：CS149 Fall 2024《Parallel Programming Basics》《Work Distribution and Scheduling》
- 必读论文：Blumofe, Leiserson, Scheduling Multithreaded Computations by Work Stealing, 1999；Frigo, Leiserson, Randall, The Implementation of the Cilk-5 Multithreaded Language, 1998
- 对应实验：[CS149 A2 任务调度器](../solutions/cs149/asst2-task-scheduler/README.md)
- 实验报告：[报告 PDF](../reports/cs149-asst2/cs149_asst2_report.pdf)（实测：每次 launch：spawn 53.3 µs、睡眠池 10.8 µs、自旋池 0.78 µs）；[A5 报告 PDF](../reports/cs149-asst5/cs149_asst5_report.pdf)（实测：RMAT 上 direction-optimizing BFS 8 线程 6.58 倍；网格上并行慢两个数量级）
- 状态：初稿

## 摘要

写并行程序分四步：分解（拆成任务）、分配（任务给 worker）、编排（通信与同步）、映射（worker 到硬件）。
分配的核心矛盾是负载均衡与调度开销：任务太粗尾部不均，太细队列被挤爆。静态分配零开销但怕不可预测，
动态分配适应性强但要付出队列和同步的代价。work stealing 用每线程双端队列加随机窃取，
在期望意义上达到 `T1/P + O(T∞)`，同时保持局部性。ML 系统里的执行器、数据加载、推理调度都是这套思想。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 四步：decomposition → assignment → orchestration → mapping。前两步决定负载均衡，第三步决定通信开销。
2. 粒度原则：任务数远大于 worker 数，单任务耗时远大于取任务的开销。
3. 长任务先调度（LPT）能把尾部从「长任务时间 + 平均负载」压到「两者最大值」。
4. work stealing：自己从 deque 底部 push/pop（LIFO，局部性好），小偷从顶部偷（最老、通常最大的任务）。
5. Cilk 用 continuation stealing（child-first）：空间 `≤ P·S1`，单核执行顺序与串行程序一致。

**必须能默写的公式**

```text
下界:     TP ≥ max(T1/P, T∞)
贪心/窃取: TP ≤ T1/P + T∞  （贪心）；  E[TP] ≤ T1/P + O(T∞)  （随机 work stealing）
窃取次数:  E[#steals] = O(P · T∞)
并行度:    T1 / T∞，要求 ≫ P 才能近似线性加速
粒度:      T ≈ (N·t + N·o)/P + t_max      N 个任务，平均 t，单次取任务开销 o，最长任务 t_max
中心队列:  吞吐上限 ≈ 1 / o_serial          o_serial = 取任务时必须串行的那段时间
空间:      continuation stealing S_P ≤ P·S1； child stealing 可能 O(N)
```

**一句话区分**

- static vs dynamic assignment：运行前决定 vs 运行时按需领取。
- child stealing vs continuation stealing：spawn 时把子任务放进队列、自己继续 vs 自己先跑子任务、把后续放进队列。
- stalling join vs greedy join：发起 spawn 的线程必须等在 sync 处 vs 最后一个完成的线程继续往下跑。
- spinning vs sleeping 线程池：空转查队列，唤醒快但烧核 vs 在条件变量上睡，省核但唤醒有延迟。

## 学习目标

- 能把一个问题按四步拆开，指出每步可能成为瓶颈的地方；
- 能给出任务粒度的代价模型并选择合理的 chunk 大小；
- 能解释 Cilk 的 spawn/sync 语义和 work stealing 的实现细节；
- 能证明或复述 `T1/P + T∞` 的界，并用数字说明并行度不够时会怎样；
- 能实现带依赖的任务图调度和 sleeping 线程池，避免 lost wakeup。

## 先修知识

- [M01](module-01-performance-models.md) 的 work/span 与 Brent 定理；
- C++ `std::thread`、`std::mutex`、`std::condition_variable`；
- 递归与分治。

## 核心概念

### 1. 四步法

- 分解：找出可以并行的任务。分解不够细，Amdahl 的串行部分大。
- 分配：把任务交给 worker。目标是负载均衡、减少通信。
- 编排：决定数据结构、通信、同步（barrier、锁、消息）。
- 映射：worker 到核、到 GPU、到节点。通常由 OS 或运行时完成，程序员可以用亲和性干预。
CS149 的网格求解器例子：N×N 网格迭代，按 red-black 着色后同色点可并行更新；每轮结束要做全局收敛判断，
用每线程局部求和再 barrier 合并，而不是每个点去抢一把全局锁。

### 2. 静态、半静态与动态分配

静态：开始前固定分配，零运行时开销。适合每个任务代价已知且相近（大矩阵按块切）。
半静态：周期性地按上一轮的测量重新划分（N-body 中粒子分布缓慢变化）。
动态：任务放在共享队列，worker 空闲就领。适合代价不可预测（Mandelbrot 不同区域迭代数差很多、树搜索）。

### 3. 粒度

任务太粗：最后一个任务决定结束时间，其他 worker 空等。任务太细：每次领任务都要原子操作或加锁，
所有 worker 挤在同一个计数器上，队列本身变成串行瓶颈。常见折中：一次领一批（chunk），
或者让每个 worker 有自己的队列（下面的 work stealing）。

### 4. Cilk 的 spawn/sync

`cilk_spawn f()` 表示 f 可以与调用者后面的代码并发执行，不保证一定并发。`cilk_sync` 等待本函数内所有
spawn 出去的子调用。函数返回前有隐式 sync。语义上去掉这两个关键字就是正确的串行程序。

```cpp
int fib(int n) {
    if (n < 2) return n;
    int x = cilk_spawn fib(n - 1);   // 子任务
    int y = fib(n - 2);              // continuation 的一部分
    cilk_sync;
    return x + y;
}
// T1 = Θ(φ^n)，T∞ = Θ(n)，并行度巨大
```

### 5. work stealing 的实现

每个 worker 一个 deque。Cilk 执行 spawn 时，当前线程立即去跑子任务，把调用者的 continuation 压到自己 deque 底部。
空闲 worker 随机选一个受害者，从它 deque 顶部偷走最老的 continuation。
分治程序中，最老的 continuation 对应递归树上最靠近根的节点，代表最大的剩余工作，偷一次就够忙很久。
自己从底部取最新的任务，数据刚用过，还在 cache 里。

### 6. 带依赖的任务图

A2 的 Part B 要求 `runAsyncWithDeps(runnable, num_tasks, deps)`：一个 bulk launch 依赖若干之前的 launch。
实现方式是维护每个 launch 的未完成前驱数（in-degree）。前驱全部完成后进入就绪队列，worker 从就绪队列取。
这与深度学习框架的执行器同构：算子是节点，张量依赖是边，入度为 0 的算子可以派发。

### 7. 线程池：spinning 与 sleeping

每次 `run()` 都创建线程代价高（线程创建约 10 µs 量级）。线程池让线程常驻。
spinning：worker 循环检查队列，延迟最低，但没活时也占满核，和别的进程或主线程抢 CPU。
sleeping：worker 在条件变量上等待，有任务时被 notify。必须在 while 循环里检查条件（防虚假唤醒），
修改条件和 notify 都要配合同一把 mutex，否则可能丢失唤醒。

## 关键推导

### 推导 1：粒度的代价模型

N 个任务，平均耗时 t，单次领取开销 o（其中 `o_serial` 部分必须串行，例如对共享计数器的原子操作），P 个 worker：

```text
T ≈ (N·t + N·o)/P + t_max            并行时间 + 领取开销 + 尾部
约束: N / T ≤ 1 / o_serial            中心队列每秒最多服务 1/o_serial 次
t 太大 → N 小 → t_max/T 大（尾部不均）
t 太小 → N·o 占比大，或 N/T 超过队列服务率
```

结论：chunk 大小选到 `o/t` 只有几个百分点，同时 `N ≥ 数倍 P`。

### 推导 2：LPT 为什么减小尾部

```python
import heapq

def makespan(tasks, P):          # 按给定顺序贪心分给最早空闲的 worker
    workers = [0.0] * P
    for t in tasks:
        heapq.heappush(workers, heapq.heappop(workers) + t)
    return max(workers)

small = [1.0] * 63
print(makespan(small + [10.0], 8))   # 17.0：长任务最后才开始
print(makespan([10.0] + small, 8))   # 10.0：长任务先开始，其他人分掉小任务
```

总工作 73，下界 `max(73/8, 10) = 10`。长任务放最后时，它在 7 ms 才开始，结束于 17 ms。

## 深入推导

### 推导 3：work stealing 的界（证明思路）

把每个处理器在每个时间步的动作记一个 token：干活记 work token，尝试偷记 steal token。

```text
总 token = P · TP = (#work tokens) + (#steal tokens) = T1 + #steals
⇒ TP = T1/P + #steals/P
```

关键引理：随机窃取下，每 O(P) 次窃取尝试，以常数概率让关键路径的剩余长度减 1
（被偷走的是 deque 顶部的、位于关键路径上的节点）。所以 `E[#steals] = O(P·T∞)`，代入得 `E[TP] ≤ T1/P + O(T∞)`。
含义：并行度 `T1/T∞ ≫ P` 时，窃取开销被 `T1/P` 淹没，线性加速。

### 推导 4：child stealing 与 continuation stealing 的空间

```text
for (i = 0; i < N; i++) spawn work(i);
child stealing:         调用者先把 N 个子任务全部压入 deque，空间 O(N)
continuation stealing:  调用者先执行 work(0)，deque 中只留一个 "i = 1 处继续" 的 continuation
                        每个 worker 的栈深不超过串行执行的栈深 S1 → S_P ≤ P · S1
```

continuation stealing 需要编译器支持（保存和恢复栈帧），所以 Cilk 是语言扩展。
TBB、Java ForkJoin 这类库实现的是 child stealing，循环里 spawn 大量任务时要改成递归二分。

### 推导 5：greedy join

stalling join：发起 spawn 的线程卡在 sync，即使子任务被别人偷走、自己无事可做也只能等。
greedy join：每个 sync 点维护一个「未完成子任务计数」，最后一个完成的线程（不管是谁）继续执行 sync 后面的代码。
没有线程因为等 join 而空闲，这是 Cilk 能达到贪心界的必要条件之一。

### 推导 6：sleeping 线程池的正确写法

```cpp
void TaskSystem::workerLoop() {
    while (true) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return shutdown_ || next_ < num_tasks_; });   // 谓词防虚假唤醒
        if (shutdown_) return;
        int id = next_++;
        lk.unlock();                                  // 执行任务时不持锁
        runnable_->runTask(id, num_tasks_);
        lk.lock();
        if (++done_ == num_tasks_) done_cv_.notify_all();
    }
}
// run(): 加锁设置 runnable_/num_tasks_/next_=0/done_=0 → cv_.notify_all() → 在 done_cv_ 上等 done_ == num_tasks_
```

丢失唤醒的典型错误：主线程不加锁修改 `num_tasks_` 后 notify，worker 恰好在检查谓词和进入睡眠之间，错过通知。
同一把 mutex 保护谓词的读写就能排除这个窗口。

## 白板估算

数字来自 [硬件参数速查](../interview/HARDWARE-NUMBERS.md) 的 CPU 部分。

```text
1) 每次 run() 都创建 8 个线程，线程创建约 10 µs:
   单次额外开销 ≈ 8 × 10 µs = 80 µs（串行创建）
   调用 1000 次 → 80 ms；若每次 run 的真实工作只有 100 µs，开销占 44%
   线程池: 只创建一次，之后每次 run 只付一次 notify + 唤醒

2) 中心队列: 每次领取是对共享计数器的原子操作，跨核 cache line 迁移约 100 ns
   服务率上限 ≈ 1 / 100 ns = 1e7 次/s
   16 个 worker、每任务 1 µs → 需求 1.6e7 次/s > 1e7 → 队列成瓶颈
   一次领 10 个 → 需求 1.6e6 次/s，开销占比降到约 1%

3) work stealing 界: T1 = 1 s，T∞ = 1 ms
   P = 64:   T1/P + T∞ = 15.6 + 1 = 16.6 ms → 加速约 60
   P = 1024: 0.98 + 1 = 1.98 ms → 加速约 505，span 开始主导
   并行度 T1/T∞ = 1000，P 接近它时加速明显偏离线性

4) 尾部: 63 个 1 ms 任务 + 1 个 10 ms 任务，8 个 worker
   长任务最后: 17 ms；长任务最先: 10 ms；下界 max(73/8, 10) = 10 ms
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 每次 run 都 spawn 线程 | [A2](../solutions/cs149/asst2-task-scheduler/README.md) Part A 第一版 | 所有测试正确；小任务测试上明显慢于线程池 |
| spinning 线程池 | A2 Part A 第二版 | 任务多时接近参考实现；任务少时 CPU 被空转占满 |
| sleeping 线程池 | A2 Part A 第三版 | 与参考实现性能相当；没有死锁和丢失唤醒 |
| 带依赖的异步 launch | A2 Part B `runAsyncWithDeps` / `sync` | 依赖顺序正确；独立 launch 能并发；`sync` 后所有任务完成 |
| 粒度与负载均衡 | A2 的 `mandelbrot_chunked`、`math_operations_in_tight_for_loop` 等测试 | 能用粒度模型解释哪些测试受开销主导 |

## 易错点与反例

- 「动态调度总比静态好」：任务均匀时动态只多付队列开销。大 GEMM 按块静态切最好。
- 「任务越多越均衡」：太细时队列和原子操作成瓶颈，见白板估算 2。
- 「spawn 就等于新线程」：spawn 只是声明可以并行，运行时可以不并行；Cilk 里单核就是串行执行。
- 「work stealing 的小偷从底部偷」：错，从顶部偷最老的任务；底部是 owner 自己用的。
- 「条件变量 wait 用 if 判断条件」：虚假唤醒和多个 worker 竞争会导致越界取任务，必须 while 或带谓词。
- 「spinning 线程池更快」：机器被超额订阅（线程数 > 核数）时，spinning 线程抢走真正干活的线程的时间片，反而更慢。
- 「任务图里只要拓扑排序一次就行」：异步 launch 不断到来，必须增量维护入度和就绪队列。

## 实验或练习

- 假设：每次创建线程的版本在 launch 次数多、单次工作小的测试上最慢；sleeping 线程池在少任务测试上优于 spinning。
- 配置：A2 Part A 三个实现分别跑官方测试脚本，记录每个测试的耗时；把线程数从 2 改到 16，观察 spinning 版本在超额订阅时的变化。
- 预期：小任务、多 launch 的测试中线程池远快于每次创建；线程数超过核数后 spinning 版本退化明显。
  Part B 中依赖链越长，可并发的 launch 越少，耗时越接近串行和。

## 高频问题

- 问：静态分配和动态分配怎么选？
  答：代价可预测且均匀用静态；不可预测或差异大用动态，配合合适粒度。
- 问：work stealing 为什么从 deque 两端操作？
  答：owner 在底部 LIFO 保局部性，小偷在顶部取最老、最大的任务，减少窃取次数和冲突。
- 问：Cilk 的 `cilk_spawn` 有什么语义保证？
  答：只允许并发，不要求并发；去掉关键字后是等价的串行程序。
- 问：为什么要线程池？
  答：线程创建约 10 µs 量级，频繁创建会让小任务被开销淹没。
- 问：条件变量为什么必须配 mutex？
  答：保证检查条件和进入睡眠是原子的，避免丢失唤醒。

## 面试追问链

1. 问：并行化一个程序的四个步骤是什么？
   答：分解、分配、编排、映射。分解决定并行度，分配决定负载均衡，编排决定通信与同步开销。
2. 问：动态调度的任务粒度怎么选？
   答：`T ≈ (Nt + No)/P + t_max`，取单任务开销占比几个百分点、任务数为 worker 数的数倍；中心队列还要满足服务率上限 `1/o_serial`。
3. 问：work stealing 的时间界是多少，怎么来的？
   答：`E[TP] ≤ T1/P + O(T∞)`。token 计数得 `TP = (T1 + #steals)/P`，随机窃取使 `E[#steals] = O(P·T∞)`。
4. 问：child stealing 和 continuation stealing 各有什么取舍？
   答：前者库就能实现，但 spawn 循环会把 O(N) 个任务压入队列；后者空间 `P·S1`、串行顺序一致，但需要编译器支持。
5. 问：深度学习框架的执行器和本讲有什么关系？
   答：计算图就是任务图。执行器维护每个算子的未完成输入计数，入度为 0 的算子进就绪队列，由线程或 CUDA stream 执行。
6. 问：MoE 训练里专家负载不均，用本讲的语言怎么描述，怎么缓解？
   答：是静态分配下的负载不均，最慢的专家决定整步时间。缓解：辅助负载均衡 loss、capacity factor 截断溢出 token、
   专家复制，本质都是降低 `t_max` 相对平均值的比例。
7. 问：LLM serving 的 continuous batching 和静态 batching 的区别，用调度语言说？
   答：静态 batching 等一批请求全部生成完才换批，长请求拖住整批，相当于粗粒度静态分配；continuous batching 以每个
   decode 迭代为粒度动态补位，相当于细粒度动态分配，代价是每步都要重新组 batch、管理 KV cache。
8. 问：在 GPU 上如何实现动态负载均衡？
   答：persistent kernel：启动刚好占满所有 SM 的 block，每个 block 循环用 `atomicAdd` 从全局计数器领 tile，
   和 CPU 的中心队列同构；计数器争用同样要靠一次领多个 tile 缓解。

## 自测题

1. 64 个任务、8 个 worker，静态按块分配，每块 8 个任务；其中一块里有一个 10 ms 任务，其余都是 1 ms。求完成时间。
   要点：那个 worker `7 + 10 = 17 ms`，其余 8 ms；动态 LPT 可降到 10 ms。
2. 写出 work stealing 的 token 论证，说明为什么窃取次数和 `T∞` 有关。
   要点：每 O(P) 次窃取以常数概率推进关键路径一步。
3. 为什么 Cilk 的单核执行顺序和串行程序一样？
   要点：child-first，遇到 spawn 直接执行子调用，continuation 留在 deque 里没人偷。
4. 设计题：实现 `runAsyncWithDeps`，说明数据结构和线程如何被唤醒。
   要点：launch 表 + 前驱计数 + 后继列表 + 就绪队列；任务完成时递减后继的计数，归零入队并 notify；`sync` 等全部 launch 完成。
5. 一个 dataloader 有 4 个 worker，每个 batch 预处理 40 ms，GPU 每步 15 ms。GPU 会不会饿？
   要点：数据产出 `4 / 40 ms = 100 batch/s`，GPU 消耗 `1 / 15 ms ≈ 67 batch/s`，不饿；worker 降到 2 个就会饿。

## 关联笔记

- [M01](module-01-performance-models.md)：work/span 下界和贪心调度的上界。
- [M02](module-02-multicore-simd-threads.md)：ISPC task 与 Mandelbrot 负载不均。
- [M04](module-04-locality-coherence-synchronization.md)：队列、锁、原子操作的硬件代价。
- [M16](module-16-llm-serving.md)：continuous batching 与请求调度。
- [M18](module-18-architectures-moe-clusters.md)：MoE 负载均衡与集群调度。

## 本讲小结

分配要在负载均衡和调度开销之间取舍，粒度模型给出量化标准。work stealing 用局部 deque 和随机窃取，
期望达到 `T1/P + O(T∞)`，并行度远大于核数时接近线性。线程池消除线程创建开销，sleeping 版本要防丢失唤醒。
框架执行器、dataloader、推理调度器都是这些模式的变体。

## 延伸阅读

- Blumofe, Leiserson, Scheduling Multithreaded Computations by Work Stealing, JACM 1999.
- Frigo, Leiserson, Randall, The Implementation of the Cilk-5 Multithreaded Language, PLDI 1998.
- Culler, Singh, Gupta, Parallel Computer Architecture: A Hardware/Software Approach, 1998, Ch. 2–3.
- Chase, Lev, Dynamic Circular Work-Stealing Deque, SPAA 2005.
