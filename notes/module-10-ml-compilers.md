# Module 10 — ML 编译器与图优化

- 课程出处：CS149 Fall 2024《Domain-Specific Programming Systems》 · CS229S Fall 2023 Tianqi Chen 客座（TVM） · CMU 15-849《ML Compilation》×2、《Graph-Level Optimizations》、《RL for Device Placement》
- 必读论文：Chen et al.，TVM，2018；Zheng et al.，Ansor，2020；Jia et al.，TASO，2019；Wang et al.，PET，2021；Tillet et al.，Triton，2019；Lattner et al.，MLIR，2021；Ansel et al.，PyTorch 2，2024；Mirhoseini et al.，Device Placement Optimization with RL，2017
- 对应实验：[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)（实测：torch.compile 的加速等于字节比（RMSNorm 9.86 倍 / 10.0））
- 状态：初稿

## 摘要

前几讲我们手写了分块 GEMM、FlashAttention 和 NKI kernel。问题是，模型有 N 种，硬件有 M 种，算子形状和融合组合更是数不过来，
手写的算子库永远覆盖不全。ML 编译器要解决的就是这个组合爆炸。

编译分两层。图级优化把计算图改写成更便宜的等价图，比如算子融合、常量折叠、CSE（公共子表达式消除）和布局变换；
TASO/PET 这类工作还能自动发现替换规则。算子级优化为每个子图决定循环结构：这沿用了 Halide 式的调度思想，
TVM/Ansor 用学习出来的代价模型在调度空间里搜索，Triton 则让人写块级程序、由编译器管块内的细节。

在两层之前，还需要一个前端把用户程序变成图。XLA 要求静态图；torch.compile 用 Dynamo 在字节码层面捕获图，
用 guard（对输入的假设检查）保证正确，遇到不支持的代码就断图。

图优化里收益最大的是融合，配套实验也印证了这一点：torch.compile 在 RMSNorm 上的加速是 9.86 倍，几乎就等于省下的字节比 10.0。

贯穿全讲的取舍有两条：搜索越充分，性能越好，但编译时间越长；图越大，融合机会越多，但也越容易被动态性打断。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 逐元素算子的强度不到 1 FLOP/B，单独跑永远受访存限制。融合把 k 次 HBM 往返压成 1 次，是图优化里收益最大的一项。
2. 融合分三类：逐元素链（vertical）、逐元素加归约（softmax、LayerNorm），以及 GEMM epilogue（bias、激活、残差在累加器里做完）。
3. TASO 自动生成候选替换，用定理证明器验证，再做代价驱动的回溯搜索。PET 更进一步，允许「部分等价」的变换，再自动补上纠正 kernel。
4. Ansor 的流程是：先生成 sketch（循环结构），再随机注解（tile 大小、并行和向量化），最后用进化搜索加学习的代价模型挑选。它不需要人写模板，而 AutoTVM 需要。
5. torch.compile 由三部分组成：Dynamo（通过 PEP 523 捕获字节码，生成 FX 图和 guard，必要时 graph break）、AOTAutograd（生成联合前反向图）、Inductor（生成 Triton/C++ 代码）。

**必须能默写的公式**

```text
k 个逐元素算子、张量 S 字节：  未融合 HBM ≈ 2kS，融合后 ≈ 2S  → 节省约 k 倍
GEMM + epilogue：  未融合多读写 2·M·N·s 字节/每个后续逐元素算子
调度空间（一条长 2^a 的轴切 l 层）：C(a + l − 1, l − 1) 种有序分解
代价模型目标：  argmin_{schedule ∈ space} cost_model(features(schedule))，只实测 top-k
TASO 搜索：  保留 cost(G') < α · cost(G_best) 的图（α 略大于 1，允许暂时变差）
```

**一句话区分**

- 图级和算子级：前者改「算什么」（有哪些算子、怎么组合），后者改「怎么算」（循环顺序、tile、并行）。
- AutoTVM 和 Ansor：前者由人写调度模板、机器调参数；后者由机器生成模板结构，再调参数。
- Triton 和 CUDA：Triton 的编程单元是 block（一个 tile），块内的线程映射、共享内存和同步由编译器管；CUDA 的编程单元是线程。
- XLA 和 torch.compile：XLA 从静态图出发，整体编译；torch.compile 从 eager Python 出发，按帧捕获，捕获不了就退回 Python 执行。

## 学习目标

- 用 roofline 算出算子融合的收益，并区分三类融合；
- 说清 TASO/PET/equality saturation 如何自动发现并验证图替换；
- 用 Halide 调度原语描述一个分块矩阵乘，并估算调度空间大小；
- 讲清 AutoTVM、Ansor 的搜索流程与代价模型的作用；
- 解释 torch.compile 的 guard、graph break、recompile 和 Inductor 生成 Triton 的流程。

## 先修知识

- [M07](module-07-matmul-hardware-aware.md) 的分块与 epilogue fusion；
- [M09](module-09-accelerators.md) 的 Halide 算法/调度分离。

## 核心概念

### 1. 计算图 IR

编译器要优化，先得有一个能「看全局」的程序表示，这就是计算图 IR（中间表示）。

图里的节点是算子（matmul、conv、add），边是张量，并带有 shape、dtype 和 layout 信息。
常见的图级 IR 有 XLA HLO、TorchFX、ONNX、TVM Relay/Relax，以及 MLIR 的 `stablehlo`/`torch` 方言。
图级 IR 的价值在于全局视角：只有看得到一个张量后面接了哪些算子，才能决定要不要融合、用什么布局。

### 2. 图级优化

有了图，就可以做一系列「换个等价写法，让它更便宜」的改写。常见的有四类：

- 算子融合：把逐元素链合成一个 kernel；归约融合把 softmax 的 max、exp、sum、除法放进同一个 kernel；epilogue fusion 把 bias、激活、残差合进 GEMM。
- 常量折叠：输入全是常量的子图在编译期就求值，例如推理时把 BN 折进卷积权重。同类的还有公共子表达式消除（CSE）和死代码消除。
- 布局变换：Tensor Core 上的卷积偏好 NHWC 布局，编译器在整张图里统一布局，并消去成对出现的 transpose。
- 代数替换：把两个共享输入的 matmul 拼成一个更大的 matmul（例如合并 QKV），让 GEMM 更大、利用率更高。

### 3. TASO 与 PET

上面这些改写规则，传统上靠专家一条条手写，既容易漏，也容易写错。TASO 和 PET 想把这件事自动化。

TASO（SOSP 2019）的流程分五步。先枚举所有不超过若干个算子的小图；再用随机输入的输出「指纹」找出可能等价的图对；
然后用 Z3 对照算子的公理化定义，形式化地验证它们确实等价；接着剪掉冗余的规则；
最后在模型图上用代价模型做回溯搜索，并允许代价暂时变差到 `α` 倍，以便跳出局部最优。
这样，「谁来写替换规则」这件事就从专家手里拿走了。

PET（OSDI 2021）进一步允许只在部分输出位置等价的变换。比如改变 reshape 方式后，边界位置的结果不对，PET 会自动生成纠正 kernel 把这些位置修补好。
它利用了张量程序的多线性性质，只需要在少量位置做验证，就能确认整体正确。

### 4. Equality saturation

有了很多规则之后，又出现一个新问题：先用哪条？传统的改写一次只选一条规则，不同的应用顺序会得到不同的结果，这叫 phase ordering 问题。

e-graph 的做法是同时表示所有等价形式。它反复应用全部规则，直到饱和或者预算用尽，然后再用代价模型（常用 ILP）从 e-graph 里抽出最优的图。
Tensat 就是把这个方法用在张量图上。代价是 e-graph 可能指数级膨胀，所以需要限制迭代次数。

### 5. 算子级：调度原语

图级优化决定了要算哪些算子；接下来要决定每个算子的循环怎么写。调度原语就是描述「怎么写循环」的词汇表。

Halide 的原语包括：`split`（拆分一个循环轴）、`reorder`（交换循环顺序）、`tile`（split 加 reorder 的二维组合）、`vectorize`、`parallel`、`unroll`，
以及 `compute_at`（把生产者放进消费者的某一层循环里计算，相当于融合加局部重计算）。
到了 GPU 上，还要加上 `bind`（把循环轴绑定到 blockIdx/threadIdx）和 `cache_read/cache_write`（用共享内存或寄存器做缓冲）。

### 6. TVM、AutoTVM 与 Ansor

调度原语组合起来，空间大得惊人（后面的推导会算出来）。TVM 这一系列工作的重点，就是怎样在这个空间里自动找到好的调度。

TVM 用 tensor expression（TE）描述算子，用调度描述循环，再降到 LLVM、CUDA 等后端。
AutoTVM 的搜索空间来自人写的调度模板，模板里留出 tile 大小之类的「旋钮」。它用 XGBoost 之类的代价模型预测性能，用模拟退火探索，
只把预测最好的一批送去实测，再用实测结果继续训练代价模型。

Ansor 去掉了模板。它按推导规则自动生成 sketch，例如对 matmul 生成多级 tiling 的结构、对逐元素消费者生成融合；
然后随机填充注解，再用进化搜索（变异 tile 大小、交叉）加代价模型做精选。
它的任务调度器还会按各个子图对端到端时间的贡献来分配调优预算。

### 7. Triton

TVM 走的是「全自动搜索」。Triton 选了一条折中的路：块与块之间怎么切由人决定，块内部的细节交给编译器。

Triton 程序以 block 为单位：一个 program 实例处理一个 tile，`tl.load` 一次加载整个 tile。
编译器负责块内的线程映射、访存合并、共享内存分配与同步、Tensor Core 指令选择，以及软件流水（`num_stages`）。
程序员只负责块级的切分和融合。它最擅长写融合 kernel：一个 matmul 加 epilogue，或者一个 FlashAttention，只要几十行。

### 8. MLIR

前面提到的编译器各自都要实现 IR、pass 和 lowering，重复劳动很多。MLIR 是为了复用这些基础设施而出现的。

MLIR 是一个构建编译器的框架，而不是某个具体的编译器。它的核心概念是方言（dialect）：同一个 IR 里可以混用
`linalg`（结构化张量运算）、`affine`/`scf`（循环）、`vector`、`gpu`、`llvm` 等方言，一级一级地往下 lowering。
好处是每一层优化都能在最合适的抽象层次上做，而且不同项目可以复用方言；Triton、IREE 和 XLA 的部分后端都基于它。

### 9. torch.compile 与 XLA

以上讲的都是「拿到图之后怎么优化」。可 PyTorch 用户写的是 eager 模式的 Python，编译器怎么拿到图？这是前端要解决的问题。

Dynamo 通过 CPython 的帧求值钩子（PEP 523），在函数执行前拿到它的字节码，符号执行一遍，把张量操作记录成 FX 图，
同时为它所做的假设生成 guard（shape、dtype、device、Python 常量）。下次调用时先检查 guard：全部满足就直接运行编译好的代码，否则重新编译（recompile）。

遇到不支持的东西，比如数据依赖的 Python 控制流、`.item()`、部分第三方库调用，Dynamo 就会 graph break：前半段编译，中间回到 Python 执行，后半段再重新捕获。

捕获到图之后，AOTAutograd 生成联合的前反向图，并决定哪些激活存下来、哪些重算；Inductor 把图降成循环级 IR 并做融合，在 GPU 上生成 Triton，在 CPU 上生成 C++/OpenMP。

XLA 走的是另一条路。它从 JAX/TF 的静态 HLO 出发，整体编译、shape 全部静态，所以融合很激进，但 shape 一变就要重新编译。

### 10. 用 RL 做设备放置

编译器的问题不只在单卡上。模型大到要放在多个设备上时，「哪个算子放在哪个设备」也是一个搜索问题。

Mirhoseini et al.（ICML 2017）用一个 seq2seq 策略网络为每个算子输出设备编号，把在真机上实测的运行时间当作奖励，用 REINFORCE 训练。
后续工作（分层 planner、图神经网络策略）降低了搜索代价。
工程上的结论是：对固定且长期运行的大模型值得这样做，但搜索本身要花很多 GPU 小时；今天的主流做法是基于代价模型的自动并行（见 M13）。

## 关键推导

### 推导 1：融合收益

融合到底能快多少？对访存受限的算子，答案就是省下的字节比。设张量大小为 `S` 字节，有 `k` 个逐元素一元算子串联：

```text
未融合：每个算子读 S、写 S → 2kS 字节，k 次 kernel 启动
融合后：读一次 S、写一次 S → 2S 字节，1 次启动
时间比 ≈ k（访存受限时）；FLOPs 不变
```

GEMM epilogue 的道理一样。GEMM 的 `M×N` 输出原本要写回 HBM，后面每个逐元素算子再把它读写一遍；
融合后，累加器在寄存器里直接把这些运算做完，只写一次最终结果。

### 推导 2：调度空间的规模

为什么不能把所有调度都试一遍？算一下调度空间有多大就明白了。

一条长 `2^a` 的轴切成 `l` 层 tile（每层都是 2 的幂），相当于把 `a` 个因子 2 分配到 `l` 个槽位里，有序分解的个数是 `C(a+l−1, l−1)`。
以 1024×1024×1024 的 matmul 为例，i、j 各切 4 层，k 切 2 层：

```text
i: C(13,3) = 286；j: 286；k: C(11,1) = 11
286 × 286 × 11 = 899,756 ≈ 9e5 种 tile 组合，还没算循环顺序、unroll、vectorize
```

如果编译加实测每个候选要 1 s，全部测完大约需要 10 天。所以必须用代价模型先筛选，只实测排在前面的 top-k。

## 深入推导

### Ansor 的搜索循环

把上面核心概念里 Ansor 的几步串起来，就是下面这个循环：

```text
for round in rounds:
    sketches  = generate_sketches(subgraph)            # 规则：多级 tiling、融合消费者、cache_write
    init      = [random_annotate(s) for s in sketches]  # tile 大小、parallel/unroll/vectorize
    pop       = evolve(init, fitness = cost_model)      # 变异 + 交叉，代价模型打分，不实测
    measured  = measure_on_hardware(top_k(pop))         # 只实测少量
    cost_model.update(measured)                         # 在线训练，预测越来越准
task_scheduler: 把下一轮预算给「对端到端延迟贡献最大且仍在进步」的子图
```

这里有一个设计上的巧思：代价模型不需要预测出绝对时间，只要把候选的快慢顺序排对就够了，所以 Ansor 和 AutoTVM 都用排序损失来训练它。

### 代码：TVM 的 tensor expression 与调度

下面用 TVM 写一个分块矩阵乘，前三行是算法，后面全是调度：

```python
import tvm
from tvm import te

N = 1024
A = te.placeholder((N, N), name="A")
B = te.placeholder((N, N), name="B")
k = te.reduce_axis((0, N), name="k")
C = te.compute((N, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")  # 算法

s = te.create_schedule(C.op)                                                      # 调度
i, j = s[C].op.axis
io, jo, ii, ji = s[C].tile(i, j, x_factor=32, y_factor=32)
ko, ki = s[C].split(k, factor=8)
s[C].reorder(io, jo, ko, ii, ki, ji)          # 最内层 ji 连续访问 B 与 C
s[C].vectorize(ji)
s[C].parallel(io)
func = tvm.build(s, [A, B, C], target="llvm")
```

这是 TVM 经典的 TE API；新版本以 TensorIR 为主，但概念是对应的。代码里 tile 的两个 32 和 split 的 8，就是 AutoTVM 模板里的「旋钮」。

### 代码：Triton 的 matmul + bias + ReLU 融合

下面是一个 GEMM 加 epilogue 的 Triton kernel。注意最后三行：bias 和 ReLU 都在累加器还在寄存器里时完成。

```python
import triton, triton.language as tl

@triton.jit
def matmul_bias_relu(A, B, bias, C, M, N, K, sam, sak, sbk, sbn, scm, scn,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(A + rm[:, None] * sam + (k0 + rk)[None, :] * sak)
        b = tl.load(B + (k0 + rk)[:, None] * sbk + rn[None, :] * sbn)
        acc += tl.dot(a, b)                               # 编译器选 Tensor Core 指令
    acc += tl.load(bias + rn)[None, :]                    # epilogue：在寄存器里加 bias
    acc = tl.maximum(acc, 0.0)                            # epilogue：ReLU
    tl.store(C + rm[:, None] * scm + rn[None, :] * scn, acc.to(C.dtype.element_ty))
```

这里省略了边界 mask，并假设 M、N、K 分别整除 BM、BN、BK。

### 代码：观察 torch.compile 的 graph break

graph break 在实际项目里很常见。下面这个例子故意写了一个数据依赖的分支，可以用 `explain` 看到断图：

```python
import torch

def f(x, w):
    y = torch.nn.functional.gelu(x @ w)
    if y.sum() > 0:          # 数据依赖的 Python 分支：Dynamo 无法静态决定 → graph break
        y = y * 2
    return y + 1

x, w = torch.randn(64, 128), torch.randn(128, 128)
print(torch._dynamo.explain(f)(x, w).graph_break_count)   # 期望 1
g = torch.compile(f)                                       # 运行时设 TORCH_LOGS="graph_breaks,recompiles"
g(x, w)
```

修复方法是把 Python 分支改写成张量运算：`y = torch.where(y.sum() > 0, y * 2, y)`。这样整个函数就能进入同一张图。

## 白板估算

参数取 H100：HBM 3.35 TB/s，bf16 989 TFLOP/s（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）。
这道题对比同一串后处理在三种融合程度下的 HBM 流量。

```text
场景：GEMM [8192,4096]×[4096,4096] 之后接 bias → GeLU → 乘 dropout mask → 加残差，bf16
张量 S = 8192·4096·2 B = 67.1 MB

GEMM：2·8192·4096·4096 = 2.75e11 FLOPs → 278 µs（计算受限）

未融合（4 个 kernel，mask 视为另一个 S 大小的张量）：
  bias   读 S 写 S              134 MB
  GeLU   读 S 写 S              134 MB
  mask   读 S、读 mask、写 S     201 MB
  残差   读 S、读残差、写 S      201 MB
  合计 ≈ 671 MB → 671e6 / 3.35e12 ≈ 200 µs，占 GEMM 时间的 72%

融合成一个逐元素 kernel：读 GEMM 输出、mask、残差，写结果 → 4S = 268 MB → 80 µs
完全 epilogue 融合进 GEMM：省掉 GEMM 输出的写和读，只多读 mask 与残差、写结果 → 3S = 201 MB → 60 µs
（mask 改成在 kernel 内用随机数种子生成，可再省 67 MB）
```

这道题最该记住的是 72% 这个比例：几个看起来不起眼的逐元素算子，不融合的话，耗时能占到后面那个大 GEMM 的 72%。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 融合收益 | [15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md) | 融合前后结果一致；记录 HBM 字节数与耗时 |
| graph break | 本讲 `torch._dynamo.explain` 片段 | 改写前 1 次断图，改写后 0 次 |
| roofline 判定 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `roofline_time` | 逐元素算子判定为访存受限 |

## 易错点与反例

- 常见错答：「融合能减少 FLOPs」。融合减少的是 HBM 字节数和 kernel 启动次数；`compute_at` 式的融合甚至会增加重计算。
- 常见错答：「两个 GEMM 也应该融合」。GEMM 本身是计算受限的，硬把两个 GEMM 融合往往放不下 tile、降低并行度；值得这样做的是 attention 这类中间结果有 `T²` 大小的特例。
- 常见错答：「torch.compile 一定加速」。频繁 recompile（每次 shape 都变）或者大量 graph break 时可能反而更慢；小模型的收益常常来自 CUDA Graphs 去掉了启动开销。
- 代价模型只在训练分布内才准。换了硬件或算子类型要重新收集数据，否则会选出很差的调度。
- TASO 的替换在 batch 或 shape 变化后可能不再划算，代价模型需要按实际的 shape 来评估。
- 以为 MLIR 是一个编译器。它是 IR 基础设施，具体优化由各个方言和 pass 提供。

## 实验或练习

这个练习用来验证「融合的加速约等于字节比」。

- 假设：对 `[8192, 4096]` bf16 的 4 步逐元素链，`torch.compile` 融合后的耗时约为 eager 的 1/3 到 1/2。
- 配置：分别跑 eager、`torch.compile(mode="default")` 和手写的 Triton 融合 kernel；用 `torch.profiler` 数 kernel 个数和耗时；用 `TORCH_LOGS="output_code"` 查看 Inductor 生成的 Triton 代码。
- 预期：eager 有 4 个逐元素 kernel，compile 后合成 1 个；耗时比接近字节数之比。具体数值以实测为准。

## 高频问题

- 问：为什么需要 ML 编译器？
  答：手写算子库覆盖不了所有形状、融合组合和新硬件；编译器能自动做融合和调度搜索。
- 问：融合会减少 FLOPs 吗？
  答：不会。它减少的是中间张量的 HBM 往返和 kernel 启动次数；`compute_at` 式融合还可能因为重计算增加 FLOPs。
- 问：TASO 和手写规则有什么区别？
  答：TASO 自动枚举并形式化验证替换规则，再用代价搜索决定用哪些，不依赖专家写规则。
- 问：为什么调度搜索里的代价模型只需要排序正确？
  答：它的作用是从大量候选里挑出 top-k 去实测，只要快慢顺序对就够了，所以用排序损失训练，而不追求预测绝对时间。
- 问：什么是 graph break？
  答：Dynamo 遇到无法捕获的代码时把图切开，中间回到 Python 执行，前后两张图之间的融合机会就丢了。

## 面试追问链

1. 问：编译器的图级优化和算子级优化各管什么？
   答：图级改算子的组合（融合、折叠、布局、代数替换）；算子级决定单个（融合后的）算子怎么写循环、怎么切 tile、怎么映射到并行单元。
2. 问：那就拿融合来说，算一下 4 个逐元素算子融合前后的 HBM 字节数。
   答：未融合约 `2kS = 8S`，融合后是 `2S`；S=67 MB 时是 537 MB 对 134 MB，在 H100 上约 160 µs 对 40 µs。
3. 问：算子级的调度空间有多大？为什么需要代价模型？
   答：一个 1024³ 的 matmul 仅 tile 组合就约有 9e5 种；每个候选实测都要编译加运行，所以必须用学习的代价模型排序，只实测 top-k。
4. 问：AutoTVM 和 Ansor 怎么取舍？
   答：AutoTVM 依赖人写的模板，空间小、收敛快，但上限受模板限制；Ansor 空间大，能找到模板之外的调度，但搜索时间更长。
5. 问：torch.compile 的 guard 检查什么？什么时候会 recompile？
   答：检查输入张量的 shape、dtype、device、stride，以及被读取的 Python 值；任何一项不满足就重新编译。动态 shape 用符号 shape，避免每个长度都编一次。
6. 问：为什么 XLA 在 TPU 上很自然，在 PyTorch 的动态模型上却难用？
   答：XLA 需要静态 shape 和完整的图，TPU 也需要静态调度；动态控制流和变长输入会触发大量重编译，或者需要 padding/bucketing。
7. 问：你负责把一个推理服务从 eager 切到 torch.compile，上线前要验证什么？
   答：数值一致性（在容差内）、graph break 数（用 `explain`）、recompile 次数（按实际请求的 shape 分布压测）、首个请求的编译延迟（预热或缓存）、显存峰值；变长 batch 需要固定几个 bucket。
8. 问：一个新 AI 芯片团队问你，编译栈应该接在 MLIR 上，还是做 Triton 后端？
   答：如果芯片是 SIMT 风格、有类似 Tensor Core 的块级指令，做 Triton 后端可以直接复用大量现成的 kernel；如果是大阵列加 scratchpad 的静态调度架构，用 MLIR 方言表达 tile 和 DMA 更合适。两者并不冲突，Triton 本身也基于 MLIR。

## 自测题

1. 一条长 4096 的轴切成 3 层，有多少种 2 的幂有序分解？
   要点：`4096 = 2^12`，`C(14, 2) = 91`。
2. 解释 `compute_at` 为什么可能增加计算量。
   要点：生产者在消费者的每个 tile 内计算，tile 边界的 halo 区域会被相邻 tile 重复计算。
3. 写出把 BN 折进前面卷积的公式。
   要点：`W' = γ·W/√(σ²+ε)`，`b' = γ·(b − μ)/√(σ²+ε) + β`。
4. 给出一个会导致 recompile 风暴的写法，并修复它。
   要点：每次用不同的序列长度调用；修复方法是 padding 到固定的 bucket，或者开启 `dynamic=True`。
5. 设计题：一个 GEMM 后面接 LayerNorm，能做 epilogue 融合吗？
   要点：LayerNorm 需要整行的统计量，而一个 CTA tile 只覆盖部分列；需要让 BN 覆盖整行（N 较小时可行），或者分两段（GEMM 输出部分和与统计量，再融合归一化）。

## 关联笔记

- [M07 矩阵乘与分块](module-07-matmul-hardware-aware.md)：调度原语描述的就是分块 GEMM。
- [M09 专用硬件与加速器编程](module-09-accelerators.md)：Halide 与 NKI，编译器的目标硬件。
- [M11 深度学习框架与自动微分](module-11-frameworks-autodiff.md)：eager 与图模式，编译器的前端来源。
- [M13 张量并行与自动并行](module-13-tensor-pipeline-parallel.md)：从设备放置到自动并行。
- [M08 FlashAttention](module-08-attention-flashattention.md)：编译器难以自动发现、需要手写的融合。
- [edge-deploy M06 调度搜索](https://gitee.com/liu-xingyan04/edge-deploy/blob/main/notes/module-06-schedule-search.md)：同一套搜索在真机测量上做：噪声与差距同量级，代价模型 Spearman 0.47–0.53

## 本讲小结

ML 编译器的收益大头来自融合：逐元素和归约算子受访存限制，少一次 HBM 往返就省一份时间，加速比约等于字节比。
图级替换（TASO/PET/e-graph）负责自动发现并验证改写；算子级调度（TVM/Ansor）在巨大的空间里靠代价模型搜索；Triton 则把块内细节交给编译器。
torch.compile 用 guard 和 graph break，在动态的 Python 上尽量捕获出图。

下一讲回到框架本身：自动微分与运行时。

## 延伸阅读

- Chen et al., TVM: An Automated End-to-End Optimizing Compiler for Deep Learning, OSDI 2018, https://arxiv.org/abs/1802.04799
- Chen et al., Learning to Optimize Tensor Programs（AutoTVM）, NeurIPS 2018, https://arxiv.org/abs/1805.08166
- Zheng et al., Ansor, OSDI 2020, https://arxiv.org/abs/2006.06762
- Jia et al., TASO, SOSP 2019；Wang et al., PET, OSDI 2021；Yang et al., Tensat, MLSys 2021.
- Tillet et al., Triton, MAPL 2019；Lattner et al., MLIR, CGO 2021, https://arxiv.org/abs/2002.11054
- Ansel et al., PyTorch 2, ASPLOS 2024；Mirhoseini et al., Device Placement Optimization with RL, ICML 2017, https://arxiv.org/abs/1706.04972
