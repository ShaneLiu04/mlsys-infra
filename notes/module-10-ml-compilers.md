# Module 10 — ML 编译器与图优化

- 课程出处：CS149 Fall 2024《Domain-Specific Programming Systems》 · CS229S Fall 2023 Tianqi Chen 客座（TVM） · CMU 15-849《ML Compilation》×2、《Graph-Level Optimizations》、《RL for Device Placement》
- 必读论文：Chen et al.，TVM，2018；Zheng et al.，Ansor，2020；Jia et al.，TASO，2019；Wang et al.，PET，2021；Tillet et al.，Triton，2019；Lattner et al.，MLIR，2021；Ansel et al.，PyTorch 2，2024；Mirhoseini et al.，Device Placement Optimization with RL，2017
- 对应实验：[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)（实测：torch.compile 的加速等于字节比（RMSNorm 9.86 倍 / 10.0））
- 状态：初稿

## 摘要

ML 编译器解决的是「N 个模型 × M 种硬件」的组合爆炸：手写算子库覆盖不了所有形状、所有融合组合、所有新硬件。
编译分两层：图级把计算图改写成更便宜的等价图（算子融合、常量折叠、CSE、布局变换，TASO/PET 自动发现替换规则）；
算子级为每个子图决定循环结构（Halide 式调度，TVM/Ansor 在调度空间里用学习的代价模型搜索，Triton 让人写块级程序、编译器管块内）。
前端负责把用户程序变成图：XLA 要求静态图，torch.compile 用 Dynamo 在字节码层面捕获图、用 guard 保证正确、遇到不支持的代码就断图。
主线取舍：搜索越充分性能越好，但编译时间越长；图越大融合机会越多，但越容易被动态性打断。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 逐元素算子强度不到 1 FLOP/B，单独跑永远访存受限；融合把 k 次 HBM 往返压成 1 次，是图优化里收益最大的一项。
2. 融合分三类：逐元素链（vertical）、逐元素 + 归约（softmax、LayerNorm）、GEMM epilogue（bias/激活/残差在累加器里做完）。
3. TASO 自动生成候选替换并用定理证明器验证，再做代价驱动的回溯搜索；PET 允许「部分等价」变换再自动补纠正 kernel。
4. Ansor：sketch 生成（结构）→ 随机注解（tile 大小、并行/向量化）→ 进化搜索 + 学习的代价模型，无需人写模板（AutoTVM 需要）。
5. torch.compile = Dynamo（PEP 523 捕获字节码 → FX 图 + guard，必要时 graph break）+ AOTAutograd（联合前反向图）+ Inductor（生成 Triton/C++）。

**必须能默写的公式**

```text
k 个逐元素算子、张量 S 字节：  未融合 HBM ≈ 2kS，融合后 ≈ 2S  → 节省约 k 倍
GEMM + epilogue：  未融合多读写 2·M·N·s 字节/每个后续逐元素算子
调度空间（一条长 2^a 的轴切 l 层）：C(a + l − 1, l − 1) 种有序分解
代价模型目标：  argmin_{schedule ∈ space} cost_model(features(schedule))，只实测 top-k
TASO 搜索：  保留 cost(G') < α · cost(G_best) 的图（α 略大于 1，允许暂时变差）
```

**一句话区分**

- 图级 vs 算子级：前者改「算什么」（哪些算子、怎么组合），后者改「怎么算」（循环顺序、tile、并行）。
- AutoTVM vs Ansor：前者人写调度模板、机器调参数；后者机器生成模板结构再调参数。
- Triton vs CUDA：Triton 的编程单元是 block（一个 tile），块内线程映射、共享内存、同步由编译器管；CUDA 的单元是线程。
- XLA vs torch.compile：XLA 从静态图出发、整体编译；torch.compile 从 eager Python 出发、按帧捕获，捕获不了就退回 Python。

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

节点是算子（matmul、conv、add），边是张量，带 shape、dtype、layout。图级 IR 的例子：XLA HLO、TorchFX、ONNX、TVM Relay/Relax、MLIR 的 `stablehlo`/`torch` 方言。图级 IR 的价值是全局视角：能看到一个张量后面接了谁，才能决定融合和布局。

### 2. 图级优化

- 算子融合：逐元素链合成一个 kernel；归约融合把 softmax 的 max、exp、sum、除法放进一个 kernel；epilogue fusion 把 bias/激活/残差合进 GEMM。
- 常量折叠：输入都是常量的子图在编译期求值（例如推理时把 BN 折进卷积权重）；另有公共子表达式消除（CSE）、死代码消除。
- 布局变换：Tensor Core 卷积偏好 NHWC；编译器在图中统一布局并消去成对的 transpose。
- 代数替换：两个共享输入的 matmul 拼成一个更大的 matmul（QKV 合并），提高 GEMM 尺寸与利用率。

### 3. TASO 与 PET

TASO（SOSP 2019）的流程：枚举不超过若干个算子的所有小图 → 用随机输入的输出指纹找候选等价对 → 用 Z3 对照算子的公理化定义验证 → 剪掉冗余规则 → 在模型图上用代价模型做回溯搜索，允许代价暂时变差到 `α` 倍以逃出局部最优。它把「谁来写替换规则」从专家手里拿走。
PET（OSDI 2021）进一步允许只在部分输出位置等价的变换（如改变 reshape 方式后边界位置不对），自动生成纠正 kernel 修补这些位置；它利用张量程序的多线性性，只需在少量位置验证即可确认整体正确。

### 4. Equality saturation

传统改写一次只选一条规则，顺序决定结果（phase ordering 问题）。e-graph 同时表示所有等价形式：反复应用全部规则直到饱和或预算用尽，再用代价模型（常用 ILP）从 e-graph 里抽出最优图。Tensat 把它用在张量图上。代价是 e-graph 可能指数膨胀，需要限制迭代次数。

### 5. 算子级：调度原语

Halide 的原语：`split`（拆轴）、`reorder`（换循环顺序）、`tile`（split + reorder 的二维组合）、`vectorize`、`parallel`、`unroll`、
`compute_at`（把生产者放进消费者的某层循环里算，即融合 + 局部重计算）。GPU 上再加 `bind`（把轴绑到 blockIdx/threadIdx）、`cache_read/cache_write`（共享内存/寄存器缓冲）。

### 6. TVM、AutoTVM 与 Ansor

TVM 用 tensor expression（TE）描述算子，用调度描述循环，再降到 LLVM/CUDA 等后端。AutoTVM 的搜索空间来自人写的调度模板（模板里留出 tile 大小等「旋钮」），用 XGBoost 之类的代价模型预测性能，模拟退火探索，只把预测最好的一批送去实测，实测结果再训练模型。
Ansor 去掉了模板：按推导规则（如对 matmul 生成多级 tiling 结构、对逐元素消费者生成融合）自动生成 sketch，随机填充注解，再用进化搜索（变异 tile 大小、交叉）加代价模型精选；任务调度器按各子图对端到端时间的贡献分配调优预算。

### 7. Triton

Triton 程序以 block 为单位：一个 program 实例处理一个 tile，`tl.load` 一次加载整个 tile。编译器负责块内线程映射、访存合并、
共享内存分配与同步、Tensor Core 指令选择、软件流水（`num_stages`）。程序员负责块级的切分与融合。它的甜点区是融合 kernel：写一个 matmul + epilogue 或 FlashAttention 只要几十行。

### 8. MLIR

MLIR 是构建编译器的框架而非某个编译器。核心是方言（dialect）：同一个 IR 里可以混用 `linalg`（结构化张量运算）、`affine`/`scf`（循环）、`vector`、`gpu`、`llvm` 等方言，逐级 lowering。好处是每一层的优化在最合适的抽象上做，且不同项目可复用方言（Triton、IREE、XLA 的部分后端都基于它）。

### 9. torch.compile 与 XLA

Dynamo 通过 CPython 的帧求值钩子（PEP 523）在执行前拿到字节码，符号执行它，把张量操作记成 FX 图，并为假设生成 guard（shape、dtype、device、Python 常量）。下次调用先查 guard，全部满足直接跑编译好的代码，否则重新编译（recompile）。
遇到不支持的东西（数据依赖的 Python 控制流、`.item()`、部分第三方库调用）就 graph break：前半段编译、中间回到 Python、后半段再捕获。
AOTAutograd 生成联合前反向图并决定存哪些激活、重算哪些；Inductor 把图降成循环级 IR、做融合，GPU 上生成 Triton，CPU 上生成 C++/OpenMP。
XLA 则从 JAX/TF 的静态 HLO 出发，整体编译、shape 静态，融合激进，但 shape 一变就重新编译。

### 10. 用 RL 做设备放置

Mirhoseini et al.（ICML 2017）用 seq2seq 策略网络为每个算子输出设备编号，在真机上实测运行时间作为奖励，用 REINFORCE 训练。
后续工作（分层 planner、图神经网络策略）降低了搜索代价。工程上的结论：对固定且长期运行的大模型值得，搜索本身要很多 GPU 小时；今天的主流做法是基于代价模型的自动并行（见 M13）。

## 关键推导

### 推导 1：融合收益

张量 `S` 字节，`k` 个逐元素一元算子串联：

```text
未融合：每个算子读 S、写 S → 2kS 字节，k 次 kernel 启动
融合后：读一次 S、写一次 S → 2S 字节，1 次启动
时间比 ≈ k（访存受限时）；FLOPs 不变
```

GEMM epilogue：GEMM 输出 `M×N` 原本要写回 HBM，后续每个逐元素算子再读写一遍；融合后累加器在寄存器里直接做完，只写一次最终结果。

### 推导 2：调度空间的规模

一条长 `2^a` 的轴切成 `l` 层 tile（每层是 2 的幂），等价于把 `a` 个 2 分配到 `l` 个槽位，有序分解数为 `C(a+l−1, l−1)`。
matmul 1024×1024×1024：i、j 各切 4 层，k 切 2 层：

```text
i: C(13,3) = 286；j: 286；k: C(11,1) = 11
286 × 286 × 11 = 899,756 ≈ 9e5 种 tile 组合，还没算循环顺序、unroll、vectorize
```

若编译加实测每个候选 1 s，全部测完要约 10 天；这就是必须用代价模型筛选、只实测 top-k 的原因。

## 深入推导

### Ansor 的搜索循环

```text
for round in rounds:
    sketches  = generate_sketches(subgraph)            # 规则：多级 tiling、融合消费者、cache_write
    init      = [random_annotate(s) for s in sketches]  # tile 大小、parallel/unroll/vectorize
    pop       = evolve(init, fitness = cost_model)      # 变异 + 交叉，代价模型打分，不实测
    measured  = measure_on_hardware(top_k(pop))         # 只实测少量
    cost_model.update(measured)                         # 在线训练，预测越来越准
task_scheduler: 把下一轮预算给「对端到端延迟贡献最大且仍在进步」的子图
```

代价模型不需要预测绝对时间，只需要排序正确，所以 Ansor/AutoTVM 用排序损失训练。

### 代码：TVM 的 tensor expression 与调度

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

这是 TVM 经典 TE API（新版本以 TensorIR 为主，概念对应）。tile 的两个 32 和 split 的 8 就是 AutoTVM 模板里的「旋钮」。

### 代码：Triton 的 matmul + bias + ReLU 融合

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

省略了边界 mask；假设 M、N、K 分别整除 BM、BN、BK。

### 代码：观察 torch.compile 的 graph break

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

改写为 `y = torch.where(y.sum() > 0, y * 2, y)` 就变成张量运算，整个函数进入一张图。

## 白板估算

H100：HBM 3.35 TB/s，bf16 989 TFLOP/s（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 融合收益 | [15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md) | 融合前后结果一致；记录 HBM 字节数与耗时 |
| graph break | 本讲 `torch._dynamo.explain` 片段 | 改写前 1 次断图，改写后 0 次 |
| roofline 判定 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `roofline_time` | 逐元素算子判定为访存受限 |

## 易错点与反例

- 常见错答：「融合能减少 FLOPs」。融合减少的是 HBM 字节和 kernel 启动次数；`compute_at` 式融合甚至会增加重计算。
- 常见错答：「两个 GEMM 也应该融合」。GEMM 本身计算受限，融合两个 GEMM 往往放不下 tile、降低并行度；值得做的是 attention 这类中间结果为 `T²` 的特例。
- 常见错答：「torch.compile 一定加速」。频繁 recompile（shape 每次都变）或大量 graph break 时可能更慢；小模型的收益常来自 CUDA Graphs 去掉启动开销。
- 代价模型只在训练分布内准：换硬件、换算子类型要重新收集数据，否则会选出很差的调度。
- TASO 的替换在 batch 或 shape 变化后可能不再划算，代价模型需要按实际 shape 评估。
- 以为 MLIR 是一个编译器：它是 IR 基础设施，具体优化由各方言和 pass 提供。

## 实验或练习

- 假设：对 `[8192, 4096]` bf16 的 4 步逐元素链，`torch.compile` 的融合版本耗时约为 eager 的 1/3 到 1/2。
- 配置：分别跑 eager、`torch.compile(mode="default")`、手写 Triton 融合 kernel；用 `torch.profiler` 数 kernel 数与耗时；`TORCH_LOGS="output_code"` 查看 Inductor 生成的 Triton 代码。
- 预期：eager 有 4 个逐元素 kernel，compile 后合成 1 个；耗时比接近字节数之比。具体数值以实测为准。

## 高频问题

- 问：为什么需要 ML 编译器？
  答：算子库覆盖不了所有形状、融合组合和新硬件；编译器自动做融合和调度搜索。
- 问：算子融合的收益从哪来？
  答：减少中间张量的 HBM 往返和 kernel 启动；对访存受限的逐元素/归约算子收益接近融合算子数。
- 问：TASO 和手写规则的区别？
  答：TASO 自动枚举并形式化验证替换规则，再用代价搜索选用；不依赖专家写规则。
- 问：Ansor 为什么不需要模板？
  答：用推导规则自动生成调度结构（sketch），再随机注解并进化搜索。
- 问：什么是 graph break？
  答：Dynamo 遇到无法捕获的代码时把图切开，中间回到 Python 执行，导致融合机会丢失。

## 面试追问链

1. 问：编译器的图级和算子级优化各管什么？
   答：图级改算子的组合（融合、折叠、布局、代数替换）；算子级决定单个（融合后）算子的循环、tile、并行映射。
2. 问：算一下 4 个逐元素算子融合前后的 HBM 字节数。
   答：未融合约 `2kS = 8S`，融合后 `2S`；S=67 MB 时 537 MB 对 134 MB，H100 上约 160 µs 对 40 µs。
3. 问：调度空间有多大，为什么需要代价模型？
   答：1024³ matmul 仅 tile 组合就约 9e5 种；每个实测要编译加运行，必须用学习的代价模型排序，只实测 top-k。
4. 问：AutoTVM 和 Ansor 的取舍？
   答：AutoTVM 依赖人写模板，空间小、收敛快，但上限受模板限制；Ansor 空间大、能找到模板外的调度，但搜索时间更长。
5. 问：torch.compile 的 guard 检查什么，什么时候会 recompile？
   答：输入张量的 shape、dtype、device、stride 以及被读取的 Python 值；任何一项不满足就重新编译。动态 shape 用符号 shape 避免每个长度都编一次。
6. 问：为什么 XLA 在 TPU 上很自然，在 PyTorch 动态模型上难用？
   答：XLA 需要静态 shape 和完整图，TPU 需要静态调度；动态控制流和变长输入会触发大量重编译或需要 padding/bucketing。
7. 问：你负责把一个推理服务从 eager 切到 torch.compile，上线前要验证什么？
   答：数值一致性（容差内）、graph break 数（`explain`）、recompile 次数（按实际请求 shape 分布压测）、首个请求的编译延迟（预热或缓存）、显存峰值；变长 batch 需固定若干 bucket。
8. 问：新 AI 芯片团队问你，编译栈应该接在 MLIR 上还是做 Triton 后端？
   答：芯片若是 SIMT 风格且有类似 Tensor Core 的块级指令，做 Triton 后端可直接复用大量现有 kernel；若是大阵列 + scratchpad 的静态调度架构，基于 MLIR 方言表达 tile 与 DMA 更合适。两者不冲突，Triton 本身也基于 MLIR。

## 自测题

1. 一条长 4096 的轴切 3 层，有多少种 2 的幂有序分解？
   要点：`4096 = 2^12`，`C(14, 2) = 91`。
2. 解释 `compute_at` 为什么可能增加计算量。
   要点：生产者在消费者的每个 tile 内计算，tile 边界的 halo 区域会被相邻 tile 重复计算。
3. 写出把 BN 折进前面卷积的公式。
   要点：`W' = γ·W/√(σ²+ε)`，`b' = γ·(b − μ)/√(σ²+ε) + β`。
4. 给出一个会导致 recompile 风暴的写法，并修复。
   要点：每次用不同序列长度调用；修复为 padding 到固定 bucket 或开启 `dynamic=True`。
5. 设计题：一个 GEMM 后接 LayerNorm，能否 epilogue 融合？
   要点：LayerNorm 需要整行统计量，CTA tile 只覆盖部分列；需 BN 覆盖整行（N 较小时可行）或两段式（GEMM 输出部分和统计量，再融合归一化）。

## 关联笔记

- [M07 矩阵乘与分块](module-07-matmul-hardware-aware.md)：调度原语描述的就是分块 GEMM。
- [M09 专用硬件与加速器编程](module-09-accelerators.md)：Halide 与 NKI，编译器的目标硬件。
- [M11 深度学习框架与自动微分](module-11-frameworks-autodiff.md)：eager 与图模式，编译器的前端来源。
- [M13 张量并行与自动并行](module-13-tensor-pipeline-parallel.md)：从设备放置到自动并行。
- [M08 FlashAttention](module-08-attention-flashattention.md)：编译器难以自动发现、需要手写的融合。

## 本讲小结

ML 编译器的收益大头来自融合：逐元素和归约算子访存受限，少一次 HBM 往返就省一份时间。图级替换（TASO/PET/e-graph）自动发现并验证改写，
算子级调度（TVM/Ansor）在巨大空间里靠代价模型搜索，Triton 把块内细节交给编译器。torch.compile 用 guard 和 graph break 在动态 Python 上捕获图。
下一讲回到框架本身：自动微分与运行时。

## 延伸阅读

- Chen et al., TVM: An Automated End-to-End Optimizing Compiler for Deep Learning, OSDI 2018, https://arxiv.org/abs/1802.04799
- Chen et al., Learning to Optimize Tensor Programs（AutoTVM）, NeurIPS 2018, https://arxiv.org/abs/1805.08166
- Zheng et al., Ansor, OSDI 2020, https://arxiv.org/abs/2006.06762
- Jia et al., TASO, SOSP 2019；Wang et al., PET, OSDI 2021；Yang et al., Tensat, MLSys 2021.
- Tillet et al., Triton, MAPL 2019；Lattner et al., MLIR, CGO 2021, https://arxiv.org/abs/2002.11054
- Ansel et al., PyTorch 2, ASPLOS 2024；Mirhoseini et al., Device Placement Optimization with RL, ICML 2017, https://arxiv.org/abs/1706.04972
