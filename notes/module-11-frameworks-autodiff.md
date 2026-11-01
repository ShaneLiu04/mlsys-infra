# Module 11 — 深度学习框架与自动微分

- 课程出处：CS229S Fall 2023《Analyzing the Performance of Transformers》(10-09，FLOPs 与反传) · CMU 15-849《Intro to DNNs》《Systems for Deep Learning》《Systems for Classical ML》
- 必读论文：Baydin et al.，Automatic Differentiation in Machine Learning: a Survey，2018；Abadi et al.，TensorFlow，2016；Paszke et al.，PyTorch，2019；Jeong et al.，JANUS，2019；Li et al.，Scaling Distributed ML with the Parameter Server，2014；Chen & Guestrin，XGBoost，2016
- 对应实验：[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md)
- 实验报告：[报告 PDF](../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)
- 状态：初稿

## 摘要

训练框架做三件事：把模型表示成计算图，自动求梯度，把成千上万个小算子高效地发到设备上。
自动微分有前向和反向两种模式，代价分别正比于输入维数和输出维数；损失是标量，所以反向模式一次就得到全部参数梯度，
代价约为前向的 2 倍，但必须保存前向的中间激活，这是训练显存的大头。框架设计在静态图（TensorFlow 1.x，可整体优化）和
eager（PyTorch，易调试）之间取舍，JANUS 与 torch.compile 试图两者兼得。运行时靠 dispatcher、异步 CUDA stream 和 caching allocator 掩盖 Python 开销。
分布式的早期形态是 parameter server，用一致性模型在吞吐与收敛之间折中；经典 ML 系统（XGBoost）和推荐系统的 embedding 表则展示了另一类以访存为主的负载。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. `f: Rⁿ → Rᵐ`：前向模式一次得 Jacobian 的一列（JVP），代价 ∝ n；反向模式一次得一行（VJP），代价 ∝ m。标量损失 m=1，选反向。
2. 线性层反向要算 `dX` 和 `dW` 两个 GEMM，FLOPs 是前向的 2 倍；训练每 token ≈ `6N` FLOPs（前向 2N + 反向 4N）。
3. 反向需要前向的中间结果，激活显存 ∝ 层数 × batch × 序列长度；GPT 式层每层约 `sbh(34 + 5as/h)` 字节（bf16，无重计算）。
4. PyTorch 的算子调用是异步的：Python 只是把 kernel 排进 CUDA stream；`.item()`、`.cpu()`、打印张量会强制同步。
5. Parameter server 的一致性：BSP 每步全同步、ASP 完全异步、SSP 允许最快与最慢 worker 相差至多 s 步。

**必须能默写的公式**

```text
JVP：  J·v，一次前向传播，n 维输入要 n 次得到完整 J
VJP：  uᵀ·J，一次反向传播，m 维输出要 m 次
线性层 Y = X·W，X[B,I]，W[I,O]：
  前向 2BIO；dX = dY·Wᵀ 2BIO；dW = Xᵀ·dY 2BIO   → 反向/前向 = 2
训练 FLOPs ≈ 6·N·D（N 参数，D token）
激活（GPT 式，每层）≈ s·b·h·(34 + 5·a·s/h) 字节；用 FlashAttention 后 ≈ 34·s·b·h
XGBoost 分裂增益：½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ)] − γ
SSP：worker 的时钟 c 满足 c − c_min ≤ s
```

**一句话区分**

- 静态图 vs eager：前者先建图再执行，能全局优化但难调试；后者逐算子执行，灵活但每个算子都付 Python 和启动开销。
- 前向 AD vs 数值差分：前向 AD 精确到机器精度，差分有截断误差且每个方向要两次求值。
- parameter server vs all-reduce：PS 是中心化存储、适合稀疏和异步；all-reduce 是对称集合通信、适合稠密同步。

## 学习目标

- 推导前向/反向模式的代价，解释为什么深度学习用反向模式；
- 推出反向 FLOPs ≈ 2 倍前向，以及训练 `6ND`；
- 估算一个 Transformer 的激活显存；
- 说清框架运行时：dispatcher、异步执行、同步点、caching allocator；
- 比较 BSP/ASP/SSP；说清 XGBoost 的 histogram、稀疏感知和 cache 感知设计；
- 估算 embedding 表的显存并说出其并行方式。

## 先修知识

- 链式法则与矩阵求导；
- [M06](module-06-data-parallel-primitives.md) 的 reduce 与 Spark，作为 parameter server 的对照。

## 核心概念

### 1. 计算图与 tape

前向时框架记录每个算子的输入、输出和反向函数，构成一张 DAG（PyTorch 中是 `grad_fn` 链）。反向从损失出发按逆拓扑序遍历，
每个节点把上游梯度乘以局部 Jacobian（VJP）传给输入。tape 里保存的张量就是「为反向保存的激活」。

### 2. 前向模式与反向模式

前向模式把切向量 `v` 和值一起向前传，每个算子同时算 `f(x)` 和 `J·v`；要得到 n 个参数的梯度要跑 n 次。
反向模式先跑一遍前向存下中间结果，再从输出往回传余切向量 `u`。7B 参数的标量损失：前向模式要约 7e9 次传播，反向模式 1 次。
前向模式的用处：输入维数少、输出维数多（如对少数超参数求导），或 Hessian-向量积（forward-over-reverse）。

### 3. 静态图与 eager

TensorFlow 1.x 先构造图再 `session.run`，图可以整体做融合、内存规划、设备放置，但 Python 控制流要改写成图算子（`tf.cond`、`tf.while_loop`），调试困难。
PyTorch eager 的每行 Python 立刻执行，控制流就是 Python 控制流。代价：每个算子要经 Python 解释器、dispatcher、kernel 启动，图级优化机会丢失。
JANUS（NSDI 2019）在 eager 程序上投机地生成静态图：根据运行时 profile 做假设（类型、分支方向、shape），把假设编译成图里的断言，
断言失败就回退到 eager。这正是 torch.compile 里 guard 的前身思路。

### 4. 框架运行时

一次 `torch.add(a, b)`：Python 绑定 → dispatcher 按 dispatch key 依次经过 Autograd（记录 `grad_fn`）、Autocast（混合精度转换）、
最后到后端 kernel（CUDA/CPU）。CUDA kernel 被放进当前 stream 后立即返回，CPU 继续往下跑，GPU 在后台执行。
只要 GPU 队列不空，Python 开销被掩盖；kernel 很小（几微秒）时 CPU 发射跟不上，GPU 空等——这时用 CUDA Graphs 或编译融合。
同步点：`.item()`、`.cpu()`、打印、`torch.cuda.synchronize()`、某些需要知道输出大小的算子（`nonzero`）。

### 5. Caching allocator

`cudaMalloc/cudaFree` 慢且会隐式同步。PyTorch 自己维护显存池：释放的块不还给驱动，按大小分级缓存，按 stream 归属复用。
副作用：`nvidia-smi` 看到的占用 ≥ 实际张量占用；不同大小反复分配会产生碎片，峰值显存比张量总和高。
跨 stream 使用张量要 `record_stream`，否则块可能被提前复用。

### 6. Parameter server

Li et al.（OSDI 2014）：服务器节点按 key 区间分片存参数，worker 计算梯度后 `push`，再 `pull` 最新参数。适合稀疏模型：
每个 worker 只拉自己 batch 里出现的 key。一致性模型：

| 模型 | 规则 | 吞吐 | 收敛 |
| --- | --- | --- | --- |
| BSP | 每步所有 worker 同步 | 受最慢 worker 拖累 | 与单机 SGD 等价 |
| ASP | 无同步 | 最高 | 梯度陈旧无界，可能发散 |
| SSP | 最快与最慢相差 ≤ s 步 | 居中 | 陈旧有界，有收敛保证 |

### 7. 经典 ML 系统：XGBoost

梯度提升树每轮用损失的一阶 `g` 和二阶 `h` 统计量找分裂点。系统设计要点：

- 近似分裂：把特征值分桶（weighted quantile sketch，以 `h` 为权重；或直接 histogram），只在桶边界评估增益；
- 稀疏感知：缺失值与零值不展开，为每个分裂学一个默认方向，复杂度 ∝ 非缺失项数；
- column block：数据按列以 CSC 格式预排序存储，各轮复用，列间可并行；
- cache 感知：按排序顺序访问时 `g/h` 是随机访存，用预取缓冲区把它们搬成连续；
- out-of-core：块压缩与跨磁盘分片，用 I/O 带宽换内存。

### 8. 推荐系统的 embedding 表

推荐模型（如 DLRM）的稀疏特征查 embedding 表，表可达数亿到数十亿行。查表是 gather，强度极低；表太大只能按行或按表切到多卡（模型并行），
MLP 部分用数据并行，两者之间用 all-to-all 交换查到的向量。热点 ID 访问极度倾斜，常用热行缓存。

## 关键推导

### 推导 1：反向 FLOPs 是前向的 2 倍

```text
前向：Y[B,O] = X[B,I]·W[I,O]                2·B·I·O
反向：dX[B,I] = dY[B,O]·Wᵀ[O,I]             2·B·I·O
      dW[I,O] = Xᵀ[I,B]·dY[B,O]             2·B·I·O
反向 = 4BIO = 2 × 前向
一个 token 经过 N 个参数：前向 2N、反向 4N → 6N；D 个 token → 6ND
```

边界：第一层不需要 `dX`；逐元素算子反向与前向同量级但可忽略；attention 的 `T²` 项另算（见 M08）。开全量重计算时硬件 FLOPs 变成 8N。

### 推导 2：激活显存（GPT 式层，bf16，无张量并行）

Korthikanti et al.（2022）逐项计数：attention 块约 `11sbh + 5as²b`，MLP 块约 `19sbh`，两个 LayerNorm `4sbh`。

```text
每层 = sbh·(34 + 5as/h) 字节
其中 5as²b 是 softmax 输入/输出与 dropout mask（∝ s²）
FlashAttention 不存 s×s 张量 → 每层 ≈ 34·sbh
```

## 深入推导

### 代码：50 行的反向模式自动微分

```python
import math

class Var:
    def __init__(self, val, parents=()):
        self.val, self.parents, self.grad = val, parents, 0.0   # parents: [(Var, 局部导数)]

    def __add__(self, o): o = o if isinstance(o, Var) else Var(o); return Var(self.val + o.val, [(self, 1.0), (o, 1.0)])
    def __mul__(self, o): o = o if isinstance(o, Var) else Var(o); return Var(self.val * o.val, [(self, o.val), (o, self.val)])
    def tanh(self): t = math.tanh(self.val); return Var(t, [(self, 1 - t * t)])

    def backward(self):
        order, seen = [], set()
        def topo(v):
            if id(v) in seen: return
            seen.add(id(v)); [topo(p) for p, _ in v.parents]; order.append(v)
        topo(self)
        self.grad = 1.0
        for v in reversed(order):                     # 逆拓扑序：一次遍历得到所有输入的梯度
            for p, local in v.parents:
                p.grad += v.grad * local              # VJP：上游梯度 × 局部导数

w, x, b = Var(0.5), Var(2.0), Var(-0.3)
y = (w * x + b).tanh()
y.backward()
assert abs(w.grad - (1 - math.tanh(0.7) ** 2) * 2.0) < 1e-12
```

`parents` 里保存的 `o.val`、`self.val`、`t` 就是「为反向保存的激活」。真实框架里它们是张量，反向结束前不能释放。

### 代码：XGBoost 式 histogram 找分裂

```python
import numpy as np

def best_split(bins, g, h, n_bins=256, lam=1.0, gamma=0.0):
    """bins: [n] 某特征的桶编号；g/h: 一阶/二阶梯度。返回 (增益, 分裂桶)。"""
    G = np.bincount(bins, weights=g, minlength=n_bins)     # 一次扫描建直方图，O(n)
    H = np.bincount(bins, weights=h, minlength=n_bins)
    GL, HL = np.cumsum(G)[:-1], np.cumsum(H)[:-1]          # 只在桶边界评估，O(n_bins)
    GT, HT = G.sum(), H.sum()
    GR, HR = GT - GL, HT - HL
    gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - GT**2 / (HT + lam)) - gamma
    k = int(np.argmax(gain))
    return float(gain[k]), k
```

建直方图是 `O(n)` 的顺序扫描，找分裂只有 `O(桶数)`；兄弟节点的直方图可以用「父 − 另一个孩子」相减得到，省一半扫描。

### 代码：SSP 的等待条件

```python
def ssp_can_proceed(my_clock: int, all_clocks: list[int], staleness: int) -> bool:
    """s=0 退化为 BSP；s=∞ 退化为 ASP。"""
    return my_clock - min(all_clocks) <= staleness
```

## 白板估算

H100 bf16 989 TFLOP/s、80 GB；IB HDR 25 GB/s（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）。

```text
1) Llama-2-7B（6.74e9 参数）训练吞吐，单卡 H100，MFU 40%
   每 token 6·6.74e9 = 4.04e10 FLOPs
   0.4·989e12 / 4.04e10 ≈ 9.8e3 token/s/卡

2) 前向 vs 反向模式
   前向模式求全梯度：6.74e9 次传播；反向模式：1 次前向 + 1 次反向 ≈ 3 倍前向 FLOPs

3) 激活显存（按 GPT 式层近似，h=4096, a=32, s=4096, b=1, 32 层）
   sbh = 4096·1·4096 = 1.68e7
   每层 1.68e7·(34 + 5·32·4096/4096) = 1.68e7·194 ≈ 3.26e9 B
   32 层 ≈ 104 GB → 单卡放不下
   FlashAttention：1.68e7·34 ≈ 5.7e8 B/层，32 层 ≈ 18.3 GB

4) embedding 表：10 亿行 × 128 维 × fp32
   1e9·128·4 = 512 GB → 至少 7 张 80 GB 卡（512/80 = 6.4）
   Adam 再加 2 份 fp32 状态 → 1.5 TB；所以推荐系统常用行级 Adagrad 等小状态优化器

5) Parameter server 带宽：1e9 个 fp32 稠密参数（4 GB），64 个 worker，BSP
   每步每 worker push 4 GB + pull 4 GB → 服务器侧总流量 64·8 GB = 512 GB
   16 台服务器、每台 25 GB/s：512e9 / (16·25e9) ≈ 1.28 s/步
   同样的同步若用 ring all-reduce，每卡收发 2·(63/64)·4 GB ≈ 7.9 GB，没有中心热点
```

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| 反向模式 AD | 本讲 `Var` 片段；[15-849 编译与压缩](../solutions/cmu15849/compiler-compression/README.md) | 梯度与有限差分在 1e-6 内一致 |
| 6ND 与 MFU | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `train_flops`、`mfu` | 与手算一致 |
| histogram 分裂 | 本讲 `best_split` | 与暴力枚举所有阈值的最优增益一致 |

## 易错点与反例

- 常见错答：「反向和前向 FLOPs 一样」。线性层要算 `dX` 和 `dW` 两个 GEMM，是 2 倍。
- 常见错答：「反向模式总是更好」。输出维数大于输入维数时（例如求网络输出对少量输入的 Jacobian）前向模式更便宜。
- 常见错答：「`torch.cuda.empty_cache()` 能降低显存峰值」。它只把缓存块还给驱动，活着的张量一字节不少。
- 用 `time.time()` 测 GPU 算子不加 `synchronize`：测到的是发射时间，不是执行时间。
- 训练循环里每步 `loss.item()` 打日志：每步一个同步点，CPU 发射无法提前，小模型上明显变慢。
- ASP 在稠密大模型上并不一定更快到达目标精度：陈旧梯度降低统计效率，需要更多步。
- 把 XGBoost 的 histogram 当成精确分裂：它只在桶边界评估，是近似算法。

## 实验或练习

- 假设：小 MLP（hidden=256）在 GPU 上 eager 训练受 CPU 发射限制，CUDA Graphs 或 `torch.compile(mode="reduce-overhead")` 能显著提速；hidden=8192 时差别消失。
- 配置：batch=64，hidden ∈ {256, 1024, 8192}；对比 eager、compile、compile+CUDA Graphs；用 profiler 看 GPU 空闲比例。
- 预期：小 hidden 时 GPU 时间线有大量空隙；大 hidden 时 GEMM 主导，三者接近。具体数值以实测为准。

## 高频问题

- 问：为什么深度学习用反向模式 AD？
  答：损失是标量，反向模式一次传播得到所有参数梯度，代价约 2 倍前向；前向模式要按参数个数重复。
- 问：训练 FLOPs 为什么是 6ND？
  答：前向每参数每 token 2 FLOPs，反向算 `dX` 和 `dW` 共 4 FLOPs。
- 问：激活显存由什么决定？
  答：层数 × batch × 序列 × 宽度，再加 attention 的 `s²` 项；重计算和 FlashAttention 分别砍掉一部分。
- 问：PyTorch 为什么 eager 也能跑得快？
  答：kernel 异步发射到 CUDA stream，只要单个 kernel 够大，Python 开销被 GPU 执行时间掩盖。
- 问：BSP、ASP、SSP 区别？
  答：全同步、完全异步、有界陈旧（相差不超过 s 步）。

## 面试追问链

1. 问：前向模式和反向模式 AD 的代价分别是多少？
   答：`f: Rⁿ→Rᵐ`，完整 Jacobian 前向要 n 次 JVP，反向要 m 次 VJP；标量损失 m=1，反向一次搞定，代价约为前向的 2–3 倍。
2. 问：推导线性层反向是前向的 2 倍，进而推出 6ND。
   答：`dX = dY·Wᵀ`、`dW = Xᵀ·dY` 各 `2BIO`，与前向 `2BIO` 合计 `6BIO`；每 token 每参数 6 FLOPs。
3. 问：反向模式的代价是什么，怎么缓解？
   答：要存前向激活，显存 ∝ 层数 × token 数；用重计算（多约 1 次前向换显存）、FlashAttention（去掉 `s²` 项）、激活 offload。
4. 问：为什么小模型在 GPU 上 eager 训练跑不满？
   答：每个 kernel 只跑几微秒，CPU 侧 Python + dispatcher + 启动开销同量级，GPU 等待发射；用 CUDA Graphs 录制整步或编译融合。
5. 问：静态图框架的优势在 PyTorch 里如何拿回来？
   答：torch.compile 在字节码层捕获图，用 guard 保证假设成立，失败就重编译或 graph break——和 JANUS 的「投机建图 + 断言 + 回退」同一思路。
6. 问：parameter server 的 SSP 为什么比 ASP 收敛更可靠？
   答：陈旧度有上界 s，梯度用到的参数最多落后 s 步，可以证明收敛并控制误差；ASP 的陈旧度无界，慢节点的梯度可能严重过期。
7. 问：你要为一个 10 亿 ID 的推荐模型设计训练系统，embedding 和 MLP 分别怎么并行？
   答：embedding 表 512 GB 级，按行或按表切到多卡/多机（模型并行或 PS），MLP 小，用数据并行；前向用 all-to-all 把查到的向量送到对应样本所在的卡，反向反方向送梯度；热点 ID 做缓存，优化器选行级小状态。
8. 问：线上 GBDT 训练数据涨到单机内存放不下，XGBoost 的哪些设计可以用？
   答：histogram 近似把每个特征压成少量桶编号（每值 1 字节），out-of-core 块压缩加分片，从磁盘流式读入；分布式时每台机器建局部直方图再 all-reduce 合并直方图，通信量与样本数无关。

## 自测题

1. 一个 4 层 MLP 的前向 FLOPs 是 F，训练一步（前向+反向）多少？开全量重计算呢？
   要点：约 3F；重计算 4F。
2. 证明一层 `y = tanh(wx + b)` 的 `∂y/∂w = (1 − y²)·x`，并用本讲 `Var` 验证。
   要点：链式法则；代码中的 assert。
3. 估算 GPT 式 24 层、h=2048、a=16、s=2048、b=8 的激活显存（不用 FlashAttention）。
   要点：sbh = 3.36e7；`34 + 5·16·2048/2048 = 114`；每层 3.83e9 B；24 层约 92 GB。
4. SSP 中 s=0 和 s=∞ 分别退化为什么？
   要点：BSP 与 ASP。
5. 设计题：为什么 XGBoost 的 histogram 方法适合分布式？
   要点：直方图大小只与桶数和特征数有关，各机器局部统计后求和即可，通信量与样本数无关。

## 关联笔记

- [M10 ML 编译器](module-10-ml-compilers.md)：从 eager 程序捕获图并编译。
- [M12 数据并行与 ZeRO](module-12-data-parallel-zero.md)：all-reduce 取代 parameter server 成为稠密训练的主流。
- [M14 显存优化](module-14-memory-optimization.md)：重计算与混合精度，处理本讲的激活显存。
- [M06 数据并行原语](module-06-data-parallel-primitives.md)：reduce 与 Spark 的对照。

## 本讲小结

反向模式 AD 让标量损失的梯度只花约 2 倍前向，训练因此是 `6ND` FLOPs，代价是保存激活。框架在静态图与 eager 之间权衡，
运行时用异步 stream 和 caching allocator 掩盖开销。parameter server 用一致性模型交换吞吐与收敛，XGBoost 和 embedding 表展示了访存主导的 ML 负载。
下一讲进入多卡：数据并行与 all-reduce。

## 延伸阅读

- Baydin et al., Automatic Differentiation in Machine Learning: a Survey, JMLR 2018, https://arxiv.org/abs/1502.05767
- Paszke et al., PyTorch: An Imperative Style, High-Performance Deep Learning Library, NeurIPS 2019, https://arxiv.org/abs/1912.01703
- Jeong et al., JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs, NSDI 2019.
- Li et al., Scaling Distributed Machine Learning with the Parameter Server, OSDI 2014；Ho et al., More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server, NIPS 2013.
- Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016, https://arxiv.org/abs/1603.02754
- Korthikanti et al., Reducing Activation Recomputation in Large Transformer Models, 2022, https://arxiv.org/abs/2205.05198
