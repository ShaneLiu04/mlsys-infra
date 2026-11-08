# Module 08 — Attention 与 FlashAttention

- 课程出处：CS149 Fall 2024《Efficiently Evaluating DNNs》（融合 attention） · CS229S Fall 2023《Hardware Aware Algorithm Design》(10-11) · CMU 15-849《Transformers and Language Models》
- 必读论文：Dao et al.，FlashAttention，2022；Dao，FlashAttention-2，2023；Shah et al.，FlashAttention-3，2024；Milakov & Gimelshein，Online normalizer calculation for softmax，2018；Ainslie et al.，GQA，2023
- 对应实验：[CS229S P1 Transformer kernels](../solutions/cs229s/project1-transformer-kernels/README.md)
- 实验报告：[报告 PDF](../reports/cs229s-project1/cs229s_project1_report.pdf)（实测：Triton FlashAttention 前向 224 TFLOP/s，与 FA2 持平，比朴素快 7.15 倍）
- 状态：初稿

## 摘要

上一讲说 GEMM 只要分好块就能计算受限。attention 里有两个 GEMM，按理也应该很快，
可实际上朴素实现一到长序列就又慢又爆显存。这一讲解释为什么，以及 FlashAttention 是怎么解决的。

问题出在两个 GEMM 中间。标准 attention 每头每层前向要做 `4T²d` FLOPs，同时还要物化一个 `T×T` 的分数矩阵。
序列一长，瓶颈就不在 FLOPs 上，而在于把这个 `T²` 大小的中间结果写进 HBM、再读出来。

FlashAttention 没有做任何近似，它只是重排了计算顺序。它用 online softmax（一种可以边读边更新的 softmax 写法）
把 softmax 拆成能分块累加的形式，于是分数矩阵 `S` 和概率矩阵 `P` 只存在于 SRAM 里，
HBM 访问量从 `Θ(Td + T²)` 降到 `Θ(T²d²/M)`（`M` 是 SRAM 能放的元素数）。

但别把这个公式读成「访存不再是二次」。d=64–128 时 `d²/M` 只是个常数，访存在 T 上**仍是二次**：
在[实验](../solutions/cs229s/project1-transformer-kernels/README.md)的模型里，它只比「S、P 各读写一次」的朴素模型少 2 倍（因果时 4 倍）。
实测比朴素实现快 7.15 倍，收益主要来自另外三处：显存降到 `O(T)`；省掉了 eager 实现里 scale、mask、softmax、dropout 各自一遍的 `T²` 读写；以及 K/V 的重读大多命中 L2。

反向传播时它不存 `P`，只存每行的 logsumexp，需要时重算。v2 调整了循环顺序和 warp 分工，v3 用上了 Hopper 的异步硬件。
整讲的取舍只有一条：多做一遍前向的 FLOPs（重计算），换掉 `T²` 的显存和带宽。在现代 GPU 上，这笔交易几乎总是划算的。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 每头每层前向 FLOPs 是 `4T²d`（`QKᵀ` 和 `PV` 各 `2T²d`），整层是 `4T²·d_model`。反向约为前向的 2.5 倍，多出来的部分是重算。
2. 朴素实现的 HBM 流量被 `T²` 项主导：写 S、读 S、写 P、读 P。T=8192 时强度约 60 FLOP/B，所以访存受限。
3. Online softmax 维护两个量：行最大值 `m` 和归一化和 `l`。新块到来时，用 `exp(m_old − m_new)` 去缩放旧的 `l` 和 `O`。
4. FlashAttention 是精确算法。HBM 访问为 `Θ(T²d²/M)`，在 T 上仍是二次；显存从 `O(T²)` 降到 `O(T)`；反向只存 logsumexp。
5. GQA/MQA 不减少 attention FLOPs。它减少的是 KV cache 的字节数和 decode 时读 KV 的量（Llama-3-8B 比 MHA 少 4 倍）。

**必须能默写的公式**

```text
S = Q·Kᵀ/√d,  P = softmax(S),  O = P·V           Q,K,V,O ∈ R^{T×d}
FLOPs_fwd/头/层 = 2T²d + 2T²d = 4T²d；因果且跳块时约减半
Online softmax（处理块 j）：
  m_new = max(m, rowmax(S_j))
  l_new = e^{m−m_new}·l + rowsum(e^{S_j−m_new})
  O_new = e^{m−m_new}·O + e^{S_j−m_new}·V_j           （FA2：最后再除以 l）
  LSE   = m + log l                                    （反向只存这一项）
HBM 访问：naive Θ(Td + T²)，Flash Θ(T²d²/M)，M = SRAM 元素数
反向：dV = Pᵀ dO,  dP = dO Vᵀ,  D = rowsum(dO∘O),  dS = P∘(dP − D),  dQ = dS K,  dK = dSᵀ Q
KV/token = 2 · L · n_kv · d_h · s 字节
```

**一句话区分**

- FA1 和 FA2：FA1 外层循环遍历 K/V 块，O 要反复读写 HBM；FA2 外层循环遍历 Q 块，O 一直留在寄存器里，并且可以沿序列长度并行。
- FlashAttention 和稀疏/线性 attention：前者是精确的，只改 IO 顺序；后者改了数学本身，换来低于 `T²` 的复杂度，也带来精度风险。
- MQA 和 GQA：MQA 让所有头共享 1 组 K/V；GQA 让每组头共享 1 组，是质量和 KV 字节数之间的折中。

## 学习目标

- 推出 attention 的 FLOPs、中间张量大小，以及朴素实现的 HBM 流量；
- 从 softmax 定义推导 online softmax 的递推，并证明结果与一次性 softmax 相等；
- 写出 FlashAttention 前向的分块循环，说清 v1 与 v2 的循环顺序差别；
- 推导反向公式，解释为什么只存 logsumexp 就够；
- 用 H100 参数估算 T=8192 时 naive 与 Flash 的瓶颈。

## 先修知识

- [M07](module-07-matmul-hardware-aware.md) 的分块与强度；
- [M05](module-05-gpu-architecture-cuda.md) 的共享内存、warp、寄存器容量。

## 核心概念

### 1. 标准 attention 的 shape 与代价

要优化 attention，先得知道它的代价在哪里：是算得多，还是存得多。

单头时 `Q,K,V ∈ R^{T×d}`。`S = QKᵀ` 是 `T×T` 的矩阵，花 `2T²d` FLOPs；`O = PV` 又是 `2T²d`；softmax 本身只有 `O(T²)`。
多头拼起来，每层前向是 `4T²·d_model`。

显存是另一回事：`S` 和 `P` 都是 `T²` 大小，和 d 无关，所以 T 翻倍，显存翻四倍。
而从 FLOPs 看，Llama-3-8B 在 T=8192 时 attention 只占单层前向的约 23%（非因果）。
所以在中等序列长度下，attention 的麻烦主要是 IO 和显存，而不是 FLOPs。

### 2. 为什么朴素实现访存受限

PyTorch 的朴素实现分成三个 kernel：matmul 写出 S，softmax 读 S 再写出 P，最后一个 matmul 读 P。
每个 kernel 都要把 `T²` 大小的张量在 HBM 上过一遍。

问题集中在中间的 softmax。它对每个元素只做 `O(1)` 次运算，强度远低于 1 FLOP/B。
所以即使两个 matmul 都用 cuBLAS 跑到很快，中间这一步也会把整体拖进访存受限的区域。

### 3. Online softmax

要把 S、P 留在 SRAM 里，就得分块计算 attention。可是 softmax 需要整行的最大值和指数和，看上去必须先看完整行才能算。
online softmax 打破了这个限制，它依赖下面这个恒等式：

```text
e^{x − m_new} = e^{x − m_old} · e^{m_old − m_new}
```

也就是说，用旧最大值算出的部分和，只要乘一个标量，就能修正成新最大值下的结果。
`O` 是 `P` 的线性函数，所以同一行的 `O` 也乘同一个因子就行。递推结束后，`O/l` 就是精确的 softmax 结果。

### 4. FlashAttention v1

有了 online softmax，就可以设计分块循环了。把 Q 切成每块 `Br` 行，把 K/V 切成每块 `Bc` 行。

v1 的外层循环遍历 K/V 块 j，把 `K_j, V_j` 装进 SRAM；内层循环遍历所有 Q 块 i，从 HBM 读出 `Q_i, O_i, m_i, l_i`，更新后再写回去。
这样 S、P 的块从不离开 SRAM。代价是 O、m、l 要被反复读写 `T/Bc` 次，而且每一步都要重新归一化 O。

### 5. FlashAttention v2 的三个改动

v1 已经省掉了 `T²` 的 HBM 往返，但离 GEMM 的速度还有差距。v2 针对剩下的开销做了三处改动：

1. 把循环顺序反过来。外层遍历 Q 块，每个线程块负责一个 Q 块，内层流式读取 K/V。这样 O 的累加器一直留在寄存器里，最后只写一次。
2. 减少非 matmul 的 FLOPs。不在每一步做 `diag(l)⁻¹` 归一化，只在末尾除一次；也只存 LSE，而不是 m 和 l 两个量。
   这很重要，因为 A100 上非 matmul 运算的吞吐比 Tensor Core 低约 16 倍（19.5 vs 312），省掉的每一个指数和除法都值钱。
3. 调整并行方式和 warp 分工。除了 batch×head，还沿序列长度（Q 块）并行，所以长序列、小 batch 时也能填满 108/132 个 SM。
   在线程块内部，把 Q 切给 4 个 warp，每个 warp 都看到全部 K/V，不需要通过共享内存交换部分和。v1 则是把 K/V 切给 warp，最后需要 warp 之间做归约。

### 6. 因果 mask 与块跳过

因果 attention 里，每个位置只能看到它之前的 token，所以 Q 块 i 只需要 K 块 `j ≤ i`。
完全落在对角线上方的块可以直接跳过，对角线上的块需要加 mask，对角线下方的块不需要 mask。

`n` 个块时，实际只算 `n(n+1)/2` 块，约为一半。但要省下这些 FLOPs，前提是按块跳过；逐元素填 `−∞` 再算全矩阵，结果是对的，却一点也不省。

### 7. FlashAttention-3（Hopper）

到了 Hopper，瓶颈变成「加载数据和计算没有完全重叠」。Hopper 提供了 TMA（异步数据搬运单元）和 wgmma（异步 Tensor Core 指令），FA3 围绕它们做 warp specialization（让不同 warp 各干一种活）：
producer warp 只负责发 TMA 加载，consumer warpgroup 只做 wgmma 和 softmax。
两个 consumer warpgroup 还会 ping-pong：一个在算 softmax 时，另一个在算 GEMM。

FP8 路径则用分块量化和 Hadamard 变换（incoherent processing，把离群值摊开）来压低误差。论文报告 H100 上 bf16 约 740 TFLOP/s，约 75% 利用率。

### 8. MQA/GQA 与近似 attention

FlashAttention 不改数学。如果还嫌 attention 贵，有两条改数学的路，它们省的东西不一样。

MQA/GQA 减少的是 K/V 的头数 `n_kv`。attention FLOPs 不变，因为每个 Q 头仍然要和 T 个 key 做点积；减少的是 KV 的存储和读取。

近似方法则直接改变计算：稀疏方法（局部窗口、块稀疏、Longformer/BigBird）、低秩方法（Linformer 把 T 投影到 k 维）、
核化方法（Performer、linear attention 用 `φ(Q)(φ(K)ᵀV)` 把复杂度降为 `O(Td²)`）。
它们在 T 不够长时墙钟上常常更慢，质量也有损。FlashAttention 出现之后，精确 attention 能实用的序列长度大大提高，这些近似方法的空间也就小了。

## 关键推导

### 推导 1：online softmax 的正确性

我们要证明：分两块算出来再合并，和一次性算整行的 softmax 结果完全相同。

对一行 `x = [x⁽¹⁾, x⁽²⁾]`，设 `m_i = max x⁽ⁱ⁾`，`l_i = Σ e^{x⁽ⁱ⁾ − m_i}`，`m = max(m_1, m_2)`：

```text
l = e^{m_1 − m} l_1 + e^{m_2 − m} l_2 = Σ_all e^{x − m}
O = [e^{m_1 − m} Õ_1 + e^{m_2 − m} Õ_2] / l,   Õ_i = Σ_k e^{x_k⁽ⁱ⁾ − m_i} v_k
  = Σ_all e^{x − m} v / Σ_all e^{x − m} = softmax(x)·V
```

两块成立，再用归纳法推广到任意块数即可。数值上，每一步指数的参数都 `≤ 0`，所以不会溢出。这正是减去最大值的目的，它不是可有可无的步骤。

### 推导 2：IO 复杂度 `Θ(T²d²/M)`

这个推导回答：分块之后，HBM 流量到底是多少，它和 SRAM 大小有什么关系？

SRAM 能放 `M` 个元素，要同时放下 `K_j, V_j`（`2·Bc·d`）和 `Q_i, O_i`（`2·Br·d`），所以取 `Bc = Θ(M/d)`，`Br = Θ(min(M/d, d))`。
在 FA1 的循环里，每个 K/V 块读一遍，共 `Θ(Td)`；而对每个 K/V 块都要把全部 Q、O 过一遍，共 `(T/Bc)·Θ(Td) = Θ(T²d²/M)`。

对典型的 `d=64–128`、`M≈10⁵` 元素，`d²/M` 远小于 1，理论上 HBM 访问量比 naive 的 `T²` 小数倍到十倍以上。
不过这仍是 T 的二次函数；摘要里提到，在实验的模型中它只比朴素实现少 2 倍（因果时 4 倍）。
论文还证明了下界：对所有 `M ∈ [d, Td]`，没有任何精确算法能做到 `o(T²d²/M)`。

### 推导 3：反向只存 LSE

反向传播需要 `P`，但存下 `P` 要 `T²` 的显存，这正是我们想避免的。

FlashAttention 的做法是只存 `LSE_i = m_i + log l_i`，一共 `T` 个数。反向时按块重算 `S_ij = Q_i K_jᵀ`，
再用 `P_ij = exp(S_ij − LSE_i)` 一步得到归一化后的概率，不需要再跑一遍 online 递推。

另一个需要的量是 `D_i = rowsum(dO_i ∘ O_i)`，它是 softmax 雅可比矩阵里的标量项：`dS = P∘(dP − D)`。
其中 `rowsum(P∘dP) = rowsum(dO∘O)`，这可以由 `O = PV` 直接得到。

## 深入推导

### FLOPs 账：前向 + 反向

重算不是免费的。下面把前向和反向的 GEMM 逐个列出来，看看多花了多少：

```text
前向:  QKᵀ 2T²d,  PV 2T²d                                 = 4T²d
反向:  重算 QKᵀ 2T²d,  dV=PᵀdO 2T²d,  dP=dO Vᵀ 2T²d,
       dQ=dS K 2T²d,  dK=dSᵀQ 2T²d                         = 10T²d
反向/前向 = 2.5（普通线性层是 2，多出的 0.5 是重算）
```

### 代码：分块前向的 numpy 参考实现

下面是 FA2 循环顺序的最小参考实现，最后一行和朴素 softmax 对拍。写 kernel 之前先让这个版本通过，能省很多调试时间。

```python
import numpy as np

def flash_fwd(Q, K, V, Br=64, Bc=64):
    T, d = Q.shape
    O = np.zeros_like(Q); LSE = np.zeros(T)
    for i in range(0, T, Br):                      # FA2 顺序：外层 Q 块
        q = Q[i:i+Br] / np.sqrt(d)
        m = np.full(q.shape[0], -np.inf); l = np.zeros(q.shape[0])
        acc = np.zeros((q.shape[0], d))
        for j in range(0, T, Bc):                  # 内层流式读 K/V
            s = q @ K[j:j+Bc].T
            m_new = np.maximum(m, s.max(axis=1))
            p = np.exp(s - m_new[:, None])
            scale = np.exp(m - m_new)
            l = scale * l + p.sum(axis=1)
            acc = scale[:, None] * acc + p @ V[j:j+Bc]
            m = m_new
        O[i:i+Br] = acc / l[:, None]               # 只在最后归一化一次
        LSE[i:i+Br] = m + np.log(l)
    return O, LSE

def ref(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1]); P = np.exp(S - S.max(1, keepdims=True))
    return (P / P.sum(1, keepdims=True)) @ V

rng = np.random.default_rng(0); Q, K, V = (rng.standard_normal((256, 64)) for _ in range(3))
assert np.allclose(flash_fwd(Q, K, V)[0], ref(Q, K, V), atol=1e-6)
```

### 代码：Triton 前向 kernel 骨架（非因果）

把上面的 numpy 版本翻译成 Triton，结构几乎一一对应：每个 program 负责一个 Q 块，内层循环流式读 K/V。
这里省略了边界 mask；因果版本要把内层循环的上界改成 `(pid_m+1)*BLOCK_M`，并在对角块上加 `tl.where`。

```python
import triton, triton.language as tl

@triton.jit
def attn_fwd(Q, K, V, O, LSE, stride_t, T, scale,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr):
    pid_m, pid_bh = tl.program_id(0), tl.program_id(1)   # 沿 Q 块 × (batch·head) 并行
    base = pid_bh * T * stride_t
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + base + offs_m[:, None] * stride_t + offs_d[None, :])
    m = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    for start in range(0, T, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[:, None] * stride_t + offs_d[None, :])
        v = tl.load(V + base + offs_n[:, None] * stride_t + offs_d[None, :])
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m - m_new)
        l = alpha * l + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    tl.store(O + base + offs_m[:, None] * stride_t + offs_d[None, :], (acc / l[:, None]).to(O.dtype.element_ty))
    tl.store(LSE + pid_bh * T + offs_m, m + tl.log(l))
```

## 白板估算

参数取 H100 SXM：bf16 989 TFLOP/s，HBM 3.35 TB/s，L2 约 10 TB/s 量级（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）。
下面依次估算单头的 naive 与 Flash、整层里 attention 的占比，以及 decode 时读 KV 的时间。

```text
单头，T = 8192，d = 128，bf16，非因果
FLOPs = 4·T²·d = 4·6.71e7·128 = 3.44e10    → t_compute = 3.44e10/989e12 ≈ 35 µs

naive：S 写、S 读、P 写、P 读 ≈ 4·T²·2 B = 5.37e8 B；Q,K,V,O = 4·T·d·2 B = 8.4e6 B（可忽略）
       I ≈ 3.44e10/5.45e8 ≈ 63 FLOP/B < 295 → 访存受限
       t_mem ≈ 5.45e8/3.35e12 ≈ 163 µs，比计算慢约 4.7 倍
       显存：S 一头 T²·2 B = 134 MB；32 头一层 ≈ 4.3 GB（batch=1）

Flash（Br = Bc = 128）：HBM 只读 Q,K,V、写 O ≈ 8.4 MB → 2.5 µs
       K/V 被每个 Q 块重读：(T/Br)·2·T·d·2 B = 64·4.19e6 ≈ 268 MB
       一头的 K,V 共 4.2 MB，远小于 50 MB L2，重读主要命中 L2：268e6/10e12 ≈ 27 µs < 35 µs
       → 变成计算受限

整层占比（Llama-3-8B，T = 8192）：
       attention 4·T²·d_model = 4·6.71e7·4096 ≈ 1.10e12
       线性层 2·T·2.18e8 ≈ 3.57e12  → attention 约 23%；因果跳块后约 13%

decode 读 KV（T = 8192，batch = 1）：
       Llama-2-7B（MHA，512 KB/token）：4.29e9 B / 3.35e12 ≈ 1.28 ms/步
       Llama-3-8B（GQA 8 组，128 KB/token）：1.07e9 B / 3.35e12 ≈ 0.32 ms/步，少 4 倍
```

Flash 这一段的关键在 L2 那一行：K/V 虽然被重读了 64 次，但一头的 K/V 装得进 L2，所以这些重读不怎么花 HBM 带宽。

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| online softmax / 分块前向 | [CS229S P1](../solutions/cs229s/project1-transformer-kernels/README.md) | 与朴素 attention 的最大误差在 bf16 容差内 |
| 融合 attention 的速度与显存 | 同上，扫 T 记录耗时和峰值显存 | 峰值显存随 T 线性而非平方增长 |
| KV 字节数 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `kv_cache_bytes` | Llama-3-8B 每 token 128 KB |

## 易错点与反例

- 常见错答：「FlashAttention 降低了 attention 的 FLOPs」。前向 FLOPs 不变，反向因为重算还多了；它降的是 HBM 访问和显存。
- 常见错答：「FlashAttention 是近似算法」。它是精确的，误差只来自浮点运算结合顺序的不同。
- 常见错答：「GQA 让 attention 计算快了 4 倍」。FLOPs 没变，prefill 几乎不受益；受益的是 decode 时的 KV 读取和 KV 显存。
- online softmax 只缩放了 `l`，忘了同时缩放 `O`。结果会错，但只有在后面的块出现更大的最大值时才暴露，所以小测试可能恰好通过。
- 因果 mask 用 `−inf` 填满再算全矩阵。结果是对的，但 FLOPs 没省。
- 一整行全被 mask 时 `m = −inf`，`exp(m − m_new)` 会得到 NaN。实现里要特判这种情况（例如 padding 行）。
- 以为 T 很小时 Flash 也一定快很多。T=128 时 S 本来就放得进 SRAM，这时的差距主要来自 kernel 融合本身。

## 实验或练习

这个练习用来亲眼看到「显存从平方变成线性」。

- 假设：朴素 attention 的峰值显存 ∝ T²，融合实现 ∝ T；墙钟上，T ≥ 1024 时融合实现明显更快。
- 配置：bf16，d=64，heads=12，batch=4，T ∈ {512, 1024, 2048, 4096, 8192}；对比手写朴素实现、`torch.nn.functional.scaled_dot_product_attention` 和自己的 Triton kernel。
- 预期：朴素实现在 T=8192 附近先 OOM 或明显变慢；用 `torch.cuda.max_memory_allocated` 记录峰值显存，验证增长的阶数。

## 高频问题

- 问：attention 每层前向多少 FLOPs？
  答：`4T²·d_model`，再加上 QKV/O 投影的 `8T·d_model²`（MHA）。
- 问：FlashAttention 省掉了什么，没省掉什么？
  答：省掉了 S/P 在 HBM 上的往返和 `O(T²)` 的显存；FLOPs 没省，反向还因为重算多了 `2T²d`。
- 问：反向为什么不存 P？
  答：存 P 要 `T²` 显存；只存 LSE，反向时重算 S，再用 `exp(S − LSE)` 得到 P。
- 问：FA2 相比 FA1 改了什么？
  答：外层循环换成 Q 块；减少非 matmul 运算；沿序列长度并行；warp 之间按 Q 切分，免去归约。
- 问：MQA/GQA 省了什么？
  答：KV cache 显存和 decode 时的 KV 读取量，比例为 `n_heads/n_kv`。

## 面试追问链

1. 问：T=8192、bf16 时，一头的分数矩阵 S 有多大？T 再翻倍呢？
   答：S、P 各 `T²` 个元素，一头 134 MB；T 翻倍变成 4 倍。它和 d 无关，所以长序列时先撞上的是显存。
2. 问：为什么朴素 attention 在 H100 上是访存受限的？算一下。
   答：T=8192、d=128 时，FLOPs 3.44e10，HBM 流量约 5.4e8 B，强度约 63 < 295；访存约需 163 µs，计算只要 35 µs。
3. 问：推导 online softmax，并说明为什么要同时维护 m 和 l。
   答：利用 `e^{x−m_new} = e^{x−m_old}·e^{m_old−m_new}`。m 保证指数不溢出，l 是分母；O 和 l 用同一个因子缩放，最后除以 l。
4. 问：FlashAttention 的 K/V 被每个 Q 块重读，HBM 流量真的变小了吗？
   答：理论上是 `Θ(T²d²/M)`，比 `T²` 小 `M/d²` 倍。实践中一头的 K/V 只有几 MB，同时运行的 Q 块共享它，重读大多命中 L2，HBM 只付一遍。
5. 问：为什么 FA2 要专门减少非 matmul 的 FLOPs？
   答：Tensor Core 的吞吐比 CUDA core 上的 fp32 高一个数量级以上。exp、除法、max 都在 CUDA core 上算，哪怕只占很少的 FLOPs，也会占掉可观的时间。
6. 问：长上下文训练（T=128K、batch=1）时，FA2 的并行度从哪来？单卡放不下怎么办？
   答：沿 Q 块并行，T/128=1000 个块足以填满 SM。单卡放不下时用 context parallel / ring attention 把序列切到多卡，K/V 块沿环传递，这其实就是 online softmax 的跨设备版本。
7. 问：你在设计推理引擎的 decode attention kernel，batch 小、上下文长，怎么并行？
   答：decode 时每个请求只有 1 个 query，Q 维没有并行度。所以沿 KV 长度切分（split-K / Flash-Decoding），每块算出局部的 m、l、O，再用 online softmax 的合并公式归约。瓶颈是读 KV，可以用 GQA 让一组 Q 头共享一次 KV 读取。
8. 问：老板想用 linear attention 替换精确 attention 来省钱，你怎么评估？
   答：先算账：T=8K 时 attention 只占约 23% 的 FLOPs，而且精确 attention 已经被 Flash 推到计算受限，收益上限有限。再看质量（检索、in-context 类任务会退化）和生态（KV cache、推测解码是否兼容）。只有 T 极长时才值得换。

## 自测题

1. 计算 T=4096、d=128、因果、块大小 128 时，实际计算的块占全部块的比例。
   要点：n=32，`n(n+1)/2 = 528`，`528/1024 ≈ 51.6%`。
2. 证明 `rowsum(P∘dP) = rowsum(dO∘O)`。
   要点：`dP = dO Vᵀ`，`Σ_j P_ij (dO_i·v_j) = dO_i·Σ_j P_ij v_j = dO_i·O_i`。
3. 写出两块 online softmax 合并的 (m, l, O) 公式，并说明它为什么可以用于 split-K decode。
   要点：公式同推导 1；合并满足结合律，所以可以并行归约。
4. 估算 Llama-3-70B 在 T=32K 时一条序列的 KV cache 大小。
   要点：每 token `2·80·8·128·2 = 327,680 B`（320 KB），× 32768 ≈ 1.07e10 B ≈ 10.7 GB。
5. 设计题：显存只够存 LSE、不够存 O 时，反向还能进行吗？
   要点：不行，`D = rowsum(dO∘O)` 需要 O；FlashAttention 反向的输入是 Q、K、V、O、dO、LSE。

## 关联笔记

- [M07 矩阵乘与分块](module-07-matmul-hardware-aware.md)：分块与强度的基础。
- [M05 GPU 架构与 CUDA](module-05-gpu-architecture-cuda.md)：SRAM 容量决定 Br、Bc。
- [M14 显存优化](module-14-memory-optimization.md)：重计算与激活显存，FlashAttention 是算子级重计算。
- [M15 KV cache 与推测解码](module-15-inference-kv-speculative.md)：decode attention 与 GQA 的收益。
## 本讲小结

attention 的瓶颈是 `T²` 中间结果的 IO 和显存，而不是 FLOPs。online softmax 让 softmax 可以分块累加，FlashAttention 据此把 S/P 留在 SRAM，
用 LSE 重算代替存 P；它的 HBM 流量在 T 上仍是二次，实际加速主要来自融合和 L2 复用。
GQA 不改 FLOPs，只减少 KV 字节数，这一点要到后面的推理系统里才真正发挥作用。

## 延伸阅读

- Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022, https://arxiv.org/abs/2205.14135
- Dao, FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023, https://arxiv.org/abs/2307.08691
- Shah et al., FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision, 2024, https://arxiv.org/abs/2407.08608
- Milakov & Gimelshein, Online normalizer calculation for softmax, 2018, https://arxiv.org/abs/1805.02867
- Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models, 2023, https://arxiv.org/abs/2305.13245
- Katharopoulos et al., Transformers are RNNs（linear attention）, 2020, https://arxiv.org/abs/2006.16236
