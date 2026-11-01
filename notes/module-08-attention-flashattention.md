# Module 08 — Attention 与 FlashAttention

- 课程出处：CS149 Fall 2024《Efficiently Evaluating DNNs》（融合 attention） · CS229S Fall 2023《Hardware Aware Algorithm Design》(10-11) · CMU 15-849《Transformers and Language Models》
- 必读论文：Dao et al.，FlashAttention，2022；Dao，FlashAttention-2，2023；Shah et al.，FlashAttention-3，2024；Milakov & Gimelshein，Online normalizer calculation for softmax，2018；Ainslie et al.，GQA，2023
- 对应实验：[CS229S P1 Transformer kernels](../solutions/cs229s/project1-transformer-kernels/README.md)
- 实验报告：[报告 PDF](../reports/cs229s-project1/cs229s_project1_report.pdf)（实测：Triton FlashAttention 前向 224 TFLOP/s，与 FA2 持平，比朴素快 7.15 倍）
- 状态：初稿

## 摘要

标准 attention 每头每层前向 `4T²d` FLOPs，还要物化 `T×T` 的分数矩阵；长序列下瓶颈不是 FLOPs，而是把 `T²` 的
中间结果写进 HBM 再读出来。FlashAttention 的贡献不是近似，而是重排计算：用 online softmax 把 softmax 拆成
可以分块累加的形式，让 `S` 和 `P` 只活在 SRAM 里，HBM 访问量从 `Θ(Td + T²)` 降到 `Θ(T²d²/M)`。
注意 d=64–128 时 `d²/M` 是常数，访存在 T 上**仍是二次**：[实验](../solutions/cs229s/project1-transformer-kernels/README.md)
里只比「S、P 各读写一次」的朴素模型少 2 倍（因果 4 倍）。实际收益来自三处：显存 `O(T)`、省掉 eager 实现里
scale/mask/softmax/dropout 各自的 `T²` 读写，以及 K/V 重读命中 L2。
反向不存 `P`，只存每行的 logsumexp 并重算。v2 调整循环顺序和 warp 分工，v3 利用 Hopper 的异步硬件。主线取舍：多做一遍前向的 FLOPs（重计算）换掉 `T²` 的显存与带宽，在现代 GPU 上几乎总是划算。

## 一页速览卡（面试前 5 分钟）

**必须记住的五件事**

1. 每头每层前向 FLOPs `4T²d`（`QKᵀ` 和 `PV` 各 `2T²d`），全层 `4T²·d_model`；反向约 2.5 倍前向（含重算）。
2. 朴素实现的 HBM 流量被 `T²` 项主导（写 S、读 S、写 P、读 P），T=8192 时强度约 60 FLOP/B，访存受限。
3. Online softmax：维护行最大值 `m` 和归一化和 `l`，新块到来时用 `exp(m_old − m_new)` 缩放旧的 `l` 和 `O`。
4. FlashAttention 是精确算法，HBM 访问 `Θ(T²d²/M)`（在 T 上仍是二次）；显存从 `O(T²)` 降到 `O(T)`；反向只存 logsumexp。
5. GQA/MQA 不减少 attention FLOPs，减少的是 KV cache 字节数和 decode 时的 KV 读取量（Llama-3-8B 比 MHA 少 4 倍）。

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

- FA1 vs FA2：FA1 外层循环 K/V 块，O 反复读写 HBM；FA2 外层循环 Q 块，O 留在寄存器，按序列长度并行。
- FlashAttention vs 稀疏/线性 attention：前者精确、只改 IO 顺序；后者改了数学，换来低于 `T²` 的复杂度和精度风险。
- MQA vs GQA：MQA 所有头共享 1 组 K/V；GQA 每组头共享 1 组，是质量与 KV 字节数的折中。

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

单头：`Q,K,V ∈ R^{T×d}`。`S = QKᵀ` 是 `T×T`，`2T²d` FLOPs；`O = PV` 又是 `2T²d`；softmax 本身 `O(T²)`。
多头拼起来，每层前向 `4T²·d_model`。显存上 `S`/`P` 是 `T²`，与 d 无关，T 翻倍显存翻四倍。
Llama-3-8B 在 T=8192 时 attention 只占单层前向约 23%（非因果），所以中等长度下的问题主要是 IO 和显存，不是 FLOPs。

### 2. 为什么朴素实现访存受限

PyTorch 朴素实现分三个 kernel：matmul 写 S，softmax 读 S 写 P，matmul 读 P，每个都把 `T²` 的张量过一遍 HBM。
softmax 每元素只做 `O(1)` 次运算，强度远低于 1，即使两个 matmul 用 cuBLAS，中间这步也把整体拖进访存区。

### 3. Online softmax

softmax 需要整行的最大值和和，看上去必须先看完整行。关键观察：

```text
e^{x − m_new} = e^{x − m_old} · e^{m_old − m_new}
```

所以用旧最大值算出的部分和，乘一个标量就能修正到新最大值下。`O` 是 `P` 的线性函数，同一行乘同一个因子即可；递推结束时 `O/l` 就是精确结果。

### 4. FlashAttention v1

把 Q 切成 `Br` 行一块，K/V 切成 `Bc` 行一块。v1 外层循环遍历 K/V 块 j，把 `K_j, V_j` 装进 SRAM；
内层循环遍历所有 Q 块 i，从 HBM 读 `Q_i, O_i, m_i, l_i`，更新后写回。S、P 块从不离开 SRAM；代价是 O、m、l 被反复读写 `T/Bc` 次，且每步都重新归一化 O。

### 5. FlashAttention v2 的三个改动

1. 循环顺序反过来：外层遍历 Q 块，每个线程块负责一个 Q 块，内层流式读 K/V；O 累加器一直留在寄存器，最后写一次。
2. 减少非 matmul FLOPs：不在每步做 `diag(l)⁻¹` 归一化，只在末尾除一次；只存 LSE 而不是 m 和 l。
   A100 上非 matmul FLOP/s 比 Tensor Core 低约 16 倍（19.5 vs 312），省掉的每一个指数和除法都值钱。
3. 并行与 warp 分工：除 batch×head 外，还沿序列长度（Q 块）并行，长序列小 batch 时也能填满 108/132 个 SM。
   线程块内把 Q 切给 4 个 warp，每个 warp 看到全部 K/V，不用经共享内存交换部分和（v1 把 K/V 切给 warp，需要 warp 间归约）。

### 6. 因果 mask 与块跳过

因果 attention 中，Q 块 i 只需要 K 块 `j ≤ i`。完全在对角线上方的块直接跳过，对角线上的块加 mask，下方的块不需要 mask。
`n` 个块时实际计算 `n(n+1)/2` 块，约为一半；省 FLOPs 的前提是按块跳过，逐元素加 `−∞` 再算并不省。

### 7. FlashAttention-3（Hopper）

Hopper 有 TMA（异步搬运）和 wgmma（异步 Tensor Core 指令）。FA3 做 warp specialization：producer warp 只发 TMA 加载，consumer warpgroup 只做 wgmma 与 softmax，两个 consumer warpgroup ping-pong（一个算 softmax 时另一个算 GEMM）；
FP8 路径用分块量化和 Hadamard 变换（incoherent processing）压低离群值误差。论文报告 H100 上 bf16 约 740 TFLOP/s，约 75% 利用率。

### 8. MQA/GQA 与近似 attention

MQA/GQA 减少 K/V 头数 `n_kv`，attention FLOPs 不变（每个 Q 头仍与 T 个 key 做点积），减的是 KV 存储与读取。
近似方法改变数学：稀疏（局部窗口、块稀疏、Longformer/BigBird）、低秩（Linformer 把 T 投影到 k）、核化（Performer、linear attention 用 `φ(Q)(φ(K)ᵀV)` 降为 `O(Td²)`）。
它们在 T 不够长时常常墙钟更慢且质量有损；FlashAttention 出现后，精确 attention 的实用长度大幅上升。

## 关键推导

### 推导 1：online softmax 的正确性

对一行 `x = [x⁽¹⁾, x⁽²⁾]`，设 `m_i = max x⁽ⁱ⁾`，`l_i = Σ e^{x⁽ⁱ⁾ − m_i}`，`m = max(m_1, m_2)`：

```text
l = e^{m_1 − m} l_1 + e^{m_2 − m} l_2 = Σ_all e^{x − m}
O = [e^{m_1 − m} Õ_1 + e^{m_2 − m} Õ_2] / l,   Õ_i = Σ_k e^{x_k⁽ⁱ⁾ − m_i} v_k
  = Σ_all e^{x − m} v / Σ_all e^{x − m} = softmax(x)·V
```

归纳到任意块数即可。数值上每一步指数的参数 `≤ 0`，不会溢出；这是减最大值的目的，不是可有可无。

### 推导 2：IO 复杂度 `Θ(T²d²/M)`

SRAM 能放 `M` 个元素。要放下 `K_j, V_j`（`2·Bc·d`）和 `Q_i, O_i`（`2·Br·d`），取 `Bc = Θ(M/d)`，`Br = Θ(min(M/d, d))`。
FA1 每个 K/V 块读一遍，共 `Θ(Td)`；对每个 K/V 块要过一遍全部 Q、O，共 `(T/Bc)·Θ(Td) = Θ(T²d²/M)`。
对典型 `d=64–128`、`M≈10⁵` 元素，`d²/M` 远小于 1，HBM 访问量比 naive 的 `T²` 小数倍到十倍以上。
论文还证明了下界：对所有 `M ∈ [d, Td]`，没有精确算法能做到 `o(T²d²/M)`。

### 推导 3：反向只存 LSE

反向需要 `P`。存 `P` 要 `T²`；FlashAttention 存 `LSE_i = m_i + log l_i`（`T` 个数），反向时按块重算 `S_ij = Q_i K_jᵀ`，
`P_ij = exp(S_ij − LSE_i)`，一步得到归一化概率，不必再跑 online 递推。`D_i = rowsum(dO_i ∘ O_i)` 是 softmax 雅可比的标量项：
`dS = P∘(dP − D)`，其中 `rowsum(P∘dP) = rowsum(dO∘O)`，由 `O = PV` 直接得到。

## 深入推导

### FLOPs 账：前向 + 反向

```text
前向:  QKᵀ 2T²d,  PV 2T²d                                 = 4T²d
反向:  重算 QKᵀ 2T²d,  dV=PᵀdO 2T²d,  dP=dO Vᵀ 2T²d,
       dQ=dS K 2T²d,  dK=dSᵀQ 2T²d                         = 10T²d
反向/前向 = 2.5（普通线性层是 2，多出的 0.5 是重算）
```

### 代码：分块前向的 numpy 参考实现

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

省略边界 mask；因果版本把内层循环上界改成 `(pid_m+1)*BLOCK_M`，并在对角块上加 `tl.where`。

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

H100 SXM：bf16 989 TFLOP/s，HBM 3.35 TB/s，L2 约 10 TB/s 量级（[硬件参数速查](../interview/HARDWARE-NUMBERS.md)）。

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

## 实现映射

| 概念 | 对应代码 | 验收标准 |
| --- | --- | --- |
| online softmax / 分块前向 | [CS229S P1](../solutions/cs229s/project1-transformer-kernels/README.md) | 与朴素 attention 的最大误差在 bf16 容差内 |
| 融合 attention 的速度与显存 | 同上，扫 T 记录耗时和峰值显存 | 峰值显存随 T 线性而非平方增长 |
| KV 字节数 | [`tools/mlsys_calc.py`](../tools/mlsys_calc.py) 的 `kv_cache_bytes` | Llama-3-8B 每 token 128 KB |

## 易错点与反例

- 常见错答：「FlashAttention 降低了 attention 的 FLOPs」。前向 FLOPs 不变，反向因重算还多了；它降的是 HBM 访问和显存。
- 常见错答：「FlashAttention 是近似算法」。它是精确的，误差只来自浮点结合顺序。
- 常见错答：「GQA 让 attention 计算快了 4 倍」。FLOPs 没变，prefill 几乎不受益；受益的是 decode 的 KV 读取和 KV 显存。
- online softmax 忘了同时缩放 `O`，只缩放 `l`：结果错，且只在后面的块出现更大最大值时才暴露，小测试可能通过。
- 因果 mask 用 `−inf` 填完再算全矩阵：结果对，但 FLOPs 没省。
- 整行全被 mask 时 `m = −inf`，`exp(m − m_new)` 得 NaN；实现里要特判（例如 padding 行）。
- 认为 T 很小时 Flash 也一定更快：T=128 时 S 本来就能放进 SRAM，差距主要来自 kernel 融合本身。

## 实验或练习

- 假设：朴素 attention 的峰值显存 ∝ T²，融合实现 ∝ T；墙钟上 T ≥ 1024 时融合实现明显更快。
- 配置：bf16，d=64，heads=12，batch=4，T ∈ {512, 1024, 2048, 4096, 8192}；对比手写朴素实现、`torch.nn.functional.scaled_dot_product_attention`、自己的 Triton kernel。
- 预期：朴素实现在 T=8192 附近先 OOM 或显著变慢；记录 `torch.cuda.max_memory_allocated` 验证增长阶数。

## 高频问题

- 问：attention 每层前向多少 FLOPs？
  答：`4T²·d_model`，加上 QKV/O 投影的 `8T·d_model²`（MHA）。
- 问：FlashAttention 为什么快？
  答：不把 `T×T` 的 S/P 写回 HBM，靠 online softmax 分块累加，访存从 `T²` 级降到 `T²d²/M`。
- 问：反向为什么不存 P？
  答：存 P 要 `T²` 显存；只存 LSE，反向重算 S 再 `exp(S − LSE)` 得到 P，多花 `2T²d` FLOPs。
- 问：FA2 相比 FA1 改了什么？
  答：外层循环换成 Q 块、减少非 matmul 运算、沿序列长度并行、warp 间按 Q 切分免去归约。
- 问：MQA/GQA 省了什么？
  答：KV cache 显存和 decode 时 KV 读取量，比例为 `n_heads/n_kv`。

## 面试追问链

1. 问：给定 T、d、头数，attention 的 FLOPs 和中间张量多大？
   答：每头 `4T²d` FLOPs，S/P 各 `T²` 个元素；T=8192、bf16 一头 134 MB，T 翻倍变 4 倍。
2. 问：为什么朴素 attention 在 H100 上是访存受限的，算一下？
   答：T=8192、d=128：FLOPs 3.44e10，HBM 约 5.4e8 B，强度约 63 < 295；时间约 163 µs，计算只要 35 µs。
3. 问：推导 online softmax，说明为什么需要同时维护 m 和 l。
   答：`e^{x−m_new} = e^{x−m_old}·e^{m_old−m_new}`；m 保证指数不溢出，l 是分母，O 与 l 用同一个因子缩放，最后除 l。
4. 问：FlashAttention 的 K/V 被每个 Q 块重读，HBM 流量真的小了吗？
   答：理论上 `Θ(T²d²/M)`，比 `T²` 小 `M/d²` 倍；实践中一头的 K/V 只有几 MB，同时运行的 Q 块共享它，重读大多命中 L2，HBM 只付一遍。
5. 问：为什么 FA2 要减少非 matmul FLOPs？
   答：Tensor Core 吞吐比 CUDA core 的 fp32 高一个数量级以上，exp、除法、max 在 CUDA core 上算，少量非 matmul 运算就会占掉可观时间。
6. 问：长上下文训练（T=128K、batch=1）时 FA2 的并行度从哪来？如果单卡放不下怎么办？
   答：沿 Q 块并行，T/128=1000 个块足以填满 SM；单卡放不下时用 context parallel / ring attention，把序列切到多卡，K/V 块沿环传递，正是 online softmax 的跨设备版本。
7. 问：你在设计一个推理引擎的 decode attention kernel，batch 小、上下文长，怎么并行？
   答：decode 时每个请求只有 1 个 query，Q 维没有并行度；沿 KV 长度切分（split-K / Flash-Decoding），每块算局部 m、l、O，再用 online softmax 的合并公式归约。瓶颈是读 KV，用 GQA 让一组 Q 头共享一次 KV 读取。
8. 问：老板想用 linear attention 替换精确 attention 来省钱，你怎么评估？
   答：先算账：T=8K 时 attention 只占约 23% 的 FLOPs，精确 attention 已被 Flash 推到计算受限，收益上限有限；再看质量（检索、in-context 任务退化）和生态（KV cache、推测解码兼容性）。只有 T 极长时才值得换。

## 自测题

1. 计算 T=4096、d=128、因果、块大小 128 时实际计算的块数比例。
   要点：n=32，`n(n+1)/2 = 528`，`528/1024 ≈ 51.6%`。
2. 证明 `rowsum(P∘dP) = rowsum(dO∘O)`。
   要点：`dP = dO Vᵀ`，`Σ_j P_ij (dO_i·v_j) = dO_i·Σ_j P_ij v_j = dO_i·O_i`。
3. 写出两块 online softmax 合并的 (m, l, O) 公式，并说明它可以用于 split-K decode。
   要点：同推导 1；合并满足结合律，所以可以并行归约。
4. 估算 Llama-3-70B 在 T=32K 时一条序列的 KV cache 大小。
   要点：每 token `2·80·8·128·2 = 327,680 B`（320 KB），× 32768 ≈ 1.07e10 B ≈ 10.7 GB。
5. 设计题：显存只够存 LSE 不够存 O 时反向能否进行？
   要点：不行，`D = rowsum(dO∘O)` 需要 O；FlashAttention 反向输入是 Q、K、V、O、dO、LSE。

## 关联笔记

- [M07 矩阵乘与分块](module-07-matmul-hardware-aware.md)：分块与强度的基础。
- [M05 GPU 架构与 CUDA](module-05-gpu-architecture-cuda.md)：SRAM 容量决定 Br、Bc。
- [M14 显存优化](module-14-memory-optimization.md)：重计算与激活显存，FlashAttention 是算子级重计算。
- [M15 KV cache 与推测解码](module-15-inference-kv-speculative.md)：decode attention 与 GQA 的收益。
## 本讲小结

attention 的瓶颈是 `T²` 中间结果的 IO 而不是 FLOPs。online softmax 让 softmax 可以分块累加，FlashAttention 据此把 S/P 留在 SRAM，
HBM 访问降到 `Θ(T²d²/M)`，反向靠 LSE 重算。v2 改循环顺序和 warp 分工，v3 用 Hopper 异步硬件。GQA 不改 FLOPs，只改 KV 字节数，
这一点在下一阶段的推理系统里才真正发力。

## 延伸阅读

- Dao et al., FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, 2022, https://arxiv.org/abs/2205.14135
- Dao, FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, 2023, https://arxiv.org/abs/2307.08691
- Shah et al., FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision, 2024, https://arxiv.org/abs/2407.08608
- Milakov & Gimelshein, Online normalizer calculation for softmax, 2018, https://arxiv.org/abs/1805.02867
- Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models, 2023, https://arxiv.org/abs/2305.13245
- Katharopoulos et al., Transformers are RNNs（linear attention）, 2020, https://arxiv.org/abs/2006.16236
