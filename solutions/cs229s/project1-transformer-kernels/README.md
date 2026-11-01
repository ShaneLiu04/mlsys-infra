# CS229S Project 1：Attention 内核、IO 模型、depthwise 卷积与手写 associative recall

对应 CS229S Fall 2023 Project 1。用 numpy 写 FlashAttention 风格的前向和反向，并对照 torch 自动微分；
用 IO 计数器校验 HBM 流量模型；depthwise 因果卷积三种写法的计时；一个**手工设定权重**、没有任何训练的
两层 attention-only transformer，用它解 associative recall。

机器：4 物理核 / 8 逻辑核的 Windows Server CPU（与其他任务共享），Python 3.12，numpy 2.5，torch 2.6 CPU。
CPU 部分的计时来自 numpy，不代表 GPU kernel 的速度。IO 模型假设每个 SM 有 100 KiB SRAM、数据为 bf16。
**GPU 部分**（下文，RTX 5090）是 Triton FlashAttention-2 前向 + 反向、CUDA depthwise 卷积和 GPT 训练的实测；
完整报告：[cs229s_project1_report.pdf](../../../reports/cs229s-project1/cs229s_project1_report.pdf)。

## 一句话结论

分块 attention 的前向和反向与 torch 一致（float64 下最大误差 4.9e-15）。但在论文的块大小下（Br=min(Bc,d)），
FlashAttention 前向的 HBM 流量**仍然是 Θ(T²)**：非因果时比朴素实现少约 2 倍（d=64），因果时少约 4 倍。
主要收益是不落盘 T×T 矩阵（显存从 O(T²) 降到 O(T)），而不是流量阶数的改变。

GPU 上实测的差距比「2 倍流量」大得多：Triton 前向 224 TFLOP/s（FA2 217.9），T=8192 时比朴素实现快 7.15 倍。
朴素 eager 实现有 4Td+6T² 次读写，本身就是一个跑满带宽（93%）的访存程序，加速比可以写成 3·ridge/d
（d=64 预测 7.3、实测 7.15）。GPT-2 small bf16 + flash 训练 MFU 0.447。

## 对应官方作业

| 官方部分 | 本目录实现 | 差异 |
| --- | --- | --- |
| 训练 Transformer LM | `gpu/train_tiny.py`：GPU 上训练 1/2 层 GPT 解 MQAR，GPT-2-small 形状测 tokens/s 与 MFU（见下方「GPU 部分 → 训练」） | 只有合成任务（MQAR），没有在真实语料上训练 |
| 手写两层 transformer 解 associative recall | `tfkernels/recall.py`：上一 token 头 + 匹配头，权重全部手工设定 | 没有训练，也没有用 MLP 或 LayerNorm |
| MHA 的算术强度推导 | `tfkernels/io_model.py`：朴素实现与 flash 的 HBM 字节数；decode 阶段 MHA/GQA/MQA 的强度（复用 `tools/mlsys_calc.py`） | 只算前向、单头 |
| 1D depthwise 卷积的 CUDA kernel | `tfkernels/conv1d.py`：参考循环、O(NK) 移位相加、滑窗点积、FFT | 没有 GPU，用 numpy 代替 CUDA；滑窗版对应「每个线程算一个输出」 |
| （补充）FlashAttention | `tfkernels/attention.py`：朴素实现、online softmax、分块前向（运行中的 max/sum，因果块跳过）、基于 logsumexp 重计算的反向 | 课上讲过，但不是 P1 的必做项 |

## 怎么跑

```powershell
cd solutions/cs229s/project1-transformer-kernels; python -m pytest -q tests; python scripts/run_experiments.py
```

33 个测试约 5 s；实验约 70 s（`meta.json` 记录为 68.2 s）。

## 结果

### 正确性（`attention_correctness.csv`，float64，对照 `torch.scaled_dot_product_attention` 的 autograd）

8 组配置（T 最大 1024，d=64/128，块大小可以不整除 T，例如 T=255、Bc=48，T=512、Bc=100，因果和非因果都有），
O、dQ、dK、dV 的最大绝对误差都 ≤4.9e-15。测试阈值是 1e-5，覆盖 T=1、Tq≠Tk、batch×head 维。
T=1024 时朴素实现的 S 矩阵在 fp64 下占 8.39 MB，flash 每块只要 0.033 MB。numpy 版 flash **并不更快**：
T=1024、d=128 非因果时 44.6 ms 对朴素的 22.9 ms；因果时靠跳块追平（23.9 对 26.1 ms）。

### HBM 流量（`attention_io.csv`、`attention_io_vs_sram.csv`、`attention_io_counter_check.csv`）

朴素实现 = 4Td + 4T²（S 和 P 各写一次、读一次）。flash（FA2 顺序）= Q 读 1 次 + K/V 读 ⌈T/Br⌉ 遍 + O 和 lse 写 1 次。
实现里的 IO 计数器在 8 组配置上与模型逐元素一致；T=2048 因果时，352 个 K/V 块里跳过了 168 个。

| d | T | 朴素 MB | flash MB | 比值 | 因果 flash 比值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4,096 | 136.3 | 68.2 | 2.00 | 3.72 |
| 64 | 65,536 | 34,393 | 17,197 | 2.00 | 3.98 |
| 128 | 4,096 | 138.4 | 88.1 | 1.57 | 3.00 |
| 128 | 65,536 | 34,427 | 22,045 | 1.56 | 3.11 |

意外点：比值不随 T 增长。原因是 Br 被 d 封顶，K/V 要重读 T/Br 遍，流量约为 2T²d/Br = 2T²。SRAM 从 16 KiB
扫到 256 KiB（T=8192，d=64）：16 KiB 时比值只有 1.004；32 KiB 起 Br 已经顶到 d=64，比值固定在 2.00，
更大的 SRAM 只是把 Bc 变大，不再减少流量。因果时朴素实现仍然算完整的 T²，flash 跳过一半块，所以接近 4 倍。

![io](artifacts/figures/attention_io.png)

### Decode 阶段的强度（`decode_attention_intensity.csv`，Llama-3-8B 形状，H100，seq=4096，bf16）

| 变体 | KV 头数 | 每 token 的 KV（KiB） | 每条序列 KV（GB） | FLOP/B | 每 token 每序列 attention 时间（µs） |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA | 32 | 512 | 2.15 | 1 | 641 |
| GQA-8 | 8 | 128 | 0.54 | 4 | 160 |
| GQA-4 | 4 | 64 | 0.27 | 8 | 80 |
| MQA | 1 | 16 | 0.067 | 32 | 20 |

强度 = 2·(H/H_kv)/bytes，与 T 和 batch 都无关（每条序列有自己的 KV），离 H100 的 ridge（295）差 1 到 2 个数量级。

### Depthwise 因果卷积（`conv1d_timing.csv`，C=16，float64）

| 设置 | 移位相加 ms | 滑窗 ms | FFT ms |
| --- | ---: | ---: | ---: |
| N=16384，K=4 | 3.90 | 2.76 | 14.34 |
| N=16384，K=16 | 14.67 | 4.48 | 15.64 |
| N=16384，K=64 | 57.11 | 14.29 | 14.30 |
| N=16384，K=1024 | 867.9 | 303.6 | 14.83 |
| K=N=4096 | 160.6 | 313.6 | 3.69 |
| K=4，N=65536 | 14.96 | 10.55 | 89.64 |

FFT 的耗时只取决于 N（约 14–15 ms），O(NK) 的写法随 K 线性增长：移位相加在 K≈16、滑窗在 K≈64 与 FFT 交叉。
短卷积（Mamba/H3 的 K=4）直接卷积快 5.2–8.5 倍；长卷积（Hyena 的 K=N）在所有 N 上都是 FFT 更快。

![conv](artifacts/figures/conv1d_timing.png)

### Associative recall（`recall_accuracy.csv`，每格 5000 条随机序列，权重手工设定，没有训练）

| 键值对数 | β=1 | β=2 | β=4 | β≥8 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.8386 | 0.9982 | 1.0 | 1.0 |
| 8 | 0.4538 | 0.8198 | 1.0 | 1.0 |
| 16 | 0.3286 | 0.4694 | 1.0 | 1.0 |
| 32 | 0.2268 | 0.2826 | 0.9988 | 1.0 |

第 1 层：query 是 β·e_pos(p)，key 是 e_pos(j+1)，所以位置 p 注意到 p−1，把 tok(p−1) 写进 prev 子空间。
第 2 层：query 是 β·tok(p)，key 是 prev(j)，最后一个位置（查询键 q）会注意到「前一个 token 是 q」的那个位置，
也就是对应的值。β 太小时 softmax 不够尖，注意力被其他位置稀释，序列越长退化越严重。

![recall](artifacts/figures/recall_attention.png)

## GPU 部分（RTX 5090）

完整分析见报告 [cs229s_project1_report.pdf](../../../reports/cs229s-project1/cs229s_project1_report.pdf)。
机器：RTX 5090（sm_120，170 SM，32 GB，96 MB L2），torch 2.8.0+cu128，Triton 3.4.0，CUDA 12.8。
实测峰值（`artifacts/raw/gpu/peaks.json`）：copy 带宽 1521.9 GB/s，BF16 matmul 236.3 TFLOP/s，ridge 155 FLOP/B。
所有计时都在共享 GPU 锁内独占完成；原始数据在 `artifacts/raw/gpu/*.jsonl`，`python scripts/summarize_gpu.py` 打印下面每张表。

| 文件 | 内容 |
| --- | --- |
| `gpu/flash_triton.py` | Triton FlashAttention-2 前向（online softmax，exp2，因果分两段）+ 反向（dK/dV 与 dQ 两个 kernel），朴素实现，fp32 分块参考 |
| `gpu/dwconv_cuda.py` | depthwise 因果卷积的三个 CUDA kernel（naive / smem+halo / 寄存器分块），`load_inline` 编译；cuDNN 与 FFT 基线 |
| `gpu/tiny_gpt.py`, `gpu/train_tiny.py` | 可切换 attention 实现的 GPT；MQAR 训练、GPT-2-small 形状吞吐与 MFU、手写 recall 的 GPU 复核 |
| `gpu/bench_attention.py`, `gpu/bench_roofline.py`, `gpu/bench_dwconv.py` | 基准；`gpu/run_gpu.sh` 依次运行全部实验 |
| `tests/test_gpu.py` | 35 个 GPU 测试（无 CUDA 时整体 skip） |

### FlashAttention（`attention_fwd.jsonl`，fp16，d=64，非因果，B·H=32；ms / TFLOP/s）

| 方法 | T=1024 | T=4096 | T=8192 | T=16384 | T=32768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Triton（本实验） | 0.079 / 108.5 | 0.739 / 185.9 | 2.558 / 214.9 | 9.818 / 224.0 | 39.18 / 224.5 |
| SDPA flash | 0.071 / 120.9 | 0.717 / 191.6 | 2.599 / 211.5 | 10.09 / 217.9 | 39.81 / 221.0 |
| SDPA efficient | 0.129 / 66.5 | 1.350 / 101.8 | 5.107 / 107.6 | 20.17 / 109.0 | 79.78 / 110.2 |
| SDPA math | 0.894 / 9.6 | 13.66 / 10.1 | 54.06 / 10.2 | OOM | — |
| 朴素（物化 S、P） | 0.222 / 38.7 | 4.943 / 27.8 | 18.28 / 30.1 | OOM | — |

- Triton 前向在长序列上与 PyTorch 的 FA2 持平（峰值 224.5 TFLOP/s，是实测 matmul 峰值的 95%），比朴素实现快 7.15 倍（T=8192），因果时快 22.2 倍。
- 朴素实现峰值额外显存是 2 个 fp16 T×T 张量（T=8192：8.62 GB），T=16384 需要 34 GB，OOM；Triton 在 T=32768 时额外显存只有 138 MB。
- 误差（对 fp32 参考）：fp16 6.6e-5、bf16 3.65e-4（T=4096），与 SDPA flash 相同；朴素实现大 2–4 倍（S 以 fp16 落盘）。
- 反向正确（dQ 误差与 SDPA flash 同为 8.8e-5），但只有 FA2 的 80%（T=16384：167.4 对 209.0 TFLOP/s）：两个 kernel 各自重算 QKᵀ 和 dO·Vᵀ，总矩阵乘 9 个对 7 个。

![fwd](artifacts/figures/gpu_attn_fwd_time_tflops.png)

**与 CPU 部分 IO 模型的对照**。朴素实现按 eager 实际的 4Td+6T² 次 fp16 读写计算，T=8192 时隐含带宽 1.417 TB/s，是 copy 带宽的 93%：它就是一个以 DRAM 带宽运行的访存程序。
加速比可以写成 `3·ridge/d`：d=64 时预测 7.3，实测 7.15；d=128 时预测 3.6，实测 3.8。
反过来，按 IO 模型计算 flash 的隐含带宽最高 6.57 TB/s（块扫描中 Br=16，DRAM 的 4.3 倍）：K/V 重读主要命中 96 MB L2，
所以 Θ(T²d²/M) 是 DRAM 流量的上界。flash 前向的 DRAM 强度约等于 Br（fp16），最佳块是 (Br,Bc)=(128,64)，215.4 TFLOP/s。
（容器无权读取性能计数器，ncu 报 `ERR_NVGPUCTRPERM`，所以这里是推断而不是直接测量。）

![blocks](artifacts/figures/gpu_attn_block_sweep.png)

### Roofline：一层 Llama-2-7B（`roofline.jsonl`，bf16）

| 形状 | 算子 | 强度 FLOP/B | ms | 实测 | roofline 比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| decode B=1 | MLP | 1.0 | 0.1851 | 1463 GB/s | 0.96 |
| decode B=1 | attention | 1.0 | 0.0868 | 773 GB/s | 0.51 |
| decode B=256 | attention | 1.0 | 10.95 | 1570 GB/s | 1.03 |
| decode B=256 | QKV 投影 | 236 | 0.2067 | 124.7 TFLOP/s | 0.53 |
| prefill T=8192 | QKV 投影 | 2234 | 3.589 | 229.8 TFLOP/s | 0.97 |
| prefill T=8192 | attention | 2048 | 2.887 | 190.5 TFLOP/s | 0.81 |

decode 的 attention 强度恒为 1（与 `decode_attention_intensity` 的 2(H/H_kv)/b 一致），B=256 时占一层时间的 94%；线性层强度约等于 B，B≈155 才越过 ridge。
prefill 的四个算子都在 ridge 右侧。decode B=4 的 QKV 投影比例 1.51>1，是因为 100 MB 权重在重复测量时部分留在 L2，真实推理不会这样。

![roofline](artifacts/figures/gpu_roofline_layer.png)

### Depthwise 因果卷积（`dwconv.jsonl`，B=4，C=768，fp32；ms）

| 设置 | naive | smem | regblock | cuDNN | FFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=16384，K=4 | 0.2785 | 0.3124 | **0.2735** | 0.8396 | 3.577 |
| N=16384，K=64 | 0.8633 | 0.5770 | **0.2776** | 1.573 | 3.581 |
| N=16384，K=1024 | 13.00 | 8.462 | **2.289** | 14.82 | 3.581 |
| K=N=512 | 0.1343 | 0.1423 | 0.0776 | 0.2768 | **0.0701** |
| K=N=8192 | 27.28 | 38.20 | 15.07 | 57.18 | **1.737** |

- K=4 时寄存器分块 kernel 达到 1472 GB/s（copy 带宽的 97%），cuDNN 的 `conv1d(groups=C)` 慢 3.07 倍。
- K=1024 时寄存器分块 45.0 TFLOP/s（直接卷积等效），比 smem 快 3.7 倍：smem 版每个 FMA 要 2 次共享内存读，停在约 12 TFLOP/s（≈共享内存带宽上限）；寄存器分块每 64 FMA 只读 23 次。
- 与 FFT 的交叉点：N=16384 时约 K≈1600，K=N 时 N≥512 用 FFT。CPU 上交叉点是 K≈16–64，GPU 上右移了 25–100 倍。
- 与 fp64 cuDNN 参考的最大误差 ≤1.4e-5（K=4096）。

![conv](artifacts/figures/gpu_dwconv_time.png)

### 训练（`train_throughput.jsonl`，`train_mqar*.jsonl`）

GPT-2-small 形状（12 层，d=768，T=1024，B=8，110.2M 参数），MFU = 6N·tokens/s ÷ 236.3 TFLOP/s（`mlsys_calc.mfu`）：

| 精度 | attention | k tokens/s | MFU（6N） | 峰值显存 GB |
| --- | --- | ---: | ---: | ---: |
| fp32 | 朴素 | 41.5 | 0.116 | 15.38 |
| fp32 | SDPA | 59.6 | 0.167 | 10.53 |
| TF32 | SDPA | 82.2 | 0.230 | 10.54 |
| bf16 autocast | 朴素 | 64.8 | 0.181 | 15.37 |
| bf16 autocast | SDPA flash | 159.6 | 0.447 | 8.27 |
| bf16 autocast | Triton（本实验，前向+反向） | 156.0 | 0.437 | 8.25 |

bf16 与 SDPA 都开时比 fp32 朴素快 3.84 倍；只开 bf16 而保留朴素 attention 只快 1.56 倍，因为朴素 attention 是访存受限的。
固定每步 16384 token 扫 T：朴素实现从 T=1024（B=16）起 OOM；SDPA 从 T=512 的 174.1k 降到 T=8192 的 121.8k tokens/s，
而按 PaLM 公式（6N + 12LTd）计入 attention 的 MFU 从 0.53 升到 0.81。12LTd 计的是非因果的全部 attention FLOPs；按因果的 6LTd 计，MFU 在 0.51–0.57 之间基本不变。

MQAR（2 层 / 1 层 GPT，d=128，4 头，batch 256，每步新采样）。先扫描 (n_keys, P, lr, wd)：
n_keys=64 时 P=4 和 P=8 在 lr 1e-3 下 6000 步都达到 1.000；lr 3e-3 卡在 0.36；P=16 和 P=32 在 6000 步内学不会（0.10–0.25）。
用选出的 n_keys=64、P=8、lr 1e-3 跑 6000 步的对照：

| 配置 | eval 准确率 | 首次 acc ≥ 0.99 | 训练时间 s |
| --- | ---: | ---: | ---: |
| 2 层，bf16 + SDPA | 1.000 | 第 1100 步 | 63.5 |
| 1 层，bf16 + SDPA | 0.172 | — | 44.8 |
| 2 层，bf16 + Triton（本实验） | 1.000 | 第 1100 步 | 75.1 |
| 2 层，fp32 + 朴素 | 1.000 | 第 1100 步 | 63.2 |

2 层模型前 1000 步与 1 层模型一样停在约 0.17 的平台，然后在 100 步内跳到 100%（induction head 形成）；1 层模型学不会，因为匹配需要「上一 token 头 + 匹配头」两步。
三种 attention 实现的曲线在跳变前完全重合，这也是 Triton 反向的端到端校验。T=32 的小模型每步约 10 ms，几乎全是启动开销，所以实现选择不影响速度。

![mqar](artifacts/figures/gpu_mqar_curves.png)

手写 associative recall 在 GPU 上复核（`train_recall.jsonl`）：24 组配置与 CPU 的准确率逐格相同，预测不一致数为 0。

## 面试讲法

**20 秒结论**：FlashAttention 用 online softmax（运行中的 max、sum 和输出缩放）把 attention 融合成分块的
一趟计算，不写出 T×T 矩阵；反向用保存的 logsumexp 重算 P。它把显存从 O(T²) 降到 O(T)，但 HBM 流量是
Θ(T²d²/M)。d=64–128 时这仍然是 Θ(T²)，在我的模型里只比朴素实现少 2 倍（因果时 4 倍）。decode 阶段的
attention 强度只有 2·(H/H_kv)/b，GQA/MQA 是靠减少 KV 字节来提速，加大 batch 没有帮助。

**白板**：
- online softmax：`m' = max(m, s)`，`l' = l·e^{m−m'} + Σe^{s−m'}`，`acc' = acc·e^{m−m'} + e^{s−m'}V`。
- 反向：`P = exp(S − lse)`，`D = rowsum(dO∘O)`，`dS = P∘(dO·Vᵀ − D)`，`dQ = dS·K/√d`，`dK = dSᵀ·Q/√d`，`dV = Pᵀ·dO`。
- 流量：朴素 `4Td + 4T²`；flash `Td + ⌈T/Br⌉·2Td + Td`。
- decode：`I = 4·H·hd·T / (2·H_kv·hd·T·b) = 2(H/H_kv)/b`。

**不能推出的结论**：
- 不能推出「FlashAttention 在 GPU 上只快 2 倍」：朴素模型只数了 S 和 P 各一次读写；实际 eager 实现还有 scale、mask、dropout 各自的 T² 读写和 kernel launch，fp32 的 S 也更大。
- 不能推出 numpy 的计时代表 GPU：numpy 版 flash 比朴素版慢，是 Python 循环的开销。
- 不能推出「训练出来的模型就是这样解 recall 的」：手写模型只证明两层 attention 足以表达这个算法（induction head）；GPU 上训练的 2 层模型学会了 MQAR、1 层学不会，与这个构造一致，但没有检查训练出的 attention 模式。

**追问**：
1. 为什么 FA2 把外层循环换成 Q 块？→ 不同 Q 块之间没有依赖，可以分给不同 thread block 并行；O 只写一次，不需要反复读写 O。
2. 反向为什么要保存 lse，而不是 m 和 l？→ 一个标量就够重算 P = exp(S − lse)，省显存，还避免了二次归一化。
3. 什么时候卷积该用 FFT？→ 核长 K 超过 log N 的若干倍时；本机 N=16384 时交叉点在 K≈16–64，短卷积直接写更快。

## 文件

```text
project1-transformer-kernels/
├── code/tfkernels/{attention,io_model,conv1d,recall}.py
├── tests/{conftest.py,test_kernels.py}
├── scripts/run_experiments.py
└── artifacts/raw/*.csv, meta.json；artifacts/figures/*.png
```
