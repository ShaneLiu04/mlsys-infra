# 硬件参数速查

白板估算只用这一张表。厂商规格里带稀疏的数字是稠密的 2 倍，面试时一律用稠密值，
并说出来。存储延迟只给数量级，不同代际差一倍很正常。计算器 `python tools/mlsys_calc.py hw`
用的是同一组数。

## GPU

| 型号 | bf16 稠密 | fp8 稠密 | HBM | HBM 带宽 | 拐点（bf16） | SM | 每 SM 共享内存上限 | L2 | 卡间互联（单向） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A100 SXM 80GB | 312 TFLOP/s | — | 80 GB | 2.04 TB/s | 153 FLOP/B | 108 | 164 KB | 40 MB | NVLink3 300 GB/s |
| H100 SXM 80GB | 989 TFLOP/s | 1979 TFLOP/s | 80 GB | 3.35 TB/s | 295 FLOP/B | 132 | 228 KB | 50 MB | NVLink4 450 GB/s |
| H200 SXM | 989 TFLOP/s | 1979 TFLOP/s | 141 GB | 4.8 TB/s | 206 FLOP/B | 132 | 228 KB | 50 MB | NVLink4 450 GB/s |
| RTX 4090 | ≈165 TFLOP/s（fp32 累加） | — | 24 GB | 1.01 TB/s | 164 FLOP/B | 128 | 100 KB | 72 MB | PCIe4 x16 ≈32 GB/s |

拐点 = 峰值算力 / HBM 带宽。算术强度低于拐点的 kernel 受访存限制，高于拐点的受算力限制。
A100 的 fp32（非 tensor core）只有 19.5 TFLOP/s，TF32 tensor core 156 TFLOP/s，所以「fp32 训练」
和「bf16 训练」差的不是 2 倍而是一个数量级。

NVLink 常被报成双向总带宽（A100 600 GB/s，H100 900 GB/s）。本仓库的公式用单向值，
对应 all-reduce 里每张卡一个方向能发出去的字节数。

## 互联与网络

| 链路 | 单向带宽 | 备注 |
| --- | ---: | --- |
| PCIe 4.0 x16 | ≈32 GB/s | 主机 ↔ GPU、无 NVLink 的消费卡之间 |
| PCIe 5.0 x16 | ≈64 GB/s | H100 PCIe 版、Grace 之外的主机 |
| InfiniBand HDR | 200 Gb/s = 25 GB/s / 端口 | A100 集群常见，每卡一端口 |
| InfiniBand NDR | 400 Gb/s = 50 GB/s / 端口 | H100 集群常见，每卡一端口 |
| 节点内 NVLink（H100） | 450 GB/s | 比跨节点 IB 快约 9 倍，所以张量并行留在节点内 |

## 存储层次的数量级（GPU）

| 层级 | 容量（H100） | 带宽量级 | 延迟量级 |
| --- | --- | --- | --- |
| 寄存器 | 256 KB / SM | — | 1 周期 |
| 共享内存 / L1 | 最多 228 KB / SM | 每 SM 每周期 128 B，全卡 ~30 TB/s 量级 | ~20–30 周期 |
| L2 | 50 MB | ~10 TB/s 量级 | ~200 周期 |
| HBM | 80 GB | 3.35 TB/s | ~500 周期 |
| 主机内存（经 PCIe） | 数百 GB | 32–64 GB/s | 微秒 |

## CPU（CS149 语境）

| 量 | 数量级 |
| --- | --- |
| L1 命中 | ~4 周期（~1 ns） |
| L2 命中 | ~12 周期 |
| L3 命中 | ~40 周期 |
| DRAM | ~100 ns，~200–300 周期 |
| 一个 DDR5 通道 | ~40–50 GB/s |
| AVX2 / AVX-512 宽度 | 8 / 16 个 fp32 lane |
| 跨核 cache line 迁移（coherence miss） | 数十到上百 ns |
| 线程创建 | ~10 µs 量级，所以线程池要复用 |

## 模型

由 `tools/mlsys_calc.py` 按结构算出，测试里和官方参数量对过。KV 按 bf16、每 token、全层计。

| 模型 | 层 | d_model | 头 / KV 头 | 参数量 | KV / token |
| --- | ---: | ---: | --- | ---: | ---: |
| GPT-2 124M | 12 | 768 | 12 / 12 | 0.124 B | 36 KB |
| Qwen2.5-0.5B | 24 | 896 | 14 / 2 | 0.49 B | 12 KB |
| Llama-2-7B | 32 | 4096 | 32 / 32 | 6.74 B | 512 KB |
| Llama-3-8B | 32 | 4096 | 32 / 8 | 8.03 B | 128 KB |
| Llama-3-70B | 80 | 8192 | 64 / 8 | 70.6 B | 320 KB |

## 十个要背下来的数

1. H100 bf16 稠密 ≈ 1 PFLOP/s，HBM ≈ 3.35 TB/s，拐点 ≈ 300 FLOP/B。
2. A100 bf16 312 TFLOP/s，HBM 2 TB/s，拐点 ≈ 150 FLOP/B。
3. 混合精度 Adam 训练静态显存 16 字节/参数：7B 要 112 GB，单卡放不下。
4. bf16 推理权重 2 字节/参数：70B 要 140 GB，至少两张 80 GB 卡。
5. Llama-3-8B 的 KV 每 token 128 KB，8K 上下文一条序列 1 GB。
6. batch=1 的 decode 算术强度约 1 FLOP/B，H100 上 8B 模型上限约 200 token/s。
7. 训练 FLOPs ≈ 6ND。8B 模型训 15T token 是 7.2×10²³ FLOP。
8. Ring all-reduce 每卡收发 2(p−1)/p × S，和卡数几乎无关。
9. 流水线气泡 (p−1)/(m+p−1)：8 级、32 个 micro-batch 时是 18%。
10. 节点内 NVLink 比跨节点 IB 快一个数量级，张量并行不出节点。

## 本仓库实测（RTX 5090 / Xeon 6459C）

上面是规格。下面是本仓库实验在同一台机器上量到的数，面试时可以说「我测过」。
GPU 规格：RTX 5090，170 SM，32 GB GDDR7，显存带宽规格 1792 GB/s。CPU：Xeon Gold 6459C，每个实验固定 12 个物理核。

| 量 | 实测 | 占规格 | 出处 |
| --- | ---: | ---: | --- |
| 显存带宽，saxpy kernel（2²⁸） | 1578 GB/s | 88% | [A3](../solutions/cs149/asst3-scan-render/README.md) |
| 显存带宽，triad（每个数组 ≥256 MiB） | 1549 GB/s | 86% | [A4](../solutions/cs149/asst4-dnn-accelerator/README.md) |
| 显存带宽，`copy_` / D2D | 1518–1522 GB/s | 85% | [P2](../solutions/cs229s/project2-kv-speculative/README.md)、[分布式训练](../solutions/cmu15849/distributed-training/README.md) |
| bf16 / fp16 大 GEMM | 223–238.7 TFLOP/s | 名义稠密 209.5 的 1.07–1.14 倍 | [P2](../solutions/cs229s/project2-kv-speculative/README.md)、[推理显存](../solutions/cmu15849/serving-memory/README.md) |
| FP8 / int8 大 GEMM（M=8192） | 456.2 / 679.4 TOP/s | — | [编译与压缩](../solutions/cmu15849/compiler-compression/README.md) |
| PCIe H2D / D2H（pinned） | 56.5 / 54.1 GB/s | PCIe 5.0 x16 的 88% | [A3](../solutions/cs149/asst3-scan-render/README.md) |
| PCIe（pageable） | 10.5 GB/s | — | 同上，多一次 CPU staging 拷贝 |
| saxpy 端到端（含 PCIe） | 52–54 GB/s | kernel 的 3.4% | 同上 |
| CPU DRAM（saxpy，含 write-allocate） | 1 线程 23 GB/s，12 线程 173 GB/s | — | [A1](../solutions/cs149/asst1-cpu-parallel/README.md) |
| 线程创建（spawn） | 每线程约 16 µs | — | [A2](../solutions/cs149/asst2-task-scheduler/README.md) |
| 线程池唤醒：自旋 / 条件变量 | 0.783 / 10.8 µs | — | 同上 |
| 共享原子计数器 `fetch_add` | 约 15 ns/次，与线程数无关 | 一次 cache line 迁移 | 同上 |
| false sharing（12 线程，同一行 vs 各占一行） | 113 vs 5.89 ns/次 | 慢 19 倍 | 同上 |

几个读法：
- 带宽类 kernel 在这张卡上能做到规格的 85–88%，再高通常是测量错误（工作集落进了 96 MB L2）。
- GEMM 峰值高于名义值，因为消费卡的 boost 频率高于算名义值时用的频率；算 MFU 时要说清分母用的是哪一个。
- 只要数据要过 PCIe，端到端就只剩 kernel 的几个百分点，所以 GPU 上的数据要常驻。
- 单核 DRAM 23 GB/s 受 Little 定律限制（每核在途 miss 数有限），不是内存通道的上限。
