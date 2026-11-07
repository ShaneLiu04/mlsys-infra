# 15-849 服务与显存：PagedAttention、iteration-level 调度、chunked prefill

CMU 15-849（Spring 2022）是论文研讨课，没有编程作业。本目录是按课程方向补的最小复现，
不是官方作业；vLLM、Sarathi 晚于该课程，但推理岗面试默认要会。分两部分：

- **CPU 部分**（4 物理核 / 8 逻辑核的 Windows Server）：模型与硬件参数来自 `tools/mlsys_calc.py`（Llama-3-8B、H100 SXM 80GB），
  **调度器里的时间是 roofline 模型输出，不是 GPU 测量**；没有建模 kernel launch、CPU 调度、采样开销，绝对延迟偏乐观。
- **GPU 部分**（下文「GPU 实测」，RTX 5090 + Qwen2.5-0.5B bf16）：真实引擎上验证同一批结论，并测出模型漏掉的东西——
  小模型 decode 受 CPU 派发限制（CUDA graph 2.16 ms，带宽下限 0.65 ms）；分页间接寻址开销 <1%；KV 浪费 86.8% → 1.26%、
  并发 7.0×；chunked prefill 把长 prompt 下的 P99 卡顿从 33.5 ms 压到 17.6 ms。

完整报告：[serving_memory_report.pdf](../../../reports/cmu15849-serving-memory/serving_memory_report.pdf)

## 一句话结论

按最大长度连续预留 KV 浪费 85.6%，80GB H100 只能并发 104 条；分页（block=16）浪费 1.25%，
并发 714 条。iteration-level 调度把饱和吞吐从 0.89k 提到 14.0k tok/s；chunked prefill 把最长一次
迭代从 121 ms 压到 9.2 ms，并把饱和吞吐再提到 16.2k tok/s。

## 对应论文

| 论文 | 实现 | 简化 |
| --- | --- | --- |
| vLLM / PagedAttention (Kwon 2023) | `block_manager.py`：定长块、块表、refcount、fork + copy-on-write、swap out/in（组内保持共享）、recompute 抢占；按块表 gather 的注意力与逐块 online-softmax 版本 | 单层单 query；CPU 块池大小固定；无前缀缓存 |
| vLLM 论文 §3 显存浪费分析 | `waste.py`：最大长度预留 / 已知最终长度 / 分页 {8,16,32,64} 的生命周期平均浪费与并发数 | 只算内部碎片与预留；外部碎片未模拟 |
| Orca (Yu 2022) | `scheduler.py::simulate_continuous(chunked=False)`：每次迭代重新组批，新请求整段 prompt 与 decode 混在一步 | 无 selective batching 细节；max 8192 prefill token/步 |
| Sarathi-Serve (Agrawal 2024) | `chunked=True`：每步 512 token 预算，先 decode 再填 prefill 块 | 固定块大小，无动态预算 |
| FasterTransformer 式静态批 | `simulate_static`：按最长 prompt 填充，跑到最长输出结束 | 批大小受 4096 预留上限约束 |

步时间：`mlsys_calc.roofline_time`，权重每步读一次、每个上下文 token 的 KV 读一次，
FLOPs = 2N·token + 4·L·d·(query,上下文)对。纯 decode 时与 `mlsys_calc.decode_step_time` 一致（测试）。

## 怎么跑

```powershell
cd solutions/cmu15849/serving-memory
python -m pytest -q tests          # 无 GPU：23 passed, 4 skipped（20 个 CPU 用例 + 3 个 GPU 代码的 CPU 用例；4 个 Triton 用例 skip）
python scripts/run_experiments.py  # 约 15 s（8 进程并行扫描），写 artifacts/raw 与 artifacts/figures
```

## 结果

### 显存浪费（10k 请求，`kv_waste.csv`，`summary.json`）

长度分布：prompt ~ LogNormal(ln256, 1.0) 截断 [8,3072]，output ~ LogNormal(ln128, 0.9) 截断 [4,1024]，
prompt+output ≤ 4096；实测 prompt 均值 415 / P99 2583，output 均值 187 / P99 1019。
KV/token = 131,072 B；权重 16.06 GB；KV 预算 = 0.9×80 GB − 权重 = 55.9 GB = 426,784 token。

| 方案 | 生命周期平均浪费 | 每序列平均占用 token | 并发序列数 |
| --- | ---: | ---: | ---: |
| 连续，按 4096 预留 | 85.6% | 4096 | 104 |
| 连续，预知最终长度 | 23.5% | 771 | 553 |
| 分页 block=8 | 0.59% | 594 | 718 |
| 分页 block=16 | 1.25% | 598 | 714 |
| 分页 block=32 | 2.56% | 606 | 704 |
| 分页 block=64 | 5.07% | 622 | 686 |
| 理想（逐 token） | 0 | 590 | 723 |

即使「预知最终长度」也浪费 23.5%：生成过程中长度从 prompt 涨到终值，前期预留空着。

![waste](artifacts/figures/kv_waste.png)

### 调度（3000 请求，Poisson，`scheduler_sweep.csv`；SLO：TTFT ≤ 1 s 且 TPOT ≤ 50 ms，事先固定）

| 指标 | static | continuous (Orca) | chunked (Sarathi) |
| --- | ---: | ---: | ---: |
| 饱和吞吐（λ=160） | 891 tok/s（4.8 req/s） | 14,050 tok/s（75.8 req/s） | 16,156 tok/s（87.2 req/s） |
| 峰值 goodput | 0.97 req/s（λ=2） | 70.2 req/s（λ=80） | 85.7 req/s（λ=100） |
| λ=100 时 P99 TTFT | 583 s | 4.58 s | 0.595 s |
| λ=160 时 P99 TPOT | 19.1 ms | 23.3 ms | 8.97 ms |
| 最长单次迭代 | 5,448 ms | 121 ms | 9.2 ms |

static 在 λ=2 时 SLO 达标率只有 47.7%（TTFT 均值 1.47 s，被填充 prefill 和整批等待拖住）。
50 ms 的 TPOT SLO 在本模型里对 continuous 从未成为约束，goodput 的崩塌全部来自 TTFT 排队；
chunked 的优势体现在 TPOT 尾部（9 vs 23 ms）和更晚饱和。满预算下所有运行抢占次数为 0。

![latency](artifacts/figures/latency_vs_load.png)

### KV 压力与抢占（λ=60，`kv_pressure.csv`）

| KV 预算 | continuous 抢占 / goodput | chunked 抢占 / goodput |
| ---: | ---: | ---: |
| 55.9 GB | 0 / 55.7 | 0 / 56.4 |
| 6.55 GB | 73 / 6.9 | 39 / 51.1 |
| 3.28 GB | 309 / 1.9 | 286 / 2.2 |

6.55 GB 时 continuous 因抢占重算 26,480 个 prefill token，goodput 掉到 6.9 req/s；chunked 只抢占 39 次。

## GPU 实测（单卡 RTX 5090，Qwen2.5-0.5B bf16）

完整报告：[serving_memory_report.pdf](../../../reports/cmu15849-serving-memory/serving_memory_report.pdf)。
代码在 `gpu/`：`qwen.py` 从 safetensors 自写 Qwen2 前向（展平不规则 token 批、attention 回调；fp32 下与 HF logits
最大差 1.5e-4、argmax 100% 一致），`paged_attn.py` Triton 分页 decode kernel，`engine.py` 分页 KV + iteration-level
调度 + chunked prefill 的真实引擎，`serve_bench.py` 五个子命令。原始结果 `artifacts/raw/gpu/*.jsonl`，
图由 `scripts/plot_gpu.py` 生成（`srv_*.{png,pdf}`），并用实测 5090 参数重跑 lab 调度模拟器做对照。

| 实验 | 关键实测 | 与模型对照 |
| --- | --- | --- |
| B1 decode | CUDA graph B=1 2.16 ms/步，eager 恒为 8.2–9.0 ms（CPU 下发）；步时间翻倍的 batch：ctx 128/512/2048/8192 → 512/256/128/64 | 带宽 roofline 0.65 ms，graph 实测是模型的 2.7–3.3 倍；拐点模型 266,846/ctx |
| B1 prefill | 8192 token 173.9 TFLOP/s（实测 GEMM 峰值 238.7 的 73%）；512 token 仅 57.1 TFLOP/s | ≳2k token 才计算受限 |
| B2 分页 kernel | 随机物理块 vs 按序块表时间差 <1%；比 torch gather 快 2.2–10.7×，比预留 4096 的 SDPA 快 3.3–6.9×；32/8/128、B=256 有效带宽 1039 GB/s | block 8–64 时间差 ±15%，无单调趋势 |
| B2 浪费（4 GiB） | 按 4096 预留 86.8%（并发 85）；分页 block=16 1.26%（并发 593，7.0×）；变长分配 reserved/allocated 平均 1.68 | `waste.py` 85.6% / 1.25%，差 ≤1.2 个百分点；(b−1)/2 token/序列 |
| B3 引擎 vs static | static 饱和于 5.36 req/s（TTFT P50 13.7–24.5 s）；continuous λ=40 时 23.2 req/s、TTFT P99 38 ms | 闭式 static 容量 5.53 req/s；continuous 容量为 static 的 7.5–15× |
| B3 chunked prefill | 10% prompt=3072 token、λ=20：逐请求最大卡顿 P99 33.5→17.6 ms，长 prompt TTFT 34→110 ms，吞吐 −2.6%；lab 短 prompt 分布下无收益 | Sarathi 命题在长 prompt 下成立 |
| B3 成本模型 | 引擎迭代 = 9.15 ms + 1.03 × roofline | lab 模型斜率对、缺 ~9 ms/迭代固定开销（中位数低估 12.2×） |

主要结论：小模型 decode 的瓶颈依次是 CPU 下发、kernel 数、最后才是 HBM 带宽；分页间接寻址几乎免费，
浪费模型在真实分配下精确成立；continuous batching 的优势在真实 GPU 上同样成立，而 chunked prefill
是用长 prompt 的 TTFT 换 decode 尾延迟，且强依赖“带前缀 prefill 块”的 attention 实现（布尔掩码 SDPA
退回 math 路径，每块多 5.8 ms；保留为 `engine_chunked_maskedsdpa.jsonl` 消融）。

![decode roofline](artifacts/figures/srv_decode_roofline.png)

![分页 kernel](artifacts/figures/srv_paged_blocksize.png)

![实测浪费](artifacts/figures/srv_waste_measured.png)

![引擎 vs 负载](artifacts/figures/srv_engine_load.png)

![decode 卡顿](artifacts/figures/srv_stall_timeline.png)

复现（需 CUDA；模型放在 `$HF_HOME/hub/models--Qwen--Qwen2.5-0.5B/snapshots/main`）：

```bash
cd solutions/cmu15849/serving-memory
python gpu/serve_bench.py verify && python gpu/serve_bench.py roofline
python gpu/serve_bench.py paged && python gpu/serve_bench.py waste
for p in static continuous chunked; do python gpu/serve_bench.py engine --policy $p --rates 5,10,20,40,80; done
for p in continuous chunked; do python gpu/serve_bench.py engine --policy $p --rates 10,20 --long-frac 0.1; done
python scripts/plot_gpu.py
```

### 对照 vLLM 0.11（RTX 6000D，同一负载，`engine_vllm_*.jsonl`、`vllm_compare.csv`）

`gpu/vllm_bench.py` 用同一个 Qwen2.5-0.5B bf16、同样的长度分布与 Poisson 到达（同种子）、同样 2 GiB KV 预算、最多 128 序列、8192 批 token，
输出强制到采样长度。V1 总是对长 prefill 分块，所以最接近的自写策略是 `chunked`。

![vllm compare](artifacts/figures/rtx6000d/vllm_compare.png)

| λ（req/s） | 自写 continuous tok/s / TPOT | 自写 chunked | vLLM eager | vLLM CUDA graph |
| ---: | --- | --- | --- | --- |
| 10 | 1666 / 10.9 ms | 1685 / 10.6 ms | 1778 / 5.7 ms | 1804 / 2.5 ms |
| 20 | 2849 / 11.4 ms | 2857 / 11.3 ms | 3310 / 6.1 ms | 3569 / 2.7 ms |
| 40 | 4140 / 11.3 ms | 3977 / 13.0 ms | 5417 / 6.5 ms | 6666 / 3.2 ms |

- **λ = 10 时吞吐差不多**（1666 对 1804）：负载还没压满，大家都在服务到达的请求；差别全在每 token 延迟，TPOT 10.9 对 2.5 ms，4.3 倍。
- **λ = 40 时 vLLM 吞吐 1.61 倍**：每步更快，同样的到达率下排队更少、批更满。
- **把 8.1 ms 的 TPOT 差拆开**：vLLM eager 6.5 ms，自写 11.3 ms——约 5 ms 是 Python 调度与逐 kernel 下发的差（上面成本模型的「每迭代 9.15 ms 固定开销」）；
  CUDA graph 再省 3.3 ms。0.5B 模型 decode 的瓶颈按顺序是：调度/下发 → kernel 数 → 带宽，与 B1 的结论一致。
- **尾 TTFT**：自写 chunked 在 λ = 40 时 p99 181 ms，vLLM 30 ms；vLLM 的分块预算与调度顺序让新请求更快拿到首 token，自写版本的块大小没有调。
- 不能推出：这是 0.5B 模型、单卡、随机 token 的提示（前缀缓存关闭）；大模型上 GEMM 时间变长，下发开销占比会小得多，差距会收窄。

```bash
<vllm venv>/bin/python gpu/vllm_bench.py --rates 10,20,40 && <vllm venv>/bin/python gpu/vllm_bench.py --rates 10,20,40 --eager
python scripts/plot_vllm.py --card rtx6000d
```

vLLM 0.11 要求 transformers < 5；venv 若继承系统的 transformers 5.x，会在 `all_special_tokens_extended` 处报错，需在 venv 里单独装 4.x。

## 面试讲法

**20 秒结论**：KV cache 是 serving 的容量瓶颈。连续预留浪费 85%，分页把浪费压到块尾（≈ block/2 每序列），
并发提升 6.9×（104→714）。有了分页才能做 iteration-level 调度；再用 chunked prefill 把 prefill
拆进 decode 步，decode 是访存受限的，捎带 prefill 的计算几乎免费，同时消除 prefill 造成的 decode 卡顿。

**白板**：
- KV/token = 2·L·n_kv·d_head·bytes = 2·32·8·128·2 = 128 KiB；并发 ≈ 预算 / (平均活跃长度)。
- 分页浪费 ≈ (block/2)/平均长度：16/2/590 ≈ 1.4%（实测 1.25%）。
- 步时间 = max(2N·tok/peak, (W + KV)/bw)；decode batch=256、ctx=512 时 9.9 ms，仍是访存受限（`step_time_model.csv`）。

**不能推出的结论**：
- 不能把 16k tok/s 当成真实 H100 吞吐：模型假设 100% 峰值算力与带宽、零调度开销。
- chunked 吞吐更高来自 roofline 的 max(计算, 访存) 假设——混合步完美重叠；真实 kernel 重叠不完美，增益会变小。
- 不能说 block 越小越好：block=8 浪费最少，但真实 kernel 的 gather 粒度与块表开销变差，这里没建模。
- 满预算下抢占为 0，不能用来说明 swap 与 recompute 谁更好；本实验只实现了 recompute 的调度路径。

**追问**：
1. swap 还是 recompute？——短序列重算比 PCIe 搬运便宜（prefill 计算受限、吞吐高）；长序列或 beam 组共享多时 swap 更划算。
2. beam search 怎么省显存？——fork 共享 prompt 块，写入时 CoW 只复制最后一块（测试里 3 个分支只多 1 块）。
3. prefill/decode 分离（PD 分离）解决什么？——chunked 仍让两者抢同一卡；分离后各自按 TTFT/TPOT 配置并行度，代价是 KV 传输。

## 文件

```text
serving-memory/
├── code/servesim/   block_manager, workload, waste, scheduler
├── gpu/             qwen, paged_attn (Triton), engine, serve_bench（RTX 5090 实测）
├── tests/           27 个用例：refcount、CoW、swap、无泄漏、注意力逐位相等、请求守恒、抢占；
│                    test_gpu_serving.py：块表划分、gather/Triton 与参考一致、三种策略贪心 token 一致（无 CUDA 时 Triton 用例 skip）
├── scripts/run_experiments.py, plot_gpu.py
└── artifacts/raw/*.csv, summary.json, raw/gpu/*.jsonl；artifacts/figures/*.png, srv_*.{png,pdf}
```
