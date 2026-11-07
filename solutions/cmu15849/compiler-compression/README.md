# 15-849 编译与压缩：自动微分、图 IR 与融合、调度搜索、量化、剪枝、LoRA

CMU 15-849（Spring 2022）是论文研讨课，没有编程作业。本目录是其必读论文的最小复现，
不是官方作业。CPU 部分（下文前半）在 4 物理核 / 8 逻辑核的 Windows Server 上运行：HBM 流量、kernel 时间与调度代价都是
模型输出（参数来自 `tools/mlsys_calc.py` 的 H100），**不是 GPU 上的测量**；量化、剪枝、
LoRA 的误差与精度是 CPU 上的真实计算。**GPU 部分**（[跳到 GPU 实测](#gpu-实测rtx-5090)）在 RTX 5090 上测量了这些模型对应的真实现象。

完整报告（中文，含推导与面试问答）：[compiler_compression_report.pdf](../../../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)

## 一句话结论

Transformer MLP 块把 bias/GeLU/残差融进 GEMM epilogue 后 HBM 流量降 49%，但 roofline 时间只降 16%，
因为两个 GEMM 是计算受限的；8/1024 个 50× 激活离群通道让 per-tensor W8A8 误差从 2.3% 涨到 10.2%，
LLM.int8 降到 0.49%、SmoothQuant 降到 2.0%；GPTQ 只在输入相关时有用（3.5% vs RTN 25.3%）。

## 对应论文

| 论文 | 实现 | 简化 |
| --- | --- | --- |
| 自动微分（框架模块；Baydin 综述、PyTorch autograd） | `autodiff.py`：numpy 张量反向模式，迭代拓扑序，add/mul/matmul/relu/exp/log/sum/softmax-CE，广播求和 | 无高阶导、无 in-place |
| TVM / XLA 图层优化 | `graph_ir.py`：常量折叠、CSE、逐元素融合、GEMM epilogue 融合、解释器 | 单输出融合组；LayerNorm 不融合 |
| TASO (Jia 2019) | `merge_shared_matmuls`：X@W1, X@W2 → slice(X@[W1\|W2]) | 只实现这一条替换，无代价驱动的搜索 |
| Halide / TVM / Ansor | `schedule_search.py`：(循环序, tm, tn, tk) 空间 + 缓存模型；grid / random / 进化搜索 | 单 SM 解析代价模型，无学习型代价模型，无多级分块 |
| Deep Compression (Han 2016) | 剪枝 + k-means 16 中心权重共享 | 无 Huffman 编码 |
| LLM.int8 / SmoothQuant / GPTQ | absmax int8/int4（per-tensor/channel/group128）、混合分解、α=0.5 平滑、Cholesky 误差补偿 | 量化后反量化在 fp64 中模拟；GPTQ 无分组、无 act-order |
| NVIDIA 2:4 / 幅值剪枝 | 非结构化 vs N:M；小 MLP 剪枝 ± 微调 | 教师网络合成数据 |
| LoRA (Hu 2021) | `LoRALinear`：B=0 初始化、α/r 缩放、合并 | 单层 |

## 怎么跑

```powershell
cd solutions/cmu15849/compiler-compression
python -m pytest -q tests          # 29 passed + 4 skipped（无 CUDA 时跳过 GPU kernel 测试），约 9 s
python scripts/run_experiments.py  # 约 65 s（剪枝与 LoRA 8 进程并行）
```

## 结果

### 自动微分（`autodiff_check.csv`）

两层 MLP + softmax-CE：梯度与 torch 最大差 ≤5.6e-17，与中心差分最大差 ≤2.3e-10。5000 层链式乘法反向不爆栈（测试）。

### 图融合（T=4096, d=4096, d_ff=16384, bf16，`fusion_totals.csv`，`fusion_kernels.csv`）

所有 pass 前后解释器输出最大差为 0（融合）或 1.7e-18（matmul 合并），见 `graph_pass_checks.csv`。

| 变体 | kernel 数 | HBM 流量 | 非 GEMM 流量 | roofline 时间 |
| --- | ---: | ---: | ---: | ---: |
| 不融合 | 7 | 1.376 GB | 0.772 GB | 1342 µs |
| 逐元素融合（add+gelu、add+add） | 5 | 1.040 GB（−24%） | 0.436 GB | 1242 µs（−7.5%） |
| + GEMM epilogue | 3 | 0.705 GB（−49%） | 0.067 GB | 1132 µs（−16%） |

每个 GEMM 556 µs 且计算受限，占总时间的 83%；融合省下的是访存受限的小 kernel。

![fusion](artifacts/figures/fusion_traffic.png)

TASO 式 QKV 合并（Llama-3-8B，`qkv_merge.csv`）：T=4096 时 GEMM 流量 201.3→134.2 MB（输入只读一次），
时间不变（208 µs，计算受限）；T=1（decode）时流量 50.37→50.35 MB，只省 2 次 kernel 启动。

### 调度搜索（2048³ GEMM，单 H100 SM：7.49 TFLOP/s、25.4 GB/s、228 KB，`schedule_search.csv`，`schedule_top.csv`）

空间 3072 个调度，只有 8 个在最优 5% 以内。最优 ijk / 256×256×64：5.99 ms = 2.87 TFLOP/s，
仅 38% 峰值（计算下限 2.29 ms）——C 块强度 tm·tn/(tm+tn)=128 FLOP/B 低于脊点 295。4×4×4 分块需 3023 ms。

| 方法 | 达到 5% 以内所需评估次数（中位 / P90，200 次） |
| --- | --- |
| grid（按枚举顺序） | 434（确定性） |
| random | 244 / 649 |
| 进化（锦标赛+交叉+变异） | 147.5 / 496 |

进化比随机快约 1.65×（中位数）；grid 的 434 完全取决于枚举顺序，不说明方法好坏。

![schedule](artifacts/figures/schedule_search.png)

### 权重量化（1024×1024，`quant_weight.csv`）

| 方案 | 高斯权重 | Student-t(4) 权重 |
| --- | ---: | ---: |
| int8 per-tensor | 1.13% | 20.6% |
| int8 per-channel | 0.79% | 1.97% |
| int8 group128 | 0.65% | 1.11% |
| int4 per-tensor | 20.6% | 97.2% |
| int4 per-channel | 14.3% | 32.1% |
| int4 group128 | 11.7% | 19.3% |
| k-means 16 中心（4 bit，整表共享） | 11.0% | 25.0% |

（输出 Y=XW 相对误差，与权重误差相差 <0.1 个百分点。）重尾使 per-tensor 失效：一个极大值决定了整表的步长。
k-means 在高斯权重上略胜 group128，在重尾上输给它（group128 每 128 个权重多一个 scale）。

### 激活离群（8/1024 通道，权重 per-channel int8，`quant_outliers.csv`）

| 离群倍数 | per-tensor 激活 | per-token 激活 | SmoothQuant α=0.5 | LLM.int8（阈值 6） | per-tensor 下非离群部分 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.28% | 2.12% | 3.29% | 2.12% | 2.28% |
| 20 | 9.11% | 4.67% | 2.59% | 1.08% | 17.5% |
| 50 | 10.2% | 5.17% | 1.99% | 0.49% | 43.6% |
| 100 | 9.73% | 5.24% | 1.61% | 0.25% | 80.8% |

意外点：整体 ‖Y‖ 误差只涨到 ~10% 就饱和，因为离群通道同时主导了 ‖Y‖；真正被毁的是普通通道的贡献（80.8%）。
per-token 只减半，因为离群是按**通道**而非按 token 出现的。无离群时 SmoothQuant 反而略差（3.29% vs 2.28%）。

![outliers](artifacts/figures/quant_outliers.png)

### GPTQ vs RTN（int4 per-channel，512×512，2048 校准 / 2048 测试 token，`gptq.csv`）

| 输入谱衰减 | RTN 测试误差 | GPTQ 测试误差 | GPTQ 校准误差 |
| ---: | ---: | ---: | ---: |
| 0（各向同性） | 26.4% | 28.2% | 24.7% |
| 0.5 | 26.1% | 13.7% | 12.8% |
| 1.0 | 25.3% | 3.5% | 3.5% |

各向同性输入下 GPTQ 在校准集略好、测试集更差：H≈cI 时没有可利用的相关性，补偿只拟合了采样噪声。GPTQ 的权重误差总是更大（0.28–0.39 vs 0.26），它优化的是输出不是权重。

### 剪枝（`pruning_matrix.csv`，`pruning_mlp.csv`）

1024² 重尾矩阵，输出误差：非结构化 50% 为 20.4%，2:4 为 29.0%，4:8 为 25.1%；非结构化 90% 为 62.0%。
小 MLP（教师网络 10 类，稠密 66.2%，稠密再训 3 轮 67.0%）：

| 方法 | 稀疏度 | 不微调 | 微调 3 轮 |
| --- | ---: | ---: | ---: |
| 幅值 | 0.5 | 63.6% | 67.1% |
| 幅值 | 0.8 | 50.5% | 66.6% |
| 幅值 | 0.9 | 36.6% | 63.6% |
| 幅值 | 0.95 | 17.6% | 53.0% |
| 2:4 | 0.5 | 62.9% | 67.2% |

![pruning](artifacts/figures/pruning.png)

### LoRA（`lora_fit.csv`，`lora_params.csv`）

初始输出与基座最大差 0；合并后输出最大差 3.3e-15。d=256、真实 ΔW 秩 8、1500 步：
r=4 误差 0.434（Eckart–Young 下界 0.424），r=8 误差 0.044，可训参数 4096 vs 全量 65,536（6.25%）。
Llama-3-8B：r=8 只挂 q、v 为 3.41M 参数（0.042%）；r=16 挂全部线性层为 41.9M（0.52%）。

## 面试讲法

**20 秒结论**：融合省的是访存受限 kernel 的 HBM 往返，大 GEMM 计算受限时端到端收益远小于流量收益（−49% 流量，−16% 时间）。
量化的敌人是动态范围：重尾权重要细粒度 scale，按通道出现的激活离群要么拆出去（LLM.int8），要么挪到权重上（SmoothQuant）。
GPTQ 用 H=XᵀX 把舍入误差推给还没量化的列，只有输入有相关性时才有收益。

**白板**：
- 融合省的字节 = 每个被消掉的中间张量 ×2（写+读）：bias→GeLU 的 [T, d_ff] 与 bias→残差的 [T, d]，2·2·(T·d_ff+T·d) = 0.336 GB（测试断言）。
- GEMM 分块强度：C 常驻、A/B 流过时 ≈ tm·tn/(tm+tn)（bf16 每元素 2 B，2 FLOP/MAC）；受限于 SRAM：2(tm·tk+tk·tn+tm·tn) ≤ 228 KB。
- SmoothQuant：s_j = max|X_j|^α / max|W_j|^(1−α)，XW = (X·diag(s)⁻¹)(diag(s)·W) 精确成立。
- LoRA：参数 r(d_in+d_out)，ΔW = (α/r)·BA，B=0 保证初始等价。

**不能推出的结论**：
- 不能把 −16% 当真实加速：模型无重叠、无 L2、epilogue 无额外寄存器压力。
- 不能说 per-token 能解决离群：它只减半误差（5.17% vs 10.2%）。
- 不能用 2:4 的 MLP 结果说明 LLM 上无损：教师任务简单，且这里有 3 轮微调。
- 调度搜索结论只在本代价模型内成立；真实 Ansor 依赖学习型代价模型与实测。

**追问**：
1. 为什么 FlashAttention 不能靠通用逐元素融合得到？——softmax 有行归约，需要 online-softmax 改写算法，不是图层替换。
2. W4A16 与 W8A8 选哪个？——decode 访存受限选 W4A16（读权重减半）；大 batch prefill 计算受限选 W8A8（int8 tensor core）。
3. 2:4 的加速从哪来？——稀疏 tensor core 跳过一半 MAC，元数据每 4 个只需 2 bit；非结构化稀疏在 GPU 上难以兑现。

## GPU 实测（RTX 5090）

RTX 5090（sm_120，170 SM，32 GB GDDR7 1.79 TB/s，96 MB L2，共享内存每 block 99 KB），PyTorch 2.8.0+cu128、Triton 3.4.0、
transformers 5.21。原始结果在 `artifacts/raw/gpu/*.jsonl`，派生数字（加速比、Spearman、实验室模型在 5090 参数下的预测）在
`artifacts/raw/gpu/summary_gpu.json`。详细分析见[报告 PDF](../../../reports/cmu15849-compiler-compression/compiler_compression_report.pdf)。

```bash
# 服务器（每步各自持有 GPU 锁，taskset 2 核）
CORES=47,48 bash gpu/run_gpu.sh            # chains mlp triton dynamic matmul qspeed quant_gpt2 quant_gpt2m prune_ppl prune_speed lora
python scripts/plot_gpu.py                 # 本地，从 raw 文件画 16 张图
python -m pytest -q tests                  # 本地 29 passed + 4 skipped（CUDA）；服务器 33 passed
```

| 问题 | 关键结果 | 原始文件 |
| --- | --- | --- |
| Q1 融合（访存受限链，16384×4096 bf16） | `torch.compile` 加速 = eager/融合字节比：RMSNorm 9.86×（字节比 10.0×），softmax 5.05×（5.0×），残差链 2.68×（2.60×）；融合后 1.41–1.61 TB/s | `fusion_chains.jsonl` |
| Q1 MLP（T=4096, d=1024） | eager 419.4 µs，default 414.4 µs，max-autotune-no-cudagraphs 356.8 µs（−15%，编译 0.59→6.45 s）；实验室图 IR 预测 435.7 / 338.1 µs | `fusion_mlp.jsonl` |
| Q1 工程代价 | max-autotune 的 CUDA graph 输入拷贝让 RMSNorm 慢 2.08×（364.0 vs 174.9 µs）；10 个序列长度：静态 8 个图 4.46 s，automatic 2 个图 0.99 s，`mark_dynamic` 1 个图 0.54 s | `fusion_chains.jsonl`, `fusion_dynamic.jsonl` |
| Q2 Triton matmul 网格（192 配置） | 8192³：网格最优 221.5 TFLOP/s（128×128×32）vs cuBLAS 214.3 vs autotune 214.7；M=256 时 cuBLAS 175.0 > 网格 165.0 > autotune 152.3；代价模型 Spearman ρ：实验室模型 0.65/0.36/0.10，波次 roofline 0.26/0.21/0.72 | `matmul_search.jsonl` |
| Q3 量化 ppl（GPT-2 124M / 355M，fp16 29.96 / 21.72） | W8 per-channel 29.98 / 21.72；W4 g128 RTN 33.27 → GPTQ 31.20；W3 g128 RTN 65.92 → AWQ 式 53.35 → GPTQ 37.24；W8A8 per-tensor 49.94，per-token 30.18，LLM.int8 30.06；SmoothQuant α=0.5 407.25，α=0.8 30.49 | `quant_ppl_gpt2*.jsonl` |
| Q3 速度 | M=8192：bf16 226.9、FP8 456.2、int8 679.4 TOP/s；batch-1 GEMV 16384×8192：bf16 163.0 µs，W8 82.5 µs，W4 62.2 µs，先反量化再 matmul 3688 µs | `quant_speed.jsonl` |
| Q4 剪枝（GPT-2 124M，一次性） | 50%：幅值 538.35 ppl，Wanda 47.21；2:4：幅值 735.95，Wanda 116.38；2:4 cuSPARSELt 大 GEMM 1.54–1.58×，M≤2048 更慢（~445 µs 下限）；CUTLASS 后端不支持 sm_120 | `prune_*.jsonl` |
| Q5 LoRA（GPT-2 355M，150 步） | 峰值显存：全量 16.07 GiB，LoRA r=16 12.17，int8 基座 11.91，int4 基座 11.77；步时 128.4 / 104.8 / 109.6 / 115.5 ms；val loss 2.891 / 2.874 / 2.876 / 2.914；合并后 logits 最大差 0.0045（最大值 340.6） | `lora_ft.jsonl` |

**融合加速就是字节比**：大尺寸下每条链的 Inductor 加速与 eager/理想字节比相差不到 7%，融合版本都顶在 ~1.53 TB/s 的带宽墙上；
黄色区域（理想字节 < 96 MB L2）里“带宽”超过 GDDR7 峰值，是 L2 命中，不是 bug。

![fusion bandwidth](artifacts/figures/gpu_fusion_bandwidth.png)

**分块地形与代价模型**：256×256 tile 在三个形状上都最差（寄存器溢出 + C 块太少），最优 tile 随形状变化；
复用型模型在方阵上排得还行，在 M=256 上失效，波次模型恰好相反——所以 Ansor 用学习型模型 + 实测。

![matmul heatmap](artifacts/figures/gpu_matmul_heatmap.png)

**量化：位宽越低越需要数据**。int3 时 GPTQ 把 RTN 的 65.92 拉回 37.24；SmoothQuant 的 α 选错比不平滑还差。

![quant ppl](artifacts/figures/gpu_quant_ppl.png)

**batch 1 下 weight-only 量化按字节加速**：大矩阵上 W8 1.8–2.0×（接近理论 2×），W4 2.3–2.7×（理论 3.76×，被解包开销限制）；不融合反量化则慢 10–23×。

![gemv](artifacts/figures/gpu_weight_only_gemv.png)

**LoRA 省的是优化器状态**：全量与 LoRA 的峰值差 3.97 GiB = fp32 梯度 + AdamW 两个矩；剩下 ~10.7 GiB 是激活，rank 4→64 峰值只涨 0.35 GiB。

![lora cost](artifacts/figures/gpu_lora_cost.png)

GPU 部分不能推出的结论：W8A8/SmoothQuant/LLM.int8 是精度模拟（fp16 GEMM），没有端到端 int8 速度；GPT-2 的离群远弱于 6.7B+ 模型；
AWQ 为简化版、GPTQ 无 act-order；LoRA 只训 150 步单种子，0.01 量级的 loss 差在噪声内；cuSPARSELt 的 ~445 µs 下限可能是软件路径问题。

## 换到 7B：Qwen2.5-7B 上的同一套量化（RTX 6000D）

GPT-2 的激活离群远弱于 6.7B+ 模型，所以把同一份 `gpu/quant_llm.py` 原样跑在 Qwen2.5-7B（bf16 载入，wikitext-2 test 29.9 万 token，
校准 128 × 1024 token）。原始结果 `artifacts/raw/rtx6000d/quant_ppl_Qwen2.5-7B.jsonl`，图与汇总由 `scripts/plot_qwen7b.py` 生成（`quant_scale_summary.csv`）。

![quant scale](artifacts/figures/gpu_quant_scale.png)

| 方法 | GPT-2 124M | GPT-2 355M | Qwen2.5-7B（fp16 7.639） |
| --- | ---: | ---: | ---: |
| W8 逐张量 / 逐通道 | +14.6% / +0.06% | +7.8% / −0.005% | +5.0% / +0.11% |
| W4 g128：RTN / AWQ / GPTQ | +11.1 / +8.0 / +4.1% | +6.1 / +5.9 / +3.7% | +5.9 / +3.8 / +2.9% |
| W3 g128：RTN / AWQ / GPTQ | +120 / +78 / +24% | +49 / +36 / +17% | +75 / +23 / +15% |
| W4 逐通道：RTN / GPTQ | +28 / +8.8% | +18 / +6.8% | +34 / +9.1% |
| W8A8 逐张量 / 逐 token | +67 / +0.71% | +10 / +0.31% | **+109** / +2.5% |
| SmoothQuant α=0.5 / 0.8（逐张量） | +1260 / +1.8% | +22 / +1.2% | +58 / +32% |
| SmoothQuant α=0.5 / 0.8（逐 token） | +0.64 / +0.65% | +0.87 / +0.23% | +1.5 / +1.3% |
| LLM.int8（阈值 6） | +0.32% | +0.23% | **+0.27%** |

- **权重量化随规模变容易，激活量化随规模变难**：W4 g128 GPTQ 从 +4.1% 降到 +2.9%，W3 GPTQ 从 +24% 降到 +15%；
  而 W8A8 逐张量从 GPT-2 355M 的 +10% 涨到 7B 的 +109%。下图解释了原因：Qwen 第 26 层 `mlp.down_proj` 的输入里有一个通道 |x| = 6368，
  是该层通道最大值中位数的 358 倍；逐张量的激活步长由它决定，其余通道只剩几个量化级。
- **7B 上方法的排序被拉开**：W3 g128 时 RTN +75%、AWQ +23%、GPTQ +15%；GPT-2 上 AWQ 与 RTN 的差距没这么大。AWQ 按激活幅度缩放通道，
  在离群明显的模型上才有东西可缩放。
- **SmoothQuant 逐张量救不了 7B**：α=0.8 仍 +32%，因为离群 6368 太大，平滑把难度挪给权重后权重也装不下；逐 token 激活 + 平滑才回到 +1.3%。
  LLM.int8 用混合精度把 |x| > 6 的列留在 fp16，+0.27% 是全部 W8A8 方法里最好的——代价是平均 700 列、最多 18702 列（某层几乎整层）走 fp16。
- **GPTQ 在 7B 上需要 float64 Cholesky**：第一次运行在 `down_proj`（18944 维输入）上报「leading minor of order 18004 is not positive-definite」，
  1% 阻尼的 fp32 Hessian 被离群通道撑得条件数太大。`gptq_quantize` 现在用 float64 分解，失败时把阻尼放大 10 / 100 倍重试；三个 GPTQ 配置各 967 s，AWQ 473–566 s。

不能推出：perplexity 协议是 HF ctx 1024，与 edge-deploy llama-cpp-7b 的 llama-perplexity ctx 512 不同，两边数字不能直接比；W8A8 全部是精度模拟（fp16 GEMM）；
只有一个 7B 模型，离群通道的位置与幅度是 Qwen2.5 的特性。

## 文件

```text
compiler-compression/
├── code/compkit/        autodiff, graph_ir, schedule_search, quantization, pruning, lora
├── gpu/                 common, triton_kernels（融合 RMSNorm/bias+GELU、分块 matmul、W8/W4 GEMV）,
│                        fusion_bench, matmul_search, llm_utils, quant_llm（GPTQ/AWQ/SmoothQuant/LLM.int8）,
│                        quant_speed, prune_bench, lora_ft, run_gpu.sh
├── tests/               33 个用例（test_gpu.py：7 个 CPU 可跑的算法测试 + 4 个 CUDA kernel 测试）
├── scripts/run_experiments.py, plot_gpu.py
└── artifacts/raw/*.csv, summary.json；artifacts/raw/gpu/*.jsonl, summary_gpu.json；artifacts/figures/*.png
```
