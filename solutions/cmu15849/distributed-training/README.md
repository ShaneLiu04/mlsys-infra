# 15-849 分布式训练：all-reduce、张量并行、ZeRO、流水线、重计算

这个目录算的是分布式训练里最常被问的几笔账：all-reduce 该用什么算法，ZeRO 每多分片一样东西要多付多少通信，
1F1B 到底省了什么，重计算用多少时间换多少显存。我先在 CPU 上用模拟器把公式逐一验证，再在单张 RTX 5090 上对上实测。
最重要的发现有两条：「谁更快」常常取决于消息大小和拓扑，比如 halving-doubling 在交换机上和 ring 一样快，到物理环上却慢 3 倍；
1F1B 并不减少气泡，它省的是在途激活。

CMU 15-849（Spring 2022）是论文研讨课，没有编程作业。本目录是其必读论文的最小复现，不是官方作业。分两部分：

- CPU 部分（4 物理核 / 8 逻辑核的 Windows Server）：多卡集群的时间是 α-β 模型（通信时间 = 启动延迟 α × 步数 + 每字节时间 β × 字节数）和事件模型的输出，
  参数取自 `tools/mlsys_calc.py` 的 H100（NVLink 单向 450 GB/s），α=5 µs 是假设值，**不是 GPU 上的测量**。真实测量只有 gloo 的 CPU all-reduce。
- GPU 部分（下文「GPU 实测」，单卡 RTX 5090）：把单卡能测的模型逐个对上实测。16 B/参数的规则实测为 16.00–16.32，
  激活显存模型误差 ≤1.3%，774M bf16 的 MFU 是 70.2%，全量重计算用 +18.9% 的时间把 batch 从 8 提到 24，另外还测了 1F1B 的气泡与在途激活。
  多卡通信（NCCL、NVLink）没有条件测，仍以模型为准。

完整报告：[distributed_training_report.pdf](../../../reports/cmu15849-distributed-training/distributed_training_report.pdf)

## 一句话结论

all-reduce 小消息看 α，大消息看 β。小消息时树和 halving-doubling 比 ring 快 2.3×，因为步数少；
大消息时 ring 与 halving-doubling 发送的字节数相同，都是 2(P−1)/P·S，但在物理环上 halving-doubling 因为链路争用慢 3×。

ZeRO-3 用 1.5× 的通信换来 1/P 的显存。1F1B 不减少气泡，只把在途激活从 m 压到 ≤p。
重计算在异构的层链上，按预算求解的 DP 比 √n 规则少 27% 的重算量。

## 对应论文

| 论文 | 实现 | 简化 |
| --- | --- | --- |
| Ring / Rabenseifner all-reduce（NCCL、Horovod 背景） | `collectives.py`：naive、ring RS+AG、递归减半-倍增、二项树；真实搬 numpy 数组，逐 rank 计字节与步数 | 单端口全双工；无流水分块；两种拓扑（交换机 / 物理环） |
| Megatron-LM (Shoeybi 2019) | `tensor_parallel.py`：列并行→行并行 MLP、按头切分 attention，显式 `f`/`g` 算子 | 单进程模拟 T 个分片；无 sequence parallel |
| ZeRO (Rajbhandari 2020) | `zero.py`：stage 0–3，ring reduce-scatter / all-gather，分片 Adam | fp64 代替混合精度；无 bucket、无通信重叠 |
| GPipe / PipeDream (1F1B) / Megatron 交错 | `pipeline.py`：离散事件模拟，固定每卡算子序 | 通信零开销；tb=2tf 均匀 |
| Chen 2016 √n、Checkmate | `checkpointing.py`：链上 no-ckpt / √n / 预算约束 DP，torch `checkpoint` 验证 | 单层级分段模型，非 Checkmate 的 MILP |
| torch.distributed | `gloo_bench.py`：2/4 进程 gloo all_reduce 实测 | 本机 CPU，`file://` 初始化 |

## 怎么跑

```powershell
cd solutions/cmu15849/distributed-training
python -m pytest -q tests          # 无 GPU：78 passed, 2 skipped（71 个 CPU 用例 + 7 个 GPU 代码的 CPU 用例；2 个 CUDA 用例 skip）
python scripts/run_experiments.py  # 约 25–55 s，写 artifacts/raw/*.csv 与 artifacts/figures/*.png
```

## 结果

### all-reduce（P=8，`allreduce_sweep.csv`）

数据并行每一步都要对梯度做 all-reduce（所有卡最后都拿到全体之和）。模拟器真的在 rank 之间搬数组，所以能同时检查结果、字节和步数。
所有算法的求和都与 fp64 参考一致：int64 测试逐位相等，fp32 的最大误差 1.8e-6 来自求和顺序。
模拟器计出的时间与闭式公式一致（测试 `rel_tol=1e-12`），ring 的闭式就是 `mlsys_calc.ring_allreduce_time`。

| 算法 | 步数 | 1 KB 时间 | 1 GB 时间（交换机） | 1 GB 时间（物理环） | 1 GB 每 rank 发送 |
| --- | ---: | ---: | ---: | ---: | ---: |
| naive（root 汇聚+广播） | 14 | 70.0 µs | 33.5 ms | 33.5 ms | 7.52 GB（root） |
| ring | 14 | 70.0 µs | 4.25 ms | 4.25 ms | 1.88 GB |
| halving-doubling | 6 | 30.0 µs | 4.21 ms | 12.56 ms | 1.88 GB |
| 二项树 | 6 | 30.0 µs | 14.3 ms | 14.3 ms | 3.22 GB |

ring 在消息约 4.27 MB 以上超过树；在物理环上，约 5.18 MB 以上超过 halving-doubling；
在交换机拓扑上，ring 永远不会严格快于 halving-doubling（`summary.json`）。
卡数变多时，ring 的步数劣势会被放大：P=64、1 KB 时 ring 要 630 µs，树只要 60 µs（`allreduce_vs_P.csv`）。
ring 的延迟项是 2(P−1)α，随卡数线性增长，这就是大规模集群上 NCCL 要引入 tree 算法的原因。

![all-reduce](artifacts/figures/allreduce_time_vs_size.png)

两张图左边的平台高度由步数 × α 决定，所以步数少的树和 halving-doubling 在下面；右边的斜线由字节 × β 决定，字节最少的算法胜出。
两图的区别只在右端：交换机上绿线（halving-doubling）和橙线（ring）重合，物理环上绿线被抬高，因为它的远距离交换要在环上共用链路。

我还用 gloo 在本机实测了 all-reduce（`gloo_allreduce.csv`，20 次取 min）。
world=2 时 1 KB 要 0.40 ms、16 MB 要 25.8 ms；world=4 时 1 KB 要 1.52 ms、16 MB 要 23.3 ms。
按 ring 公式拟合，α≈181/232 µs，等效带宽 0.97/1.42 GB/s。本机还有其他进程，噪声大，中位数在 4–16 MB 处比 min 大 2×，所以拟合只能说明数量级。

![gloo](artifacts/figures/gloo_allreduce.png)

实测和虚线（α-β 拟合）形状吻合，说明模型的结构是对的；但平台高度远大于假设的 α=5 µs，所以这些数字不能拿去推 GPU 集群。

### 张量并行（`tp_equivalence.csv`，`tp_comm_llama3_8b.csv`）

张量并行把一层的权重矩阵切到多张卡上。要确认两件事：结果与不切是否一致，每层多付多少通信。

T∈{1,2,4,8} 时，MLP 与 attention 的前向、输入梯度、参数梯度与未切分模块的最大差 ≤8.9e-16（fp64）。
每个块前向做 1 次、反向做 1 次 all-reduce，消息是 b·s·h 个元素，与 T 无关。
放到 Llama-3-8B（batch 1、bf16）上，每层 4 次 all-reduce；TP=8、seq=8192 时每次 67.1 MB、0.331 ms，32 层每步通信 42.4 ms（模型值）。

### ZeRO（P=4，20 步，`zero_stages.csv`，`zero_paper_scale.csv`）

ZeRO 把数据并行里每张卡各存一份的东西分片：stage 1 分优化器状态，stage 2 再分梯度，stage 3 连参数也分。

| stage | 与单进程最大差 | 每 rank 常驻字节 | 相对 DP | 每步通信相对 DP |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.9e-16 | 43,136 | 1.000 | 1.0 |
| 1 | 2.9e-16 | 26,960 | 0.625 | 1.0 |
| 2 | 2.9e-16 | 18,872 | 0.4375 | 1.0 |
| 3 | 2.9e-16 | 10,784 | 0.25 | 1.5 |

fp64 下 params:grads:m:v = 1:1:1:1，所以显存比例应为 1、(2+2/P)/4、(1+3/P)/4、1/P，实测吻合。
换成论文的混合精度（每参数 2+2+12 字节）、Ψ=7.5B、N_d=64，`mlsys_calc` 给出 120 / 31.4 / 16.6 / 1.88 GB，与 ZeRO 论文图 1 一致。

![zero](artifacts/figures/zero_memory_comm.png)

左图每升一级就把一种颜色压到 1/P：stage 1 压 Adam 的两个矩，stage 2 压梯度，stage 3 压参数。
右图的通信量到 stage 3 才跳到 1.5，因为参数分片后，前向和反向各要多一次 all-gather。

### 流水线（`pipeline_bubble.csv`，`pipeline_peak_activations.csv`）

流水线并行把模型按层切成 p 段，batch 切成 m 个 micro-batch 依次流过；开头和结尾总有卡在空等，这部分叫气泡。
这一节比较 GPipe、1F1B 和交错调度的气泡与在途激活（做完前向、还在等反向的 micro-batch 所占的激活）。

模拟出的气泡与 (p−1)/(m+p−1) 一致到 1e-12；交错 v=2 与 `mlsys_calc.pipeline_bubble(p,m,v)` 一致。

| p | m | GPipe / 1F1B 气泡 | 交错 v=2 气泡 | 设备 0 在途激活 GPipe / 1F1B / 交错 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 8 | 0.273 | 0.158 | 8 / 4 / 5.5 |
| 8 | 8 | 0.467 | 0.304 | 8 / 8 / 8 |
| 8 | 32 | 0.179 | 0.099 | 32 / 8 / 11.5 |

有一个初看意外、但和论文一致的结果：交错调度把气泡减半，设备 0 上的在途激活却比 1F1B 多（p=8,m=32 时 11.5 对 8）。

![gantt](artifacts/figures/pipeline_gantt.png)

GPipe 和 1F1B 两张甘特图总长度一样，空白面积也一样，所以气泡相同，不同的只是空白的位置。
GPipe 做完 8 个前向才开始反向，设备 0 要攥着 8 份激活；1F1B 在稳态里一前一后交替，激活做完反向就释放，设备 0 最多攥 4 份。
交错调度把每张卡的工作切得更碎，开头的空白变短，整条时间线也缩短了。

![bubble](artifacts/figures/pipeline_bubble.png)

模拟点精确落在公式曲线上；p 越大，压低气泡需要的 m 越多，交错调度整体在下方。

### 重计算（`ckpt_frontier.csv`，`ckpt_torch.csv`）

重计算（gradient checkpointing）前向只存少数层的激活，反向时把中间的再算一遍。问题是给定显存预算，怎样选保存点最省重算。

在均匀的 64 层上，不 checkpoint 时峰值是 64、重算是 0；按 √n 规则每 8 层存一次，峰值 15、重算 56；DP 能达到的最低峰值也是 15，
说明在这种均匀模型下 √n 已经在 Pareto 前沿上。层的大小和代价不均匀时就不一样了：异构的 32 层（大小、代价 ∈[1,8]），同样的预算 53 下 √n 要重算 117，DP 只要 85。

![ckpt](artifacts/figures/ckpt_frontier.png)

蓝色阶梯是 DP 前沿，即每个显存预算下的最少重算。左图（均匀层）里 √n 的红三角正好落在阶梯左端；
右图（异构层）里它浮在阶梯上方，这段竖直距离就是 √n 看不到层间差异而多付的重算。

再用 torch 的 `checkpoint` 在 16 层 Linear+Tanh（d=256，batch 64，fp32）上实测，梯度与不 checkpoint 时逐位相等：

| 每段层数 | 峰值保存激活（层数单位） | 前后向中位时间相对不 ckpt |
| ---: | ---: | ---: |
| 不 ckpt | 17 | 1.00 |
| 2 | 10 | 1.73 |
| 4 | 8 | 1.52 |
| 8 | 10 | 1.39 |
| 16 | 17 | 1.34 |

4 层一段等于 √16，峰值最低，与模型的预测一致（存 4 个段的输入，加上重算时 4 层的激活）。
但时间比远高于理论的约 1.33×，因为 CPU 上这些层很小，Python 钩子和 checkpoint 调用的开销占了主导，所以这个时间不能外推到 GPU。

## GPU 实测（单卡 RTX 5090，32 GB）

这一部分问的是：这些公式放到真实的 GPT-2 训练上还准不准，误差来自哪里。

完整报告：[distributed_training_report.pdf](../../../reports/cmu15849-distributed-training/distributed_training_report.pdf)。
代码在 `gpu/`（`gpt.py` 是自写的 GPT-2，带 checkpoint 模式；`train_bench.py` 有六个子命令），原始结果在 `artifacts/raw/gpu/*.jsonl`，
图由 `scripts/plot_gpu.py` 生成到 `artifacts/figures/gpu_*.{png,pdf}`。
单卡上无法测多卡通信，所以做了两种替代：ZeRO 用真实的扁平分片张量测每个 rank 的显存，通信则用实测的 PCIe/D2D 带宽去参数化 α-β 模型；
流水线的 4 个 stage 放在同一张卡上，按 1F1B/GPipe 的串行化顺序执行真实 kernel，用 CUDA event 计时后再按依赖关系重放。

| 实验 | 关键实测 | 与模型对照 |
| --- | --- | --- |
| A1 静态显存（GPT-2 124M/355M/774M） | fp32、AMP、bf16+fp32 master：16.00–16.32 B/参数；纯 bf16：8.01–8.31 | 16N / 8N 规则，误差 ≤2% |
| A1 激活（b×s 扫描） | 124M AMP 斜率 536,843 B/token，355M 1,093,862 | `L·36·sbh + 4bsV` 模型 532,804 / 1,085,764（差 <1%） |
| A1 MFU（774M，b=4，s=1024） | 纯 bf16 70.2%（28.2k tok/s），AMP 65.3%，fp32 23.7% | 分母 209.5 TFLOP/s 名义；实测 GEMM 峰值 238.7 |
| A2 重计算（774M AMP，b=8，s=1024） | 每块全量：激活 16.88→3.35 GB，步时间 ×1.189，最大 batch 8→24（吞吐 28.9k→24.7k tok/s）；只 MLP：×1.093、激活 9.13 GB | 理论 4/3；MLP 节省 7.749 GB vs 逐张量模型 7.739 GB |
| A3 ZeRO 分片（774M，N_d=8） | 每 rank 12.384 / 4.260 / 2.905 / 1.548 GB，与公式最大差 0.099%；分片 AdamW 逐位相等 | (2+2+12)Ψ 分片公式 |
| A3 通信参数 | PCIe pinned 56.5 GB/s，D2D 1518 GB/s，单卡 ring α≈7 µs/launch | N_d=8 PCIe 上 DP 47.97 ms、ZeRO-3 71.96 ms |
| A4 流水线（355M，p=4，b=4，s=512） | 1F1B m=8/16/32 气泡 0.343/0.245/0.233，峰值 7.64 GB 与 m 无关；GPipe m=8 气泡 0.481、24.40 GB，m=16 OOM | 均衡公式 (p−1)/(m+p−1) 低估；按实测各级时间的不均衡模型 0.354/0.254/0.203 |

这张表里值得带走的结论：
- 16 字节/参数在三种精度布局下都成立；混合精度省的是激活和带宽，不是静态显存。
- 激活公式要按 flash attention 实际保存的张量来数（AMP 36、纯 bf16 32 字节/sbh），误差才降到 1% 以内。
- 1F1B 不减少气泡，只把在途 micro-batch 压到每级 [4,3,2,1]；最后一级多了 LM head，各级不均衡，m 增大时气泡停在约 0.23，降不到公式的 0.086。

![显存分解](artifacts/figures/gpu_memory_breakdown.png)

柱子从下往上是权重、梯度、优化器状态和激活。下面三层基本贴着 16N 的虚线（纯 bf16 约为一半），四种精度之间真正变化的是红色的激活。
黑色短线是实测峰值，它可以低于堆叠总和，因为到达峰值时一部分激活已经释放。

![激活扫描](artifacts/figures/gpu_activation_sweep.png)

右图里激活随 b·s 是直线，没有 s² 项，因为 flash attention 不保存 T×T 的注意力矩阵；AMP 曲线在小 b·s 处上翘，是与 token 数无关的固定部分。

![重计算](artifacts/figures/gpu_recompute.png)

左图里全量重计算在最左上方，省得最多也最慢，但仍低于 4/3 的理论线；只重算 MLP 或 attention 是中间的折中。
右图说明这笔交换为什么值得：能放下的 batch 变成原来的三倍，吞吐只掉了一截。

![流水线甘特图](artifacts/figures/gpu_pipeline_gantt.png)

这是用真实 kernel 耗时重放的时间线。stage 3 多了 LM head，块明显更长，其他 stage 不时停下来等它，这就是不均衡带来的额外气泡。

![气泡与显存](artifacts/figures/gpu_pipeline_bubble_memory.png)

左图里实测气泡（实心点）和不均衡模型（空心点）对得很好，点线（均衡公式）一路低估，m 越大差得越多。
右图里 GPipe 的峰值显存随 m 上涨到 OOM，1F1B 的峰值则很快走平，这才是它真正的价值。

复现（需 CUDA；`--quick` 为小规模冒烟测试）：

```bash
cd solutions/cmu15849/distributed-training
python gpu/train_bench.py hw && python gpu/train_bench.py memory && python gpu/train_bench.py actsweep
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gpu/train_bench.py recompute
python gpu/train_bench.py zero && python gpu/train_bench.py pipeline
python scripts/plot_gpu.py
```

## 面试讲法

20 秒结论：all-reduce 的时间 = 步数·α + 每 rank 字节·β。ring 的字节数最优，是 2(P−1)/P·S，但步数是 2(P−1)，
所以小消息用树，大消息用 ring。ZeRO-1/2 通信不变、显存下降；ZeRO-3 多一次参数 all-gather，通信变为 1.5×。
流水线气泡是 (p−1)/(m+p−1)，1F1B 只省激活，交错调度才省气泡。

白板上可以写这几行：
- ring：RS 走 P−1 步、每步 S/P；AG 再 P−1 步 → 2(P−1)α + 2(P−1)/P·Sβ。
- Megatron MLP：A 列切 → GeLU 无需通信 → B 行切 → 一次 all-reduce；反向在输入处再一次。
- ZeRO 每步：DP = RS+AG = 2Ψ；ZeRO-3 = AG(前向) + AG(反向) + RS(梯度) = 3Ψ。

**不能推出的结论**：
- 不能说「halving-doubling 总比 ring 好」。交换机拓扑上两者的带宽项相同，物理环上它慢 3×。
- 不能用本机 gloo 的带宽（约 1 GB/s）去推 NVLink；α=5 µs 也是假设，交叉点 4 MB 会随 α/β 线性移动。
- 不能说「1F1B 气泡更小」。算子均匀时两者的气泡完全相同（p=4,m=8 都是 0.273）。
  GPU 实测里 m=8 时 GPipe 的气泡是 0.481，比 1F1B 的 0.343 大，这个差距我没有单独定位原因，可能来自 GPipe 一次攥 24.40 GB 激活带来的分配开销，也可能是计时口径，
  所以它不能当成 1F1B 调度本身的优势来讲。
- CPU 上 checkpoint 1.5× 的时间开销不代表 GPU；GPU 上主要是多 1 次前向的 FLOPs。

追问：
1. 为什么 TP 一般限制在单机 8 卡？每层 4 次 b·s·h 的 all-reduce 都在关键路径上，而跨机 IB 的带宽约为 NVLink 的 1/9。
2. ZeRO-3 和 FSDP 的通信能和计算重叠吗？能，按层预取下一层的 all-gather。总量 1.5× 不变，暴露出来的时间取决于每层计算能不能把通信盖住。
3. 交错流水线的代价是什么？p2p 通信次数变成 v 倍，而且在途激活增加（表中 5.5 对 4）。

## 文件

```text
distributed-training/
├── code/disttrain/   collectives, tensor_parallel, zero, pipeline, checkpointing, gloo_bench
├── gpu/              gpt.py, train_bench.py（RTX 5090 实测）
├── tests/            80 个用例（gloo 用例超时/失败则 skip；test_gpu_training.py 中 CUDA 用例无 GPU 时 skip）
├── scripts/run_experiments.py, plot_gpu.py
└── artifacts/raw/*.csv, summary.json, raw/gpu/*.jsonl；artifacts/figures/*.png, gpu_*.{png,pdf}
```
