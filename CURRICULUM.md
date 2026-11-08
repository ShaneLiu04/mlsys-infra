# 课程合并设计

三门课各管一层。CS149 讲一台机器怎么并行，CS229S 讲 Transformer 在这台机器上怎么算得快，
CMU 15-849 讲框架、编译器和分布式系统怎么把模型拆到很多台机器上。

面试官不会问「某门课第几讲讲了什么」，他们问的是「这个数字从哪来、瓶颈在哪、换一种做法会怎样」。
所以这个仓库不按课程分章，而是按系统栈从下往上排成 18 个模块，每个模块标出它在三门课里的出处。

## 三门课的版本

| 课程 | 版本 | 形式 | 官方入口 |
| --- | --- | --- | --- |
| Stanford CS149 Parallel Computing | Fall 2024（Kayvon Fatahalian、Kunle Olukotun） | 20 讲、5 个编程作业、6 份书面作业 | https://gfxcourses.stanford.edu/cs149/fall24 |
| Stanford CS229S Systems for Machine Learning | Fall 2023，第一期（Azalia Mirhoseini、Simran Arora） | 20 讲（含 7 场客座）、2 个项目、期末项目 | https://cs229s.stanford.edu/fall2023/ |
| CMU 15-849 Machine Learning Systems | 当前公开页面为 Spring 2022 版（Zhihao Jia） | 论文研讨制，26 次课，没有编程作业 | https://www.cs.cmu.edu/~zhihaoj2/15-849/ |

15-849 没有编程作业，`solutions/cmu15849/` 里是我按它的必读论文做的最小复现，不是官方作业。

## 为什么是这个顺序

几乎所有 ML 系统的面试题，拆到最后都是下面三个问题之一：

1. **算得过来吗？** 要会算 FLOPs、算术强度，会画 roofline。这一层没打牢，后面「为什么 decode 是访存受限」这类结论就只能死记。
2. **放得下吗？** 权重、梯度、优化器状态、激活、KV cache 各占多少字节，放不下时该怎么切。
3. **传得过来吗？** all-reduce 要传多少数据，流水线有多少气泡，PCIe、NVLink、InfiniBand 各有多快。

三门课正好对应这三个问题。CS149 给第一个问题打地基：并行模型、局部性、GPU。CS229S 把这些用到 Transformer 上：FlashAttention、KV cache、推测解码。
15-849 处理后两个问题：ZeRO、流水线、重计算、编译器。

所以模块的顺序是：先讲硬件，再讲单卡上的算子，然后是编译器，接着是训练系统和推理系统，最后是模型效率和集群。
每一层都要用到下一层的结论，倒过来读会到处碰到没解释过的概念。

## 18 个模块

| # | 模块 | CS149 Fall 2024 | CS229S Fall 2023 | 15-849 | 实验 |
| ---: | --- | --- | --- | --- | --- |
| 01 | 为什么并行：性能模型与 roofline | Why Parallelism | Hardware Aware Algorithm Design (10-04) | Intro to DL Systems | — |
| 02 | 多核、SIMD 与多线程 | A Modern Multi-Core Processor；Multi-Core II + ISPC | — | — | [CS149 A1](solutions/cs149/asst1-cpu-parallel/) |
| 03 | 并行编程与调度 | Parallel Programming Basics；Work Distribution and Scheduling | — | — | [CS149 A2](solutions/cs149/asst2-task-scheduler/) |
| 04 | 局部性、通信、一致性与同步 | Locality/Communication/Contention；Cache Coherence；Locks/Lock-Free；Relaxed Consistency；Transactional Memory | — | — | [CS149 A2](solutions/cs149/asst2-task-scheduler/) |
| 05 | GPU 架构与 CUDA | GPU Architecture and CUDA Programming | 10-04 GPU 存储层次 | — | [CS149 A3](solutions/cs149/asst3-scan-render/) |
| 06 | 数据并行原语：scan、reduce、Spark | Data-Parallel Thinking；Spark | — | Parameter Server（对照） | [CS149 A3](solutions/cs149/asst3-scan-render/) |
| 07 | 矩阵乘、分块与硬件感知算法 | Efficiently Evaluating DNNs | Hardware Aware Algorithm Design | ML Hardware（Winograd） | [CS149 A4](solutions/cs149/asst4-dnn-accelerator/) |
| 08 | Attention 与 FlashAttention | Efficiently Evaluating DNNs（融合） | Hardware Aware Algorithm Design (10-11) | Transformers and LMs | [CS229S P1](solutions/cs229s/project1-transformer-kernels/) |
| 09 | 专用硬件与加速器编程 | Hardware Specialization；Programming Specialized Hardware I/II | — | ML Hardware（TPU） | [CS149 A4](solutions/cs149/asst4-dnn-accelerator/) |
| 10 | ML 编译器与图优化 | Domain-Specific Programming Systems | Tianqi Chen 客座（TVM） | ML Compilation ×2；Graph-Level Optimizations；RL for Device Placement | [15-849 编译与压缩](solutions/cmu15849/compiler-compression/) |
| 11 | 深度学习框架与自动微分 | — | 10-09 FLOPs 与反传 | Intro to DNNs；Systems for DL；Systems for Classical ML | [15-849 编译与压缩](solutions/cmu15849/compiler-compression/) |
| 12 | 数据并行训练、all-reduce 与 ZeRO | — | Parallelism Fundamentals；Ce Zhang 客座 | Data and Model Parallelism；Zero Redundancy | [15-849 分布式训练](solutions/cmu15849/distributed-training/) |
| 13 | 张量并行、流水线并行与自动并行 | — | Parallelism Fundamentals；Deepak Narayanan 客座 | Pipeline Parallelism；Megatron-LM 客座；GShard/GSPMD | [15-849 分布式训练](solutions/cmu15849/distributed-training/) |
| 14 | 显存优化：重计算、混合精度、offload | — | Tim Dettmers 客座 | Tensor Rematerialization；ZeRO-Offload/Infinity | [15-849 分布式训练](solutions/cmu15849/distributed-training/) |
| 15 | Transformer 推理：KV cache 与推测解码 | — | Analyzing the Performance of Transformers (10-09) | — | [CS229S P2](solutions/cs229s/project2-kv-speculative/) |
| 16 | LLM 推理服务系统 | — | Scheduling ML Clusters | Library to accelerate training and inference（客座） | [15-849 服务与显存](solutions/cmu15849/serving-memory/) |
| 17 | 模型压缩：量化、剪枝、蒸馏、PEFT | — | Memory Efficient Neural Networks；Adapting LLMs | Model Compression | [15-849 编译与压缩](solutions/cmu15849/compiler-compression/) |
| 18 | 高效架构、MoE、集群与数据 | — | Attention-Free Architectures；Fedus（MoE）；Scheduling ML Clusters；Data Pipelines | GNN；ML for Systems；Federated Learning | — |

每个模块的笔记文件名见 [notes/README.md](notes/README.md)。

## 删了什么，补了什么

三门课加起来六十多讲，全讲一遍既没必要，也会冲淡重点。取舍的标准只有一条：面试会不会问，问到时要答到什么深度。

| 内容 | 怎么处理 | 理由 |
| --- | --- | --- |
| CS149 的 cache coherence、锁、事务内存 | 压进模块 04，只保留 MESI、false sharing、内存序、CAS 和 ABA 问题 | 面试问的是 false sharing 和原子操作的代价，事务内存极少被问到 |
| CS149 的 Spark | 并进模块 06，当作「数据并行原语放到集群上是什么样子」 | 和 all-reduce、parameter server 对照着讲才有意义 |
| 15-849 的经典 ML 系统（XGBoost）和 GNN | 模块 11、18 各留一节 | 推荐和广告方向会问 parameter server 和 embedding 表；GNN 只留采样和稀疏聚合 |
| 15-849 的 RL for device placement、ML for systems | 放进模块 10、18 | 研究岗的加分项，工程岗知道结论就够了 |
| CS229S 的 RLHF、scaling laws | 模块 17、18 只讲系统那一面（显存、生成吞吐） | 算法细节在另一个仓库 [cs229-r](https://gitee.com/liu-xingyan04/cs229-r) |
| 课程之后才出现的系统：vLLM、SGLang、FlashAttention-3、FP8、chunked prefill、PD 分离 | 补进模块 08、15、16 | 2025 年之后的推理岗面试默认你知道这些 |

## 每个模块交付什么

1. 一篇笔记（`notes/`）。结构由 `notes/check_notes.py` 检查：速览卡、推导、反例、面试追问链、自测题都要有。
2. 至少一条能当场写在白板上的公式，收进 [interview/FORMULAS.md](interview/FORMULAS.md)。
3. 有实验的模块，在 `solutions/` 里有代码、测试、原始数据和图。README 里的数字只能来自这些原始数据。
4. 至少一道带「常见错答」的题，收进 [interview/QUESTION-BANK.md](interview/QUESTION-BANK.md)。

## 按岗位看重点

| 岗位 | 主线模块 | 最先准备 |
| --- | --- | --- |
| 训练框架、大模型训练 infra | 07、11–14、18 | 显存账、3D 并行、通信量 |
| 推理引擎、serving | 05、07、08、15–17 | roofline、KV cache、调度、量化 |
| 算子、编译器、AI 芯片 | 02、04–10 | CUDA 内存层次、分块、Triton 和 TVM、systolic array |
| 通用 ML 工程 | 01、08、12、15、16 | 每个模块的一页速览卡 |
