# 阅读路线

按岗位和时间预算选一条。每条路线都以「速览卡 → 白板估算 → 追问链」为单位，
不要求从头读到尾。

## 按时间

| 预算 | 读什么 | 产出 |
| --- | --- | --- |
| 90 分钟（明天面试） | [HARDWARE-NUMBERS](../interview/HARDWARE-NUMBERS.md) 的「十个数」→ [FORMULAS](../interview/FORMULAS.md) F2、F10、F15、F16 → 岗位对应的一道 [设计题](../interview/SYSTEM-DESIGN.md) | 能当场算出 7B 训练显存、8B decode 上限、70B 部署卡数 |
| 一个周末 | 18 讲的速览卡和白板估算 + 题库 Q1–Q42 | 每讲能讲 60 秒，每道题能说出常见错答 |
| 两周 | 按下面岗位路线精读 6–8 讲，跑对应实验，改一个参数看数字怎么变 | 两个能讲 90 秒的作品集故事 |
| 一学期 | 18 讲全读，补读必读论文，把 CPU 实验迁到 GPU（CUDA/Triton） | 能独立写 FlashAttention 前向 Triton kernel、跑通 3D 并行 |

## 按岗位

**训练框架 / 大模型训练 infra**

1. [M01](module-01-performance-models.md) roofline → [M07](module-07-matmul-hardware-aware.md) GEMM 与 MFU
2. [M11](module-11-frameworks-autodiff.md) 自动微分与框架运行时
3. [M12](module-12-data-parallel-zero.md) all-reduce 与 ZeRO → [M13](module-13-tensor-pipeline-parallel.md) TP/PP → [M14](module-14-memory-optimization.md) 重计算与混合精度
4. [M18](module-18-architectures-moe-clusters.md) MoE 与集群
5. 实验：[15-849 分布式训练](../solutions/cmu15849/distributed-training/README.md)；设计题 D2、D5

**推理引擎 / serving**

1. [M01](module-01-performance-models.md) → [M05](module-05-gpu-architecture-cuda.md) GPU 存储层次 → [M07](module-07-matmul-hardware-aware.md)
2. [M08](module-08-attention-flashattention.md) FlashAttention
3. [M15](module-15-inference-kv-speculative.md) KV cache 与推测解码 → [M16](module-16-llm-serving.md) 服务系统
4. [M17](module-17-compression-quantization.md) 量化
5. 实验：[CS229S P2](../solutions/cs229s/project2-kv-speculative/README.md)、[15-849 服务与显存](../solutions/cmu15849/serving-memory/README.md)；设计题 D1、D4、D6

**算子 / 编译器 / AI 芯片**

1. [M02](module-02-multicore-simd-threads.md) → [M04](module-04-locality-coherence-synchronization.md) → [M05](module-05-gpu-architecture-cuda.md) → [M06](module-06-data-parallel-primitives.md)
2. [M07](module-07-matmul-hardware-aware.md) 分块 → [M08](module-08-attention-flashattention.md) → [M09](module-09-accelerators.md) systolic 与 Trainium
3. [M10](module-10-ml-compilers.md) 编译器
4. 实验：CS149 [A1](../solutions/cs149/asst1-cpu-parallel/README.md)、[A3](../solutions/cs149/asst3-scan-render/README.md)、[A4](../solutions/cs149/asst4-dnn-accelerator/README.md)，[CS229S P1](../solutions/cs229s/project1-transformer-kernels/README.md)

**通用 ML 工程（不专做 infra）**

[M01](module-01-performance-models.md)、[M08](module-08-attention-flashattention.md)、[M12](module-12-data-parallel-zero.md)、[M15](module-15-inference-kv-speculative.md)、[M16](module-16-llm-serving.md) 的速览卡 + 题库第三、四部分。

## 必读论文（按模块）

| 模块 | 论文 |
| --- | --- |
| M01 | Williams et al., Roofline (2009) |
| M03 | Blumofe & Leiserson, Scheduling Multithreaded Computations by Work Stealing (1999) |
| M06 | Blelloch, Prefix Sums and Their Applications (1990)；Zaharia et al., Resilient Distributed Datasets (2012) |
| M08 | Dao et al., FlashAttention (2022)；Dao, FlashAttention-2 (2023) |
| M09 | Jouppi et al., In-Datacenter Performance Analysis of a TPU (2017) |
| M10 | Ragan-Kelley et al., Halide (2013)；Chen et al., TVM (2018)；Zheng et al., Ansor (2020)；Jia et al., TASO (2019)；Tillet et al., Triton (2019) |
| M11 | Abadi et al., TensorFlow (2016)；Paszke et al., PyTorch (2019)；Li et al., Parameter Server (2014)；Chen & Guestrin, XGBoost (2016) |
| M12 | Rajbhandari et al., ZeRO (2020)；Li et al., PyTorch Distributed (2020) |
| M13 | Shoeybi et al., Megatron-LM (2019)；Narayanan et al., Efficient Large-Scale LM Training (2021)；Huang et al., GPipe (2019)；Narayanan et al., PipeDream (2019)；Xu et al., GSPMD (2021)；Zheng et al., Alpa (2022) |
| M14 | Chen et al., Training Deep Nets with Sublinear Memory Cost (2016)；Jain et al., Checkmate (2020)；Micikevicius et al., Mixed Precision Training (2018)；Rajbhandari et al., ZeRO-Infinity (2021) |
| M15 | Pope et al., Efficiently Scaling Transformer Inference (2022)；Leviathan et al., Speculative Decoding (2023)；Chen et al., Speculative Sampling (2023) |
| M16 | Yu et al., Orca (2022)；Kwon et al., vLLM / PagedAttention (2023)；Agrawal et al., Sarathi-Serve (2024)；Zhong et al., DistServe (2024)；Zheng et al., SGLang (2024) |
| M17 | Han et al., Deep Compression (2016)；Dettmers et al., LLM.int8() (2022)；Xiao et al., SmoothQuant (2023)；Frantar et al., GPTQ (2023)；Hu et al., LoRA (2021)；Dettmers et al., QLoRA (2023) |
| M18 | Fedus et al., Switch Transformers (2021)；Gu & Dao, Mamba (2023)；Hoffmann et al., Chinchilla (2022)；Qiao et al., Pollux (2021) |
