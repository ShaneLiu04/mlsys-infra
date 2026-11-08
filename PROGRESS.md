# 学习进度

每完成一项，链接到对应的笔记、实验或报告，而不只是打个勾。没做完的事也写清楚卡在哪里。

## 建仓（2026-10-28）

- [x] 核对三门课的官方大纲、日程和作业列表，合并成 18 个模块，写在 [CURRICULUM.md](CURRICULUM.md)
- [x] 18 讲笔记，每讲有速览卡、深入推导、白板估算、追问链和自测题，`python notes/check_notes.py` 通过
- [x] [术语表](notes/GLOSSARY.md)和[阅读路线](notes/READING-ROADMAP.md)
- [x] 面试材料：[速查表](interview/HARDWARE-NUMBERS.md)、[公式卡](interview/FORMULAS.md)、[设计题](interview/SYSTEM-DESIGN.md)、[作品集](interview/PORTFOLIO.md)、[手写题](interview/CODING.md)、[题库](interview/QUESTION-BANK.md)、[课程问答](interview/COURSE-QA.md)、[工程手册](interview/ENGINEERING.md)
- [x] 白板估算计算器和采样参考实现 [`tools/`](tools/mlsys_calc.py)，带测试
- [x] 统一的测试入口 `scripts/run-all-tests.ps1` 和仓库检查 `scripts/check-repo.ps1`
- [ ] 镜像官方 starter：公司网络访问 github.com 返回 407，等 `scripts/sync-upstream.ps1` 在能访问的网络下跑

## 实验：CPU 版

每个实验先在 4 核 CPU 上做一个能跑能测的版本，把机制讲清楚。

- [x] CS149 A1 多核与 SIMD — [README](solutions/cs149/asst1-cpu-parallel/README.md)
- [x] CS149 A2 任务系统 — [README](solutions/cs149/asst2-task-scheduler/README.md)
- [x] CS149 A3 scan 与保序渲染 — [README](solutions/cs149/asst3-scan-render/README.md)
- [x] CS149 A4 加速器分块与 systolic 阵列 — [README](solutions/cs149/asst4-dnn-accelerator/README.md)
- [x] CS229S P1 FlashAttention 与 IO 模型 — [README](solutions/cs229s/project1-transformer-kernels/README.md)
- [x] CS229S P2 KV cache 与推测解码 — [README](solutions/cs229s/project2-kv-speculative/README.md)
- [x] 15-849 分布式训练 — [README](solutions/cmu15849/distributed-training/README.md)
- [x] 15-849 服务与显存 — [README](solutions/cmu15849/serving-memory/README.md)
- [x] 15-849 编译与压缩 — [README](solutions/cmu15849/compiler-compression/README.md)
- [x] CS149 A5（加分题）方向自适应 BFS 与 PageRank — [README](solutions/cs149/asst5-big-graphs/README.md) — [报告](reports/cs149-asst5/cs149_asst5_report.pdf)

## 实验：GPU 和 C++ 版（2026-10-28，AutoDL 上的 RTX 5090 + 2 × Xeon Gold 6459C）

和 CPU 版放在一起，README 写明卡型；报告的汇总见 [README 的报告表](README.md#报告与-gpu-实测)。跑完已归档、关机。

- [x] CS149 A1 用 C++、AVX-512、CUDA 重写，服务器上 3 轮实验 — [报告](reports/cs149-asst1/cs149_asst1_report.pdf)
- [x] CS149 A2 C++17 任务系统（spawn、spin、sleep、异步任务图），12 核 3 轮 — [报告](reports/cs149-asst2/cs149_asst2_report.pdf)
- [x] CS149 A3 真正的 CUDA：saxpy、scan、find_repeats、保序圆形渲染器 — [报告](reports/cs149-asst3/cs149_asst3_report.pdf)
- [x] CS149 A4 SGEMM 优化阶梯、WMMA、im2col 和 implicit GEMM 卷积、实测 roofline — [报告](reports/cs149-asst4/cs149_asst4_report.pdf)
- [x] FlashAttention 前向和反向的 Triton kernel，对照 SDPA 与 FA2；MQAR 与 GPT-2 训练 — [报告](reports/cs229s-project1/cs229s_project1_report.pdf)
- [x] 真实 GPT-2 上的 KV cache、CUDA graph 和推测解码（验证代价 v ≈ 1.11，证实了「验证近乎免费」的假设） — [报告](reports/cs229s-project2/cs229s_project2_report.pdf)
- [x] 单卡实测训练显存 16N、激活、重计算、ZeRO、1F1B — [报告](reports/cmu15849-distributed-training/distributed_training_report.pdf)
- [x] 自写分页 KV 引擎，实测 continuous batching 和 chunked prefill — [报告](reports/cmu15849-serving-memory/serving_memory_report.pdf)
- [x] torch.compile 融合、matmul 分块搜索、GPTQ、AWQ、SmoothQuant、剪枝、LoRA — [报告](reports/cmu15849-compiler-compression/compiler_compression_report.pdf)
- [x] 面试知识综述（CCF-A 格式） — [survey.pdf](reports/survey/survey.pdf)

## 换卡复测

同一批实验在另外两张卡上重跑，看哪些结论跟着带宽走、哪些跟着算力走、哪些和卡无关。汇总在 [cross_gpu/comparison.md](artifacts/cross_gpu/comparison.md)。

- [x] RTX 4080 SUPER（2026-10-28）：各实验的原始数据放在 `artifacts/raw/ada4080/`；补了一组 FP8 Triton GEMM 扫描，最好的配置和 cuBLAS 打平 — [A4 README](solutions/cs149/asst4-dnn-accelerator/README.md)
- [x] RTX 6000D（2026-11-04）：持续和突发算力探测，发现这张卡的 GEMM 算力不是一个稳定的数 — [A4 README](solutions/cs149/asst4-dnn-accelerator/README.md)
- [x] RTX 6000D：同一负载下自写引擎对照 vLLM 0.11 — [服务与显存 README](solutions/cmu15849/serving-memory/README.md)
- [x] RTX 6000D：同一套量化从 GPT-2 换到 Qwen2.5-7B，GPTQ 在 7B 上要改用 float64 Cholesky — [编译与压缩 README](solutions/cmu15849/compiler-compression/README.md)

## 文档

- [x] 全部材料按 [写作约定](templates/WRITING.md) 改写：先讲动机再给数字，拆开过密的句子，术语首次出现时解释；报告重新编译（2026-11-08）

## 还没做的

- [ ] Nsight Compute 计数器实测：容器返回 `ERR_NVGPUCTRPERM`。各报告改用 occupancy API、`ptxas -v` 和解析模型，并标注了哪些是推算。
- [ ] 2–4 卡的 NCCL all-reduce 总线带宽：一直只租到单卡实例。
