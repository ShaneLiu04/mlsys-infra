# 学习进度

更新时链接到对应笔记、实验或实现，不只勾选状态。

## 建仓（2026-10-28）

- [x] 核对三门课的官方大纲、日程、作业列表，写成 [CURRICULUM.md](CURRICULUM.md)（18 个模块）
- [x] 18 讲笔记（速览卡、深入推导、白板估算、追问链、自测题；`python notes/check_notes.py` 通过）
- [x] [术语表](notes/GLOSSARY.md)、[阅读路线](notes/READING-ROADMAP.md)
- [x] 面试材料：[速查表](interview/HARDWARE-NUMBERS.md)、[公式卡](interview/FORMULAS.md)、[设计题](interview/SYSTEM-DESIGN.md)、[作品集](interview/PORTFOLIO.md)、[手写题](interview/CODING.md)、[题库](interview/QUESTION-BANK.md)、[课程问答](interview/COURSE-QA.md)、[工程手册](interview/ENGINEERING.md)
- [x] 白板估算计算器与采样参考实现：[`tools/`](tools/mlsys_calc.py)，25 个测试
- [x] 统一测试入口 `scripts/run-all-tests.ps1`、仓库检查 `scripts/check-repo.ps1`
- [ ] 镜像官方 starter（github.com 在公司网络下 407，待 `scripts/sync-upstream.ps1`）

## 实验（CPU 版）

- [x] CS149 A1 多核与 SIMD — [README](solutions/cs149/asst1-cpu-parallel/README.md)
- [x] CS149 A2 任务系统 — [README](solutions/cs149/asst2-task-scheduler/README.md)
- [x] CS149 A3 scan 与保序渲染 — [README](solutions/cs149/asst3-scan-render/README.md)
- [x] CS149 A4 加速器分块与 systolic — [README](solutions/cs149/asst4-dnn-accelerator/README.md)
- [x] CS229S P1 FlashAttention 与 IO 模型 — [README](solutions/cs229s/project1-transformer-kernels/README.md)
- [x] CS229S P2 KV cache 与推测解码 — [README](solutions/cs229s/project2-kv-speculative/README.md)
- [x] 15-849 分布式训练 — [README](solutions/cmu15849/distributed-training/README.md)
- [x] 15-849 服务与显存 — [README](solutions/cmu15849/serving-memory/README.md)
- [x] 15-849 编译与压缩 — [README](solutions/cmu15849/compiler-compression/README.md)
- [x] CS149 A5（EC）方向自适应 BFS 与 PageRank — [README](solutions/cs149/asst5-big-graphs/README.md) — [报告](reports/cs149-asst5/cs149_asst5_report.pdf)

## GPU / C++ 版（2026-10-28，AutoDL：RTX 5090 + 2 × Xeon Gold 6459C，跑完已归档、关机）

每项与 CPU 版并存，README 写明卡型，报告见 [README 的报告表](README.md#报告与-gpu-实测)。

- [x] CS149 A1 C++ / AVX-512 / CUDA 重写，服务器 3 轮实验 — [报告](reports/cs149-asst1/cs149_asst1_report.pdf)
- [x] CS149 A2 C++17 任务系统（spawn / spin / sleep / 异步 DAG），12 核 3 轮 — [报告](reports/cs149-asst2/cs149_asst2_report.pdf)
- [x] CS149 A3 真 CUDA：saxpy、scan、find_repeats、保序圆形渲染器 — [报告](reports/cs149-asst3/cs149_asst3_report.pdf)
- [x] CS149 A4 SGEMM 优化阶梯、WMMA、卷积 im2col / implicit GEMM、实测 roofline — [报告](reports/cs149-asst4/cs149_asst4_report.pdf)
- [x] FlashAttention 前向 / 反向 Triton kernel，对比 SDPA 与 FA2；MQAR 与 GPT-2 训练 — [报告](reports/cs229s-project1/cs229s_project1_report.pdf)
- [x] 真实 GPT-2 KV cache / CUDA graph / 推测解码（v ≈ 1.11，验证了假设） — [报告](reports/cs229s-project2/cs229s_project2_report.pdf)
- [x] 训练显存 16N、激活、重计算、ZeRO、1F1B 单卡实测 — [报告](reports/cmu15849-distributed-training/distributed_training_report.pdf)
- [x] 自写分页 KV 引擎：continuous batching / chunked prefill 实测 — [报告](reports/cmu15849-serving-memory/serving_memory_report.pdf)
- [x] torch.compile 融合、matmul 分块搜索、GPTQ / AWQ / SmoothQuant、剪枝、LoRA — [报告](reports/cmu15849-compiler-compression/compiler_compression_report.pdf)
- [x] 面试知识综述（CCF-A 格式，34 页） — [survey.pdf](reports/survey/survey.pdf)
- [ ] Nsight Compute 计数器实测：容器返回 `ERR_NVGPUCTRPERM`，各报告改用 occupancy API、`ptxas -v` 与解析模型，已标注
- [ ] 2–4 卡 NCCL all-reduce bus bandwidth（单卡实例做不了）
- [ ] vLLM 对照（用自写引擎代替）
