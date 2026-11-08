# 资料索引

这里只放链接，不转载内容，访问日期 2026-10-28。如果只能挑一份课外读物，我推荐下面「书与长文」里的 How to Scale Your Model：
它把 roofline、分片和通信估算讲得最系统，和这个仓库的白板估算是同一套思路。

## 三门课

| 课程 | 链接 |
| --- | --- |
| Stanford CS149 Fall 2024 | https://gfxcourses.stanford.edu/cs149/fall24 |
| Stanford CS149 作业仓库 | https://github.com/stanford-cs149 |
| Stanford CS229S Fall 2023 | https://cs229s.stanford.edu/fall2023/ |
| CS229S 日程与 slides | https://cs229s.stanford.edu/fall2023/calendar/ |
| HazyResearch AI 系统资料合集 | https://github.com/HazyResearch/aisys-building-blocks |
| CMU 15-849 | https://www.cs.cmu.edu/~zhihaoj2/15-849/ |
| CMU 15-849 日程与论文 | https://www.cs.cmu.edu/~zhihaoj2/15-849/schedule.html |

## 相邻课程（查漏补缺）

| 课程 | 补什么 | 链接 |
| --- | --- | --- |
| CMU 10-414/714 Deep Learning Systems（Tianqi Chen、Zico Kolter） | 从零写自动微分框架 needle、CUDA 后端 | https://dlsyscourse.org/ |
| MIT 6.5940 TinyML and Efficient Deep Learning（Song Han） | 量化、剪枝、蒸馏、高效 LLM 部署 | https://efficientml.ai/ |
| Stanford CS336 Language Modeling from Scratch | 从零训练 LM，含 Triton kernel 和并行作业 | https://stanford-cs336.github.io/ |
| UC Berkeley CS294 / CS267 | 分布式系统与 HPC | https://sites.google.com/lbl.gov/cs267-spr2024 |

如果还想对照着看，CS336 补的是从零训练和 Triton，nano-vLLM 补的是一个小而完整的推理引擎。它们不在这个仓库里，这里只留课程和论文的链接。

## 书与长文

| 资料 | 用途 |
| --- | --- |
| Google DeepMind, How to Scale Your Model（2025） https://jax-ml.github.io/scaling-book/ | TPU/GPU 上的 roofline、分片、通信估算，面试估算题的最佳参考 |
| Horace He, Making Deep Learning Go Brrrr From First Principles https://horace.io/brrr_intro.html | 计算受限、访存受限、开销受限三分法 |
| Stas Bekman, Machine Learning Engineering Open Book https://github.com/stas00/ml-engineering | 大规模训练的工程细节、故障排查 |
| NVIDIA CUDA C++ Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | 内存层次、warp、occupancy 的权威定义 |
| NVIDIA H100 白皮书 https://resources.nvidia.com/en-us-tensor-core | HARDWARE-NUMBERS 的数字来源 |
| GPU MODE 讲座 https://github.com/gpu-mode/lectures | CUDA、Triton、FlashAttention 的社区讲座 |

## 工具

| 工具 | 用途 |
| --- | --- |
| PyTorch Profiler / Nsight Systems | 时间线：空洞、通信 overlap、CPU 发射开销 |
| Nsight Compute | 单 kernel：occupancy、DRAM/SM 吞吐、bank conflict |
| nccl-tests | all-reduce 等集合通信的实测 bus bandwidth |
| py-spy | 分布式挂住时看每个 rank 的 Python 栈 |
| Triton | block 级 GPU kernel |
| vLLM / SGLang | 推理服务基线 |
| Megatron-LM / DeepSpeed / FSDP | 训练并行基线 |
