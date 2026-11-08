# 个人实现

这里是十个实验的个人实现，对应笔记里的各讲。笔记讲的是公式和机制，实验做的是把这些公式放到代码里跑一遍，
看数字和推导对不对得上，对不上的地方又是为什么。全部实验都能在 CPU 上运行和测试；
很多实验还有在 GPU 上实测的部分，运行方式写在各自的 README 里。

每个目录的结构相同：

```text
<lab>/
  README.md                 结论、对应作业、怎么跑、结果、面试讲法
  code/<package>/           实现
  tests/                    pytest，含与参考实现逐元素对比的正确性测试
  scripts/run_experiments.py  生成 artifacts/raw/*.csv 和 artifacts/figures/*.png
  artifacts/
```

README 里的数字只来自 `artifacts/raw/`，每个数字都能追到一个原始文件。官方 starter 是 C++/ISPC/CUDA/NKI，建仓时拿不到，
所以这里是按作业要求独立实现的 Python 版本；和官方作业的差异，写在每个 README 的「对应官方作业」一节里。

我写每个 README 都按同一个顺序：开头一段是这个实验要回答的问题和结论，「结果」里每张图下面写它说明了什么，最后是「不能推出的结论」。引用数字之前先读最后那一节。有些数是这台机器上测的，有些是按公开硬件参数算出来的，混着讲会被问穿。

| 课程 | 实验 | 内容 | 笔记 |
| --- | --- | --- | --- |
| CS149 | [asst1-cpu-parallel](cs149/asst1-cpu-parallel/README.md) | Mandelbrot 划分与负载不均、SIMD 发散利用率、saxpy 带宽 | [M02](../notes/module-02-multicore-simd-threads.md) |
| CS149 | [asst2-task-scheduler](cs149/asst2-task-scheduler/README.md) | 四种任务系统、带依赖的异步任务图、spin vs sleep | [M03](../notes/module-03-parallel-programming-scheduling.md) [M04](../notes/module-04-locality-coherence-synchronization.md) |
| CS149 | [asst3-scan-render](cs149/asst3-scan-render/README.md) | 三种 scan、find_repeats、保序的分块圆形渲染器 | [M05](../notes/module-05-gpu-architecture-cuda.md) [M06](../notes/module-06-data-parallel-primitives.md) |
| CS149 | [asst4-dnn-accelerator](cs149/asst4-dnn-accelerator/README.md) | scratchpad 分块的访存模型、systolic array 模拟、im2col | [M07](../notes/module-07-matmul-hardware-aware.md) [M09](../notes/module-09-accelerators.md) |
| CS149 | [asst5-big-graphs](cs149/asst5-big-graphs/README.md) | OpenMP 方向自适应 BFS（top-down / bottom-up / hybrid）、PageRank pull 与 push | [M06](../notes/module-06-data-parallel-primitives.md) |
| CS229S | [project1-transformer-kernels](cs229s/project1-transformer-kernels/README.md) | FlashAttention 前向/反向、IO 模型、depthwise conv、联想回忆 | [M08](../notes/module-08-attention-flashattention.md) |
| CS229S | [project2-kv-speculative](cs229s/project2-kv-speculative/README.md) | KV cache、推测解码与分布无偏检验 | [M15](../notes/module-15-inference-kv-speculative.md) |
| 15-849 | [distributed-training](cmu15849/distributed-training/README.md) | 集合通信模拟、Megatron TP、ZeRO 0–3、流水线调度、重计算 | [M12](../notes/module-12-data-parallel-zero.md) [M13](../notes/module-13-tensor-pipeline-parallel.md) [M14](../notes/module-14-memory-optimization.md) |
| 15-849 | [serving-memory](cmu15849/serving-memory/README.md) | PagedAttention 块管理、碎片研究、continuous batching 调度模拟 | [M16](../notes/module-16-llm-serving.md) |
| 15-849 | [compiler-compression](cmu15849/compiler-compression/README.md) | 自动微分、图 IR 融合、调度搜索、量化、剪枝、LoRA | [M10](../notes/module-10-ml-compilers.md) [M11](../notes/module-11-frameworks-autodiff.md) [M17](../notes/module-17-compression-quantization.md) |

表格按课程和系统栈从下往上排：CS149 的五个实验是 CPU 并行、GPU 和加速器的基础，CS229S 的两个实验落在单卡的 attention 和推理上，
15-849 的三个实验分别是训练、服务和编译压缩。如果只想挑一两个看，可以按岗位选：做推理的先看 CS229S P2 和 serving-memory，
做训练的先看 distributed-training，做算子和编译器的先看 CS229S P1 和 compiler-compression。更完整的阅读顺序见 [READING-ROADMAP](../notes/READING-ROADMAP.md)。

全部测试：`./scripts/run-all-tests.ps1`。
