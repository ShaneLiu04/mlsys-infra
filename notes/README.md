# 机器学习系统十八讲

这是一套原创的中文笔记，英文术语保留原样，不逐页翻译官方 slides。十八讲按系统栈从下往上排：
从硬件和并行讲起，经过单卡算子、编译器和框架，再到训练系统、推理系统和模型效率。每一讲都标出它在 CS149、CS229S、CMU 15-849 里的出处，
三门课为什么这样合并，见 [CURRICULUM.md](../CURRICULUM.md)。如果你是为面试而来，入口在仓库根目录的 [README](../README.md)，
公式卡和题库在 [interview/](../interview/README.md)。

第一次来，建议先读 [M01](module-01-performance-models.md)。后面几乎每一讲都会用到它的 roofline 和算术强度，
比如判断一个 kernel 被算力还是带宽卡住、decode 为什么要加 batch，没有这把尺子，后面的数字很难看懂。
读完 M01，再按你的方向挑：做训练的往 M12–M14 走，做推理的往 M15–M16 走，做算子和编译器的往 M05–M10 走。
时间不多、或者想按岗位走一条完整路线的话，看 [READING-ROADMAP](READING-ROADMAP.md)。

## 课程索引

| # | 部分 | 主题 | 笔记 | 实验 |
| ---: | --- | --- | --- | --- |
| 1 | 硬件与并行 | 为什么并行：性能模型与 roofline | [M01](module-01-performance-models.md) | — |
| 2 | 硬件与并行 | 多核、SIMD 与多线程 | [M02](module-02-multicore-simd-threads.md) | CS149 A1 |
| 3 | 硬件与并行 | 并行编程与调度 | [M03](module-03-parallel-programming-scheduling.md) | CS149 A2 |
| 4 | 硬件与并行 | 局部性、通信、一致性与同步 | [M04](module-04-locality-coherence-synchronization.md) | CS149 A2 |
| 5 | 硬件与并行 | GPU 架构与 CUDA | [M05](module-05-gpu-architecture-cuda.md) | CS149 A3 |
| 6 | 硬件与并行 | 数据并行原语：scan、reduce、Spark | [M06](module-06-data-parallel-primitives.md) | CS149 A3 |
| 7 | 单卡算子 | 矩阵乘、分块与硬件感知算法 | [M07](module-07-matmul-hardware-aware.md) | CS149 A4 |
| 8 | 单卡算子 | Attention 与 FlashAttention | [M08](module-08-attention-flashattention.md) | CS229S P1 |
| 9 | 单卡算子 | 专用硬件与加速器编程 | [M09](module-09-accelerators.md) | CS149 A4 |
| 10 | 编译与框架 | ML 编译器与图优化 | [M10](module-10-ml-compilers.md) | 15-849 编译与压缩 |
| 11 | 编译与框架 | 深度学习框架与自动微分 | [M11](module-11-frameworks-autodiff.md) | 15-849 编译与压缩 |
| 12 | 训练系统 | 数据并行训练、all-reduce 与 ZeRO | [M12](module-12-data-parallel-zero.md) | 15-849 分布式训练 |
| 13 | 训练系统 | 张量并行、流水线并行与自动并行 | [M13](module-13-tensor-pipeline-parallel.md) | 15-849 分布式训练 |
| 14 | 训练系统 | 显存优化：重计算、混合精度、offload | [M14](module-14-memory-optimization.md) | 15-849 分布式训练 |
| 15 | 推理系统 | Transformer 推理：KV cache 与推测解码 | [M15](module-15-inference-kv-speculative.md) | CS229S P2 |
| 16 | 推理系统 | LLM 推理服务系统 | [M16](module-16-llm-serving.md) | 15-849 服务与显存 |
| 17 | 模型效率 | 模型压缩：量化、剪枝、蒸馏、PEFT | [M17](module-17-compression-quantization.md) | 15-849 编译与压缩 |
| 18 | 模型效率 | 高效架构、MoE、集群与数据 | [M18](module-18-architectures-moe-clusters.md) | — |

最右一列是对应的实验（[solutions/](../solutions/README.md)），读到一个公式时，可以去那里看它在实测数据上准不准。

## 每讲结构

每一讲的小节名字是固定的，方便检索，但不用每次都从头读到尾。读的时候只需要知道它们分成三组：前面几节告诉你这一讲在讲什么、需要什么基础；从「核心概念」到「实验或练习」是正文；最后的问答和自测是用来检查自己会不会讲。

| 部分 | 作用 |
| --- | --- |
| 摘要 | 3–5 句话：问题、核心结论、主线上的取舍 |
| 一页速览卡 | 面试前 5 分钟：五件事、必须默写的公式、一句话区分 |
| 学习目标 / 先修知识 | 本讲在系统栈里的位置 |
| 核心概念 | 概念、公式、张量 shape、直觉 |
| 关键推导 | 主要公式的推导与边界条件 |
| 深入推导 | M02–M17 必有：完整步骤、代价模型、和代码的对应 |
| 白板估算 | 用 A100/H100 的真实参数把公式算成数字 |
| 实现映射 | 这个概念在哪个实验文件里，怎样算做对 |
| 易错点与反例 | 常见误解与会失效的场景 |
| 实验或练习 | 可执行的最小实验 |
| 高频问题 | 基础问答 |
| 面试追问链 | 由浅入深的连续追问和回答框架 |
| 自测题 | 手算、证明、设计题，带要点 |
| 关联笔记 | 前后讲的依赖 |
| 本讲小结 / 延伸阅读 | 收束与论文入口 |

## 辅助导航

- [术语与符号表](GLOSSARY.md)：读到不认识的词或符号时查这里，每个词一句话解释，并指向讲它的那一讲。
- [按岗位和时间预算的阅读路线](READING-ROADMAP.md)：不打算全读时，看这里挑。
- [硬件参数速查](../interview/HARDWARE-NUMBERS.md)：做白板估算时要用的 A100/H100 参数。

## 笔记使用法

一讲不用一遍读完。我自己一般是这样过的：

先只读摘要、速览卡和学习目标，弄清这一讲在栈的哪一层、要解决什么问题。然后读核心概念和两节推导，读完合上笔记，自己把公式重推一遍。推不出来的地方就是还没懂的地方。

接着做白板估算。先不看答案，用硬件参数自己算，再去对。对完打开「实现映射」里的实验，跑一遍测试，改一个参数，看数字怎么变。

面试前不用重读全文。把追问链和自测题的答案说出来，说顺了就够了；临场只翻速览卡、白板估算和追问链。

## 校验

改动笔记后，可以用下面的脚本检查结构、公式块、链接、索引和术语表是否完整：

```bash
python notes/check_notes.py            # 结构 / 公式 / 链接 / 索引 / 术语
python notes/check_notes.py --verbose  # 每讲行数、公式块数、问题数
```

笔记只写原创摘要与短引用，不复制官方材料。
