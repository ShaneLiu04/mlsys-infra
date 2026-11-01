# 机器学习系统十八讲

原创中文笔记，保留英文术语，不逐页翻译官方 slides。顺序按系统栈从下往上排，
每讲标出它在 CS149、CS229S、CMU 15-849 里的出处，合并理由见 [CURRICULUM.md](../CURRICULUM.md)。
面试入口在仓库根目录的 [README](../README.md)，公式卡和题库在 [interview/](../interview/README.md)。

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

## 每讲结构

| 部分 | 作用 |
| --- | --- |
| 摘要 | 3–5 句话：问题、核心结论、主线上的取舍 |
| 一页速览卡 | 面试前 5 分钟：五件事、必须默写的公式、一句话区分 |
| 学习目标 / 先修知识 | 本讲在系统栈里的位置 |
| 核心概念 | 概念、公式、张量 shape、直觉 |
| 关键推导 | 主要公式的推导与边界条件 |
| 深入推导 | M02–M17 必有：完整步骤、代价模型、和代码的对应 |
| 白板估算 | 用 A100/H100 的真实参数把公式算成数字 |
| 实现映射 | 概念 → 实验文件 → 验收标准 |
| 易错点与反例 | 常见误解与会失效的场景 |
| 实验或练习 | 可执行的最小实验 |
| 高频问题 | 基础问答 |
| 面试追问链 | 由浅入深的连续追问和回答框架 |
| 自测题 | 手算、证明、设计题，带要点 |
| 关联笔记 | 前后讲的依赖 |
| 本讲小结 / 延伸阅读 | 收束与论文入口 |

## 辅助导航

- [术语与符号表](GLOSSARY.md)
- [按岗位和时间预算的阅读路线](READING-ROADMAP.md)
- [硬件参数速查](../interview/HARDWARE-NUMBERS.md)

## 笔记使用法

1. 首读「摘要 → 速览卡 → 学习目标」，确定这一讲在栈里的哪一层；
2. 二读「核心概念 → 关键推导 → 深入推导」，合上笔记重推公式；
3. 做「白板估算」：不看答案，用硬件参数算出数字，再对答案；
4. 打开「实现映射」里的实验，跑测试，改一个参数看数字怎么变；
5. 用「面试追问链 → 自测题」做输出式复习；
6. 面试前只过「速览卡 + 白板估算 + 追问链」。

## 校验

```bash
python notes/check_notes.py            # 结构 / 公式 / 链接 / 索引 / 术语
python notes/check_notes.py --verbose  # 每讲行数、公式块数、问题数
```

笔记只写原创摘要与短引用，不复制官方材料。
