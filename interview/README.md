# 面试准备

| 文件 | 什么时候打开 |
| --- | --- |
| [HARDWARE-NUMBERS.md](HARDWARE-NUMBERS.md) | 所有估算的起点。先背「十个数」 |
| [FORMULAS.md](FORMULAS.md) | 对方说「写一下」「估一下」。22 条公式，每条带推导、口算和常见错答 |
| [SYSTEM-DESIGN.md](SYSTEM-DESIGN.md) | 设计题：70B 推理服务、70B 预训练、GRPO 系统、多 LoRA、容错、长上下文 |
| [PORTFOLIO.md](PORTFOLIO.md) | 讲自己做过的实验。每条有 20 秒结论、白板和「不能推出的结论」 |
| [CODING.md](CODING.md) | 手写题：online softmax、FlashAttention 前向、ring all-reduce、KV cache、块分配器、线程池、量化 |
| [QUESTION-BANK.md](QUESTION-BANK.md) | 42 道按系统栈分组的概念题，每题带常见错答 |
| [COURSE-QA.md](COURSE-QA.md) | 18 讲的一句话答案 |
| [ENGINEERING.md](ENGINEERING.md) | 「慢了/OOM/挂住了怎么查」的排查顺序 |

## 三种面试怎么准备

**概念面（30–45 分钟）**：先过 [COURSE-QA](COURSE-QA.md)，再按岗位挑题库的两部分。
每题先说结论，再说为什么，最后主动说一个常见错答——这是区分「背过」和「懂」的最快方式。

**估算 / 设计面（45–60 分钟）**：背 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md) 的十个数，
把 [FORMULAS](FORMULAS.md) 的 F2、F10、F11、F15、F16 默写一遍，然后限时 20 分钟做一道
[设计题](SYSTEM-DESIGN.md)。先报估算，再画框图，最后讲取舍。

**手写代码面（45–60 分钟）**：[CODING](CODING.md) 里的题都有本仓库的参考实现和测试。
练法是先关掉参考实现写一遍，跑对应测试，再对比。

## 讲实验的规则

本仓库的实验都在 4 物理核（8 逻辑核）的 CPU 上跑。能讲的是**正确性**（和参考实现逐元素相等）、**相对比较**
（同机器上 A 比 B 快几倍、浪费少多少）和**由公式推出的模型输出**（模拟器）。不能讲成
「我在 GPU 上测到了多少 token/s」。每个故事的「不能推出的结论」就是为这个准备的。
