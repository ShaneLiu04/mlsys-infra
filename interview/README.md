# 面试准备

面试前我自己是这样用这个目录的：先分清对面在考概念、考估算，还是考写代码，然后只打开对应的那一个文件。十个文件都扫一遍，不如把一个文件讲顺。下面的表是按「遇到哪种问题」排的，不是按阅读顺序排的。

| 文件 | 什么时候打开 |
| --- | --- |
| [HARDWARE-NUMBERS.md](HARDWARE-NUMBERS.md) | 所有估算都从这里起步，先把「十个数」背下来 |
| [FORMULAS.md](FORMULAS.md) | 对方说「写一下」「估一下」的时候。22 条公式，每条有推导、口算和常见错答 |
| [SYSTEM-DESIGN.md](SYSTEM-DESIGN.md) | 设计题：70B 推理服务、70B 预训练、GRPO 系统、多 LoRA、容错、长上下文 |
| [PORTFOLIO.md](PORTFOLIO.md) | 讲自己做过的实验。每个故事有 20 秒版本、白板推导和「不能推出的结论」 |
| [CODING.md](CODING.md) | 手写题：online softmax、FlashAttention 前向、ring all-reduce、KV cache、块分配器、线程池、量化 |
| [QUESTION-BANK.md](QUESTION-BANK.md) | 42 道按系统栈分组的概念题，每题附常见错答 |
| [COURSE-QA.md](COURSE-QA.md) | 18 讲，每讲三个一句话就能答完的问题 |
| [ENGINEERING.md](ENGINEERING.md) | 被问「慢了、OOM 了、挂住了怎么查」时的排查顺序 |

## 三种面试怎么准备

概念面一般 30–45 分钟。先把 [COURSE-QA](COURSE-QA.md) 过一遍，再按岗位从题库里挑两部分练。
每道题的答法都是：先说结论，再说为什么，最后主动提一个常见的错误说法。最后这一步最能让面试官分清你是背过还是真懂。

估算和设计面一般 45–60 分钟。先背 [HARDWARE-NUMBERS](HARDWARE-NUMBERS.md) 的十个数，
再把 [FORMULAS](FORMULAS.md) 里的 F2、F10、F11、F15、F16 默写一遍，然后给自己限时 20 分钟做一道
[设计题](SYSTEM-DESIGN.md)。现场的顺序是先报估算结果，再画框图，最后讲取舍。

手写代码面也是 45–60 分钟。[CODING](CODING.md) 里每道题在本仓库都有参考实现和测试。
练的时候先别看参考实现，自己写一遍，跑对应的测试，过了再去对比。

## 讲实验的规则

本仓库的数字分两类，讲的时候要先说清是哪一类。一类是实测，来自 RTX 5090 和 Xeon 6459C（后来又在 RTX 4080 SUPER、RTX 6000D 上复测过），
这类数字可以说「我测到了」，但要带上卡型、精度和 batch。另一类是模型，包括用 H100 参数算的 roofline、α-β 和事件模拟，
以及早期在 4 核笔记本 CPU 上的 numpy 复现，这类只能讲机制和相对关系，不能说成「我在 H100 上测到了多少 token/s」。
每个故事后面都有一段「不能推出的结论」，就是为了守住这条线。
