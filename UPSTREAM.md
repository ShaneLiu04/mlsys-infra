# 上游材料

这个仓库只收录我自己写的笔记、面试材料和实现。官方材料请按下表的地址获取。
`scripts/sync-upstream.ps1` 会把 starter code 克隆到 `assignments/`（只读，不改动），并记下每个仓库的 commit。

## 状态

2026-10-28 建仓时，公司网络访问 github.com 返回 407（代理要求认证），所以 starter code 至今没有镜像。
三门课的大纲、日程和作业列表是从官方网站逐项核对的，写在 [CURRICULUM.md](CURRICULUM.md)。

因为拿不到 starter，`solutions/` 里的实验都是按作业要求独立实现的。每个 README 里「对应官方作业」那一节写明了和原作业的差异。

## Stanford CS149 Parallel Computing（Fall 2024）

| 材料 | 地址 | 本仓库位置 |
| --- | --- | --- |
| 课程主页与日程 | https://gfxcourses.stanford.edu/cs149/fall24 | 已核对，见 CURRICULUM |
| Slides | 课程主页每讲链接 | `slides/cs149/`（未镜像） |
| A1 Analyzing Parallel Program Performance on a Quad-Core CPU | https://github.com/stanford-cs149/asst1 | `assignments/cs149/asst1` |
| A2 Scheduling Task Graphs on a Multi-Core CPU | https://github.com/stanford-cs149/asst2 | `assignments/cs149/asst2` |
| A3 A Simple Renderer in CUDA | https://github.com/stanford-cs149/asst3 | `assignments/cs149/asst3` |
| A4 Optimizing DNN Performance on DNN Accelerator Hardware | https://github.com/stanford-cs149/asst4-trainium | `assignments/cs149/asst4-trainium` |
| A5 (EC) Big Graph Processing in OpenMP | https://github.com/stanford-cs149/biggraphs-ec | `assignments/cs149/biggraphs-ec` |
| 往年 cs149gpt（fused / flash attention on CPU） | https://github.com/stanford-cs149/cs149gpt | `assignments/cs149/cs149gpt` |

## Stanford CS229S Systems for Machine Learning（Fall 2023）

| 材料 | 地址 | 本仓库位置 |
| --- | --- | --- |
| 课程主页 | https://cs229s.stanford.edu/fall2023/ | 已核对 |
| 日程与 slides | https://cs229s.stanford.edu/fall2023/calendar/ | 已核对，slides 未镜像 |
| Project 1：Transformer 训练、两层 Transformer 硬编码联想回忆、MHA 算术强度、1D depthwise conv CUDA kernel | 课程 Canvas / 主页 | 题面未公开，按主页描述实现 |
| Project 2：KV cache 与推测解码（nanoGPT） | 课程 Canvas / 主页 | 题面未公开，按主页描述实现 |
| 研究资料合集 | https://github.com/HazyResearch/aisys-building-blocks | 链接 |
| nanoGPT（Project 2 的骨架） | https://github.com/karpathy/nanoGPT | 链接 |

## CMU 15-849 Machine Learning Systems（Zhihao Jia）

| 材料 | 地址 | 本仓库位置 |
| --- | --- | --- |
| 课程主页 | https://www.cs.cmu.edu/~zhihaoj2/15-849/ | 已核对 |
| 日程与必读论文 | https://www.cs.cmu.edu/~zhihaoj2/15-849/schedule.html | 已核对（页面为 Spring 2022 日程） |
| 编程作业 | 无（论文研讨课） | `solutions/cmu15849/` 为论文最小复现 |

## 同步

```powershell
# 需要能访问 github.com；公司网络下先设置可用代理
./scripts/sync-upstream.ps1
```

脚本把每个仓库的 HEAD commit 追加到本文件末尾的「已同步」表。
