# 仓库完整性检查

这份文件记录「怎么知道仓库里的东西是对的」：跑了哪些检查，结果如何，数字的适用范围在哪里，还有哪些已知的缺口。

最后检查：2026-11-08。

## 自动检查

| 检查什么 | 命令 | 结果 |
| --- | --- | --- |
| 笔记的结构、公式块、追问链、链接、术语表 | `python notes/check_notes.py` | 18 讲全部通过 |
| 全部测试 | `./scripts/run-all-tests.ps1` | 11 套全部通过：446 个通过，14 个 GPU 测试因本机没有 CUDA 而跳过；RTX 5090 服务器上 GPU 测试全部运行并通过 |
| Markdown 相对链接、实验目录结构、报告（PDF 存在、图能找到、每条参考文献都被引用且没有未定义的引用）、CS149 两份共享头文件一致、密钥扫描 | `./scripts/check-repo.ps1`（报告部分由 `scripts/check_reports.py` 完成） | 通过，11 份报告 |
| 文档里引用的关键数字与原始数据一致 | `python scripts/check_claims.py`（对照 `claims.json`） | 20 条全部吻合 |
| 综述的参考文献 | 按标题查 Crossref 和 arXiv，检查 URL 能否打开；查不到的条目人工核对标题、年份、出处和卷期页码 | 131 篇：69 篇自动匹配，62 篇人工核对，没有发现错误 |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：Ubuntu + Python 3.12，安装 `requirements.txt`（CPU 版 torch），运行 `check-repo.ps1` 和 `run-all-tests.sh` | 配置就绪。Gitee 不原生运行 GitHub Actions，镜像到 GitHub 后生效 |

## 每套测试检查了什么

测试数是「通过 + GPU 跳过」。

| 套件 | 测试数 | 最关键的正确性检查 |
| --- | ---: | --- |
| tools | 37 | 参数量与官方一致（Llama-3-8B 8.03B、70B 70.6B）；每 token 的 KV 字节数；16N 显存；ring 通信量；采样分布 |
| CS149 A1 | 24 | 三种划分覆盖了全部行；SIMD 利用率公式；C++ 六个 program 与标量参考逐位相等 |
| CS149 A2 | 87 | 随机任务图的依赖顺序；不丢任务；`sync` 会等全部任务结束；带超时防止死锁；C++ 每个系统跑 20 个官方测试 |
| CS149 A3 | 96 + 1 GPU | 三种 scan 与 `np.cumsum` 一致（含长度不是 2 的幂的情况）；分块渲染与顺序参考逐位相等；CUDA 版现场编译运行 |
| CS149 A4 | 22 + 1 GPU | 分块矩阵乘、systolic 模拟、im2col 与参考一致；bank conflict 模型；GPU kernel 对照 fp64 的 torch |
| CS149 A5 | 1 | 链、星、网格、ER、RMAT 五种图上，五种 BFS 与串行距离逐元素相等；PageRank 的 pull 与 push 两种写法的 \(\ell_1\) 差和归一化 |
| CS229S P1 | 33 + 1 GPU | Flash 前向和反向与 torch autograd 的差 < 1e-5（fp64 下实测 4.9e-15）；IO 计数器与解析模型一致；Triton 版对照 SDPA |
| CS229S P2 | 15 + 1 GPU | 有无 KV cache 的贪心输出和 logits 一致；推测采样的分布检验；GPU 版对照 HF 的 GPT-2 |
| 15-849 分布式 | 78 + 2 GPU | 四种 all-reduce 求和正确；张量并行的前向和梯度与不切分时一致；ZeRO 0–3 与单进程一致；GPU 显存模型 |
| 15-849 服务 | 24 + 4 GPU | 引用计数与 copy-on-write；没有泄漏；分页 attention 与连续 attention 一致；调度前后请求守恒；Triton 分页 kernel |
| 15-849 编译 | 29 + 4 GPU | 自动微分对照有限差分和 torch；每个图优化 pass 前后输出一致；LoRA 初始输出等于基座，合并后仍一致；Triton 融合 kernel |

## 数字的适用范围

读任何一个数字之前，先确认它是在哪台机器上、用什么方法得到的：

- **CPU 版实验**：4 物理核 / 8 逻辑核的 Windows Server 虚拟机，没有 GPU，Python 3.12、numpy 2.5、torch 2.6 CPU。
  建仓时多个实验同时运行，所以 5–8 个 worker 的计时有噪声，各 README 里已注明。
- **H100 上的时间、吞吐和时延**：CPU 版 README 里出现的这类数字来自 `tools/mlsys_calc.py` 的 roofline 和 α-β 模型，是模型输出，不是实测。
- **GPU 实测**：AutoDL 单卡 RTX 5090（sm_120，32 GB，CUDA 12.8，torch 2.8）和 2 × Xeon Gold 6459C（cgroup 16 个 CPU，固定 12 个物理核）。
  宿主机是多租户的，CPU 实验跑 3 轮取中位数；GPU 计时用 CUDA event。原始数据在各实验的 `artifacts/raw/gpu/` 和 `artifacts/raw/cpp/`。
  换卡复测的 RTX 4080 SUPER 和 RTX 6000D 数据分别在 `artifacts/raw/ada4080/` 和 `artifacts/raw/rtx6000d/`。
- **性能计数器**：容器不开放（ncu 报 `ERR_NVGPUCTRPERM`），所以 occupancy、寄存器、bank conflict、DRAM 字节数都是理论值或模型推算，报告里已标注。
- **硬件参数**取稠密计算的数值，来源见 [HARDWARE-NUMBERS](interview/HARDWARE-NUMBERS.md)。

## 已知缺口

- 官方 starter code 和 slides 没有镜像，因为 github.com 访问不到，见 [UPSTREAM.md](UPSTREAM.md)。
- CS229S 的项目题面没有公开，实验按课程主页上的描述实现。
- CS149 A5 在本机 4 核上完成（OpenMP 的 BFS 和 PageRank），没有官方 grader 给的性能分。
- 只有单卡，多卡 NCCL 带宽没测，见 [PROGRESS.md](PROGRESS.md)。
- 服务器上有一个现象在本机 4 核上不复现：新建或刚被唤醒的工作线程，每条指令似乎更贵（A1 每次 launch 多 8.9 ms，A2 睡眠池的扩展性有缺口）。
  唤醒开销、负载不均和实现缺陷都已经排除。剩下两个候选是 CPU 频率管理和 SMT 兄弟线程被别的租户占用，要用 turbostat 才能区分，容器内没有权限。

## 自己复验

```powershell
python notes/check_notes.py
./scripts/run-all-tests.ps1
./scripts/check-repo.ps1
```
