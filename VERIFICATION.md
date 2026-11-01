# 仓库完整性检查

最后检查：2026-10-28。

## 自动检查

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 笔记结构、公式块、追问链、链接、术语 | `python notes/check_notes.py` | 18 讲通过 |
| 全部测试 | `./scripts/run-all-tests.ps1` | 11 套全部 PASS：430 通过，14 个 GPU 测试在本机（无 CUDA）跳过；RTX 5090 服务器上 GPU 测试全部运行通过 |
| Markdown 相对链接、实验目录结构、报告（PDF 存在、图能解析、每条参考文献都被引用且无未定义引用）、CS149 共享头文件一致、密钥扫描 | `./scripts/check-repo.ps1`（报告部分是 `scripts/check_reports.py`） | 通过，11 份报告 |
| 综述参考文献 | 按标题查询 Crossref / arXiv、URL 可达性；查询受限的条目人工核对标题、年份、出处、卷期页码 | 131 篇：69 篇自动匹配，62 篇人工核对，无错误 |
| CI | [`.github/workflows/ci.yml`](.github/workflows/ci.yml)：Ubuntu + Python 3.12，`requirements.txt`（CPU torch），`check-repo.ps1` + `run-all-tests.sh` | 配置就绪（Gitee 不原生运行 GitHub Actions，镜像到 GitHub 后生效） |

## 测试清单

| 套件 | 测试数 | 关键的正确性检查 |
| --- | ---: | --- |
| tools | 25 | 参数量与官方一致（Llama-3-8B 8.03B、70B 70.6B）；KV/token；16N；ring 通信量；采样分布 |
| CS149 A1 | 24 | 三种划分覆盖全部行；SIMD 利用率公式；C++ 六个 program 与标量参考逐位相等 |
| CS149 A2 | 87 | 随机 DAG 的依赖顺序；无丢失任务；`sync` 等待全部；带超时防死锁；C++ 每个系统 × 20 个官方测试 |
| CS149 A3 | 94 + 1 GPU | 三种 scan 与 `np.cumsum` 一致（含非 2 的幂）；分块渲染与顺序参考逐位相等；CUDA 版现场编译运行 |
| CS149 A4 | 21 + 1 GPU | 分块矩阵乘、systolic 模拟、im2col 与参考一致；bank conflict 模型；GPU kernel 对 fp64 torch |
| CS229S P1 | 33 + 1 GPU | Flash 前向/反向与 torch autograd 差 < 1e-5（fp64 实测 4.9e-15）；IO 计数器与解析模型一致；Triton 版对 SDPA |
| CS229S P2 | 15 + 1 GPU | KV cache 与无 cache 的贪心输出和 logits 一致；推测采样分布检验；GPU 版对 HF GPT-2 |
| 15-849 分布式 | 78 + 2 | 四种 all-reduce 求和正确；TP 前向和梯度与未切分一致；ZeRO 0–3 与单进程一致；GPU 显存模型 |
| 15-849 服务 | 23 + 4 GPU | 引用计数与 CoW；无泄漏；分页 attention 与连续 attention 一致；调度守恒；Triton 分页 kernel |
| 15-849 编译 | 29 + 4 GPU | 自动微分对有限差分和 torch；每个图 pass 前后输出一致；LoRA 初始输出等于基座、合并后一致；Triton 融合 kernel |
| CS149 A5 | 1 | 链、星、网格、ER、RMAT 上五种 BFS 与串行距离逐元素相等；PageRank pull 与 push 的 \(\ell_1\) 和归一化 |

## 数字的边界

- 实验机器：4 物理核 / 8 逻辑核的 Windows Server 虚拟机，无 GPU，Python 3.12、numpy 2.5、torch 2.6 CPU。
  建仓时多个实验并行运行，5–8 worker 的计时有噪声，各 README 已注明。
- CPU 版 README 中 H100 上的时间、吞吐、时延来自 `tools/mlsys_calc.py` 的 roofline / α-β 模型，标注为模型输出。
- GPU 实测平台：AutoDL 单卡 RTX 5090（sm_120，32 GB，CUDA 12.8，torch 2.8）与 2 × Xeon Gold 6459C（cgroup 16 CPU，固定 12 个物理核）。
  多租户宿主机，CPU 实验 3 轮取中位数；GPU 计时用 CUDA event。原始指标在各实验 `artifacts/raw/gpu|cpp/`。
- 容器不开放 GPU 性能计数器（ncu `ERR_NVGPUCTRPERM`）：occupancy、寄存器、bank conflict、DRAM 字节数为理论值或模型推算，报告中已标注。
- 硬件参数取稠密值，来源见 [HARDWARE-NUMBERS](interview/HARDWARE-NUMBERS.md)。

## 已知缺失

- 官方 starter code 与 slides 未镜像（github.com 不可达），见 [UPSTREAM.md](UPSTREAM.md)；
- CS229S 项目题面未公开，实验按课程主页描述实现；
- CS149 A5 已在本机 4 核上完成（OpenMP BFS 与 PageRank），没有官方 grader 的性能分；
- 只有单卡：多卡 NCCL 带宽未测；没有 vLLM 对照（用自写分页引擎代替）。见 [PROGRESS.md](PROGRESS.md)。
- 服务器上「新建/被唤醒的工作线程每条指令更贵」（A1 每次 launch +8.9 ms、A2 睡眠池扩展缺口）在本机 4 核上不复现，
  已排除唤醒开销、负载不均和实现缺陷；剩余的频率管理与 SMT 兄弟被占两个候选需要 turbostat，容器内无权限。

## 复验

```powershell
python notes/check_notes.py
./scripts/run-all-tests.ps1
./scripts/check-repo.ps1
```
