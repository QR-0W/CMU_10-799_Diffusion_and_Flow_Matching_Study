<div align="center">

<sub>INDEPENDENT COURSE STUDY</sub>

# CMU 10-799

### Diffusion & Flow Matching

从概率生成建模出发，沿着扩散模型、score-based models 与 flow matching，走向现代生成模型的训练、采样和研究实践。

[课程网站](https://kellyyutonghe.github.io/10799S26/) · [课程笔记](notes.md) · [复习笔记](复习笔记.md) · [作业工作区](homeworks/project/) · [引用记录](homeworks/project/code/REFERENCES.md)

![Course](https://img.shields.io/badge/Course-Spring%202026-6D5BD0?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-2E8B57?style=flat-square)

</div>

---

## 关于课程

CMU 10-799 是 Yutong He 于 Spring 2026 开设的七周生成模型课程。课程从 DDPM 的概率基础与工程实现开始，进一步覆盖 score-based models、flow matching、快速采样、可控生成、蒸馏、离散扩散和离散流匹配。

本仓库是非官方自学记录，不代表 Carnegie Mellon University，也不是课程官方解答。四次作业在同一代码库中持续演进，重点不只是复现结果，还包括推导、调试、实验设计和可复现性。

| 概率基础 | 扩散模型 | 流匹配 | 前沿主题 |
|:---:|:---:|:---:|:---:|
| Likelihood<br>VAE & ELBO | DDPM<br>Score SDE | Flow Matching<br>ODE Solvers | Guidance<br>Distillation<br>Discrete Models |

## 课程安排

| 周次 | 日期 | 主题 |
|---:|---|---|
| 1 | Jan 13 | Basics of Probabilistic & Generative Modeling |
| 1 | Jan 15 | Denoising Diffusion Models |
| 1 | Jan 16 | Guest: Training and serving models on Modal |
| 2 | Jan 20 | Score-Based Models |
| 2 | Jan 22 | Flow Matching |
| 3 | Jan 27 | Diffusion design space and fast sampling solvers |
| 3 | Jan 29 | Guidance & Controllable Generation |
| 4 | Feb 3 | Guest: Diffusion and flow for robotics and control |
| 4 | Feb 5 | SOTA diffusion/flow models for text-to-image generation |
| 5 | Feb 10 | Distillation, Consistency Models & Flow Maps |
| 5 | Feb 12 | Guest: Linqi "Alex" Zhou, Luma AI |
| 6 | Feb 17 | Discrete Diffusion & Masked Diffusion |
| 6 | Feb 19 | Discrete Flow Matching & Edit Flow |
| 7 | Feb 24 | Poster preparation, no class |
| 7 | Feb 26 | Final Poster Presentation |

| 考核 | 比例 | 实践重点 |
|---|---:|---|
| HW1 | 15% | DDPM、U-Net、KID 与采样步数消融 |
| HW2 | 15% | Flow Matching、DDIM 与方法比较 |
| HW3 | 20% | 保真度、可控性或速度方向的 baseline |
| HW4 | 20% | 改进方法、对比实验与消融研究 |
| Quizzes | 15% | 课程概念与推导 |
| Final Presentation | 15% | Poster 与研究总结 |

具体日期与要求以[课程网站](https://kellyyutonghe.github.io/10799S26/)和作业 PDF 为准。

## 仓库结构

```text
.
├── notes.md                       # 按讲次整理的课程笔记
├── 复习笔记.md                    # 按概念依赖整理的复习材料
├── VAE_Notes/                     # VAE 与相关生成模型专题
└── homeworks/
    ├── assignments/               # 课程发布的 HW1-HW4 PDF
    └── project/
        ├── code/                  # 模型实现与引用记录
        │   └── configs/           # 训练和评估使用的实验配置
        ├── scripts/workstation/   # 单卡工作站任务队列与辅助脚本
        ├── docs/hw1...hw4/        # 实验步骤、结果、图表与作业答案
        ├── docs/public/           # 跨作业规范与资源说明
        └── artifacts/             # 数据、模型与日志，不进入 Git
```

公开仓库只保存复现所需的代码、配置、随机种子、commit 和环境版本。主机地址、账户、存储路径、凭据、实时资源状态和私有日志不得提交。

## 环境需求

| 类别 | 要求 |
|---|---|
| 系统 | Linux 或 WSL2，Git |
| Python | Python 3.10+，使用 `uv` 或 Conda 管理环境 |
| GPU | 支持 CUDA 的 NVIDIA GPU；完整训练建议至少 24 GB 显存 |
| 存储 | 建议预留 100-200 GB，用于数据、checkpoint 和评估样本 |

开发、文档、静态检查、训练、采样和评估统一在当前单卡 RTX 3090 工作站完成。精确环境由 `code/environments/requirements-cu126-lock.txt` 固定，并记录 Python、PyTorch、CUDA 与驱动版本；正式实验通过 commit、配置文件和随机种子保证可复现性，运行时大文件统一保存在 `homeworks/project/artifacts/`。

计算预算与单 GPU 执行策略见 [`homeworks/project/docs/public/compute-resources.md`](homeworks/project/docs/public/compute-resources.md)。

## 参考资料

- [CMU 10-799: Diffusion & Flow Matching](https://kellyyutonghe.github.io/10799S26/)
- [Official starter code](https://github.com/KellyYutongHe/cmu-10799-diffusion)
- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Deep Learning](https://www.deeplearningbook.org/)

每讲的阅读材料与视频链接收录在 [`notes.md`](notes.md)。论文、外部代码、教程和 AI 工具统一登记在 [`homeworks/project/code/REFERENCES.md`](homeworks/project/code/REFERENCES.md)，并在对应作业报告中保留相关条目。

## 许可声明

本仓库用于个人学习与研究记录。课程名称、讲义、作业 PDF、数据集及其他第三方材料的版权归原作者和权利人所有；仓库中的整理不构成 CMU 或课程团队的认可与背书。

仓库当前未附带开源许可证，因此除法律明确允许的情形外，不授予复制、修改或再分发原创代码与文字的许可。引用本仓库内容时请注明来源，并遵守课程的学术诚信要求。原始课程材料请从课程网站获取。
