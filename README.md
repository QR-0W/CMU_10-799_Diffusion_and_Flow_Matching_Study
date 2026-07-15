# CMU 10-799: Diffusion & Flow Matching — 自学笔记与代码实现

> 📚 **自学仓库** | 基于 Carnegie Mellon University 官方课程  
> Spring 2026, Mini 3 (7 周) | Tue & Thu 5:00 PM - 6:20 PM, SH 105  
> 讲师：Prof. Yutong He (Kelly He)  
> 课程网站：[10799S26](https://kellyyutonghe.github.io/10799S26/schedule/)

---

## 📖 关于本仓库

本仓库是我自学 **CMU 10-799: Diffusion & Flow Matching** 课程的学习记录，包含：

- 📝 课程笔记与知识点总结（见 [`notes.md`](./notes.md)）
- 💻 课后作业的完整代码实现
- 🔬 三个专业方向的实验与探索
- 📄 原始作业 PDF 及参考资料

如果你也想自学这门课，欢迎参考本仓库的结构和内容，一起交流学习！

---

## 🎯 课程概述

Stable Diffusion、DALL-E、Sora —— 这些震惊世界的生成式 AI 模型背后，究竟是什么样的数学原理？这门 7 周的 Mini 课程从概率建模的数学基础出发，系统讲解**扩散模型（Diffusion Models）** 与**流匹配（Flow Matching）** 两大前沿框架，一路深入到最新的离散扩散与多模态生成。

### 三大专业方向

从作业 3 开始，可选择以下方向深入：

| 方向 | 目标 | 关键词 |
|------|------|--------|
| 🎨 **保真度 (Fidelity)** | 追求照片级真实感 | 高分辨率、细节增强、感知损失 |
| 🎮 **可控性 (Controllability)** | 精确的用户控制 | 条件生成、ControlNet、图像编辑 |
| ⚡ **速度 (Speed)** | 实时生成 | 蒸馏、一步生成、高效采样 |

---

## 📂 仓库结构

```
CMU_10-799_Diffusion_and_Flow_Matching_Study/
├── README.md                 # 本文件
├── notes.md                  # 课程笔记（按课时组织）
├── homeworks/
│   ├── assignments/          # HW1-HW4 原始作业 PDF
│   └── project/              # 四次作业共享代码与实验工作区
│       ├── code/             # 持续演进的课程代码库
│       ├── experiments/      # 按作业和实验拆分的配置与记录
│       ├── scripts/          # 本地、远端和评估脚本
│       ├── docs/             # 推导、设计与算力评估
│       └── reports/          # 各次作业报告材料
└── resources/                # 参考资料与论文链接
```

---

## 🗓️ 课程安排（7 周 · 13 讲 + 3 场 Guest Lecture）

### 第 1 周：概率基础与扩散模型入门

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L1 | Jan 13 | **Basics of Probabilistic & Generative Modeling** | [📝](./notes.md#lecture-1--0113-basics-of-probabilistic--generative-modeling) |
| L2 | Jan 15 | **Denoising Diffusion Models** | [📝](./notes.md#lecture-2--0115-denoising-diffusion-models) |
| L3 | Jan 16 | **Guest (Modal):** How to train & serve your models on Modal | [📝](./notes.md#lecture-3-guest--0116-how-to-train--serve-your-models-on-modal) |

> 📌 **Quiz 1** (L4)

### 第 2 周：分数模型与流匹配

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L4 | Jan 20 | **Score-Based Models** | [📝](./notes.md#lecture-4--0120-score-based-models) |
| L5 | Jan 22 | **Flow Matching** | [📝](./notes.md#lecture-5--0122-flow-matching) |

> 📌 **Quiz 2** · **⏰ HW 1 (15%) Due Jan 24 Sat**

### 第 3 周：设计空间与可控生成

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L6 | Jan 27 | **The Design Space of Diffusion Models & Solvers for Fast Sampling** | [📝](./notes.md#lecture-6--0127-the-design-space-of-diffusion-models--solvers-for-fast-sampling) |
| L7 | Jan 29 | **Guidance & Controllable Generation** | [📝](./notes.md#lecture-7--0129-guidance--controllable-generation) |

> 📌 **Quiz 3**

### 第 4 周：前沿图像生成 & 机器人应用

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L8 | Feb 3 | **Guest (Max Simchowitz):** Diffusion & Flow for Robotics, Control & Decision Making | [📝](./notes.md#lecture-8-guest--0203-qa-with-max-simchowitz--diffusion--flow-for-robotics-control--decision-making) |
| L9 | Feb 5 | **SOTA Diffusion/Flow Models for Text-to-Image Generation** | [📝](./notes.md#lecture-9--0205-sota-diffusionflow-models-for-text-to-image-generation) |

> 📌 **Quiz 4** · **⏰ HW 2 (15%) Due Feb 5 Thu**

### 第 5 周：蒸馏与一致性模型

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L10 | Feb 10 | **Distillation, Consistency Models & Flow Maps** | [📝](./notes.md#lecture-10--0210-distillation-consistency-models--flow-maps) |
| L11 | Feb 12 | **Guest (Linqi "Alex" Zhou, Luma AI)** | [📝](./notes.md#lecture-11-guest--0212-linqi-alex-zhou-from-luma-ai) |

> 📌 **Quiz 5** · **⏰ HW 3 (20%) Due Feb 15 Sun**

### 第 6 周：离散扩散与离散流匹配

| # | 日期 | 主题 | 笔记 |
|:--:|------|------|:--:|
| L12 | Feb 17 | **Discrete Diffusion & Masked Diffusion** | [📝](./notes.md#lecture-12--0217-discrete-diffusion--masked-diffusion) |
| L13 | Feb 19 | **Discrete Flow Matching & Edit Flow** | [📝](./notes.md#lecture-13--0219-discrete-flow-matching--edit-flow) |

> 📌 **Quiz 6 · Quiz 7**

### 第 7 周：最终展示

| # | 日期 | 主题 |
|:--:|------|------|
| L14 | Feb 24 | **No Class**（准备 Poster） |
| L15 | Feb 26 | **Final Poster Presentation** |

> 📌 **Poster Due Feb 25 Wed** · **⏰ HW 4 (20%) Due Feb 27 Fri**

---

## 📊 成绩组成

| 项目 | 占比 | 说明 |
|------|:---:|------|
| HW 1 | 15% | DDPM 基础实现 |
| HW 2 | 15% | Flow Matching 基础实现 |
| HW 3 | 20% | 专业方向实现（保真度/可控性/速度） |
| HW 4 | 20% | 高级课题探索 |
| Final Presentation | 15% | Poster 展示 |
| Quizzes (×7) | 15% | 随堂测验 |

---

## 📝 学习进度

| 周次 | 日期范围 | 内容 | 作业 | 状态 |
|:----:|----------|------|:----:|:----:|
| 1 | Jul 07 - Jul 13 | L1-L3: 概率基础、DDPM、Modal Guest | HW1 开始 | 🔄 |
| 2 | Jul 14 - Jul 20 | L4-L5: Score Models & Flow Matching | ⏰ HW1 Due | ⬜ |
| 3 | Jul 21 - Jul 27 | L6-L7: Fast Sampling & Guidance | HW2 开始 | ⬜ |
| 4 | Jul 28 - Aug 03 | L8-L9: Robotics Guest & SOTA T2I | ⏰ HW2 Due | ⬜ |
| 5 | Aug 04 - Aug 10 | L10-L11: Distillation & Luma AI Guest | HW3 开始 | ⬜ |
| 6 | Aug 11 - Aug 17 | L12-L13: Discrete Diffusion & Flow | ⏰ HW3 Due | ⬜ |
| 7 | Aug 18 - Aug 24 | L14-L15: Poster 准备 & Final Present | HW4 | ⬜ |
| 🎯 | Aug 25 - Aug 27 | — | ⏰ Poster & HW4 Due | ⬜ |

---

## 🔧 环境配置

```bash
# 创建虚拟环境
conda create -n diffusion-course python=3.10
conda activate diffusion-course

# 安装核心依赖
pip install torch torchvision
pip install diffusers accelerate
pip install matplotlib numpy tqdm
pip install pytorch-fid  # 用于 FID 评估
```

---

## 📚 参考资料

> 📖 每讲的完整阅读清单（论文 + 教程）见 **[notes.md](./notes.md)**，均来自课程官网。

### 核心教材

- [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890) — Lai, Song, Kim, Mitsufuji, Ermon（课程核心参考书）
- [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville（Ch.3 概率论, Ch.20 生成模型）

### 姊妹课程

- [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/)
- [MIT 6.S184: Flow Matching & Diffusion Models](https://diffusion.csail.mit.edu/2025/index.html)
- [CMU 10-423/10-623: Generative AI](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/)

### 开源工具

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [FLUX (Black Forest Labs)](https://github.com/black-forest-labs/flux)

---

## 📄 许可与声明

- 本仓库为个人自学记录，原始课程内容版权归 CMU 及课程讲师所有。
- 作业代码为个人实现，仅供参考学习。
- 如需访问原始课程资源，请访问 [课程官网](https://kellyyutonghe.github.io/10799S26/)。

---

> 💡 **关于本课程**：这门课**无考试**且**鼓励使用 AI 工具**（ChatGPT、Copilot 等）辅助编程——重点在于理解和构建，而非死记硬背。适合任何对生成式 AI 底层原理感兴趣的学习者！

---

⭐ 如果这个仓库对你有所帮助，欢迎 Star & Fork！
