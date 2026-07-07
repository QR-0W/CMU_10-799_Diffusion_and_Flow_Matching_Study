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
├── homeworks/                # 原始作业 PDF
│   ├── hw1.pdf               # 作业 1：DDPM 基础实现 (15%)
│   ├── hw2.pdf               # 作业 2：Flow Matching 基础实现 (15%)
│   ├── hw3.pdf               # 作业 3：专业方向实现 (20%)
│   └── hw4.pdf               # 作业 4：高级课题探索 (20%)
├── code/                     # 作业代码实现
│   ├── hw1/                  # 作业 1 代码
│   ├── hw2/                  # 作业 2 代码
│   ├── hw3/                  # 作业 3 代码（含方向分支）
│   └── hw4/                  # 作业 4 代码
└── resources/                # 参考资料与论文链接
```

---

## 🗓️ 课程安排（7 周 · 13 讲 + 3 场 Guest Lecture）

### 第 1 周：概率基础与扩散模型入门

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 1 | Jan 13 | **Basics of Probabilistic & Generative Modeling** | 生成建模基础、概率论回顾、VAE、GAN | [📝](./notes.md#lecture-1) |
| 2 | Jan 15 | **Denoising Diffusion Models (DDPM)** | VAE→扩散连接、DDPM 正向/反向过程、Karras 设计空间 | [📝](./notes.md#lecture-2) |
| 👥 | Jan 16 | **Guest: Modal** — 如何训练与部署扩散/流模型 | | |

> 📌 **Quiz 1** · **HW1 发布**

### 第 2 周：分数模型与流匹配

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 4 | Jan 20 | **Score-Based Models** | Score Matching、NCSN、SDE 统一框架 | [📝](./notes.md#lecture-4) |
| 5 | Jan 22 | **Flow Matching** | Conditional Flow Matching、Rectified Flow、Stochastic Interpolants | [📝](./notes.md#lecture-5) |

> 📌 **Quiz 2** · **⏰ HW1 Due (Jan 24 Sat)**

### 第 3 周：设计空间与可控生成

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 6 | Jan 27 | **Design Space & Fast Sampling** | DDIM、DPM-Solver、Progressive Distillation | [📝](./notes.md#lecture-6) |
| 7 | Jan 29 | **Guidance & Controllable Generation** | Classifier Guidance、CFG、SDEdit、RePaint、DPS | [📝](./notes.md#lecture-7) |

> 📌 **Quiz 3**

### 第 4 周：前沿图像生成 & 机器人应用

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 👥 | Feb 3 | **Guest: Max Simchowitz** — 扩散与流在机器人/控制/决策中的应用 | | |
| 9 | Feb 5 | **SOTA Text-to-Image Models** | LDM/SD、DiT、SD3/MMDiT、FLUX、Transfusion、GPT-4o | [📝](./notes.md#lecture-9) |

> 📌 **Quiz 4** · **⏰ HW2 Due (Feb 5 Thu)**

### 第 5 周：蒸馏与一致性模型

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 10 | Feb 10 | **Distillation, Consistency Models & Flow Maps** | Progressive Distillation、Consistency Models、CTM、Flow Map Distillation | [📝](./notes.md#lecture-10) |
| 👥 | Feb 12 | **Guest: Linqi "Alex" Zhou (Luma AI)** | 工业界实践分享 | |

> 📌 **Quiz 5** · **⏰ HW3 Due (Feb 15 Sun)**

### 第 6 周：离散扩散与离散流匹配

| # | 日期 | 主题 | 核心内容 | 笔记 |
|---|------|------|----------|:--:|
| 12 | Feb 17 | **Discrete Diffusion & Masked Diffusion** | D3PM、连续时间离散扩散、SEDD、MDLM、LLaDA | [📝](./notes.md#lecture-12) |
| 13 | Feb 19 | **Discrete Flow Matching & Edit Flow** | CTMC 离散流匹配、Edit Flows、OneFlow、Block Diffusion | [📝](./notes.md#lecture-13) |

> 📌 **Quiz 6 · Quiz 7**

### 第 7 周：最终展示

| # | 日期 | 主题 |
|---|------|------|
| — | Feb 24 | **无课**（准备最终展示） |
| 🎯 | Feb 26 | **Final Poster Presentation**（课堂 Poster Session） |

> 📌 **Poster Due (Feb 25 Wed)** · **⏰ HW4 Due (Feb 27 Fri)**

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
| 1 | Jul 07 - Jul 13 | L1-L2: 概率基础与 DDPM | HW1 | 🔄 进行中 |
| 2 | Jul 14 - Jul 20 | L4-L5: Score Models & Flow Matching | HW1 Due | ⬜ |
| 3 | Jul 21 - Jul 27 | L6-L7: Fast Sampling & Guidance | HW2 | ⬜ |
| 4 | Jul 28 - Aug 03 | L9: SOTA T2I Models | HW2 Due | ⬜ |
| 5 | Aug 04 - Aug 10 | L10: Distillation & Consistency | HW3 | ⬜ |
| 6 | Aug 11 - Aug 17 | L12-L13: Discrete Diffusion/Flow | HW3 Due | ⬜ |
| 7 | Aug 18 - Aug 24 | Final Project | HW4 | ⬜ |
| 🎯 | Aug 25 - Aug 27 | Poster & HW4 Due | — | ⬜ |

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

## 📚 核心参考资料

### 必读论文（按课程顺序）

| 讲次 | 论文 |
|:----:|------|
| L1 | Kingma & Welling, *VAE* (2013); Goodfellow et al., *GAN* (2014) |
| L2 | Ho et al., *DDPM* (2020); Sohl-Dickstein et al., *Deep Unsupervised Learning* (2015); Karras et al., *Design Space* (2022) |
| L4 | Song & Ermon, *NCSN* (2019); Song et al., *Score-SDE* (2021) |
| L5 | Lipman et al., *Flow Matching* (2023); Liu et al., *Rectified Flow* (2023); Albergo et al., *Stochastic Interpolants* (2023) |
| L6 | Song et al., *DDIM* (2021); Lu et al., *DPM-Solver* (2022); Nichol & Dhariwal, *Improved DDPM* (2021) |
| L7 | Dhariwal & Nichol, *Classifier Guidance* (2021); Ho & Salimans, *CFG* (2022); Meng et al., *SDEdit* (2022) |
| L9 | Rombach et al., *LDM* (2022); Peebles & Xie, *DiT* (2023); Esser et al., *SD3* (2024); Labs, *FLUX* |
| L10 | Salimans & Ho, *Progressive Distillation* (2022); Song et al., *Consistency Models* (2023) |
| L12 | Austin et al., *D3PM* (2021); Campbell et al., *CT Discrete Diffusion* (2022); Lou et al., *SEDD* (2024) |
| L13 | Campbell et al., *Discrete Flow Matching* (2024); Gat et al., *Edit Flow* (2025) |

### 推荐教程

- [What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Understanding Diffusion Models (Calvin Luo)](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html)
- [Flow Matching Guide (Jakub M. Tomczak)](https://jmtomczak.github.io/blog/18/18_fm.html)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

---

## 📄 许可与声明

- 本仓库为个人自学记录，原始课程内容版权归 CMU 及课程讲师所有。
- 作业代码为个人实现，仅供参考学习。
- 如需访问原始课程资源，请访问 [课程官网](https://kellyyutonghe.github.io/10799S26/)。

---

> 💡 **关于本课程**：这门课**无考试**且**鼓励使用 AI 工具**（ChatGPT、Copilot 等）辅助编程——重点在于理解和构建，而非死记硬背。适合任何对生成式 AI 底层原理感兴趣的学习者！

---

⭐ 如果这个仓库对你有所帮助，欢迎 Star & Fork！
