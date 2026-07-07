# CMU 10-799: Diffusion & Flow Matching — 自学笔记与代码实现

> 📚 **自学仓库** | 基于 Carnegie Mellon University 官方课程  
> Spring 2026, Mini 3 | Tue & Thu 5:00 PM - 6:20 PM, SH 105  
> 讲师：Prof. Yifan Guo

---

## 📖 关于本仓库

本仓库是我自学 **CMU 10-799: Diffusion & Flow Matching** 课程的学习记录，包含：

- 📝 课程笔记与知识点总结
- 💻 课后作业的完整代码实现
- 🔬 三个专业方向的实验与探索
- 📄 原始作业 PDF 及参考资料

如果你也想自学这门课，欢迎参考本仓库的结构和内容，一起交流学习！

---

## 🎯 课程概述

Stable Diffusion、DALL-E、Sora —— 这些震惊世界的生成式 AI 模型背后，究竟是什么样的数学原理？本课程从概率建模的数学基础出发，系统讲解**扩散模型（Diffusion Models）** 与**流匹配（Flow Matching）** 两大前沿框架。

### 覆盖主题

| 主题 | 说明 |
|------|------|
| 去噪扩散模型 (DDPM) | 正向加噪 & 反向去噪的数学原理 |
| 基于分数的随机微分方程 (Score-based SDE) | 连续时间视角下的扩散过程 |
| 流匹配 (Flow Matching) | 从概率路径到生成模型的新范式 |
| 快速采样算法 | DDIM、蒸馏等方法加速推理 |
| 可控生成 (Controllable Generation) | Classifier-Free Guidance 等 |
| 流蒸馏 (Flow Distillation) | 模型压缩与加速 |
| 离散扩散模型 | 文本等离散数据的扩散建模 |

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
CMU_10-799_Diffusion&Flow_Matching_Study/
├── README.md                 # 本文件
├── notes.md                  # 课程笔记（按课时组织）
├── homeworks/                # 原始作业 PDF
│   ├── hw1.pdf               # 作业 1：DDPM 基础实现
│   ├── hw2.pdf               # 作业 2：Flow Matching 基础实现
│   ├── hw3.pdf               # 作业 3：专业方向实现
│   └── hw4.pdf               # 作业 4：高级课题探索
├── code/                     # 作业代码实现
│   ├── hw1/                  # 作业 1 代码
│   ├── hw2/                  # 作业 2 代码
│   ├── hw3/                  # 作业 3 代码（含方向分支）
│   └── hw4/                  # 作业 4 代码
└── resources/                # 参考资料与论文链接
```

---

## 🗓️ 学习路线图

### 第一阶段：基础入门（作业 1-2）

- [ ] **作业 1：DDPM — 去噪扩散概率模型**
  - 实现正向扩散过程（加噪调度）
  - 实现 U-Net 去噪网络
  - 训练与采样（MNIST / CIFAR-10）
  - 评估指标：FID、Inception Score

- [ ] **作业 2：Flow Matching — 流匹配**
  - 理解概率路径与向量场
  - 实现 Conditional Flow Matching (CFM)
  - 对比 DDPM vs Flow Matching
  - 在相同数据集上训练与评估

### 第二阶段：专业深入（作业 3-4）

- [ ] **作业 3：选择你的升级路径**
  - 🎨 保真度：提升生成质量
  - 🎮 可控性：添加条件控制
  - ⚡ 速度：加速采样过程

- [ ] **作业 4：高级课题**
  - 在所选方向上进一步探索
  - 阅读并复现相关论文
  - 最终项目展示

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

## 📚 推荐参考资料

### 核心论文
- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through SDEs (Song et al., 2021)](https://arxiv.org/abs/2011.13456)
- [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)

### 教程与博客
- [What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Understanding Diffusion Models (Calvin Luo)](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html)
- [Flow Matching Guide (Jakub M. Tomczak)](https://jmtomczak.github.io/blog/18/18_fm.html)

### 开源代码库
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## 📝 学习日志

| 日期 | 内容 | 状态 |
|------|------|------|
| 2026-07-07 | 仓库初始化，整理课程资料 | ✅ |
| - | 作业 1：DDPM 实现 | ⬜ |
| - | 作业 2：Flow Matching 实现 | ⬜ |
| - | 作业 3：专业方向 | ⬜ |
| - | 作业 4：高级课题 | ⬜ |

---

## 📄 许可与声明

- 本仓库为个人自学记录，原始课程内容版权归 CMU 及课程讲师所有。
- 作业代码为个人实现，仅供参考学习。
- 如需访问原始课程资源，请访问 CMU 官方课程页面。

---

> 💡 **关于本课程**：这门课**无考试**且**鼓励使用 AI 工具**（ChatGPT、Copilot 等）辅助编程——重点在于理解和构建，而非死记硬背。适合任何对生成式 AI 底层原理感兴趣的学习者！

---

⭐ 如果这个仓库对你有所帮助，欢迎 Star & Fork！
