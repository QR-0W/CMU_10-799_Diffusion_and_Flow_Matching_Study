# CMU 10-799: Diffusion & Flow Matching — 课程笔记

> 📝 按课时组织的详细笔记 | 配合 [README](./README.md) 中的课程安排使用

---

## 目录

- [Lecture 1: Basics of Probabilistic & Generative Modeling](#lecture-1-basics-of-probabilistic--generative-modeling)
- [Lecture 2: Denoising Diffusion Models (DDPM)](#lecture-2-denoising-diffusion-models-ddpm)
- [Guest: Modal — Training & Serving Models](#guest-modal---training--serving-models)
- [Lecture 4: Score-Based Models](#lecture-4-score-based-models)
- [Lecture 5: Flow Matching](#lecture-5-flow-matching)
- [Lecture 6: Design Space & Fast Sampling](#lecture-6-design-space--fast-sampling)
- [Lecture 7: Guidance & Controllable Generation](#lecture-7-guidance--controllable-generation)
- [Guest: Max Simchowitz — Diffusion/Flow for Robotics & Control](#guest-max-simchowitz---diffusionflow-for-robotics--control)
- [Lecture 9: SOTA Text-to-Image Models](#lecture-9-sota-text-to-image-models)
- [Lecture 10: Distillation, Consistency Models & Flow Maps](#lecture-10-distillation-consistency-models--flow-maps)
- [Guest: Linqi "Alex" Zhou (Luma AI)](#guest-linqi-alex-zhou-luma-ai)
- [Lecture 12: Discrete Diffusion & Masked Diffusion](#lecture-12-discrete-diffusion--masked-diffusion)
- [Lecture 13: Discrete Flow Matching & Edit Flow](#lecture-13-discrete-flow-matching--edit-flow)

---

## Lecture 1: Basics of Probabilistic & Generative Modeling

> 📅 Jan 13 | 第 1 周

### 1.1 概率建模基础

- [ ] 生成模型的定义与目标：学习数据分布 $p_{\text{data}}(x)$
- [ ] 似然函数与最大似然估计 (MLE)
- [ ] 潜变量模型 (Latent Variable Models)
- [ ] KL 散度与证据下界 (ELBO)

### 1.2 经典生成模型回顾

- [ ] **Variational Autoencoders (VAE)**
  - Encoder-Decoder 架构
  - Reparameterization Trick
  - 损失函数推导：重构损失 + KL 正则
  - 优缺点：模糊样本、mode collapse

- [ ] **Generative Adversarial Networks (GAN)**
  - Generator vs Discriminator 的博弈框架
  - 训练不稳定性与 mode collapse
  - 与 VAE 的对比

### 1.3 似然基模型 vs 隐式模型

- [ ] 基于似然的模型（VAE, Flow-based, Autoregressive）
- [ ] 隐式模型（GAN）
- [ ] 扩散模型的定位：似然基 + 迭代式

### 📄 参考阅读

- Kingma & Welling, *Auto-Encoding Variational Bayes* (2013)
- Goodfellow et al., *Generative Adversarial Nets* (2014)

---

## Lecture 2: Denoising Diffusion Models (DDPM)

> 📅 Jan 15 | 第 1 周

### 2.1 从 VAE 到扩散模型

- [ ] 层级 VAE 与多步潜变量
- [ ] 扩散模型的直觉：逐步加噪 → 逐步去噪

### 2.2 DDPM 正向过程 (Forward/Diffusion Process)

- [ ] 马尔可夫链定义：$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$
- [ ] 噪声调度 (Noise Schedule)：$\beta_1 < \beta_2 < \cdots < \beta_T$
- [ ] 闭式采样：$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$
- [ ] $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ 的推导

### 2.3 DDPM 反向过程 (Reverse Process)

- [ ] 反向条件分布 $q(x_{t-1} | x_t, x_0)$ 的推导
- [ ] 变分下界 (Variational Lower Bound) 分解
- [ ] 简化训练目标：$\mathbb{E}_{t, x_0, \epsilon}[||\epsilon - \epsilon_\theta(x_t, t)||^2]$
- [ ] 为什么预测噪声 $\epsilon$ 而非均值？

### 2.4 架构设计

- [ ] U-Net 基础结构
- [ ] 时间步嵌入 (Time Embedding)
- [ ] Karras et al. 设计空间分析要点

### 📄 参考阅读

- Ho et al., *Denoising Diffusion Probabilistic Models* (2020)
- Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics* (2015)
- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative Models* (2022)

---

## Guest: Modal — Training & Serving Models

> 📅 Jan 16 | Guest Lecture

- [ ] 扩散/流模型的训练基础设施
- [ ] 模型部署与服务化
- [ ] 实践技巧与经验分享

---

## Lecture 4: Score-Based Models

> 📅 Jan 20 | 第 2 周

### 4.1 Score Matching

- [ ] Score 的定义：$\nabla_x \log p(x)$ — 指向高密度区域的方向
- [ ] Hyvärinen Score Matching：隐式匹配，无需知道归一化常数
- [ ] Denoising Score Matching：$\nabla_x \log p(x_t | x_0) = -\frac{x_t - x_0}{\sigma^2}$
- [ ] Sliced Score Matching

### 4.2 Noise Conditional Score Networks (NCSN)

- [ ] 多噪声级别的必要性：低密度区域问题
- [ ] 退火 Langevin Dynamics 采样
- [ ] 噪声条件网络设计

### 4.3 SDE 统一框架

- [ ] 正向 SDE：$dx = f(x,t)dt + g(t)dw$
- [ ] 反向 SDE：$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$
- [ ] SDE 视角下 DDPM 与 NCSN 的统一
- [ ] 概率流 ODE (Probability Flow ODE)：确定性采样

### 4.4 SDE 类型

- [ ] Variance Exploding (VE) SDE：对应 NCSN
- [ ] Variance Preserving (VP) SDE：对应 DDPM
- [ ] Sub-VP SDE

### 📄 参考阅读

- Song & Ermon, *Generative Modeling by Estimating Gradients of the Data Distribution* (2019)
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations* (2021)

---

## Lecture 5: Flow Matching

> 📅 Jan 22 | 第 2 周

### 5.1 连续归一化流 (Continuous Normalizing Flows)

- [ ] ODE 定义的变换：$\frac{dx}{dt} = v_t(x)$
- [ ] 瞬时变量变换定理 (Instantaneous Change of Variables)
- [ ] CNF 的训练困难：需要 ODE 求解器反传

### 5.2 Conditional Flow Matching (CFM)

- [ ] 核心思想：对条件向量场做回归
- [ ] CFM 损失：$\mathbb{E}_{t, p_t(x|x_1), q(x_1)}[||v_t(x) - u_t(x|x_1)||^2]$
- [ ] 与 score matching 损失的等价性
- [ ] 关键优势：无需 ODE 反传，训练稳定

### 5.3 概率路径设计

- [ ] Gaussian 路径：$p_t(x|x_1) = \mathcal{N}(x; \mu_t(x_1), \sigma_t^2 I)$
- [ ] 条件向量场：$u_t(x|x_1) = \frac{\dot{\sigma}_t}{\sigma_t}(x - \mu_t) + \dot{\mu}_t$
- [ ] Optimal Transport (OT) 路径：直线路径 + 常数方差

### 5.4 变体与扩展

- [ ] **Rectified Flow**：直线路径 + 重加权机制
- [ ] **Stochastic Interpolants**：统一的插值框架
- [ ] Flow Matching vs Score Matching 的比较

### 📄 参考阅读

- Lipman et al., *Flow Matching for Generative Modeling* (2023)
- Liu et al., *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow* (2023)
- Albergo et al., *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions* (2023)

---

## Lecture 6: Design Space & Fast Sampling

> 📅 Jan 27 | 第 3 周

### 6.1 扩散模型设计空间

- [ ] Elucidating the Design Space (Karras et al., 2022)
  - 网络架构选择
  - 训练配置（噪声调度、损失加权）
  - 采样器选择
- [ ] Improved DDPM (Nichol & Dhariwal, 2021)
  - 可学习的噪声调度
  - 余弦噪声调度

### 6.2 快速采样：DDIM

- [ ] 非马尔可夫正向过程
- [ ] DDIM 确定性采样：$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(x_t, t)$
- [ ] 子序列加速：用少量步数（~50 步）替代全步数（~1000 步）
- [ ] DDIM 反演 (Inversion) 用于图像编辑

### 6.3 高级 ODE 求解器

- [ ] DPM-Solver：针对扩散 ODE 的专用高阶求解器
- [ ] DPM-Solver++：改进的数值稳定性
- [ ] 半线性结构利用
- [ ] 10-20 步即可达到高质量采样

### 6.4 Progressive Distillation（预览）

- [ ] 逐步将步数减半：教师模型 → 学生模型
- [ ] 与快速采样器结合使用

### 📄 参考阅读

- Song et al., *Denoising Diffusion Implicit Models* (2021)
- Lu et al., *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling* (2022)
- Nichol & Dhariwal, *Improved Denoising Diffusion Probabilistic Models* (2021)

---

## Lecture 7: Guidance & Controllable Generation

> 📅 Jan 29 | 第 3 周

### 7.1 Classifier Guidance

- [ ] 利用预训练分类器引导生成
- [ ] 分数修正：$\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$
- [ ] 引导强度 (Guidance Scale) 的权衡：质量 vs 多样性

### 7.2 Classifier-Free Guidance (CFG)

- [ ] 隐式分类器：$p(y|x) \propto \frac{p(x|y)}{p(x)}$
- [ ] 联合训练条件/无条件模型
- [ ] CFG 采样：$\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$
- [ ] 为什么 CFG 是主流方案？

### 7.3 可控生成方法

- [ ] **SDEdit**：以噪声为桥梁，从引导图像生成
- [ ] **RePaint**：扩散模型的 Inpainting
- [ ] **Diffusion Posterior Sampling (DPS)**：逆问题求解
- [ ] **Manifold Preserving Guidance**：在流形上操作
- [ ] **FreeDoM**：无需训练的引导方法

### 7.4 条件注入方式

- [ ] 文本条件（Cross-Attention）
- [ ] 图像条件（ControlNet 式附加输入）
- [ ] 布局/结构条件

### 📄 参考阅读

- Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis* (2021)
- Ho & Salimans, *Classifier-Free Diffusion Guidance* (2022)
- Meng et al., *SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations* (2022)

---

## Guest: Max Simchowitz — Diffusion/Flow for Robotics & Control

> 📅 Feb 3 | Guest Lecture

- [ ] 扩散/流模型在机器人学中的应用
- [ ] 行为克隆与扩散策略 (Diffusion Policy)
- [ ] 规划与控制中的生成式方法
- [ ] Q&A 讨论

---

## Lecture 9: SOTA Text-to-Image Models

> 📅 Feb 5 | 第 4 周

### 9.1 潜空间扩散模型 (Latent Diffusion Models)

- [ ] 从像素空间到潜空间：感知压缩
- [ ] VAE Encoder → U-Net in Latent Space → VAE Decoder
- [ ] **Stable Diffusion** 系列架构剖析

### 9.2 Transformer 架构

- [ ] **DiT** (Diffusion Transformer)：用 Transformer 替代 U-Net
  - Patchify → Transformer Blocks → Unpatchify
  - AdaLN (Adaptive Layer Norm) 条件机制
- [ ] **SD3 / MMDiT**：多模态扩散 Transformer
  - 双流注意力（文本流 + 图像流）
  - 联合建模文本/图像表示

### 9.3 最新模型

- [ ] **FLUX**：Black Forest Labs 的前沿模型
- [ ] **Transfusion**：统一多模态生成
- [ ] **Gemini / GPT-4o 的多模态能力**

### 9.4 文本编码器与条件机制

- [ ] CLIP：对比语言-图像预训练
- [ ] T5：强大的文本编码器
- [ ] 文本-图像对齐的关键技术

### 📄 参考阅读

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022)
- Peebles & Xie, *Scalable Diffusion Models with Transformers* (2023)
- Esser et al., *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (2024)

---

## Lecture 10: Distillation, Consistency Models & Flow Maps

> 📅 Feb 10 | 第 5 周

### 10.1 Progressive Distillation

- [ ] 逐步蒸馏：将采样步数从 N 降至 N/2
- [ ] 教师 → 学生的知识迁移
- [ ] 蒸馏损失设计

### 10.2 Consistency Models

- [ ] 核心思想：直接从噪声映射到数据，一步生成
- [ ] 一致性函数：$f(x_t, t) \mapsto x_0$
- [ ] 自洽性条件：$f(x_t, t) = f(x_{t'}, t')$ 对同一条轨迹
- [ ] 训练方式：Consistency Distillation (CD) vs Consistency Training (CT)

### 10.3 Consistency Trajectory Models (CTM)

- [ ] 扩展到任意轨迹点之间的一致性
- [ ] 灵活控制采样步数（1 步到多步）

### 10.4 Flow Map Distillation

- [ ] 流图的概念
- [ ] 一步流匹配蒸馏
- [ ] 质量与速度的权衡

### 📄 参考阅读

- Salimans & Ho, *Progressive Distillation for Fast Sampling of Diffusion Models* (2022)
- Song et al., *Consistency Models* (2023)
- Kim et al., *Consistency Trajectory Models* (2024)

---

## Guest: Linqi "Alex" Zhou (Luma AI)

> 📅 Feb 12 | Guest Lecture

- [ ] Luma AI 的 Dream Machine
- [ ] 工业级视频生成模型
- [ ] 实际部署的挑战与解决方案

---

## Lecture 12: Discrete Diffusion & Masked Diffusion

> 📅 Feb 17 | 第 6 周

### 12.1 为什么需要离散扩散？

- [ ] 文本、代码、分子等离散数据的建模
- [ ] 连续扩散在离散空间中的局限性
- [ ] 离散扩散 vs 自回归模型的对比

### 12.2 D3PM: 离散去噪扩散概率模型

- [ ] 离散状态空间上的扩散：转移矩阵
- [ ] 正向过程：均匀噪声 / 掩码 (Mask) 转移
- [ ] 反向过程：预测干净数据

### 12.3 连续时间离散扩散

- [ ] CTMC (Continuous-Time Markov Chain) 视角
- [ ] 速率矩阵 $R_t$ 与转移概率
- [ ] 反向 CTMC 的模拟

### 12.4 最新离散扩散模型

- [ ] **SEDD** (Score Entropy Discrete Diffusion)
  - 离散 Score 的定义
  - 具体分数熵目标
- [ ] **MDLM** (Masked Diffusion Language Models)
  - 掩码扩散语言模型
  - 用 [MASK] token 的简单而有效的方式
- [ ] **LLaDA**：大语言离散扩散模型
- [ ] 简化掩码扩散

### 📄 参考阅读

- Austin et al., *Structured Denoising Diffusion Models in Discrete State Spaces* (2021)
- Campbell et al., *Continuous-Time Discrete Diffusion* (2022)
- Lou et al., *Score Entropy Discrete Diffusion Models* (2024)

---

## Lecture 13: Discrete Flow Matching & Edit Flow

> 📅 Feb 19 | 第 6 周

### 13.1 离散流匹配

- [ ] CTMC 上的流匹配框架
- [ ] 连续时间离散空间中的概率路径
- [ ] 条件离散流与采样

### 13.2 Edit Flows

- [ ] 编辑操作作为离散状态变化
- [ ] 对文本编辑序列建模
- [ ] 编辑流的训练与推理

### 13.3 混合模态生成

- [ ] **OneFlow**：混合连续-离散模态的统一流
- [ ] 同时生成图像+文本
- [ ] 跨模态对齐

### 13.4 Block Diffusion

- [ ] 分块去噪生成
- [ ] 介于自回归与一次性生成之间
- [ ] 灵活控制生成顺序

### 📄 参考阅读

- Campbell et al., *Generative Flows on Discrete State-Spaces* (2024)
- Gat et al., *Edit Flow* (2025)
- 相关 OneFlow 与 Block Diffusion 论文

---

## 附录

### A. 数学基础速查

- [ ] 高斯分布及其性质
- [ ] KL 散度的计算
- [ ] 重参数化技巧
- [ ] 伊藤积分的直观理解
- [ ] ODE/SDE 求解器基础

### B. 评估指标

- [ ] FID (Fréchet Inception Distance)
- [ ] Inception Score (IS)
- [ ] CLIP Score（文本-图像对齐）
- [ ] Precision & Recall（质量 vs 多样性）

### C. 常用数据集

- [ ] MNIST
- [ ] CIFAR-10
- [ ] ImageNet
- [ ] LAION（文本-图像）
- [ ] MS-COCO

---

> 📌 **使用方式**：每学完一讲，将勾选框 `[ ]` 改为 `[x]`，并在对应小节下补充自己的理解和推导。
