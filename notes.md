# CMU 10-799: Diffusion & Flow Matching — 课程笔记

> 📝 内容来自 [课程官网](https://kellyyutonghe.github.io/10799S26/schedule/) | 笔记由学习者自行填充

---

## Lecture 1 — 01/13: Basics of Probabilistic & Generative Modeling

### 资源

- [Lecture 1 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture1_Basics.pdf)
- [YouTube Video](https://youtu.be/p7Q77S_ZhdA)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=64500c0f-77c8-4610-a3ab-b3d101815eac)

### 阅读材料

**Tutorials:**

1. [Stanford CS236: Deep Generative Models](https://deepgenerativemodels.github.io/) — Stefano Ermon, Aditya Grover
2. [CMU 10-423/10-623: Generative AI](https://www.cs.cmu.edu/~mgormley/courses/10423-s25/) — Matt Gormley, Yuanzhi Li, Henry Chai, Pat Virtue, Aran Nayebi
3. [CMU 18-789: Deep Generative Modeling](https://cmu-dgm.github.io/index.html) — Beidi Chen, Xun Huang
4. [CMU 10-708: Probabilistic Graphical Models](https://andrejristeski.github.io/10708F25/) — Andrej Risteski, Albert Gu
5. [Stanford CS228: Probabilistic Graphical Models](https://ermongroup.github.io/cs228/) — Stefano Ermon
6. [The Principles of Diffusion Models - Chapter 1: Deep Generative Modeling](https://arxiv.org/pdf/2510.21890#page=20.19) — Lai, Song, Kim, Mitsufuji, Ermon
7. [Deep Learning - Chapter 3: Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) — Goodfellow, Bengio, Courville
8. [Deep Learning - Chapter 20: Deep Generative Models](https://www.deeplearningbook.org/contents/generative_models.html) — Goodfellow, Bengio, Courville
9. [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) — Kingma, Welling
10. [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — Carl Doersch

**Papers:**

1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma, Welling (foundational VAE paper)
2. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Goodfellow et al. (foundational GAN paper)

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 2 — 01/15: Denoising Diffusion Models

### 资源

- [Lecture 2 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture2_Diffusion.pdf)
- [YouTube Video](https://youtu.be/H-RbhdiWzto)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=dba3e0c7-7f13-447a-accf-b3d301838c94)

### 阅读材料

**Tutorials:**

1. [The Principles of Diffusion Models - Chapter 2: Variational Perspective: From VAEs to DDPMs](https://arxiv.org/pdf/2510.21890#page=37.19) — Lai, Song, Kim, Mitsufuji, Ermon
2. [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Lilian Weng
3. [Understanding Diffusion Models: A Unified Perspective](https://calvinyluo.com/2022/08/26/diffusion-tutorial.html) — Calvin Luo

**Papers:**

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho, Jain, Abbeel (foundational DDPM paper)
2. [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — Sohl-Dickstein et al. (original diffusion paper)
3. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Karras, Aittala, Aila, Laine

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 3 (Guest) — 01/16: How to train & serve your models on Modal

### 资源

- [YouTube Video](https://youtu.be/dPVpmv4eFnM)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=47b2faf1-c30d-4918-8df9-b3d7000d9e77)

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 4 — 01/20: Score-Based Models

> 📌 **Quiz 1 Due**

### 资源

- [Lecture 4 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture4_Score.pdf)
- [YouTube Video](https://youtu.be/UEJxHpFEb04)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d3616939-c5a3-48ad-8d68-b3d8018650c1)

### 阅读材料

**Tutorials:**

1. [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) — Yang Song
2. [The Principles of Diffusion Models - Appendix A: Crash Course on Differential Equations](https://arxiv.org/pdf/2510.21890#page=386.11) — Lai, Song, Kim, Mitsufuji, Ermon
3. [The Principles of Diffusion Models - Chapter 3: Score-Based Perspective: From EBMs to NCSN](https://arxiv.org/pdf/2510.21890#page=61.19) — Lai, Song, Kim, Mitsufuji, Ermon
4. [The Principles of Diffusion Models - Chapter 4: Diffusion Models Today: Score SDE Framework](https://arxiv.org/pdf/2510.21890#page=91.19) — Lai, Song, Kim, Mitsufuji, Ermon
5. [Generative Modeling by Estimating Gradients of the Data Distribution](https://andrewcharlesjones.github.io/journal/21-score-matching.html) — Andy Jones

**Papers:**

1. [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf) — Aapo Hyvärinen (original score matching paper)
2. [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) — Pascal Vincent (original denoising score matching paper)
3. [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Song, Ermon (annealed Langevin dynamics + NCSN)
4. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (foundational SDE unification paper)

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 5 — 01/22: Flow Matching

> 📌 **Quiz 2 Due** · **⏰ HW 1 (15%) Due 01/24 Sat**

### 资源

- [Lecture 5 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture5_Flow.pdf)
- [YouTube Video](https://youtu.be/_OOITDB2VCY)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=a75c4020-502f-4162-893e-b3da018375fa)

### 阅读材料

**Tutorials:**

1. [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) — Lipman et al.
2. [The Principles of Diffusion Models - Chapter 5: Flow-Based Perspective: From NFs to Flow Matching](https://arxiv.org/pdf/2510.21890#page=131.11) — Lai, Song, Kim, Mitsufuji, Ermon
3. [MIT 6.S184: Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2025/index.html) — Holderrieth, Erives
4. [An Introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html) — Fjelde, Mathieu, Dutordoir
5. [Flow Matching: A visual introduction](https://peterroelants.github.io/posts/flow_matching_intro/) — Peter Roelants
6. [Flow With What You Know](https://drscotthawley.github.io/blog/posts/FlowModels.html) — Scott H. Hawley

**Papers:**

1. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman, Chen, Ben-Hamu, Nickel, Le
2. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu, Gong, Liu
3. [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571) — Albergo, Vanden-Eijnden

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 6 — 01/27: The Design Space of Diffusion Models & Solvers for Fast Sampling

### 资源

- [Lecture 6 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture6_Fast.pdf)
- [YouTube Video](https://youtu.be/6-gp8fR9r8w)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=12d33dab-008b-4162-a402-b3df0186aae6)

### 阅读材料

**Papers:**

1. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song, Meng, Ermon (first fast deterministic sampling paper)
2. [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927) — Lu et al.
3. [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095) — Lu et al.
4. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Karras et al.
5. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — Nichol, Dhariwal
6. [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) — Kingma, Salimans, Poole, Ho
7. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Salimans, Ho

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 7 — 01/29: Guidance & Controllable Generation

> 📌 **Quiz 3 Due**

### 资源

- [Lecture 7 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture7_Guidance.pdf)
- [YouTube Video](https://youtu.be/lPipzIG6rkc)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=5d89c60e-0505-4852-ad73-b3e10187c916)

### 阅读材料

**Papers:**

1. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — Dhariwal, Nichol (classifier guidance)
2. [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho, Salimans
3. [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — Meng, He, Song, Song, Wu, Zhu, Ermon
4. [RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — Lugmayr et al.
5. [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687) — Chung et al.
6. [Manifold Preserving Guided Diffusion](https://arxiv.org/abs/2311.16424) — He et al.
7. [FreeDoM: Training-Free Energy-Guided Conditional Diffusion Model](https://arxiv.org/abs/2303.09833) — Yu et al.
8. [The Riemannian Geometry of Deep Generative Models](https://arxiv.org/abs/1711.08014) — Shao, Kumar, Fletcher
9. [Improving Diffusion Models for Inverse Problems using Manifold Constraints](https://arxiv.org/abs/2206.00941) — Chung et al.

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 8 (Guest) — 02/03: Q&A with Max Simchowitz — Diffusion & Flow for Robotics, Control & Decision Making

- Speaker: [Max Simchowitz](https://msimchowitz.github.io/)

### 资源

- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3e74017c-75d5-415d-8da4-b3e60184d9b7) *(No YouTube)*

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 9 — 02/05: SOTA Diffusion/Flow Models for Text-to-Image Generation

> 📌 **Quiz 4 Due** · **⏰ HW 2 (15%) Due 02/05 Thu**

### 资源

- [Lecture 9 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture9_SOTA.pdf)
- [YouTube Video](https://youtu.be/LHNPAtd7cU4)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=70f4eb33-b85f-4a03-a750-b3e80188564d)

### 阅读材料

**Papers:**

1. [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al. (Stable Diffusion / LDM)
2. [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — van den Oord, Vinyals, Kavukcuoglu (VQ-VAE)
3. [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841) — Esser, Rombach, Ommer (VQ-GAN)
4. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) — Radford et al. (CLIP)
5. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683) — Raffel et al. (T5)
6. [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — Peebles, Xie (DiT)
7. [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Esser et al. (SD3 / MMDiT)
8. [FLUX.1](https://github.com/black-forest-labs/flux) — Black Forest Labs
9. [FLUX.2: Frontier Visual Intelligence](https://bfl.ai/blog/flux-2) — Black Forest Labs
10. [Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer](https://arxiv.org/abs/2511.22699) — Tongyi
11. [HunyuanImage 3.0 Technical Report](https://arxiv.org/abs/2509.23951) — Tencent
12. [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039) — Zhou et al.
13. [Nano Banana (Gemini 2.5 Flash Image)](https://deepmind.google/models/gemini-image/) — Google DeepMind
14. [Introducing 4o Image Generation](https://openai.com/index/introducing-4o-image-generation/) — OpenAI

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 10 — 02/10: Distillation, Consistency Models & Flow Maps

> 📌 **Quiz 5 Due**

### 资源

- [Lecture 10 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture10_Distillation.pdf)
- [YouTube Video](https://youtu.be/L9nsCHHMv-c)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=bdf7b265-d319-4212-a58e-b3ed018b1f4f)

### 阅读材料

**Papers:**

1. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) — Salimans, Ho
2. [Consistency Models](https://arxiv.org/abs/2303.01469) — Song, Dhariwal, Chen, Sutskever (single-step generation)
3. [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](https://arxiv.org/abs/2310.02279) — Kim et al.
4. [How to build a consistency model: Learning flow maps via self-distillation](https://arxiv.org/abs/2505.18825) — Boffi, Albergo, Vanden-Eijnden
5. [Align Your Flow: Scaling Continuous-Time Flow Map Distillation](https://arxiv.org/abs/2506.14603) — Sabour, Fidler, Kreis
6. [Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models](https://arxiv.org/abs/2512.02636) — Ai et al.

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 11 (Guest) — 02/12: Linqi "Alex" Zhou from Luma AI

- Speaker: [Linqi (Alex) Zhou](https://alexzhou907.github.io/) · [Luma AI](https://lumalabs.ai/)

### 资源

- [Lecture 11 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/luma.pdf)
- [YouTube Video](https://youtu.be/H7MxR3XDt30)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=f47e7820-51c1-4c8f-b582-b3ef0189dd51)

> ⏰ **HW 3 (20%) Due 02/15 Sun**

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 12 — 02/17: Discrete Diffusion & Masked Diffusion

> 📌 **Quiz 6 Due**

### 资源

- [Lecture 12 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture12_Discrete_Diffusion.pdf)
- [YouTube Video](https://youtu.be/mXEjZblUBPs)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=ff97986e-c219-48f7-b67e-b3f401866d7f)

### 阅读材料

**Tutorials:**

1. [Discrete Diffusion (SEDD) blog post](https://aaronlou.com/blog/2024/discrete-diffusion) — Aaron Lou
2. [MDLM project page and video tutorial](https://s-sahoo.com/mdlm) — Subham Sekhar Sahoo
3. [Notes on D3PMs](https://beckham.nz/2022/07/11/d3pms.htm) — Christopher Beckham

**Papers:**

1. [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006) — Austin et al. (D3PM)
2. [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) — Campbell et al. (CTMC discrete diffusion)
3. [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) — Lou, Meng, Ermon (SEDD)
4. [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.03396) — Sahoo et al. (MDLM)
5. [Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329) — Shi et al.
6. [LLaDA: Large Language Diffusion with mAsking](https://arxiv.org/abs/2502.09992) — Nie et al.

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 13 — 02/19: Discrete Flow Matching & Edit Flow

> 📌 **Quiz 7 Due**

### 资源

- [Lecture 13 Slides](https://cmu-diffusion-10799.github.io/10799S26/assets/slides/Lecture13_Discrete_Flow.pdf)
- [YouTube Video](https://youtu.be/bK-LfpKLv0g)
- [Panopto Recording (CMU only)](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=3a289219-c1c7-4b82-bbd7-b3f60185f7dc)

### 阅读材料

**Papers:**

1. [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987) — Campbell et al.
2. [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997) — Campbell et al.
3. [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) — Gat et al.
4. [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) — Havasi et al.
5. [OneFlow: Concurrent Mixed-Modal and Interleaved Generation with Edit Flows](https://arxiv.org/abs/2510.03506) — Nguyen et al.
6. [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573) — Arriola et al.
7. [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.03396) — Sahoo et al. (MDLM)

### 📝 笔记

<!-- 在此记录你的笔记 -->

---

## Lecture 14 — 02/24: No Class

> ⏰ **Final Presentation (15%) Poster submission due 02/25 Wed**

---

## Lecture 15 — 02/26: Final Poster Presentation

> ⏰ **HW 4 (20%) Due 02/27 Fri**

---

## 附录：成绩组成

| 项目 | 占比 | 截止日期 |
|------|:---:|------|
| HW 1 | 15% | 01/24 Sat |
| HW 2 | 15% | 02/05 Thu |
| HW 3 | 20% | 02/15 Sun |
| HW 4 | 20% | 02/27 Fri |
| Final Presentation (Poster) | 15% | 02/25 Wed |
| Quizzes (×7) | 15% | 随堂 |
