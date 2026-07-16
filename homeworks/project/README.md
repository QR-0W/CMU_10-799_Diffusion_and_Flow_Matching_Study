# Coursework Workspace

四次作业在同一代码库中增量实现，实验配置、结果和报告按作业隔离。

## 快速入口

- [作业索引](../README.md)
- [实验记录规范](docs/public/experiment-conventions.md)
- [公开计算资源说明](docs/public/compute-resources.md)
- [引用与工具披露](code/REFERENCES.md)

## 工作区

- `code/`：官方 starter code、后续模型实现与引用记录
- `code/configs/`：训练和评估直接读取的可复现实验配置
- `scripts/`：本地运行、远端队列与评估脚本
- `docs/hw1/` 至 `docs/hw4/`：实验说明、设计、步骤、运行记录、诊断、指标摘要和结论
- `docs/public/`：跨作业的公开规范和资源说明
- `docs/private/`：主机、账户与部署信息；由 Git 忽略
- `reports/`：作业报告材料
- `artifacts/`：数据、checkpoint、日志、样本和完整指标；由 Git 忽略

## 核心原则

源码在 `code/` 中持续演进，不为每次作业复制一份。机器可读配置统一放在 `code/configs/`，所有叙述性实验信息统一放在 `docs/hwN/`；同一比较组固定数据划分、checkpoint、采样种子与评估参数。

具体命名、文档、图片、产物和信息披露规则统一定义在 [`docs/public/experiment-conventions.md`](docs/public/experiment-conventions.md)，本页不重复维护。
