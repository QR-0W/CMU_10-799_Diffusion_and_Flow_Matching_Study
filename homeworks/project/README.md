# Coursework Workspace

四次作业在同一代码库中增量实现，开发、检查、训练和评估均在当前单卡 RTX 3090 工作站完成，实验配置、结果和报告按作业隔离。

## 快速入口

- [作业索引](../README.md)
- [实验记录规范](docs/public/experiment-conventions.md)
- [公开计算资源说明](docs/public/compute-resources.md)
- [引用与工具披露](code/REFERENCES.md)

## 工作区

```text
project/
├── code/
│   ├── configs/hw1...hw4/          # 按作业提交的实验配置
│   ├── environments/
│   │   └── requirements-cu126-lock.txt
│   ├── scripts/                    # 训练与评估入口
│   ├── src/                        # 模型、方法、数据与路径定义
│   ├── train.py
│   ├── sample.py
│   └── download_dataset.py
├── scripts/workstation/            # 当前工作站的单卡任务队列
├── docs/hw1...hw4/                 # 实验步骤、结果、图表与作业答案
├── docs/public/                    # 跨作业规范
└── artifacts/                      # Git 忽略的运行时大文件
    ├── datasets/
    ├── checkpoints/
    ├── logs/
    ├── samples/
    ├── metrics/
    └── queue/
```

## 核心原则

源码在 `code/` 中持续演进，不为每次作业复制一份。机器可读配置统一放在 `code/configs/`，实验步骤、结果、图表和作业答案统一写入 `docs/hwN/` 的同一组文档，不再维护独立报告目录；大文件统一写入 `artifacts/`。正式实验只使用已提交的 commit，同一比较组固定数据划分、checkpoint、采样种子与评估参数。

所有项目命令从 `code/` 执行。相对路径由 `src/utils/paths.py` 统一按 `code/` 解析，默认数据集位于 `../artifacts/datasets/celeba-subset/`，训练日志、checkpoint、样本和指标分别写入对应的 `artifacts/` 子目录。

具体命名、文档、图片、产物和信息披露规则统一定义在 [`docs/public/experiment-conventions.md`](docs/public/experiment-conventions.md)，本页不重复维护。
