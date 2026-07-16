# Experiment Conventions

本文定义四次作业共享的实验与文档规范。各目录的 `README.md` 只承担导航职责，具体规则以本文为准。

## 单一代码库

- 所有模型实现统一维护在 `code/`。
- HW2-HW4 直接扩展前序代码，不复制源码目录。
- 外部代码、论文、教程和 AI 工具统一登记在 `code/REFERENCES.md`。
- 正式结果必须能够由仓库中的 commit 和配置文件复现。

## 执行边界

- 本地只负责 coding、文档、静态检查和轻量 smoke test。
- 训练、采样、KID 和所有写入报告的正式结果统一在远端 RTX 3090 上运行，默认只使用一张 GPU。
- 远端只运行已提交的 commit，不以未提交文件生成正式结果。
- 本地 smoke test 不能记为 GPU hours，也不能替代远端实验记录。
- 远端运行前记录 commit、配置和随机种子，运行后记录退出状态与产物位置。

## 实验编号

实验 ID 使用 `hwN_expNN_name`，例如 `hw1_exp02_ddpm_baseline`。

每个实验在对应的 `docs/hwN/` 文档中至少记录：

- 目标与假设
- 代码 commit
- 配置文件
- 随机种子
- 环境版本
- GPU 数与有效 batch size
- 训练迭代数与 GPU hours
- checkpoint 标识
- 评估参数与指标摘要
- 结论和异常情况

实验状态使用 `planned`、`running`、`completed` 或 `failed`。失败实验同样保留原因和诊断，不用成功结果覆盖。

## 文档

作业过程文档放在 `docs/hwN/`，文件名使用 `homeworkN-<topic>.md`，例如：

```text
docs/hw1/homework1-part0-setup.md
docs/hw1/homework1-data-exploration.md
docs/hw1/homework1-ddpm-implementation.md
```

文档按实际执行顺序记录命令、输入、输出、判断和下一步，不把尚未执行的计划写成已完成事实。

## 图片与 Typora

只有文档确实需要图片时，才在对应作业目录创建 `.assets/`：

```text
docs/hw1/
├── homework1-data-exploration.md
└── .assets/
    └── homework1-data-grid.png
```

使用相对路径和 Typora 兼容语法：

```markdown
![CelebA 训练样本](.assets/homework1-data-grid.png)
```

不使用绝对路径、`file://` URL、Base64 图片或散落在文档目录外的截图。图片名使用小写 ASCII、连字符和明确的实验含义。

## 配置与比较

- 训练和评估直接读取的配置统一提交到 `code/configs/hwN/`。
- `docs/hwN/` 负责解释配置的设计理由、变更过程和结果，不复制一份机器可读配置。
- 同一比较组固定数据划分、checkpoint、采样种子、样本数和评估实现。
- HW1 KID 使用 1,000 个生成样本。
- 采样步数比较固定为 100、300、500、700、900 和 1,000 步。
- 临时命令行覆盖必须回写到最终配置或实验记录。

仓库不设置独立的 `experiments/` 目录。实验 ID 同时用于关联配置、文档和产物，例如：

```text
code/configs/hw1/hw1_exp02_ddpm_baseline.yaml
docs/hw1/homework1-ddpm-baseline.md
artifacts/checkpoints/hw1_exp02_ddpm_baseline/
artifacts/metrics/hw1_exp02_ddpm_baseline/
```

## 产物

数据集、checkpoint、完整日志、生成样本和指标明细放在 `artifacts/`，不提交 Git。仓库只提交报告需要的小型图片、配置、指标摘要和产物索引。

中间 checkpoint 设置保留策略。正式结果至少保留最终 checkpoint、配置、随机种子、指标文件和代表性样本。

## 公开与私有信息

公开文档可以记录硬件类型、GPU 数、软件版本、GPU hours 和复现参数。以下内容只能放在被 Git 忽略的 `docs/private/`：

- 主机地址和 SSH 别名
- 账户名和个人目录
- 挂载路径和权限细节
- 凭据、Token 与密钥
- 实时进程、其他用户任务和内部日志

公开命令使用占位符或仓库相对路径。不得伪装系统服务、规避资源审计或干扰共享服务器监控。
