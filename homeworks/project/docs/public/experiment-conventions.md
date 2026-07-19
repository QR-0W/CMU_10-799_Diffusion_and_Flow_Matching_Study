# Experiment Conventions

本文定义四次作业共享的实验、结果与作业文档规范。各目录的 `README.md` 只承担导航职责，具体规则以本文为准。

## 单一代码库

- 所有模型实现统一维护在 `code/`。
- HW2-HW4 直接扩展前序代码，不复制源码目录。
- 外部代码、论文、教程和 AI 工具统一登记在 `code/REFERENCES.md`。
- 正式结果必须能够由仓库中的 commit 和配置文件复现。

## 单机执行边界

- coding、文档、静态检查、smoke test、训练、采样和评估均在当前 RTX 3090 工作站完成。
- 交互式开发允许使用未提交修改，但任何写入报告的正式结果必须来自已提交的 commit。
- smoke test 不计入正式 GPU hours，也不能替代正式实验记录。
- 正式运行前记录 commit、配置和随机种子，运行后记录退出状态与产物位置。
- 同一时间只调度一个主要 GPU 任务；长任务通过 `scripts/workstation/gpu-queue.sh` 排队，避免与交互式检查争用显存。
- 项目命令统一从 `code/` 执行，相对路径统一以该目录为基准。

## 实验编号

实验 ID 使用 `hwN_expNN_name`，例如 `hw1_exp02_ddpm_baseline`。

每个实验在对应的 `docs/hwN/` 文档中至少记录：

- 目标与假设
- 实际执行步骤与命令
- 代码 commit
- 配置文件
- 随机种子
- 环境版本
- GPU 数与有效 batch size
- 训练迭代数与 GPU hours
- checkpoint 标识
- 评估参数与指标摘要
- 结果图表及对应分析
- 结论和异常情况

实验状态使用 `planned`、`running`、`completed` 或 `failed`。失败实验同样保留原因和诊断，不用成功结果覆盖。

## 文档

作业的实验步骤、执行记录、结果分析和题目答案统一放在 `docs/hwN/`，不再维护独立的 `reports/`。文件名使用 `homeworkN-<topic>.md`，例如：

```text
docs/hw1/homework1-part0-setup.md
docs/hw1/homework1-data-exploration.md
docs/hw1/homework1-ddpm-implementation.md
```

同一实验从计划到最终结果始终更新同一份文档，按实际执行顺序记录命令、输入、输出、图表、判断、题目答案和结论，不复制一份报告，也不把尚未执行的计划写成已完成事实。

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
artifacts/logs/hw1_exp02_ddpm_baseline/
artifacts/samples/hw1_exp02_ddpm_baseline/
artifacts/metrics/hw1_exp02_ddpm_baseline/
```

## 产物

数据集、checkpoint、完整日志、生成样本和指标明细放在 `artifacts/`，不提交 Git。仓库只提交文档需要的小型图片、配置、指标摘要和产物索引。

配置中的运行时路径使用相对 `code/` 的 `../artifacts/...` 形式。默认路径由 `code/src/utils/paths.py` 定义；脚本参数可以覆盖默认值，但不得把正式产物写回源码目录。

中间 checkpoint 设置保留策略。正式结果至少保留最终 checkpoint、配置、随机种子、指标文件和代表性样本。

## 环境快照

- `environments/requirements.txt` 表达课程依赖范围，用于有意识地升级依赖。
- `environments/requirements-cu126-pinned.txt` 固定 PyTorch、torchvision 和 torchaudio 的 CUDA 构建。
- `environments/requirements-cu126-lock.txt` 固定当前工作站验证通过的完整环境，正式复现优先使用该文件。
- 更新锁定文件后必须重新运行 `uv pip check`、CUDA 张量计算以及 `train.py --help`、`sample.py --help`。

## 信息边界

公开文档可以记录硬件类型、GPU 数、软件版本、GPU hours 和复现参数。主机地址、账户名、绝对路径、凭据、Token、密钥、实时进程和其他用户信息不写入仓库。

文档和脚本优先使用仓库相对路径，敏感值只通过环境变量或系统凭据存储提供。不得伪装系统服务、规避资源审计或干扰共享服务器监控。
