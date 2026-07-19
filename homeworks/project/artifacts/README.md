# Artifacts

该目录是当前工作站唯一的运行时大文件根目录，不提交 Git：

| 目录 | 内容 |
|---|---|
| `datasets/celeba-subset/` | 下载后的课程数据集及其 split |
| `checkpoints/<run>/` | 中间与最终模型 checkpoint |
| `logs/<run>/` | 配置快照、文本日志与 W&B 本地文件 |
| `samples/<run>/` | 训练中采样、独立采样与评估生成图片 |
| `metrics/` | KID、FID、IS 输出及 torch-fidelity 缓存 |
| `queue/` | 当前工作区 GPU 队列状态与任务日志 |

所有 Python 相对路径以 `code/` 为基准解析。正式配置应使用 `../artifacts/...` 形式，不在 `code/` 下创建 `data/`、`logs/`、`checkpoints/` 或 `samples/`。

服务器空间紧张时，优先保留最终 checkpoint、实验配置、指标 JSON/CSV 和报告所需样本；中间 checkpoint 应按保留策略清理。
