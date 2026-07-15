# Artifacts

该目录用于本地或远端运行时的大文件，不提交 Git：

- `datasets/`
- `checkpoints/`
- `logs/`
- `samples/`
- `metrics/`

服务器空间紧张时，优先保留最终 checkpoint、实验配置、指标 JSON/CSV 和报告所需样本；中间 checkpoint 应按保留策略清理。
