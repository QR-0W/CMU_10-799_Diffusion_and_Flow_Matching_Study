# Coursework Project

## 目录职责

- `code/`：官方 starter code 及后续持续演进的模型实现。
- `experiments/`：每个实验的配置、运行说明、指标摘要和轻量图表。
- `scripts/local/`：本机开发、测试和小规模调试脚本。
- `scripts/remote/`：同步到 3090 服务器、启动训练和拉取结果的脚本。
- `scripts/evaluation/`：采样、KID/FID 和实验汇总脚本。
- `docs/`：算法推导、实现设计、调试记录和算力评估。
- `reports/`：每次作业最终报告使用的文字、表格和图片索引。
- `artifacts/`：数据、checkpoint、日志、生成样本和完整指标等大文件；默认不提交 Git。

## 实验约定

每个实验目录只保存可复现实验所需的轻量内容：配置、命令、随机种子、代码 commit、环境信息和指标摘要。大文件统一放在 `artifacts/`，并通过实验 ID 关联。

实验 ID 建议使用 `hwN_expNN_name`，例如 `hw1_exp02_ddpm_baseline`。同一比较组固定数据划分、checkpoint、采样种子和评估参数。
