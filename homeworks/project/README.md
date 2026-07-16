# Coursework Workspace

四次作业在同一代码库中增量实现，实验配置、结果和报告按作业隔离。

## 目录

- `code/`：即将引入的 starter code、后续模型实现与引用记录
- `configs/`：可提交的实验配置
- `experiments/`：实验说明、种子、commit 和指标摘要
- `scripts/`：本地运行、远端队列与评估脚本
- `docs/public/`：可公开的设计和资源说明
- `docs/private/`：主机、账户与部署信息；由 Git 忽略
- `reports/`：作业报告材料
- `artifacts/`：数据、checkpoint、日志、样本和完整指标；由 Git 忽略

## 约定

每次实验使用 `hwN_expNN_name` 格式的 ID，并记录配置文件、随机种子、代码 commit、环境版本与指标摘要。同一比较组固定数据划分、checkpoint、采样种子和评估参数。

公开文件不得包含主机地址、SSH 别名、账户名、挂载路径、凭据、实时资源状态或私有日志。计算资源的公开说明见 [`docs/public/compute-resources.md`](docs/public/compute-resources.md)。
