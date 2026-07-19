# Compute Resources

## Baseline

当前开发与实验环境是一台配备 NVIDIA GeForce RTX 3090 24 GB 的工作站。HW1 starter config 明确说明 `num_gpus: 1 should be ok`，训练脚本也原生支持单卡执行。

四次作业均以单卡为固定执行边界，不依赖 DDP，也不设计跨主机同步流程。公开仓库不记录主机名、地址、账户、绝对路径、实时利用率、进程信息或凭据。

## Execution strategy

- coding、文档、静态检查、smoke test、训练、采样和 KID 均在同一工作区进行。
- 交互式检查可以基于工作树，写入报告的正式实验必须基于已提交 commit。
- HW1 至 HW4 始终使用一张 GPU。
- 使用自动混合精度，初始 batch size 设为 16，再根据峰值显存尝试 32。
- 每次正式训练前运行短程 smoke test，并记录 `torch.cuda.max_memory_allocated()`。
- GPU 被其他进程占用时不启动正式任务，长任务通过项目队列串行执行。

## Expected feasibility

建议的 HW1 初始配置为 `base_channels=128`、`channel_mult=[1,2,2,4]`、每层两个 residual blocks，并在 16×16 或更低分辨率使用 attention。该规模配合 AMP 和 batch size 16 预计能够放入 24 GB 显存。

最终显存和速度必须在 U-Net 实现完成后实测。单卡 smoke test 的通过标准为：

- 完成一次 forward、backward 和 optimizer step。
- 峰值显存低于约 22 GB，保留运行波动空间。
- 无 OOM、NaN 或数据加载错误。
- 能保存并恢复 checkpoint。

若显存不足，按顺序降低 batch size、减少 attention resolution、降低 base channels，最后才考虑 gradient checkpointing。不要引入多卡训练作为默认解决方案。

## Assignment plan

| Assignment | Workload | Single-GPU plan |
|---|---|---|
| HW1 | DDPM、1,000-sample KID、采样步数消融 | 单卡训练；分组顺序执行采样与评估 |
| HW2 | Flow Matching、DDIM 和 KID 对比 | 单卡分别训练与评估各方法 |
| HW3 | 自选方向 baseline | 将方法限制在 24 GB 单卡预算内 |
| HW4 | 改进方法与消融 | 只保留必要对比和消融，支持断点续训 |

HW1 使用 1,000 个生成样本计算 KID，并比较 100、300、500、700、900 和 1,000 个采样步数。目标是 KID mean 低于 0.005。

## Reproducibility

每次正式运行记录：

- Git commit 与配置文件
- 随机种子
- GPU 型号和数量
- PyTorch 与 CUDA 版本
- batch size 与是否启用 AMP
- 训练迭代数与 GPU hours
- 评估样本数和采样参数
- 退出状态与产物位置

同一比较组固定评估参数。机器标识不是复现所必需的信息，不得提交。

## Workstation queue

`scripts/workstation/gpu-queue.sh` 用于协调当前工作区的长任务。它等待 GPU 0 空闲、通过 `CUDA_VISIBLE_DEVICES` 绑定设备，并将队列状态和日志保存在被 Git 忽略的 `artifacts/queue/`。它只协调通过该脚本提交的任务，不能替代系统级资源调度器。

从 `homeworks/project/code/` 提交训练任务时，队列会记录提交时的工作目录，并在相同目录执行命令：

```bash
../scripts/workstation/gpu-queue.sh start 0
../scripts/workstation/gpu-queue.sh submit -- \
  .venv-cu126/bin/python train.py --method ddpm --config configs/hw1/example.yaml
../scripts/workstation/gpu-queue.sh status
```
