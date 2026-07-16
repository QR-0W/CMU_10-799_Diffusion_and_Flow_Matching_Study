# Compute Resources

## Baseline

一张 NVIDIA GeForce RTX 3090 具有 24 GB 显存，足以作为本课程作业的基线计算资源。HW1 starter config 明确说明 `num_gpus: 1 should be ok`，训练脚本也原生支持单卡执行。

计算节点可以是空闲的单卡服务器，也可以是双卡服务器中的一张空闲卡。完成作业不依赖双卡 DDP。

公开仓库不记录主机名、地址、账户、挂载点、实时利用率、进程信息或凭据。实际连接与路径只保存在被 Git 忽略的 `docs/private/`。

## Execution strategy

- 本地只负责 coding、文档、静态检查和轻量检查。
- 训练、采样、KID 和报告中的正式结果统一在远端 RTX 3090 上运行。
- HW1 与 HW2 的主模型默认使用一张 GPU。
- 使用自动混合精度，初始 batch size 设为 16，再根据峰值显存尝试 32。
- 每次正式训练前运行短程 smoke test，并记录 `torch.cuda.max_memory_allocated()`。
- 如果同时获得第二张空闲卡，优先并行运行独立采样、评估或消融，而不是立即启用 DDP。
- 不在已有计算进程的 GPU 上启动任务。

## Expected feasibility

建议的 HW1 初始配置为 `base_channels=128`、`channel_mult=[1,2,2,4]`、每层两个 residual blocks，并在 16×16 或更低分辨率使用 attention。该规模配合 AMP 和 batch size 16 预计能够放入 24 GB 显存。

最终显存和速度必须在 U-Net 实现完成后实测。单卡 smoke test 的通过标准为：

- 完成一次 forward、backward 和 optimizer step。
- 峰值显存低于约 22 GB，保留运行波动空间。
- 无 OOM、NaN 或数据加载错误。
- 能保存并恢复 checkpoint。

若显存不足，按顺序降低 batch size、减少 attention resolution、降低 base channels，最后才考虑 gradient checkpointing。不要仅为增大 batch size 引入双卡训练。

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

## Local queue

`scripts/remote/gpu-queue.sh` 用于协调当前账户自己的任务。它等待空闲设备、通过 `CUDA_VISIBLE_DEVICES` 绑定单张 GPU，并将队列文件保持为私有。它不是系统调度器，不能阻止其他用户绕过队列。
