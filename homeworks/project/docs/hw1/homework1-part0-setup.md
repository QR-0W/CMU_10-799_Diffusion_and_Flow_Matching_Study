# Homework 1 Part 0: Setup

## 目标

完成官方 starter code 的 fork、固定版本导入和远端运行环境配置。

执行边界：

- 本地负责 coding、文档和静态检查。
- 训练、采样、KID 和正式 smoke test 在远端 RTX 3090 上运行。
- 远端只运行可追溯的 Git commit。

## 最终环境

| 项目 | 值 |
|---|---|
| 官方仓库 | `KellyYutongHe/cmu-10799-diffusion` |
| 课程 fork | `QR-0W/cmu-10799-diffusion` |
| 上游 commit | `e666fe97654e7463de43134a5e7317de6cb50f9a` |
| GPU | NVIDIA GeForce RTX 3090, 24 GB |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu126 |
| torchvision | 0.26.0+cu126 |
| torchaudio | 2.11.0+cu126 |
| torch-fidelity | 0.4.0 |

## 1. Fork 与导入

创建课程 fork：

```bash
gh repo fork KellyYutongHe/cmu-10799-diffusion --clone=false
```

将固定 commit 的工作树导入 `homeworks/project/code/`，不保留嵌套 `.git/`。HW1-HW4 后续都在这一份代码上增量开发。

源码来源、commit 和许可说明记录在 `code/REFERENCES.md`。

## 2. 远端工作区

远端工作区使用容量足够的存储卷、权限为 `700` 的中性目录。公开文档不记录主机别名、账户名和实际路径，这些信息只保存在被 Git 忽略的 `docs/private/`。

Part 0 使用 starter fork 建立官方基线：

```bash
umask 077
git clone --branch main \
  https://github.com/QR-0W/cmu-10799-diffusion.git \
  "$PRIVATE_RUNTIME_DIR"
git -C "$PRIVATE_RUNTIME_DIR" checkout \
  e666fe97654e7463de43134a5e7317de6cb50f9a
```

同步本仓库维护的固定依赖文件：

```bash
scp homeworks/project/code/environments/requirements-cu126-pinned.txt \
  "$PRIVATE_HOST:$PRIVATE_RUNTIME_DIR/environments/requirements-cu126-pinned.txt"
```

## 3. 安装环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv "$PRIVATE_RUNTIME_DIR/.venv-cu126" --python 3.12

uv pip install \
  --python "$PRIVATE_RUNTIME_DIR/.venv-cu126/bin/python" \
  -r "$PRIVATE_RUNTIME_DIR/environments/requirements-cu126-pinned.txt"

uv pip install \
  --python "$PRIVATE_RUNTIME_DIR/.venv-cu126/bin/python" \
  -r "$PRIVATE_RUNTIME_DIR/environments/requirements.txt"
```

先安装固定 PyTorch/CUDA 栈，再安装课程基础依赖。这样不会被 PyPI 上的其他 PyTorch 构建替换。

## 4. 验证

```bash
uv pip check --python "$PRIVATE_RUNTIME_DIR/.venv-cu126/bin/python"

cd "$PRIVATE_RUNTIME_DIR"
.venv-cu126/bin/python train.py --help
.venv-cu126/bin/python sample.py --help
```

验证结果：

```text
CUDA available: True
CUDA device count: 2
GPU: NVIDIA GeForce RTX 3090
uv pip check: All installed packages are compatible
train.py --help: exit 0
sample.py --help: exit 0
```

检测到两张 GPU 只代表当前节点配置。HW1 默认使用单张 RTX 3090，第二张卡不是完成作业的前置条件。

## 后续工作流

1. 本地修改代码、配置和 `docs/hw1/` 文档。
2. 提交 Git commit。
3. 远端检出当前主仓库的指定 commit。
4. 从 `homeworks/project/code/` 提交单卡任务。
5. 将指标摘要和报告图片整理回仓库，大文件保留在远端 artifacts。

## 状态

| 检查项 | 状态 |
|---|---|
| 官方仓库 fork | completed |
| 固定 commit 导入 | completed |
| 私有远端工作区 | completed |
| 独立远端环境 | completed |
| CUDA 与依赖检查 | completed |
| 训练和采样 CLI 检查 | completed |
| 冗余本地 CUDA 环境清理 | completed |

Part 0 完成。下一步进入 Part I，准备并探索课程提供的 CelebA 子集。

本文不需要图片，因此没有创建 `.assets/`。
