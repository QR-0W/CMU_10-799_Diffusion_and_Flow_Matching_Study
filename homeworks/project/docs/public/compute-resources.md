# Compute Resources

## Resource profile

Course experiments use one shared compute node with two NVIDIA GeForce RTX 3090 GPUs. Each GPU has 24 GB of memory. No other remote compute target is part of the current plan.

The public repository intentionally does not record hostnames, addresses, account names, mount points, live utilization, process information, credentials, or operational commands tied to the machine. Those details belong in `docs/private/`, which is excluded from Git.

## Execution strategy

- Train the HW1 and HW2 primary models on one GPU by default.
- Use the second GPU for an independent experiment, sampling run, or evaluation when both devices are available.
- Consider distributed data parallel training only after measuring single-GPU throughput and verifying that the effective batch size and optimization settings remain comparable.
- Use automatic mixed precision, bounded checkpoint retention, and scripted evaluation.
- Do not run on a partially occupied GPU. Shared-resource coordination takes priority over throughput.

This strategy favors independent jobs over synchronous two-GPU training. It avoids unnecessary communication overhead and makes interrupted experiments easier to resume.

## Coursework feasibility

| Assignment | Workload | Plan |
|---|---|---|
| HW1 | DDPM training, 1,000-sample KID, sampling-step ablation | One GPU for training; parallelize independent sampling groups when possible |
| HW2 | Flow Matching, DDIM, sampling and KID comparisons | One GPU per independent method or evaluation group |
| HW3 | Chosen specialization baseline | Keep the method within a 24 GB single-GPU budget |
| HW4 | Proposed method and ablations | Limit the study to one primary method and the necessary ablations |

HW1 evaluation uses 1,000 generated samples and compares 100, 300, 500, 700, 900, and 1,000 sampling steps. The target is KID mean below 0.005. The optional alternative parameterization is treated as extra work rather than part of the required compute budget.

## Reproducibility

Every reported run records:

- Git commit hash
- Configuration file
- Random seed
- GPU count and effective batch size
- PyTorch and CUDA versions
- Training iterations and measured GPU hours
- Evaluation sample count and sampling parameters

Comparison groups keep evaluation parameters fixed. Machine-specific identifiers are not needed to reproduce the result and must not be committed.

## Local queue

`scripts/remote/gpu-queue.sh` is a user-level queue for coordinating this repository's own jobs. It waits for an unoccupied device, binds a job with `CUDA_VISIBLE_DEVICES`, and keeps queue files private. It is not a system scheduler and cannot reserve resources against other users.
