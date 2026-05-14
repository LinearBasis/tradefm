---
name: Cluster training scripts must support multi-GPU DDP
description: When writing cluster training scripts for tradefm, design for torchrun/DDP up to 8 GPUs from the start
type: feedback
---

When the user asks for cluster-ready training scripts, they must support multi-GPU via `torchrun` / DDP from the outset (not single-GPU only). Target: 1–8 GPUs (cluster has up to 8 H100, in practice usually fewer).

**Why:** User flagged this explicitly when reviewing a single-GPU draft of `scripts/train.py` for the closed-environment cluster. Hardware ceiling is 8 GPUs but realistic usage is 1–4 — so scripts must scale down to 1 GPU cleanly while not requiring rewrites to scale up.

**How to apply:**
- Use `torch.distributed` + `DistributedSampler` + `DistributedDataParallel`
- Wrap with `init_process_group("nccl")` guarded by env vars (`RANK`, `WORLD_SIZE`)
- All print/checkpoint/TensorBoard logic gated on `rank == 0`
- Single-GPU should still work via plain `python -m scripts.train` (no DDP init when WORLD_SIZE unset)
- Document `torchrun --nproc-per-node=N` invocation in script docstring
