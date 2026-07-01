# GPT Ablation Study

This repository conducts a systematic ablation study on a ~66M parameter GPT-style model trained on 10M FineWeb tokens. The goal is to isolate the impact of key training hyperparameters—learning rate, weight decay, dropout, and document shuffling—by changing exactly one variable per run while holding all other recipe settings constant.

## Baseline Hyperparameters (2nd Run)

| Hyperparameter        | Value                        |
| --------------------- | ---------------------------- |
| Train Tokens          | 10M                          |
| Val Tokens            | 1M                           |
| Sequence Length       | 2048                         |
| Total Batch Size      | 16,384                       |
| Device Batch Size     | 2                            |
| Epochs                | 20                           |
| Architecture          | 4 layers, 512 embed dim, 8 heads |
| Learning Rate (multiplier) | 0.6                     |
| Weight Decay          | 0.8                          |
| Dropout               | 0.1                          |
| Document Shuffle      | Off                          |

## Experiment Setup

| Run          | Train Tokens | Val Tokens | LR  | WD  | Dropout | Shuffle |
| ------------ | ------------ | ---------- | --- | --- | ------- | ------- |
| Baseline     | 10M          | 1M         | 0.6 | 0.8 | 0.1     | Off     |
| LR High      | 10M          | 1M         | 1.0 | 0.8 | 0.1     | Off     |
| LR Low       | 10M          | 1M         | 0.4 | 0.8 | 0.1     | Off     |
| Shuffle      | 10M          | 1M         | 0.6 | 0.8 | 0.1     | On      |
| WD Low       | 10M          | 1M         | 0.6 | 0.2 | 0.1     | Off     |
| WD High      | 10M          | 1M         | 0.6 | 1.2 | 0.1     | Off     |
| Dropout Low  | 10M          | 1M         | 0.6 | 0.8 | 0.0     | Off     |
| Dropout High | 10M          | 1M         | 0.6 | 0.8 | 0.2     | Off     |

## Hardware & Environment

| Component     | Details                                      |
| ------------- | -------------------------------------------- |
| Device        | SAUDI-PC                                     |
| CPU           | 12th Gen Intel Core i5-12500H @ 2.50 GHz     |
| RAM           | 32 GB                                        |
| GPU           | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |
| OS            | Windows                                      |
| PyTorch       | CUDA 12.8 build                              |
| torch.compile | Disabled (`--no_torch_compile`)              |
| Runtime       | ~5 min/epoch on CUDA                         |

## Results

| Run          | Val Loss | Best Val Loss | Status  | W&B Link |
| ------------ | -------- | ------------- | ------- | -------- |
| Baseline     | 6.1553   | 6.1553        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/atcc23zv) |
| LR High      | 6.7309   | 6.7309        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/bkgu7cnh) |
| LR Low       | 6.0326   | 6.0326        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/rabskmre) |
| Shuffle      | 6.1694   | 6.1694        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/gzl6bjst) |
| WD Low       | —        | —             | Pending | —        |
| WD High      | —        | —             | Pending | —        |
| Dropout Low  | —        | —             | Pending | —        |
| Dropout High | 6.2938   | 6.2938        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/hmcn70i4) |

### Leaderboard vs Ablation Baseline

The ablation runs use a significantly reduced configuration compared to the
full slowrun leaderboard recipe, constrained by a single 4 GB GPU. The
training recipe (architecture, optimizer, regularization) is identical
between the two — only the scale and a few hardware-related defaults differ.
This explains why absolute val loss (~6.0) is higher than leaderboard results
(~4.0) — the difference is driven by model capacity and data budget, not
recipe modifications.

| Feature | Leaderboard Baseline | Ablation Baseline | Reason for Change |
| --- | --- | --- | --- |
| Layers | 16 | 4 | 4 GB VRAM can't fit a 16-layer model |
| Embed dim | 1024 | 512 | Halved to reduce memory per layer |
| Parameters | ~350M+ | ~66M | Consequence of fewer layers + smaller dim |
| Training tokens | ~100M | 10M | Shorter training time on a single slow GPU |
| GPUs | 8× H100 | 1× RTX 3050 (4 GB) | Hardware available for this study |
| MLP activation | SwiGLU | SwiGLU | — |
| RoPE | Half-truncated | Half-truncated | — |
| Attention gate | Per-head, zero-init | Per-head, zero-init | — |
| Key offset | Partial, long-window layers | Partial, long-window layers | — |
| U-Net skips | Yes, learnable weights | Yes, learnable weights | — |
| Value residuals | ResFormer, alternating layers | ResFormer, alternating layers | — |
| Optimizer | Muon (matrices) + AdamW (embed/scalars) | Muon (matrices) + AdamW (embed/scalars) | — |
| Weight decay | 3-phase: hold → decay → ramp | 3-phase: hold → decay → ramp | — |
| EMA | Yes | Disabled (`--update-ema-every 0`) | Save memory/time on single GPU |
| SWA | Yes, last 4 epochs | Disabled (`--swa-last-epochs 0`) | Save memory/time on single GPU |
| Warmup ratio | 0.0 | 0.05 | Smaller batch benefits from warmup |
| LR multiplier | 0.8 | 0.6 | Tuned for smaller batch / single GPU |
| Epochs | 16 | 20 | More epochs to compensate for fewer steps per epoch |
| `torch.compile` | Enabled | Disabled (`--no_torch_compile`) | Windows/Inductor unsupported |
| Device batch size | 32 | 2 | 4 GB VRAM limit |
| Total batch size | 524,288 | 16,384 | Single GPU, no multi-GPU accumulation |

### Weight Decay Schedule

Weight decay follows a 3-phase schedule: **hold** at the base value during
early training, **decay** to near-zero during mid-training to let the model
learn freely, then **ramp up** at the end to compress the model before
evaluation. The `--wd-mid` and `--wd-end` values are scaled proportionally
to `--weight-decay` so the schedule shape is preserved across all WD runs.

| Phase | Epochs | Baseline (WD=0.8) | WD Low (WD=0.2) | WD High (WD=1.2) |
| --- | --- | --- | --- | --- |
| 1 — Hold | 0–2 | 0.8 | 0.2 | 1.2 |
| 2 — Decay | 2–8 | 0.8 → 0.1 | 0.2 → 0.025 | 1.2 → 0.15 |
| 3 — Ramp | 8–20 | 0.1 → 1.25 | 0.025 → 0.3125 | 0.15 → 1.875 |
