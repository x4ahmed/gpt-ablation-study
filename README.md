# GPT-Slowrun Ablation Study

This repository reproduces and ablates the open-source
[NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) Tiny Track recipe. Three
small-scale phases are screened on a laptop GPU, followed by scale transfer to
an 8xH100 setup. The upstream repository was cloned when the Tiny Track record
was 3.332 with document-level shuffling.

`Best Val Loss` is the minimum across raw epoch checkpoints, final EMA weights,
and the weighted average of the last four checkpoints. `Final Val Loss` is the
raw validation loss at epoch 16.

## Small-Scale Setup

| Hyperparameter | Value |
| --- | --- |
| Train / validation tokens | 10M / 1M |
| Sequence length | 2048 |
| Architecture | 4 layers, 512 embedding dimension, 8 heads (~66M parameters) |
| Hardware | 1x NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) |
| Total batch size | 16,384 tokens |
| Device batch size | 2 sequences |
| Gradient accumulation | 4 steps |
| Optimizer steps per epoch | ~610 |
| Epochs | 16 |
| Baseline LR multiplier | 0.8 |
| Baseline weight decay | 0.8 |
| Baseline dropout | 0.1 |
| Document shuffle | On |
| Baseline optimizer | Muon for hidden matrices; AdamW for remaining parameters |
| EMA | Every 10 optimizer steps |
| Checkpoint averaging / SWA | Last 4 epochs, recency weights 1:2:3:4 |
| Warmup ratio | 0.0 |
| `torch.compile` | Disabled |

## Phase 1: Optimizer Ablation

### Setup

| Run | Hidden matrices | Remaining parameters | LR multiplier |
| --- | --- | --- | --- |
| Muon + AdamW | Muon | AdamW | 0.8 |
| AdamW only | AdamW | AdamW | 0.8 |
| AdamW (low LR) | AdamW | AdamW | 0.6 |

### Results

All other model, data, schedule, and evaluation settings are fixed.

| Run | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status | W&B Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Muon + AdamW | 4.7624 | 4.8851 | 4.6681 | 13 (EMA) | 6,432 | 4,717 MiB | 429.0m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| AdamW only | 6.1809 | NaN | 6.1400 | 2 (raw) | ~6,500 | 4,717 MiB | ~358m | Diverged | [View](https://wandb.ai/i-learn/slowrun/runs/hac9lp4b) |
| AdamW (low LR) | 4.8975 | 5.0426 | 4.8525 | 16 (EMA) | 6,684 | 4,780 MiB | 414.9m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/p5rnu5bp) |

## Phase 2: Hyperparameter Ablation

### Setup

| Run | Train Tokens | Val Tokens | LR | WD | Dropout | Shuffle |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 10M | 1M | 0.8 | 0.8 | 0.1 | On |
| LR High | 10M | 1M | 1.0 | 0.8 | 0.1 | On |
| LR Low | 10M | 1M | 0.4 | 0.8 | 0.1 | On |
| WD Low | 10M | 1M | 0.8 | 0.2 | 0.1 | On |
| WD High | 10M | 1M | 0.8 | 1.2 | 0.1 | On |
| Dropout Low | 10M | 1M | 0.8 | 0.8 | 0.0 | On |
| Dropout High | 10M | 1M | 0.8 | 0.8 | 0.2 | On |
| No Shuffle | 10M | 1M | 0.8 | 0.8 | 0.1 | Off |

### Results

Each run changes one hyperparameter from the baseline.

| Run | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status | W&B Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 4.7624 | 4.8851 | 4.6681 | 13 (EMA) | 6,432 | 4,717 MiB | 429.0m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| LR High | 4.8138 | 4.9550 | 4.7461 | 13 (EMA) | 6,305 | 4,717 MiB | 450.8m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/gq6kr8we) |
| LR Low | 4.6436 | 4.7302 | 4.4751 | 13 (Ckpt avg) | 6,205 | 4,717 MiB | 480.3m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/79wkc6pa) |
| WD Low | 4.4606 | 4.5044 | 3.9026 | 16 (Ckpt avg) | 5,669 | 4,717 MiB | 455.7m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/yfhvegka) |
| WD High | 4.9253 | 5.0719 | 4.8846 | 13 (EMA) | 6,353 | 4,717 MiB | 452.8m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/cvug1pug) |
| Dropout Low | 4.7150 | 4.8431 | 4.6002 | 13 (EMA) | 6,792 | 4,712 MiB | 414.0m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/ywvy5jnj) |
| Dropout High | 4.7854 | 4.8939 | 4.7119 | 13 (EMA) | 6,221 | 4,717 MiB | 445.2m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/aof41ihb) |
| No Shuffle | 4.7597 | 4.8747 | 4.7382 | 13 (EMA) | 6,567 | 4,717 MiB | 430.5m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/rjxf93em) |

## Phase 3: Architecture Ablation

### Setup

| Run | Component changed | Variant / flag |
| --- | --- | --- |
| Baseline | None | Full baseline recipe |
| No Gate | Attention gate | Remove gate; `--no-attn-gate` |
| Identity Gate | Attention gate | `1 + 0.25*tanh(...)`; `--attn-gate-variant identity` |
| Strong Gate | Attention gate | `2*sigmoid(...)`; `--attn-gate-variant strong` |
| No U-Net Skips | U-Net skips | Remove skips; `--no-skip-connections` |
| No Value Residual | Value residual | Remove value residual; `--no-value-residual` |
| No Key Offset | Partial key offset | Remove offset; `--no-key-offset` |
| Full RoPE | RoPE and key offset | Full RoPE with no offset; `--rope-variant full --no-key-offset` |

### Results

Each run changes one architecture component while retaining the common training setup.

| Run | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status | W&B Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 4.7624 | 4.8851 | 4.6681 | 13 (EMA) | 6,432 | 4,717 MiB | 429.0m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| No Gate | 4.7555 | 4.8721 | 4.6515 | 13 (EMA) | 6,609 | 4,705 MiB | 411.3m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/6oz3f8yg) |
| Identity Gate | 4.7743 | 4.9141 | 4.6876 | 13 (EMA) | 6,339 | 4,717 MiB | 397.6m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/sytr3g5q) |
| Strong Gate | 4.7802 | 4.9014 | 4.6729 | 13 (EMA) | 6,842 | 4,717 MiB | 403.8m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/t60uuvsw) |
| No U-Net Skips | 4.7642 | 4.8888 | 4.6807 | 13 (EMA) | 6,860 | 4,717 MiB | 409.6m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/sskx1di3) |
| No Value Residual | 4.8079 | 4.9188 | 4.6975 | 13 (EMA) | 6,500 | 4,704 MiB | 455.6m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/cwgybdzv) |
| No Key Offset | 4.7778 | 4.8964 | 4.6877 | 13 (EMA) | 7,012 | 4,717 MiB | 402.8m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/yy4bgdei) |
| Full RoPE | 4.7708 | 4.8919 | 4.6745 | 13 (EMA) | 4,639 | 4,717 MiB | 437.0m | Done | [View](https://wandb.ai/i-learn/slowrun/runs/js78r6w8) |

## Scale Transfer: 8xH100

### Setup

| Hyperparameter | Value |
| --- | --- |
| Architecture | 16 layers, 1024 embedding dimension, 8 heads, head dimension 128 |
| Parameters | 316,935,720 (~317M) |
| FLOPs per token | 1.756 x 10^9 |
| GPUs | 8x H100 (Hopper, Flash Attention 3) |
| Train / validation tokens | ~100M / ~10M |
| Sequence length | 2048 |
| Total batch size | 524,288 tokens |
| Device batch size | 16 sequences per GPU |
| Gradient accumulation | 2 steps |
| Epochs | 16 |
| LR multiplier | 0.8 |
| Baseline weight decay | 0.8, scheduled as hold -> 0.1 -> 1.25 |
| Dropout | 0.1 |
| EMA | Every 10 optimizer steps |
| Checkpoint averaging / SWA | Last 4 epochs, recency weights 1:2:3:4 |
| Document shuffle | On |

### Results

The four selected configurations use the same H100 model, data, and training schedule.

| Run | Optimizer | Gate Variant | Weight Decay | Final Train Loss | Best Val Loss | Final Val Loss | Best Val BPB | Best Epoch | Status | W&B Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | Muon + AdamW | sigmoid (default) | 0.8 | 2.9533 | **3.3343** | 3.3484 | **1.0835** | 16 (Ckpt avg) | Done | [View](https://wandb.ai/i-learn/slowrun/runs/vryw9bxv) |
| Strong Gate | Muon + AdamW | strong (`2*sigmoid`) | 0.8 | 2.9521 | **3.3376** | 3.3524 | **1.0846** | 16 (Ckpt avg) | Done | [View](https://wandb.ai/i-learn/slowrun/runs/vp6dxkif) |
| Strong Gate + Low WD | Muon + AdamW | strong (`2*sigmoid`) | 0.2 | 2.3208 | 3.4735 | 3.6652 | 1.1670 | 8 (raw) | Done | [View](https://wandb.ai/i-learn/slowrun/runs/1l93yfpf) |
| Baseline + Low WD | Muon + AdamW | sigmoid (default) | 0.2 | 2.2568 | 3.4836 | 3.7445 | 1.1912 | 8 (raw) | Done | [View](https://wandb.ai/i-learn/slowrun/runs/cy7besr1) |
