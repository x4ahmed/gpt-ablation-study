# GPT Ablation Study

This repository conducts a systematic ablation study on a ~66M parameter GPT-style model trained on 10M FineWeb tokens. The goal is to isolate the impact of key training hyperparameters—learning rate, weight decay, dropout, and document shuffling—by changing exactly one variable per run while holding all other recipe settings constant.

## Baseline Hyperparameters

| Hyperparameter        | Value                        |
| --------------------- | ---------------------------- |
| Train Tokens          | 10M                          |
| Val Tokens            | 1M                           |
| Sequence Length       | 2048                         |
| Total Batch Size      | 16,384                       |
| Device Batch Size     | 2                            |
| Grad Accum Steps      | 4                            |
| Optimizer Steps/Epoch | ~610                         |
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
| Runtime       | ~26 min/epoch (~8.8 hours/run, avg across 7 runs) |

## Results

| Run          | Val Loss | Best Val Loss | Status  | W&B Link |
| ------------ | -------- | ------------- | ------- | -------- |
| Baseline     | 6.1553   | 6.1553        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/atcc23zv) |
| LR High      | 6.7309   | 6.7309        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/bkgu7cnh) |
| LR Low       | 6.0326   | 6.0326        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/rabskmre) |
| Shuffle      | 6.1694   | 6.1694        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/gzl6bjst) |
| WD Low       | 5.7697   | 5.7697        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/nk5j7ng1) |
| WD High      | 6.3105   | 6.3105        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/6pxhm623) |
| Dropout Low  | 6.1938   | 6.1938        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/j2ad0xji) |
| Dropout High | 6.2938   | 6.2938        | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/hmcn70i4) |

### Leaderboard vs Ablation Baseline

The ablation runs use a significantly reduced configuration compared to the
full slowrun leaderboard recipe, constrained by a single 4 GB GPU. The
training recipe (architecture, regularization) is identical between the two
— only the scale, optimizer, and a few hardware-related defaults differ.
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
| Dataloader | Chunk-based (pre-packed sequences) | Document-based (flat tokens + doc boundaries) | Document-based supports per-epoch doc reshuffling; chunk-based is frozen at preprocessing |
| Optimizer | Muon (matrices) + AdamW (embed/scalars) | AdamW (all params) | Single-GPU fallback bypasses Muon; all params use AdamW |
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

## Scaled Experiment Results

After the ablation study, the best configuration (WD Low: `lr_multiplier=0.6,
weight_decay=0.2, dropout=0.1`) will be re-run at full leaderboard scale on
a rented H100 node. The `h100_runs/` folder contains standalone scripts with
the full-scale defaults (16 layers, 1024 dim, 524K batch, Muon optimizer,
EMA + SWA enabled).

| Run | Setup | Val Loss | Status | W&B Link |
| --- | --- | --- | --- | --- |
| H100 Baseline | Full leaderboard config (16L, 1024d, WD=0.8, lr=0.6) | — | Pending | — |
| H100 Best Ablation | Full leaderboard config + best ablation params (WD=0.2) | — | Pending | — |

## H100 Run Setup

After the ablation study, the best configuration will be re-run at full
leaderboard scale on a rented 8× H100 node (e.g. Lambda Labs). The
`h100_runs/` folder contains standalone scripts with the full-scale defaults
(16 layers, 1024 dim, 524K batch, Muon optimizer, EMA + SWA enabled).

### Prerequisites

- **HF token**: Create a read token at https://huggingface.co/settings/tokens
- **W&B API key**: Get it from https://wandb.ai/authorize

### Setup sequence

```bash
# 1. Clone and enter the h100_runs directory
git clone https://github.com/x4ahmed/gpt-ablation-study.git
cd gpt-ablation-study/h100_runs

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install numpy tiktoken wandb datasets tqdm kernels

# 3. Set HF token (persists across sessions)
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
source ~/.bashrc

# 4. Login to W&B (prompts for API key, saves to ~/.netrc)
wandb login

# 5. Prepare data (100M train tokens, 10M val tokens — ~10-20 min)
python prepare_data.py

# 6. Train (8 GPUs, Muon optimizer, EMA + SWA enabled)
torchrun --standalone --nproc_per_node=8 train.py \
  --run-name h100_baseline \
  --wandb_entity i-learn \
  --no-doc-shuffle
```

### Notes

- `torch.compile` is always enabled on H100 (Linux/Inductor works fine)
- Muon optimizer is used for matrix weights, AdamW for embeddings/scalars
- EMA (every 10 steps) and SWA (last 4 epochs) are enabled by default
- Results, checkpoints, and model are saved to `runs/<run_name>/`
- Monitor live metrics at https://wandb.ai/i-learn/slowrun
