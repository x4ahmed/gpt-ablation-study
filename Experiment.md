# Muon Optimizer Ablation Study

This experiment conducts a systematic ablation study using the **Muon optimizer**
(for matrices) + **AdamW** (for embeddings/scalars), matching the full-scale
leaderboard recipe. The goal is to determine whether hyperparameter findings
from the AdamW ablation study transfer when using the Muon optimizer, and to
establish a fair comparison with the H100 baseline run.

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
| Epochs                | 16                           |
| Architecture          | 4 layers, 512 embed dim, 8 heads |
| Learning Rate (multiplier) | 0.8                     |
| Weight Decay          | 0.8                          |
| Dropout               | 0.1                          |
| Document Shuffle      | On                           |
| Optimizer             | Muon (matrices) + AdamW (embed/scalars) |
| EMA                   | Every 10 steps                |
| SWA                   | Last 4 epochs                 |
| Warmup Ratio          | 0.0                           |

## Experiment Setup

| Run          | Train Tokens | Val Tokens | LR  | WD  | Dropout | Shuffle |
| ------------ | ------------ | ---------- | --- | --- | ------- | ------- |
| Baseline     | 10M          | 1M         | 0.8 | 0.8 | 0.1     | On      |
| LR High      | 10M          | 1M         | 1.0 | 0.8 | 0.1     | On      |
| LR Low       | 10M          | 1M         | 0.4 | 0.8 | 0.1     | On      |
| WD Low       | 10M          | 1M         | 0.8 | 0.2 | 0.1     | On      |
| WD High      | 10M          | 1M         | 0.8 | 1.2 | 0.1     | On      |
| Dropout Low  | 10M          | 1M         | 0.8 | 0.8 | 0.0     | On      |
| Dropout High | 10M          | 1M         | 0.8 | 0.8 | 0.2     | On      |
| No Shuffle   | 10M          | 1M         | 0.8 | 0.8 | 0.1     | Off     |

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
| Optimizer     | Muon (matrices) + AdamW (embed/scalars)      |
| Runtime       | ~26 min/epoch (estimated)                     |

## Results

| Run          | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status  | W&B Link |
| ------------ | ------------- | -------------- | ---------- | ---------- | ---------- | --------- | ---------- | ------- | -------- |
| Baseline     | 4.7624        | 4.8851         | 4.6681     | 13 (EMA)   | 6,432      | 4,717 MiB | 429.0m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| LR High      | 4.8138        | 4.9550         | 4.7461     | 13 (EMA)   | 6,305      | 4,717 MiB | 450.8m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/gq6kr8we) |
| LR Low       | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| WD Low       | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| WD High      | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| Dropout Low  | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| Dropout High | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| No Shuffle   | —             | —              | —          | —          | —          | —         | —          | Pending | —        |

### Leaderboard vs Muon Ablation Baseline

The Muon ablation uses the same optimizer (Muon + AdamW) and recipe as the
full-scale leaderboard baseline. The only differences are hardware-driven
scale reductions — the training recipe is identical.

| Feature | Leaderboard Baseline | Muon Ablation Baseline | Reason for Change |
| --- | --- | --- | --- |
| Layers | 16 | 4 | 4 GB VRAM can't fit a 16-layer model |
| Embed dim | 1024 | 512 | Halved to reduce memory per layer |
| Parameters | ~317M | ~66M | Consequence of fewer layers + smaller dim |
| Training tokens | ~100M | 10M | Shorter training time on a single slow GPU |
| GPUs | 8× H100 | 1× RTX 3050 (4 GB) | Hardware available for this study |
| Device batch size | 32 | 2 | 4 GB VRAM limit |
| Total batch size | 524,288 | 16,384 | Single GPU, no multi-GPU accumulation |
| `torch.compile` | Enabled | Disabled (`--no_torch_compile`) | Windows/Inductor unsupported |

---

## Phase 2: Architecture Component Ablation

This phase investigates which architectural features in the slowrun recipe
actually contribute to model performance. While Phase 1 held the architecture
fixed and varied training hyperparameters, Phase 2 holds all training
hyperparameters constant (using the best config from Phase 1) and removes or
modifies one architectural component at a time. This isolates the contribution
of each design choice — attention gating, U-Net skip connections, value
residuals, key offset, and RoPE type — to the final val loss.

The current baseline recipe uses the following architectural components:
**per-head attention gate** (zero-init `Linear(12, n_head)`, applied as
`y * sigmoid(gate(x))` starting at 0.5), **U-Net skip connections** (encoder
layers push activations, decoder layers pop + add with learnable weights),
**value residuals** (ResFormer — alternating layers project `x0` into the
value stream via a gated residual), **partial key offset** (stationary dims
of keys shifted forward by 1 on long-window and last layers), and
**half-truncated RoPE** (rotates only `head_dim//4` frequency pairs, leaving
the rest stationary).

### Experiment Setup

All runs use the Muon ablation baseline config from Phase 1 (best hyperparameters),
with one architectural component changed per run.

| Run              | Component Changed | Variant | Purpose |
| ---------------- | ----------------- | ------- | ------- |
| Baseline         | —                 | Current recipe | Reference |
| No Gate          | Attention gate    | Remove gate entirely | Is the per-head gate needed at all? |
| Identity Gate    | Attention gate    | `1 + 0.25*tanh(...)` | Start at ~1, learn small adjustments |
| Strong Gate      | Attention gate    | `2*sigmoid(...)` | Start near 1 but can strongly suppress or amplify heads |
| No U-Net Skips   | U-Net skips       | Remove skip connections | Do encoder-decoder skips help? |
| No Value Residual | Value residuals   | Remove ResFormer VE | Do value embeddings contribute? |
| No Key Offset    | Key offset        | Remove partial key shift | Does stationary dim shifting help? |
| Full RoPE        | RoPE              | Rotate all dims (standard) | Does half-truncated RoPE outperform full? |

### Results

| Run              | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status  | W&B Link |
| ---------------- | ------------- | -------------- | ---------- | ---------- | ---------- | --------- | ---------- | ------- | -------- |
| Baseline         | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| No Gate          | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| Identity Gate    | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| Strong Gate      | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| No U-Net Skips   | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| No Value Residual | —            | —              | —          | —          | —          | —         | —          | Pending | —        |
| No Key Offset    | —             | —              | —          | —          | —          | —         | —          | Pending | —        |
| Full RoPE        | —             | —              | —          | —          | —          | —         | —          | Pending | —        |