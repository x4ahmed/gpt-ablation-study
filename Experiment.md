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

| Run          | Val Loss | Best Val Loss | Status  | W&B Link |
| ------------ | -------- | ------------- | ------- | -------- |
| Baseline     | —        | —             | Pending | —        |
| LR High      | —        | —             | Pending | —        |
| LR Low       | —        | —             | Pending | —        |
| WD Low       | —        | —             | Pending | —        |
| WD High      | —        | —             | Pending | —        |
| Dropout Low  | —        | —             | Pending | —        |
| Dropout High | —        | —             | Pending | —        |
| No Shuffle   | —        | —             | Pending | —        |

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