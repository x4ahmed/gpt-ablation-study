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
| LR High      | —        | —             | Pending | —        |
| LR Low       | —        | —             | Pending | —        |
| Shuffle      | —        | —             | Pending | —        |
| WD Low       | —        | —             | Pending | —        |
| WD High      | —        | —             | Pending | —        |
| Dropout Low  | —        | —             | Pending | —        |
| Dropout High | —        | —             | Pending | —        |
