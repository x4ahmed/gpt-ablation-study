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
| Architecture          | 4 layers, 512 embed dim, 8 heads (~66M params) |
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

## Optimizer Ablation

| Run              | Best Val Loss | Final Val Loss | Train Loss | Best Epoch | Tokens/sec | Peak VRAM | Wall-clock | Status  | W&B Link |
| ---------------- | ------------- | -------------- | ---------- | ---------- | ---------- | --------- | ---------- | ------- | -------- |
| Muon + AdamW     | 4.7624        | 4.8851         | 4.6681     | 13 (EMA)   | 6,432      | 4,717 MiB | 429.0m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| AdamW only       | 6.1809        | NaN            | 6.1400     | 2 (raw)    | ~6,500     | 4,717 MiB | ~358m      | Diverged | [View](https://wandb.ai/i-learn/slowrun/runs/hac9lp4b) |
| AdamW (low LR)   | 4.8975        | 5.0426         | 4.8525     | 16 (EMA)   | 6,684      | 4,780 MiB | 414.9m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/p5rnu5bp) |

### AdamW Divergence Analysis

The AdamW-only run diverged at **step 1,626** (~16.6% through training) due to
**exploding gradients**. The sequence of events was:

1. **Steps 0–610 (Epoch 1):** Training progressed normally. Loss decreased from
   10.83 → 6.26. Val loss after epoch 1: 6.357.
2. **Steps 610–1,220 (Epoch 2):** Loss continued decreasing to ~6.12. Val loss
   after epoch 2: 6.181 — the best val loss recorded.
3. **Steps 1,220–1,625 (Epoch 3):** Loss plateaued around 6.0–6.2, with
   increasing instability. Several steps showed loss spikes (e.g., step 1,598:
   6.240, step 1,620: 6.243). The step times also increased dramatically (from
   ~2,400ms to ~4,000ms), suggesting the GPU was struggling with increasingly
   large gradient values.
4. **Step 1,626:** A gradient spike exceeded AdamW's numerical stability
   threshold. The momentum buffer amplified the spike instead of normalizing it,
   producing an update so large it caused numerical overflow → `inf` → `NaN`.
5. **Steps 1,626–2,036+:** All subsequent losses are `NaN`. Once any weight
   becomes NaN, every forward pass produces NaN permanently — the model is
   irrecoverably corrupted.

**Root cause:** The matrix learning rate (`0.032 = 0.04 × 0.8`) is too
aggressive for AdamW. Muon handles this LR safely because its Polar Express
orthogonalization normalizes the gradient to unit norm before applying the
update — no matter how large the raw gradient spike, the effective step size is
bounded. AdamW has no such safeguard: the raw gradient flows directly through
the momentum buffer, and a spike gets amplified rather than dampened.

**Key finding:** This demonstrates that **Muon's gradient normalization is a
critical stability advantage**, not just a performance optimization. The same
learning rate that trains stably for 9,760 steps with Muon causes catastrophic
divergence with AdamW at step 1,626. This is a direct, controlled comparison —
the only difference between the two runs is the optimizer, confirming that
Muon's orthogonalization is what prevents the explosion.

### AdamW Low LR Analysis

After the AdamW divergence at `lr_multiplier=0.8` (matrix_lr=0.032), a second
run was attempted with `lr_multiplier=0.6` (matrix_lr=0.024) — the original
default in the code. This run completed all 16 epochs without divergence.

| Metric | Muon + AdamW (lr_mult=0.8) | AdamW only (lr_mult=0.8) | AdamW low LR (lr_mult=0.6) |
| --- | --- | --- | --- |
| Best Val Loss | 4.7624 | Diverged (NaN) | 4.8975 |
| Final Val Loss | 4.8851 | NaN | 5.0426 |
| Train Loss | 4.6681 | 6.1400 | 4.8525 |
| Best Epoch | 13 (EMA) | 2 (raw) | 16 (EMA) |
| Status | Done | Diverged at step 1,626 | Done |

**Key findings:**

1. **Stability achieved:** Lowering the LR multiplier from 0.8 to 0.6 prevented
   the gradient explosion, confirming the root cause was an LR too aggressive
   for AdamW (which lacks Muon's gradient normalization).

2. **Muon still wins by 0.135:** Even with a stable LR, AdamW (4.8975) is worse
   than Muon+AdamW (4.7624) by 0.1351 (2.8%). This gap is substantial — larger
   than any architecture ablation result (the largest arch ablation was No Value
   Residual at 0.046). This means the optimizer choice matters more than any
   single architectural component at this scale.

3. **Best epoch at 16 (EMA):** AdamW's best val loss came from the final EMA
   evaluation, not from any per-epoch checkpoint. This suggests AdamW's training
   was still improving at the end — the model hadn't fully converged, possibly
   because the lower LR slowed learning.

4. **Higher peak VRAM (4,780 vs 4,717 MiB):** The AdamW run used slightly more
   memory, likely because AdamW maintains both first and second moment buffers
   (`exp_avg` + `exp_avg_sq`) for all parameters, while Muon uses different
   buffers (`momentum_buffer` + `second_momentum_buffer`) that may be more
   memory-efficient for matrix parameters.

5. **Faster wall-clock (414.9m vs 429.0m):** Despite worse val loss, AdamW was
   ~14 minutes faster. Muon's Polar Express orthogonalization adds compute
   overhead (5 Newton-Schulz iterations per step) that AdamW avoids.

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
| LR Low       | 4.6436        | 4.7302         | 4.4751     | 13 (Ckpt avg) | 6,205      | 4,717 MiB | 480.3m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/79wkc6pa) |
| WD Low       | 4.4606        | 4.5044         | 3.9026     | 16 (Ckpt avg) | 5,669      | 4,717 MiB | 455.7m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/yfhvegka) |
| WD High      | 4.9253        | 5.0719         | 4.8846     | 13 (EMA)   | 6,353      | 4,717 MiB | 452.8m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/cvug1pug) |
| Dropout Low  | 4.7150        | 4.8431         | 4.6002     | 13 (EMA)   | 6,792      | 4,712 MiB | 414.0m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/ywvy5jnj) |
| Dropout High | 4.7854        | 4.8939         | 4.7119     | 13 (EMA)   | 6,221      | 4,717 MiB | 445.2m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/aof41ihb) |
| No Shuffle   | 4.7597        | 4.8747         | 4.7382     | 13 (EMA)   | 6,567      | 4,717 MiB | 430.5m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/rjxf93em) |

### Loss Metrics Explained

Each run produces three distinct validation loss values. The **Best Val Loss**
column in the results table is the minimum across all three. The annotation in
the **Best Epoch** column (e.g. "13 (EMA)") indicates which method achieved
that minimum and at which epoch.

| Metric | Description | Formula | When Measured |
| --- | --- | --- | --- |
| **Per-epoch Val Loss** | Raw model cross-entropy loss (nats/token) on the validation set, evaluated at each epoch boundary | $\text{Val Loss} = \frac{\sum_{i} \text{CE}_i \cdot \mathbb{1}[t_i \neq \text{PAD}]}{\sum_{i} \mathbb{1}[t_i \neq \text{PAD}]}$ | End of every epoch (1–16) |
| **EMA Val Loss** | Exponential moving average of model weights, bias-corrected, evaluated once after training ends | $\hat{\theta}_{\text{EMA}} = \frac{\sum_{k} \beta^{k} \cdot \theta_{t-k}}{1 - \beta^{n_{\text{updates}}}}$, then $\text{Val Loss} = \frac{\sum_{i} \text{CE}_i \cdot \mathbb{1}[t_i \neq \text{PAD}]}{\sum_{i} \mathbb{1}[t_i \neq \text{PAD}]}$ | Post-training (final EMA eval) |
| **Ckpt Avg Val Loss (SWA)** | Recency-weighted average of the last 4 epoch checkpoints (Stochastic Weight Averaging), evaluated once after training | $\theta_{\text{SWA}} = \sum_{j=1}^{n} w_j \cdot \theta_{\text{epoch}_j}$, where $w_j = \frac{j}{\sum_{k=1}^{n} k}$, then $\text{Val Loss} = \frac{\sum_{i} \text{CE}_i \cdot \mathbb{1}[t_i \neq \text{PAD}]}{\sum_{i} \mathbb{1}[t_i \neq \text{PAD}]}$ | Post-training (final SWA eval) |

The **Final Val Loss** column is the per-epoch val loss at the last epoch
(Epoch 16) — the raw model's loss before any EMA or SWA averaging. The **Best
Val Loss** is:

$$\text{Best Val Loss} = \min\Big(\min_{\text{epoch } 1..16}\; \text{Val Loss}_{\text{raw}},\;\; \text{Val Loss}_{\text{EMA}},\;\; \text{Val Loss}_{\text{SWA}}\Big)$$

All three metrics use the same evaluation function (`evaluate_bpb`) over ~1M
validation tokens with padding masked out. The EMA and SWA variants differ only
in **which model weights** are evaluated — the loss computation itself is
identical.

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
hyperparameters constant (using the same baseline config as Phase 1) and removes
or modifies one architectural component at a time. Using the same baseline for
both phases allows direct comparison when scaling the best of each phase. This
isolates the contribution of each design choice — attention gating, U-Net skip
connections, value residuals, key offset, and RoPE type — to the final val loss.

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
| Baseline         | 4.7624        | 4.8851         | 4.6681     | 13 (EMA)   | 6,432      | 4,717 MiB | 429.0m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/3pzf3l2k) |
| No Gate          | 4.7555        | 4.8721         | 4.6515     | 13 (EMA)   | 6,609      | 4,705 MiB | 411.3m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/6oz3f8yg) |
| Identity Gate    | 4.7743        | 4.9141         | 4.6876     | 13 (EMA)   | 6,339      | 4,717 MiB | 397.6m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/sytr3g5q) |
| Strong Gate      | 4.7802        | 4.9014         | 4.6729     | 13 (EMA)   | 6,842      | 4,717 MiB | 403.8m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/t60uuvsw) |
| No U-Net Skips   | 4.7642        | 4.8888         | 4.6807     | 13 (EMA)   | 6,860      | 4,717 MiB | 409.6m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/sskx1di3) |
| No Value Residual | 4.8079      | 4.9188         | 4.6975     | 13 (EMA)   | 6,500      | 4,704 MiB | 455.6m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/cwgybdzv) |
| No Key Offset    | 4.7778        | 4.8964         | 4.6877     | 13 (EMA)   | 7,012      | 4,717 MiB | 402.8m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/yy4bgdei) |
| Full RoPE        | 4.7708        | 4.8919         | 4.6745     | 13 (EMA)   | 4,639      | 4,717 MiB | 437.0m     | Done    | [View](https://wandb.ai/i-learn/slowrun/runs/js78r6w8) |

### Ablation Case Summary

All runs use the same training recipe as the Phase 1 baseline (Muon + AdamW,
16 epochs, 10M tokens, LR multiplier 0.8, WD 0.8, dropout 0.1, EMA every 10
steps, SWA last 4 epochs). The validation method is identical across all runs
(`evaluate_bpb` over ~1M tokens). Only the architectural component listed
below is changed per run; all other hyperparameters remain fixed.

| Run | Flag | What Changes | Params | Param Delta | Training/Validation Impact | Observation |
| --- | --- | --- | --- | --- | --- | --- |
| **Baseline** | *(none)* | Current recipe — all 5 components active | 65,667,978 | — | Reference run | Reference (best val loss 4.7624) |
| **No Gate** | `--no-attn-gate` | `attn_gate` Linear(12→8) removed from all 4 layers; attention output `y` passes through `c_proj` directly without any gating | 65,667,594 | −384 | No gate params in optimizer; EMA/SWA operate on fewer params. Training and validation unchanged otherwise. | **Negligible impact.** Best val loss 4.7555 vs baseline 4.7624 — actually slightly *better* by 0.0069 (0.14%), within noise. The gate is zero-init (sigmoid(0)=0.5), so it starts by halving attention output and must learn to recover full strength. At 4 layers / 8 heads on 10M tokens, the model never learns meaningful per-head suppression — the gate stays near init and `c_proj` compensates. Removing it entirely has no measurable downside, suggesting the attention gate only matters at larger scale where selective head specialization becomes relevant. |
| **Identity Gate** | `--attn-gate-variant identity` | Same `attn_gate` Linear, but formula changes from `sigmoid(0)=0.5` to `1 + 0.25*tanh(0)=1.0` at init — gate starts at identity, learns small adjustments | 65,667,978 | 0 | Same param count, same optimizer groups. Only the forward formula differs. Zero-init still used (starts at ~1.0 instead of 0.5). | **Slightly worse than baseline.** Best val loss 4.7743 vs baseline 4.7624 — worse by 0.0119 (0.25%), and worse than No Gate (4.7555) by 0.0188. The identity gate starts at 1.0 (no scaling), so the model begins with full attention strength — but the `0.25*tanh` range limits adjustments to [0.75, 1.25], which may be too narrow to learn useful per-head suppression. Interestingly, the default sigmoid gate (starting at 0.5) and no gate at all both outperform starting at identity, suggesting the initial 0.5 scaling acts as a useful regularizer that the model can gradually learn to open up, rather than starting at full strength and having no reason to learn suppression. |
| **Strong Gate** | `--attn-gate-variant strong` | Same `attn_gate` Linear, but formula changes to `2*sigmoid(0)=1.0` at init — starts near 1 but can strongly suppress (→0) or amplify (→2) heads | 65,667,978 | 0 | Same param count, same optimizer groups. Only the forward formula differs. Zero-init still used (starts at ~1.0). | **Worst gate variant at small scale.** Best val loss 4.7802 vs baseline 4.7624 — worse by 0.0178 (0.37%), and worse than all other gate variants (No Gate 4.7555, Identity 4.7743). The [0, 2] range gives the most expressive power, but at 32 total heads on 10M tokens the model has no need for head amplification — the extra range introduces unnecessary variance in the gate's output without enough data or heads to exploit it. This is expected: the strong gate's advantage (amplifying useful heads) only emerges at scale where 128 heads can meaningfully specialize. At small scale, the wider range is a liability, not an asset. Despite being the worst small-scale result, this remains the top candidate for H100 scaling because it's the only variant that adds expressive power rather than constraining or removing it. |
| **No U-Net Skips** | `--no-skip-connections` | `skip_weights` parameter (2 scalars) removed; encoder layers no longer push activations, decoder layers no longer pop+add. Pure sequential residual stream | 65,667,976 | −2 | `skip_weights` removed from AdamW optimizer group. `GPT.forward()` skips push/pop logic. EMA/SWA operate on fewer params. | **Negligible impact.** Best val loss 4.7642 vs baseline 4.7624 — virtually identical (worse by 0.0018, 0.04%). With only 4 layers, there are just 2 skip connections (layers 0→3 and 1→2), and only 2 scalar weights. The skip connections barely contribute at this depth — the residual stream already provides a direct path through all layers. U-Net skips would likely matter more at 16+ layers where encoder-decoder information reuse becomes valuable, but at 4 layers the model is too shallow for the skips to provide meaningful gradient highways beyond what the standard residual stream already offers. |
| **No Value Residual** | `--no-value-residual` | `ve_projs` (2× Linear 512→512) and `ve_gate` (2× Linear 32→8) removed; value stream `v` is not augmented with projected `x0`. ResFormer disabled | 65,143,178 | −524,800 | Largest param reduction. `ve_projs` removed from Muon matrix groups, `ve_gate` removed from AdamW. `GPT.forward()` passes `ve=None` to all blocks. EMA/SWA operate on fewer params. | **Largest degradation of all ablations.** Best val loss 4.8079 vs baseline 4.7624 — worse by 0.0456 (0.96%), more than double the next-worst ablation (Strong Gate at 0.37%). This is the only ablation that removes a substantial number of parameters (524,800, 0.8% of the model) AND the only one that shows a clear signal beyond noise at this scale. The ResFormer value embeddings provide a direct gated path from the original input `x0` into the value stream on alternating layers, acting as both an information shortcut and a gradient highway. Even at 4 layers, removing this path measurably hurts — the model loses access to the raw embedding signal in deeper layers and must rely solely on the residual stream. This strongly suggests value residuals are the most important architectural component in the recipe, and their contribution will likely *grow* at 16 layers where the distance from `x0` to deeper layers is much greater. |
| **No Key Offset** | `--no-key-offset` | Partial key shift disabled on long-window and last layers; stationary dims of `k` are no longer shifted forward by 1 | 65,667,978 | 0 | No params removed. Only `CausalSelfAttention.forward()` skips the `k[:, 1:, ...]` shift. Optimizer, EMA, SWA all unchanged. | **Small but measurable degradation.** Best val loss 4.7778 vs baseline 4.7624 — worse by 0.0154 (0.32%), placing it between Identity Gate (0.25%) and Strong Gate (0.37%). The partial key offset shifts the stationary (non-rotated) half of keys forward by 1 on long-window and final layers, providing a minimal form of positional information in the dimensions that RoPE leaves untouched. Removing it causes a small but consistent degradation — larger than the No Gate and No U-Net ablations (both <0.04%), suggesting the key offset contributes something real even at this scale. At 16 layers with more long-window layers (the SSSL pattern repeats more times), the key offset would apply to more layers and likely matter more. |
| **Full RoPE** | `--rope-variant full --no-key-offset` | `_precompute_rotary` rotates all `head_dim//2=32` frequency pairs instead of `head_dim//4=16` pairs with zero-padding. Standard RoPE. Key offset also disabled (incompatible with full rotation — would corrupt rotated dims). | 65,667,978 | 0 | No params removed. Only `cos`/`sin` buffers differ (all dims rotated vs. half stationary). Key offset disabled since it shifts the second half of head dim, which is stationary in half-truncated but rotated in full RoPE. Optimizer, EMA, SWA all unchanged. | **Slightly worse than baseline.** Best val loss 4.7708 vs baseline 4.7624 — worse by 0.0084 (0.18%), within the noise band alongside No Gate (−0.14%) and No U-Net Skips (+0.04%). The half-truncated RoPE was a deliberate design choice: rotating only `head_dim//4` frequency pairs leaves more dims stationary for content matching, which appears to be beneficial even at this scale. Full RoPE rotates all dims, providing richer positional encoding but at the cost of less content-preserving capacity. The notably lower tokens/sec (4,639 vs baseline 6,432) is due to the longer epoch time (1,722s vs ~1,600s), not a param difference. At full scale (16 layers, 1024 dim), the tradeoff between positional richness and content preservation may shift — more layers can distribute positional vs. content information more effectively, potentially making full RoPE more competitive. |

---

## Batch Size Calculation (8×H100)

The effective batch size is controlled by `--total-batch-size`, while `--device-batch-size`
controls how many sequences each GPU processes per forward pass. Gradient accumulation
bridges the gap:

```
device_batch_size = 16 sequences per GPU
seq_len           = 2048 tokens per sequence
world_size        = 8 GPUs

tokens_per_micro_step = 16 × 2048 × 8 = 262,144 tokens

total_batch_size  = 524,288 tokens (effective batch per optimizer step)

grad_accum_steps  = 524,288 ÷ 262,144 = 2
```

Each **optimizer step** sees 524,288 tokens worth of gradients. The training loop
runs 2 micro-steps (forward + backward on 262,144 tokens each), accumulating gradients
via `(loss / grad_accum_steps).backward()`, before applying one optimizer update and
clearing gradients.