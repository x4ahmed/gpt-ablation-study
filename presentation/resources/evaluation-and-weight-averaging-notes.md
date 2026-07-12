# Evaluation and Weight Averaging Notes

This note explains how the experiment calculates validation loss and how it
constructs the raw, EMA, and checkpoint-averaged models. It uses the same
notation as slides 12 and 13.

## 1. The complete evaluation pipeline

The process can be understood as five steps:

1. Give the model a sequence of input tokens.
2. At every position, ask it to predict the following token.
3. Calculate cross-entropy for every valid target position.
4. Average those token losses to obtain the validation loss of one model.
5. Repeat the same evaluation for raw, EMA, and checkpoint-averaged weights,
   then report the lowest result.

The important distinction is that the **loss function evaluates a model**, while
EMA and checkpoint averaging create **different sets of model weights** to be
evaluated by that same loss function.

## 2. Inputs and next-token targets

Consider the tokenized sentence:

```text
[The, cat, sleeps]
```

For next-token prediction, the input and target are shifted by one position:

```text
Input:  [The, cat]
Target: [cat, sleeps]
```

The model performs two prediction tasks:

```text
The -> cat
cat -> sleeps
```

At position `i`, the correct next token is written as `t_i`, and all earlier
tokens are written as `t_<i`.

## 3. Per-token cross-entropy

For a target token `t_i`, the per-token cross-entropy is

$$
\mathrm{CE}_i(\theta_e)
=
-\log p_{\theta_e}(t_i\mid t_{<i}).
$$

Here, `theta_e` is the raw model after epoch `e`.

Cross-entropy considers the model's entire vocabulary distribution. Because the
target is one-hot, only the probability assigned to the correct token remains:

$$
\mathrm{CE}_i
=
-\sum_{v\in V} y_{i,v}\log p(v)
=
-\log p(t_i).
$$

Therefore, `CE_i` and the negative log-likelihood of the correct token are the
same quantity in this setting.

### Numerical example

Suppose the context is `The cat` and the correct next token is `sleeps`.

If the model assigns

$$
p(\text{sleeps}\mid\text{The cat})=0.8,
$$

then

$$
\mathrm{CE}=-\log(0.8)\approx 0.223.
$$

If it assigns only probability `0.1`, then

$$
\mathrm{CE}=-\log(0.1)\approx 2.303.
$$

Thus:

- high probability for the correct token gives low loss;
- low probability for the correct token gives high loss.

## 4. Real targets and ignored positions

A real target represents text that the model should predict. A padding or
ignored position is an artificial placeholder and must not affect the loss.

Example with unequal sequence lengths:

```text
Sequence A: [The, cat, sleeps, PAD]
Sequence B: [The, dog, runs, outside]
```

For sequence A, `PAD` is not a word from the original sentence. A mask marks
which positions participate in evaluation:

```text
Target: [cat, sleeps, PAD]
Mask:   [ 1,      1,   0]
```

Conceptually, the mask is

$$
m_i=\mathbb{1}[t_i\neq\mathrm{PAD}].
$$

The indicator function returns `1` for a valid target and `0` for an ignored
position.

In the repository, the ignored target is represented by the integer `-1`, not
by a tokenizer vocabulary item named `PAD`. The implementation uses:

```python
ignore_index = -1
mask = y != -1
```

Therefore, a code-literal version of the mask is

$$
m_i=\mathbb{1}[t_i\neq -1].
$$

## 5. Raw validation loss

Let `theta_e` denote the raw parameters at the end of epoch `e`. The masked mean
validation loss is

$$
\mathcal{L}_{\mathrm{val}}(\theta_e)
=
\frac{
  \sum_i \mathrm{CE}_i(\theta_e)m_i
}{
  \sum_i m_i
}.
$$

The numerator sums cross-entropy only over valid positions. The denominator is
the number of valid target tokens. This prevents padding from lowering or
raising the reported mean.

The raw model is evaluated at the end of each epoch:

$$
\theta_1,\theta_2,\ldots,\theta_E,
$$

where `E = 16` in these experiments. This produces 16 raw validation-loss
candidates.

## 6. Raw weights, EMA weights, and checkpoint-averaged weights

These are three different parameter sets evaluated with the same validation
loss.

### 6.1 Raw weights

`theta_e` is the model exactly as produced at the end of epoch `e`. It contains
no weight averaging.

The best raw result is

$$
\min_{1\le e\le E}\mathcal{L}_{\mathrm{val}}(\theta_e).
$$

### 6.2 Exponential moving average (EMA)

EMA maintains a separate, smoothed copy of the parameters:

$$
\bar\theta_s
=
\beta\bar\theta_{s-1}
+
(1-\beta)\theta_s.
$$

Here, `s` indexes optimizer steps and `S` is the final optimizer step.

Interpretation:

- `beta` controls how slowly the EMA changes;
- `beta * bar(theta)_(s-1)` retains most of the previous smoothed model;
- `(1-beta) * theta_s` adds a small contribution from the current model.

EMA averages **parameters**, not losses or predictions. It does not replace the
training model and does not directly change its gradients. The repository
updates the EMA every 10 optimizer steps and evaluates the final
`bar(theta)_S` after training.

EMA is useful because optimization can oscillate around a good region. The
smoothed weights may lie in a more stable part of that region than the last raw
update.

### 6.3 Checkpoint averaging / SWA phase

During epochs 13--16, cosine learning-rate cycles produce four nearby but
different raw checkpoints. The repository combines them using recency weights:

$$
\theta_{\mathrm{avg}}
=
\frac{
  1\theta_{13}
  +2\theta_{14}
  +3\theta_{15}
  +4\theta_{16}
}{10}.
$$

Equivalently,

$$
\theta_{\mathrm{avg}}
=
\sum_{j=1}^{4}\frac{j}{10}\theta_{E-4+j}.
$$

The newest checkpoint receives the largest weight. This combines model
parameters, not their losses.

The presentation calls this the SWA window because the late cosine cycles are
used to generate checkpoints for averaging. Strictly speaking, the final model
is a **recency-weighted checkpoint average**, whereas classical SWA commonly
uses a uniform average.

## 7. EMA versus checkpoint averaging

| Property | EMA | Checkpoint average |
|---|---|---|
| Inputs | Many parameter states during training | Final four saved checkpoints |
| Weighting | Exponentially favors recent steps | Linear weights `1, 2, 3, 4` |
| Update time | Every 10 optimizer steps | Once after training |
| Final candidate | `bar(theta)_S` | `theta_avg` |
| What is averaged | Model parameters | Model parameters |

Neither method averages validation-loss numbers. Each constructs a model first;
that model is then evaluated normally.

## 8. Selecting the reported best validation loss

The experiment compares three sources:

1. every raw epoch model `theta_e`;
2. the final EMA model `bar(theta)_S`;
3. the recency-weighted checkpoint average `theta_avg`.

The reported value is

$$
\mathcal{L}_{\mathrm{best}}
=
\min\left\{
  \min_{1\le e\le E}\mathcal{L}_{\mathrm{val}}(\theta_e),
  \mathcal{L}_{\mathrm{val}}(\bar\theta_S),
  \mathcal{L}_{\mathrm{val}}(\theta_{\mathrm{avg}})
\right\}.
$$

The result should always be reported together with its source, for example:

```text
Best validation loss: 4.7624 (EMA, epoch 16 training run)
```

This matters because a good raw checkpoint and a good averaged model represent
different training behavior.

## 9. Notation glossary

| Symbol | Meaning |
|---|---|
| `i` | Token-position index used during loss calculation |
| `t_i` | Correct target token at position `i` |
| `t_<i` | Tokens before position `i`, used as context |
| `V` | Vocabulary containing all possible token IDs |
| `v` | One candidate token in the vocabulary |
| `y_(i,v)` | One-hot target value for vocabulary token `v` at position `i` |
| `p_theta(t_i \| t_<i)` | Probability assigned to the correct token by model weights `theta` |
| `CE_i(theta_e)` | Cross-entropy for token position `i` using raw epoch weights `theta_e` |
| `m_i` | Valid-target mask: `1` means include, `0` means ignore |
| `PAD` | Conceptual padding or ignored target position |
| `-1` | Actual ignored-target marker used by the repository |
| `theta` | A complete set of model parameters |
| `theta_e` | Raw model parameters at the end of epoch `e` |
| `e` | Epoch index |
| `E` | Final epoch; `E = 16` here |
| `theta_s` | Current model parameters at optimizer step `s` |
| `s` | Optimizer-step index |
| `S` | Final optimizer step |
| `bar(theta)_s` | EMA parameter set after optimizer step `s` |
| `beta` | EMA retention coefficient between zero and one |
| `theta_avg` | Recency-weighted average of the last four checkpoints |
| `j` | Index from 1 to 4 used for checkpoint weights |
| `L_val(theta)` | Validation loss obtained by evaluating parameter set `theta` |
| `L_best` | Minimum validation loss across raw, EMA, and averaged models |

## 10. Short mental model

Remember the pipeline as:

```text
tokens
  -> next-token probabilities
  -> per-token cross-entropy
  -> ignore invalid targets
  -> mean validation loss for one model
  -> repeat for raw, EMA, and averaged weights
  -> report the lowest value and its source
```

The loss tells us **how good one model is**. EMA and checkpoint averaging tell
us **which version of the model weights we evaluate**.
