# 01 — Concepts: Activation Functions & Softmax

## Why every hidden layer needs one

Lesson 035 showed that without nonlinear activations, stacked layers
collapse into a single linear transformation. Activation functions are what
gives depth its power — each one has different tradeoffs.

## Sigmoid — mostly historical for hidden layers now

`sigmoid(x) = 1/(1+e^-x)` (Lesson 013). Squashes to `(0,1)`. Problem:
**vanishing gradients** — for large `|x|`, `sigmoid'(x) ≈ 0` (it saturates),
so gradients shrink to almost nothing flowing backward through many layers,
making deep networks with sigmoid hidden layers very slow or impossible to
train well. Still used at **output** layers for binary classification
(Lesson 023), just rarely for hidden layers anymore.

## Tanh

`tanh(x) = 2*sigmoid(2x) - 1`. Squashes to `(-1,1)`, zero-centered (a real
advantage over sigmoid — keeps activations balanced around 0). Still
saturates at extremes, so still suffers vanishing gradients, just less
severely than sigmoid.

## ReLU — the modern default

```
relu(x) = max(0, x)
relu'(x) = 1 if x > 0 else 0
```

Simple, cheap to compute, and **doesn't saturate for positive inputs** — the
gradient is exactly 1 wherever the unit is active, solving the vanishing
gradient problem for those units. This is the single biggest reason deep
networks became practically trainable.

**Dying ReLU problem**: if a unit's weights end up such that it always
outputs 0 (input always negative), its gradient is *always* 0 too — it can
never recover during training ("dead"). Variants address this:

- **Leaky ReLU**: `max(0.01x, x)` — small non-zero slope for negative
  inputs, so dead units can still get gradient signal and recover.
- **GELU**: a smooth approximation used in most modern Transformers
  (Lesson 060) — `x * Φ(x)` where `Φ` is the standard normal CDF; behaves
  like ReLU for large `|x|` but smoother near 0.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    from scipy.stats import norm
    return x * norm.cdf(x)
```

## Softmax — generalizing sigmoid to multiple classes

For `k`-class classification, the output layer produces `k` raw scores
(**logits**), and softmax converts them into a valid probability
distribution (Lesson 007):

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Every output is in `(0,1)` and they sum to exactly 1 — a Categorical
distribution over classes (Lesson 007). Paired with categorical cross-entropy
loss (Lesson 016's multi-class generalization), this is the standard output
setup for every multi-class classifier and, crucially, for an LLM's
next-token prediction (Lesson 063) — softmax over the entire vocabulary.

```python
def softmax(z):
    shifted = z - np.max(z, axis=-1, keepdims=True)   # Lesson 007's stability trick
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)
```

## Choosing an activation in practice

| Layer type | Typical choice |
|---|---|
| Hidden layers (general) | ReLU (or a variant: Leaky ReLU, GELU) |
| Transformer hidden layers | GELU (or SwiGLU in newer architectures) |
| Binary classification output | Sigmoid |
| Multi-class classification output | Softmax |
| Regression output | None (linear/identity) |
| RNN/LSTM internal gates | Sigmoid/Tanh (Lesson 046) |

## Temperature-scaled softmax (a preview for LLM sampling)

```
softmax(z / T)_i = exp(z_i/T) / Σ_j exp(z_j/T)
```

`T=1` is standard softmax. `T < 1` sharpens the distribution (more
confident, closer to picking the single max — "greedy"); `T > 1` flattens it
(more random/diverse). This exact mechanism is how LLM text generation
controls "creativity" (Lesson 066) — same softmax function, one added
parameter.
