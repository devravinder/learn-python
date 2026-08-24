# 01 — Concepts: Entropy, Cross-Entropy & KL Divergence

## Information content ("surprise") of an event

An event with probability `p` carries information content:

```
I(p) = -log(p)
```

Low-probability events are "more surprising" (higher information) — a coin
landing on its edge is far more surprising than heads. Log base 2 gives
units of **bits**; natural log gives **nats** (ML frameworks typically use
nats, i.e. natural log, unless stated otherwise).

## Entropy — average surprise of a distribution

**Entropy** `H(P)` is the expected information content of a distribution `P`:

```
H(P) = -Σ p_i * log(p_i)
```

Entropy is maximized when `P` is uniform (maximum uncertainty — you can't
predict the outcome at all) and minimized (zero) when `P` is a point mass
(one outcome certain, no uncertainty). In ML, entropy of a model's predicted
distribution measures how "confident" (low entropy) or "unsure" (high
entropy) it is.

```python
import numpy as np

def entropy(p):
    p = np.asarray(p)
    p = p[p > 0]   # 0 * log(0) is defined as 0, avoid log(0)
    return -np.sum(p * np.log(p))
```

## Cross-entropy — comparing a true distribution to a predicted one

Cross-entropy measures the average surprise of outcomes from the **true**
distribution `P`, using probabilities **predicted** by a different
distribution `Q`:

```
H(P, Q) = -Σ p_i * log(q_i)
```

This is exactly the classification loss function. For a single labeled
example, `P` is a one-hot vector (probability 1 on the true class, 0
elsewhere), so the sum collapses to a single term:

```
loss = -log(q_true_class)
```

— "how much probability did the model assign to the correct answer, in log
space." This is why cross-entropy loss decreases as the model gets more
confident *and correct*, and increases sharply as it gets more confident and
*wrong* (since `-log(q)` blows up as `q -> 0`).

```python
def cross_entropy(y_true_onehot, y_pred_probs):
    y_pred_probs = np.clip(y_pred_probs, 1e-12, 1)   # avoid log(0)
    return -np.sum(y_true_onehot * np.log(y_pred_probs))
```

## KL Divergence — the "gap" between two distributions

**Kullback-Leibler divergence** measures how different `Q` is from `P`:

```
KL(P || Q) = Σ p_i * log(p_i / q_i) = H(P, Q) - H(P)
```

`KL(P||Q) >= 0` always, and `= 0` only when `P == Q` exactly. It is **not
symmetric**: `KL(P||Q) != KL(Q||P)` in general, so it's not a true
"distance," but it does measure directional divergence. Note the identity
above: **cross-entropy = entropy + KL divergence**. Since `H(P)` (the true
distribution's own entropy) is fixed and doesn't depend on your model,
**minimizing cross-entropy is exactly equivalent to minimizing KL
divergence** between your model's predictions and the true distribution —
which is why cross-entropy loss is theoretically well-motivated, not just a
convenient formula.

## Where this shows up later in the curriculum

- **Every classifier's loss function** (Lessons 023, 036): softmax output +
  cross-entropy loss against the true label.
- **Language model training** (Lesson 063): cross-entropy between the
  model's predicted next-token distribution and the actual next token —
  literally the same formula, applied per token.
- **Perplexity** (Lesson 073): `exp(average cross-entropy loss)` — the
  standard way LLMs report how well they predict held-out text.
- **RLHF/DPO** (Lesson 072): KL divergence is used directly as a penalty
  term to keep a fine-tuned model's output distribution from drifting too
  far from the original model.

## Mutual information (brief mention)

Mutual information `I(X; Y)` measures how much knowing `X` reduces
uncertainty about `Y` — it's `KL` between the joint distribution `P(X,Y)` and
the product of marginals `P(X)P(Y)`. Zero exactly when `X` and `Y` are
independent (Lesson 006). Occasionally used for feature selection ("how much
does this feature actually tell you about the label").
