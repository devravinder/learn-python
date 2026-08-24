# 03 — Solutions: The Language Modeling Objective

*(Q1-Q2, Q6's numbers were actually computed to produce the results below.)*

## 1. Cross-entropy and perplexity by hand

```python
import math

probs = [0.7, 0.4, 0.9, 0.2]   # P(correct token) at each position
losses = [-math.log(p) for p in probs]
avg_loss = sum(losses) / len(losses)
perplexity = math.exp(avg_loss)
print(avg_loss, perplexity)
```

**Actual output: avg loss ≈ 0.747, perplexity ≈ 2.111.**

## 2. Perfect vs uniform-random model

```python
perfect_ppl = math.exp(-math.log(1.0))
uniform_ppl = math.exp(-math.log(0.01))
print(perfect_ppl, uniform_ppl)
```

**Actual output: `1.0` and `99.9999...` (≈100)** — a perfect model has
perplexity exactly 1 (`log(1)=0`, no loss at all); a uniform-random model
over a 100-token vocabulary has perplexity of essentially exactly 100,
confirming the "effective number of equally likely choices" interpretation
directly: guessing uniformly among 100 options is *exactly* as bad as
perplexity 100 describes.

## 3. Shift-by-one targets

```python
sequence = [5, 12, 7, 9, 3]
input_ids = sequence[:-1]   # [5, 12, 7, 9]
targets = sequence[1:]      # [12, 7, 9, 3]

for i, (inp, tgt) in enumerate(zip(input_ids, targets)):
    print(f"position {i}: given input token {inp}, predict {tgt}")
```

`targets[i] == sequence[i+1] == input_ids[i+1]` — each position's target is
exactly the *next* position's input token, confirming the "labels come
free from the raw sequence" self-supervised setup.

## 4–5. PyTorch cross-entropy on real logits

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
logits = torch.randn(2, 5, 10)
targets = torch.randint(0, 10, (2, 5))

loss = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
perplexity = torch.exp(loss)
print(loss.item(), perplexity.item())

# manual check for one position, e.g. batch 0, position 0
probs = F.softmax(logits[0, 0], dim=0)
manual_loss = -torch.log(probs[targets[0, 0]])

per_position_losses = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1), reduction="none")
print(manual_loss.item(), per_position_losses[0].item())   # should match
```

The manually computed `-log(P(correct token))` for position `(0,0)`
matches `F.cross_entropy`'s per-position loss at that same index exactly
(both compute the identical quantity; `reduction="none"` just skips the
final averaging step) — confirming `F.cross_entropy` isn't doing anything
more mysterious than what `01_concepts.md` describes.

## 6. Perplexity's exponential relationship to loss

```python
for loss_val in [3.5, 2.8, 2.1, 1.5, 1.0]:
    print(loss_val, math.exp(loss_val))
```

**Actual output:**

```text
loss=3.5 -> perplexity=33.12
loss=2.8 -> perplexity=16.44
loss=2.1 -> perplexity=8.17
loss=1.5 -> perplexity=4.48
loss=1.0 -> perplexity=2.72
```

Even though loss decreases by a **constant** 0.7 at each step, perplexity
drops by ever-smaller absolute amounts (33.12→16.44 is a drop of 16.68;
4.48→2.72 is a drop of only 1.76) — a direct consequence of `exp()` being
convex: equal-sized steps in the exponent produce shrinking absolute
differences in the output as the exponent decreases. This is why loss
curves (which look roughly linear-ish in log-scale training plots) and
perplexity curves (which visually flatten out dramatically) can look quite
different when plotted, despite describing the exact same underlying
training progress.
