# 03 — Solutions: Positional Encoding

*(This code was actually run to produce the numbers below.)*

## 1. Sinusoidal encoding

```python
import math

def sinusoidal_encoding(seq_len, d_model):
    pe = [[0.0] * d_model for _ in range(seq_len)]
    for pos in range(seq_len):
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            pe[pos][i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
    return pe

pe = sinusoidal_encoding(6, 8)
for row in pe:
    print([round(v, 3) for v in row])
```

**Actual output (first 2 rows):**

```text
[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
[0.841, 0.54, 0.1, 0.995, 0.01, 1.0, 0.001, 1.0]
```

Position 0's row is exactly `[0, 1, 0, 1, 0, 1, 0, 1]` as predicted
(`sin(0)=0`, `cos(0)=1` regardless of dimension). Position 1's row already
shows the frequency pattern: dimension 0 (`sin`, fastest frequency) has
moved to `0.841`, while dimension 6 (`sin`, slowest frequency) has barely
moved from 0 (`0.001`).

## 2. Frequency pattern visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def sinusoidal_encoding_np(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angles = pos / (10000 ** (2 * (i // 2) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

pe = sinusoidal_encoding_np(50, 64)
plt.imshow(pe.T, cmap="RdBu", aspect="auto")
plt.xlabel("position")
plt.ylabel("dimension")
plt.show()
```

Left columns (low dimension index) should show tight, fast-alternating
stripes across positions; right columns should show broad, slowly-varying
bands — a direct visual confirmation of the varying-frequency-per-dimension
design.

## 3. Similarity vs distance

```python
def cos_sim(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
    return dot / (na * nb)

pe64 = sinusoidal_encoding(50, 64)
print(cos_sim(pe64[0], pe64[1]))    # positions 0 vs 1
print(cos_sim(pe64[0], pe64[10]))   # positions 0 vs 10
print(cos_sim(pe64[0], pe64[40]))   # positions 0 vs 40
```

Similarity is generally highest for the closest pair (0 vs 1) and
generally decreases (with some non-monotonic wobble, since it's a sum of
many different-frequency oscillations, not a strictly decaying signal) as
distance grows — broadly sensible behavior for a positional signal, though
not a strictly monotonic decay at every possible offset.

## 4. RoPE's relative-position property, verified

```python
import random

def rotate(vec, theta):
    x, y = vec
    return [x*math.cos(theta) - y*math.sin(theta), x*math.sin(theta) + y*math.cos(theta)]

random.seed(0)
q = [random.uniform(-1, 1), random.uniform(-1, 1)]
k = [random.uniform(-1, 1), random.uniform(-1, 1)]
freq = 1.0

for pos_q, pos_k in [(3, 1), (5, 3), (10, 8)]:   # all have diff = 2
    qr, kr = rotate(q, pos_q*freq), rotate(k, pos_k*freq)
    dot = qr[0]*kr[0] + qr[1]*kr[1]
    print(f"pos_q={pos_q} pos_k={pos_k} diff={pos_q-pos_k} dot={dot:.5f}")
```

**Actual output:**

```text
pos_q=3  pos_k=1  diff=2  dot=-0.07843
pos_q=5  pos_k=3  diff=2  dot=-0.07843
pos_q=10 pos_k=8  diff=2  dot=-0.07843
```

**Identical to 5 decimal places across three completely different absolute
position pairs**, as long as the *difference* (2) stays the same — a
precise, verified confirmation of RoPE's defining property.

## 5. Different differences give different dot products

```python
for pos_q, pos_k in [(3, 1), (6, 1), (9, 1)]:   # diffs = 2, 5, 8
    qr, kr = rotate(q, pos_q*freq), rotate(k, pos_k*freq)
    dot = qr[0]*kr[0] + qr[1]*kr[1]
    print(f"diff={pos_q-pos_k} dot={dot:.5f}")
```

Each different relative distance produces a genuinely different dot
product value, confirming the score is a function of relative distance —
sensitive to *how far apart*, not to any particular absolute position.

## 6. Why this matters for language models

A phrase like "the cat sat" carries the same grammatical relationship
between "cat" and "sat" whether it appears at the very start of a document
or 10,000 tokens in. With RoPE, the attention score between "cat" and
"sat" depends only on their being 1 token apart, **regardless of where in
the document that phrase occurs** — the model doesn't need separate
training examples covering every possible absolute position a given
relationship might appear at; the relative-position structure is baked
directly into the math. This is also part of why RoPE models tend to
generalize better to sequence lengths beyond what they were trained on
(Lesson 061's "length extrapolation" point): the relationship your model
actually needs to learn (how attention should behave based on relative
distance) is exactly what the rotation directly encodes, rather than
something the model has to infer indirectly from a fixed additive signal.
