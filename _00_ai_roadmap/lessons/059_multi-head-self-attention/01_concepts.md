# 01 — Concepts: Multi-Head Self-Attention

## Why one attention computation isn't enough

A single attention computation produces *one* weighted blend per position
— but a word can be relevant to its sentence in multiple, simultaneously
different ways (its grammatical role, its topical meaning, which pronoun
refers to it, etc.). Forcing all of this through one shared Q/K/V
projection means these different relationship types have to compete for
the same limited representational space.

## The fix: run several attention "heads" in parallel

Split the model dimension into `h` smaller chunks, run **independent**
scaled dot-product attention (Lesson 058) in each chunk (each with its own
learned `W_Q`, `W_K`, `W_V` projections), then concatenate the results and
project back to the original dimension:

```
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
MultiHead(Q,K,V) = concat(head_1, ..., head_h) @ W_O
```

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Q,K,V: (batch, n_heads, seq_len, d_k)

        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ V   # (batch, n_heads, seq_len, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)   # concat heads
        return self.W_o(out)
```

## Why splitting the dimension (not just running full-size heads h times)

Note `d_k = d_model / n_heads` — each head operates on a *smaller* slice of
the full dimension, not the full dimension repeated `h` times. This keeps
total compute roughly the same as single-head attention at the full
dimension, while still giving each head its own learned projection and
therefore the freedom to specialize — a deliberate design choice trading
per-head capacity for the diversity of having multiple independent
"perspectives."

## What different heads empirically learn

In trained Transformers, different heads are commonly observed to
specialize — some attend mostly to adjacent words (local syntax), some
track long-range dependencies (e.g. a pronoun attending back to what it
refers to), some attend broadly across the whole sequence. This isn't
programmed in explicitly; it emerges purely from training multiple
independently-initialized heads on the same objective — a similar
"specialization emerges from diversity + a shared objective" story to
Lesson 027's Random Forest (different trees specializing via different
bootstrap samples/feature subsets).

## Masking: controlling what a position is allowed to attend to

`scores.masked_fill(mask == 0, float("-inf"))` sets certain positions'
scores to `-inf` *before* softmax, so they get **exactly 0** attention
weight after softmax (`exp(-inf) = 0`). Two masking patterns matter for
Module 11:
- **Padding mask**: ignore `[PAD]` tokens (Lesson 055) added to make
  batched sequences the same length — they carry no real information and
  shouldn't influence real tokens' representations.
- **Causal mask**: for language modeling (Lesson 063), position `t` must
  only attend to positions `<= t` — it cannot "see the future" tokens it's
  supposed to be predicting, or the training objective becomes trivial
  (just copy the answer). This is implemented as a lower-triangular mask,
  and is exactly what makes GPT-style models autoregressive.

```python
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1,0,0,0,0],
#  [1,1,0,0,0],
#  [1,1,1,0,0],
#  [1,1,1,1,0],
#  [1,1,1,1,1]]
```

## Computational cost: the quadratic tradeoff

Computing `Q @ K^T` is `O(seq_len^2 * d_model)` — attention cost grows
**quadratically** with sequence length. This is the price paid for
attention's "any position to any position, in one step" advantage over
RNNs' linear-but-sequential cost — a real, actively-researched tradeoff
(sparse/linear attention variants exist specifically to address this for
very long sequences), worth being aware of even though this curriculum's
models stay small enough not to hit this limit in practice.

## Setting up Lesson 060

Multi-head attention is one of two sub-layers in a full Transformer block —
the other is a position-wise feedforward network. Lesson 060 assembles
both, plus residual connections (Lesson 044) and layer normalization
(Lesson 042's BatchNorm, adapted), into the complete block architecture.
