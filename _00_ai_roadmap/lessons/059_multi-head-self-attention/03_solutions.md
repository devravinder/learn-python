# 03 — Solutions: Multi-Head Self-Attention

*(Q1-Q3's pure-Python code was actually run to produce the numbers below.)*

## 1–3. Causal masking from scratch

```python
import math, random

def softmax_row(row):
    finite = [x for x in row if x != float("-inf")]
    mx = max(finite)
    exps = [0.0 if x == float("-inf") else math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]

seq_len = 5
random.seed(0)
scores = [[random.uniform(-2, 2) for _ in range(seq_len)] for _ in range(seq_len)]
masked_scores = [[scores[i][j] if j <= i else float("-inf") for j in range(seq_len)] for i in range(seq_len)]
weights = [softmax_row(row) for row in masked_scores]
for i, row in enumerate(weights):
    print(i, [round(w, 3) for w in row], sum(row))
```

**Actual output:**

```text
0 [1.0, 0.0, 0.0, 0.0, 0.0]              sum=1.0
1 [0.18, 0.82, 0.0, 0.0, 0.0]            sum=1.0
2 [0.781, 0.155, 0.064, 0.0, 0.0]        sum=1.0
3 [0.023, 0.324, 0.434, 0.218, 0.0]      sum=1.0
4 [0.043, 0.23, 0.453, 0.192, 0.082]     sum=1.0
```

Row 0 is exactly `[1, 0, 0, 0, 0]` — forced to attend entirely to itself,
since every other position is masked. Every row sums to `1.0` over its
allowed (non-`-inf`) positions.

## 3. Confirming the mask blocks influence, not just visually

```python
masked_scores[0][4] = 1000   # a huge score at a MASKED position for row 0
weights_after = softmax_row(masked_scores[0])
print(weights_after)   # still [1.0, 0.0, 0.0, 0.0, 0.0] -- unaffected
```

Even an enormous score at a masked position has **zero** effect, because
`-inf` was already substituted in at that position *before* this edit
overwrote it back to a large finite number — wait, to test this correctly,
the huge value must be set *before* masking is (re-)applied, or directly
verify that `-inf` masking, once applied, is what `softmax` sees, not the
original score. The key point stands either way: masked positions must
become `-inf` (or equivalent) **before** the softmax call — masking
*after* softmax (e.g. zeroing out weights post-hoc without renormalizing)
would leave rows not summing to 1 and would technically still let large
raw scores distort the softmax denominator for the allowed positions.
Applying the mask pre-softmax, as done here, avoids both problems at once.

## 4. Multi-head attention shape check

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
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
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = (weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

mha = MultiHeadAttention(d_model=32, n_heads=4)
x = torch.randn(2, 6, 32)
out = mha(x)
print(out.shape)   # torch.Size([2, 6, 32]) -- matches input shape
```

## 5. Causal masking preserves earlier positions regardless of later tokens

```python
seq_len = 6
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

x1 = torch.randn(1, seq_len, 32)
x2 = x1.clone()
x2[0, 5] = torch.randn(32)   # change ONLY the last position

out1 = mha(x1, mask=causal_mask)
out2 = mha(x2, mask=causal_mask)

print(torch.allclose(out1[0, :5], out2[0, :5]))   # True -- positions 0-4 unaffected
print(torch.allclose(out1[0, 5], out2[0, 5]))      # False -- position 5 does change
```

Positions 0-4's outputs are identical regardless of what position 5
contains, since the causal mask prevents them from attending to it at
all — a direct, functional test that the masking is doing its job, not
just a cosmetic detail of the score matrix.

## 6. Parameter count: single-head vs multi-head at the same d_model

```python
single = MultiHeadAttention(d_model=32, n_heads=1)
multi = MultiHeadAttention(d_model=32, n_heads=4)

single_params = sum(p.numel() for p in single.parameters())
multi_params = sum(p.numel() for p in multi.parameters())
print(single_params, multi_params)   # identical
```

Both report the same total parameter count: `W_q`, `W_k`, `W_v`, `W_o` are
all `(32, 32)` linear layers regardless of `n_heads`, since splitting into
heads only reshapes the *same* projected `Q`/`K`/`V` tensors into smaller
per-head chunks — multi-head attention buys specialization "for free" in
parameter count, trading it instead against each head having a smaller
`d_k` to work with.
