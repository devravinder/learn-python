# 03 — Solutions: The Transformer Architecture

*(Q1's pure-Python code was actually run to produce the numbers below.)*

## 1. LayerNorm from scratch

```python
import math

def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [gamma * (v - mean) / math.sqrt(var + eps) + beta for v in x]

x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
normed = layer_norm(x)
print(normed)
mean_after = sum(normed) / len(normed)
var_after = sum((v - mean_after) ** 2 for v in normed) / len(normed)
print(mean_after, var_after)
```

**Actual output:** `normed = [-1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0]`,
`mean_after ≈ 0.0`, `var_after ≈ 0.999998` (the tiny deviation from exactly
1.0 is just the `eps=1e-5` stability term in the denominator) — confirming
LayerNorm does exactly what it claims: re-centers and re-scales a single
position's feature vector to mean 0, variance 1.

## 2. Why LayerNorm suits padded batches

Suppose a batch has 3 sequences of lengths 4, 6, and 10, padded to length
10 with `[PAD]` tokens' (typically zero) embeddings. **BatchNorm** computes
statistics *per feature, across the batch* at each position — at position
7 (past the end of the first two sequences), two of the three "sequences"
in the batch contribute meaningless padding values, which directly
corrupts that position's batch mean/variance. **LayerNorm** computes
statistics *per position, across features, independently for each
sequence* — a padded position in sequence 1 never has its statistics mixed
with sequence 2 or 3's real data at all, since each position's
normalization only ever looks at that one position's own feature vector.
This is the concrete reason every Transformer uses LayerNorm, not
BatchNorm, despite BatchNorm being the default elsewhere in this
curriculum (Lesson 042).

## 3–4. Assembling and stacking blocks

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)   # from Lesson 059
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

block = TransformerBlock(d_model=32, n_heads=4, d_ff=128)
x = torch.randn(2, 8, 32)
print(block(x).shape)   # torch.Size([2, 8, 32]) -- unchanged

blocks = nn.ModuleList([TransformerBlock(32, 4, 128) for _ in range(4)])
for b in blocks:
    x = b(x)
print(x.shape)   # still torch.Size([2, 8, 32]) after 4 stacked blocks
```

## 5. Full GPT assembly

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(positions)
        causal_mask = torch.tril(torch.ones(T, T, device=idx.device))
        for block in self.blocks:
            x = block(x, mask=causal_mask)
        x = self.ln_final(x)
        return self.head(x)

model = GPT(vocab_size=50, d_model=32, n_heads=4, n_layers=2, d_ff=128, max_len=20)
idx = torch.randint(0, 50, (1, 10))
logits = model(idx)
print(logits.shape)   # torch.Size([1, 10, 50])
```

## 6. Parameter count vs GPT-2

```python
n_params = sum(p.numel() for p in model.parameters())
print(n_params)   # a few tens of thousands, at this toy scale
```

This toy model has on the order of tens of thousands of parameters;
GPT-2 small has **~124 million** — roughly **1,000-4,000x** more
parameters, from a much larger `d_model` (768 vs 32), more layers (12 vs
2), more heads (12 vs 4), and a real ~50,000-token vocabulary (vs this
toy's 50). The *architecture* you just built is structurally identical to
GPT-2's — Module 11 scales these same exact building blocks up to
something that can meaningfully generate text, still far smaller than
GPT-2 itself but built from unchanged first principles.
