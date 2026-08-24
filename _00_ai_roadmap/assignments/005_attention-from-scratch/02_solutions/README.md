# Reference Solutions

```bash
python attention_scratch.py
```

*(Output below was actually produced by running this exact script.)*

## Q4: causal masking blocks future influence

```text
position 0 unaffected by change at position 4: True
```

## Q5: masked vs unmasked, position by position

```text
position 0: masked == unmasked -> False
position 1: masked == unmasked -> False
position 2: masked == unmasked -> False
position 3: masked == unmasked -> False
position 4: masked == unmasked -> True
```

**Only position 4 (the last position) matches between masked and
unmasked.** This makes sense once you look at the causal mask's rows: row
4 (`[1,1,1,1,1]`) already allows attending to *every* position, since with
5 total positions there's nothing after position 4 to exclude — so masking
changes nothing for the last position specifically. Every earlier position
*does* have something to lose (positions after it get excluded by the
mask), so positions 0-3 all differ between the masked and unmasked
versions. If you predicted "position 0 stays the same" before running this
(a natural but incorrect guess — position 0 is actually the *most*
restricted by the mask, not the least), this is worth sitting with: the
mask's effect scales with how much *future* context a position would
otherwise be able to see, which is largest for early positions, not last
ones.

## Q6: PyTorch cross-check (verify yourself — needs PyTorch)

```python
import torch

def to_tensor(mat):
    return torch.tensor(mat, dtype=torch.float32)

# reuse the exact same W_q, b_q, ... generated in attention_scratch.py
Q = to_tensor(x) @ to_tensor(W_q) + to_tensor(b_q)
K = to_tensor(x) @ to_tensor(W_k) + to_tensor(b_k)
V = to_tensor(x) @ to_tensor(W_v) + to_tensor(b_v)

def torch_mha(Q, K, V, n_heads, mask=None):
    seq_len, d_model = Q.shape
    d_k = d_model // n_heads
    Qh = Q.view(seq_len, n_heads, d_k).transpose(0, 1)
    Kh = K.view(seq_len, n_heads, d_k).transpose(0, 1)
    Vh = V.view(seq_len, n_heads, d_k).transpose(0, 1)
    scores = Qh @ Kh.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = (weights @ Vh).transpose(0, 1).reshape(seq_len, d_model)
    return out @ to_tensor(W_o) + to_tensor(b_o)

mask_t = to_tensor(causal_mask)
out_torch = torch_mha(Q, K, V, n_heads=2, mask=mask_t)
out_scratch = to_tensor(out_masked)
print(torch.allclose(out_torch, out_scratch, atol=1e-4))   # should be True
```

If this prints `False` instead of `True`, the most common culprits are:
weight matrix orientation (`x @ W` vs `x @ W.T` — PyTorch's `nn.Linear`
actually stores weights transposed relative to the naive `x @ W` form used
here), or a mismatch in how heads are split/concatenated (make sure both
implementations split the *same* contiguous chunks of the feature
dimension per head, in the same order).
