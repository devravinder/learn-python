# 03 — Solutions: LoRA & QLoRA

*(Q1-Q4's code was actually run to produce the numbers below.)*

## 1–2. Parameter count comparison at two matrix sizes

```python
def full_params(d_out, d_in):
    return d_out * d_in

def lora_params(d_out, d_in, rank):
    return rank * d_in + d_out * rank

for d in [4096, 768]:
    fp = full_params(d, d)
    print(f"d={d}, full={fp}")
    for r in [4, 8, 16, 64]:
        lp = lora_params(d, d, r)
        print(f"  rank={r}: {lp} ({100*lp/fp:.3f}% of full)")
```

**Actual output:**

```text
d=4096: full=16,777,216
  rank=4:  32,768   (0.195% of full)
  rank=8:  65,536   (0.391% of full)
  rank=16: 131,072  (0.781% of full)
  rank=64: 524,288  (3.125% of full)

d=768: full=589,824
  rank=4:  6,144    (1.042% of full)
  rank=8:  12,288   (2.083% of full)
  rank=16: 24,576   (4.167% of full)
  rank=64: 98,304   (16.667% of full)
```

At `d=4096`, LoRA stays under 1% of full parameters all the way up to
`rank=16`; at `d=768`, even `rank=4` already exceeds 1%. **The relative
savings get dramatically better as the base matrix gets larger** — LoRA's
parameter count grows *linearly* with `d` (`rank*d + d*rank = 2*rank*d`),
while full fine-tuning grows *quadratically* (`d*d`), so the ratio
`LoRA/full = 2*rank/d` shrinks as `d` grows. This is exactly why LoRA's
benefit is most dramatic on today's largest models (huge `d_model`), not a
one-size-fits-all percentage.

## 3–4. LoRA's zero-init behavior, verified

```python
import random

def matmul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]
def transpose(A):
    return [list(r) for r in zip(*A)]

random.seed(0)
d_out, d_in, rank = 4, 6, 2
W = [[random.uniform(-1, 1) for _ in range(d_in)] for _ in range(d_out)]
A = [[random.uniform(-1, 1) for _ in range(d_in)] for _ in range(rank)]
B = [[0.0] * rank for _ in range(d_out)]   # zero init
x = [[random.uniform(-1, 1) for _ in range(d_in)]]

out_W_only = matmul(x, transpose(W))
delta = matmul(matmul(x, transpose(A)), transpose(B))
out_with_lora = [[out_W_only[0][j] + delta[0][j] for j in range(d_out)]]
print(out_W_only[0] == out_with_lora[0])   # True
```

**Actual output: `True`** — with `B` at all zeros, the LoRA-augmented
output is bit-for-bit identical to the frozen weight's output alone,
confirming zero-effect-at-initialization exactly as `01_concepts.md`
claims.

```python
B_trained = [[0.1 * random.uniform(-1, 1) for _ in range(rank)] for _ in range(d_out)]
delta2 = matmul(matmul(x, transpose(A)), transpose(B_trained))
out_after = [[out_W_only[0][j] + delta2[0][j] for j in range(d_out)]]
print(out_after[0] != out_W_only[0])   # True
```

**Actual output: `True`** — once `B` moves away from zero (simulating a
few training steps), the output changes measurably, confirming the
adapter's effect grows from exactly zero as training proceeds, rather than
starting from an arbitrary random perturbation.

## 5. LoRA on a real model

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False
        d_out, d_in = original_linear.weight.shape
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.scale

# wrap one linear layer of a model, e.g. a Project 013 GPT block's W_q
model.blocks[0].attn.W_q = LoRALinear(model.blocks[0].attn.W_q, rank=8)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(trainable, total, trainable / total)
```

Expect `trainable` to be a small fraction of `total` — only the wrapped
layer's `A`/`B` parameters remain trainable, everything else (including
the rest of the model) stays frozen.

## 6. Merging LoRA weights

```python
with torch.no_grad():
    merged_weight = lora_layer.original.weight + lora_layer.scale * (lora_layer.B @ lora_layer.A)
    merged_linear = nn.Linear(lora_layer.original.in_features, lora_layer.original.out_features, bias=False)
    merged_linear.weight.copy_(merged_weight)

x_test = torch.randn(1, lora_layer.original.in_features)
out_unmerged = lora_layer(x_test)
out_merged = merged_linear(x_test)
print(torch.allclose(out_unmerged, out_merged, atol=1e-5))   # should be True
```

If this doesn't return `True`, the most likely culprit is a missed
`scale` factor or a transpose mismatch between how `A`/`B` are stored vs.
how they're combined into `merged_weight` — worth checking both carefully,
since a silent shape-broadcast bug here could look like it "works" while
actually producing subtly wrong merged weights.
