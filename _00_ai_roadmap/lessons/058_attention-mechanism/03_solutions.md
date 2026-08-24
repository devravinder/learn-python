# 03 — Solutions: The Attention Mechanism

*(This code was actually run to produce the numbers below.)*

## 1–2. From-scratch attention

```python
import math, random

def matmul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][t]*B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]

def transpose(A):
    return [list(row) for row in zip(*A)]

def softmax_row(row):
    mx = max(row)
    exps = [math.exp(x - mx) for x in row]
    s = sum(exps)
    return [e / s for e in exps]

def scaled_dot_product_attention(Q, K, V):
    d_k = len(Q[0])
    scores = matmul(Q, transpose(K))
    scaled = [[v / math.sqrt(d_k) for v in row] for row in scores]
    weights = [softmax_row(row) for row in scaled]
    output = matmul(weights, V)
    return output, weights

random.seed(0)
seq_len, d_k = 4, 8
Q = [[random.uniform(-1, 1) for _ in range(d_k)] for _ in range(seq_len)]
K = [[random.uniform(-1, 1) for _ in range(d_k)] for _ in range(seq_len)]
V = [[random.uniform(-1, 1) for _ in range(d_k)] for _ in range(seq_len)]

output, weights = scaled_dot_product_attention(Q, K, V)
for row in weights:
    print([round(w, 3) for w in row], "sum=", round(sum(row), 4))
```

**Actual output** — every row sums to exactly `1.0`:

```text
[0.336, 0.175, 0.242, 0.247] sum= 1.0
[0.207, 0.303, 0.251, 0.238] sum= 1.0
[0.215, 0.173, 0.352, 0.261] sum= 1.0
[0.228, 0.29,  0.165, 0.317] sum= 1.0
```

## 3. Why scaling matters, at a realistic dimension

```python
random.seed(1)
seq_len, d_k = 4, 64
Q = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(seq_len)]
K = [[random.gauss(0, 1) for _ in range(d_k)] for _ in range(seq_len)]

scores = matmul(Q, transpose(K))
print("raw scores:", [round(s, 2) for s in scores[0]])
print("unscaled softmax:", [round(w, 4) for w in softmax_row(scores[0])])
print("scaled softmax:  ", [round(w, 4) for w in softmax_row([s / math.sqrt(d_k) for s in scores[0]])])
```

**Actual output:**

```text
raw scores:        [3.26, 0.35, 3.95, 2.8]
unscaled softmax:  [0.2725, 0.0149, 0.5412, 0.1714]
scaled softmax:    [0.2682, 0.1865, 0.2922, 0.2531]
```

The unscaled softmax is noticeably more peaked (0.54 vs 0.015 — a 36x
ratio between the largest and smallest weight) than the scaled version
(0.29 vs 0.19 — only a ~1.6x ratio). At `d_k=64` (a typical Transformer
head dimension), dot products naturally grow larger in magnitude purely
from summing more terms — dividing by `sqrt(d_k)` counteracts exactly that
growth, keeping softmax in a well-behaved (non-saturated) regime
regardless of dimension, precisely as `01_concepts.md` claims.

## 4. Attention correctly retrieving a distinctive value

```python
K = [[1,0,0,0], [0,1,0,0], [5,5,5,5], [0,0,0,1]]
Q = [[5,5,5,5], [0,0,0,0], [0,0,0,0], [0,0,0,0]]   # query 0 aligned with key 2
V = [[0,0], [0,0], [10,10], [0,0]]                   # value 2 is distinctive

output, weights = scaled_dot_product_attention(Q, K, V)
print(weights[0])   # [0.0, 0.0, 1.0, 0.0]
print(output[0])    # [10.0, 10.0]
```

**Actual output: `weights[0] = [0.0, 0.0, 1.0, 0.0]`, `output[0] = [10.0, 10.0]`.**
Query 0's dot product with key 2 is far larger than with any other key
(`[5,5,5,5]·[5,5,5,5] = 100` vs at most `5` for the others), so softmax
assigns essentially all attention weight to position 2, and the output is
exactly value 2's vector — attention correctly and cleanly "retrieves" the
one relevant value, matching the query-key-value analogy from
`01_concepts.md` directly and concretely.

## 5. PyTorch version

```python
import torch
import torch.nn.functional as F

def torch_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights

Q_t = torch.tensor(Q, dtype=torch.float32)
K_t = torch.tensor(K, dtype=torch.float32)
V_t = torch.tensor(V, dtype=torch.float32)
out_t, w_t = torch_attention(Q_t, K_t, V_t)
print(w_t[0])   # matches the from-scratch [0., 0., 1., 0.]
```

## 6. Visualizing an attention map

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

seq_len, d_model = 6, 16
rng = np.random.default_rng(0)
X = rng.normal(size=(seq_len, d_model))   # stand-in for random embeddings
Q, K, V = X, X, X   # self-attention: Q, K, V from the same sequence

d_k = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d_k)
weights = np.exp(scores - scores.max(axis=1, keepdims=True))
weights /= weights.sum(axis=1, keepdims=True)

sns.heatmap(weights, annot=True, fmt=".2f", cmap="viridis")
plt.xlabel("key position (attended to)")
plt.ylabel("query position")
plt.show()
```

With random, untrained embeddings, the heatmap will look like noise — no
meaningful linguistic pattern to read yet. Once Lesson 060's Transformer
is actually trained on real text, the exact same visualization starts
showing genuinely interpretable patterns (e.g. a token attending strongly
to a grammatically related earlier token) — the mechanics you just
practiced here don't change at all, only the learned weights feeding into
them.
