# 03 — Solutions: Sampling & Generation

*(Q1-3's pure-Python code was actually run to produce the numbers below.)*

## 1. Temperature scaling

```python
import math

def softmax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

logits = [2.0, 1.0, 0.5, 0.1]
for t in [0.1, 0.5, 1.0, 2.0]:
    scaled = [l / t for l in logits]
    print(t, [round(p, 4) for p in softmax(scaled)])
```

**Actual output:**

```text
T=0.1: [1.0,    0.0,    0.0,    0.0]     -- essentially one-hot, fully greedy
T=0.5: [0.8282, 0.1121, 0.0412, 0.0185]  -- still strongly peaked
T=1.0: [0.5745, 0.2114, 0.1282, 0.0859]  -- the model's raw distribution
T=2.0: [0.4056, 0.246,  0.1916, 0.1569]  -- visibly flatter, closer to uniform
```

Exactly the predicted pattern: `T=0.1` collapses to almost-certain
selection of the top logit; `T=2.0` noticeably compresses the gaps between
probabilities, making less-likely tokens meaningfully more samplable.

## 2. Top-k filtering

```python
def top_k_filter(logits, k):
    indexed = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    keep = set(indexed[:k])
    return [logits[i] if i in keep else float("-inf") for i in range(len(logits))]

filtered = top_k_filter(logits, 2)
print(softmax(filtered))
```

**Actual output: `[0.7311, 0.2689, 0.0, 0.0]`** — exactly 2 nonzero
probabilities, renormalized between just the top 2 candidates.

## 3. Top-p filtering

```python
def top_p_filter(logits, p):
    indexed = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
    sorted_logits = [logits[i] for i in indexed]
    sorted_probs = softmax(sorted_logits)
    cum, keep_count = 0.0, 0
    for sp in sorted_probs:
        cum += sp
        keep_count += 1
        if cum > p:
            break
    return keep_count

logits2 = [2.0, 1.8, 1.5, 1.0, 0.5, 0.1]
print(softmax(logits2))
print("p=0.9:", top_p_filter(logits2, 0.9))
print("p=0.5:", top_p_filter(logits2, 0.5))
```

**Actual output:**

```text
softmax: [0.316, 0.259, 0.192, 0.116, 0.07, 0.047]
p=0.9: keeps 5 tokens
p=0.5: keeps 2 tokens
```

With this fairly spread-out distribution, `p=0.9` needs 5 of the 6 tokens
to accumulate 90% probability, while `p=0.5` only needs the top 2 — a
direct demonstration of top-p's *adaptive* behavior: the threshold
responds to how peaked or flat the actual distribution is, unlike top-k's
fixed count regardless of shape.

## 4–6. Full generation loop (PyTorch)

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, greedy=False):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = probs.argmax(dim=-1, keepdim=True) if greedy else torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx

start = torch.tensor([[stoi["t"]]])   # single-character prompt
out = generate(model, start, max_new_tokens=50, block_size=block_size)
print(decode(out[0].tolist()))
```

## 5. Greedy determinism vs sampling variety

```python
for _ in range(5):
    print(decode(generate(model, start, 20, block_size, greedy=True)[0].tolist()))
print("---")
for _ in range(5):
    print(decode(generate(model, start, 20, block_size, temperature=1.0, greedy=False)[0].tolist()))
```

The 5 greedy outputs should be **byte-for-byte identical** every time
(no randomness anywhere in `argmax`); the 5 sampled outputs should
generally differ from each other (`torch.multinomial` draws randomly each
call) — a direct, observable confirmation of which decoding strategy is
deterministic.

## 6. Temperature's qualitative effect on real generated text

Expect low temperature (0.3) to produce more repetitive, "safe" text —
possibly looping on common short phrases; temperature 0.8 to look
reasonably natural (a common default in practice); temperature 1.5 to look
noticeably more erratic, with more unusual word choices and a higher
chance of nonsensical fragments — a qualitative echo of the same sharpening
/flattening pattern verified numerically in Q1, now visible directly in
generated text rather than a probability table.
