# 03 — Solutions: Build a GPT, Part 1

*(Q1-3, Q6's calculations were actually run to produce the numbers below.)*

## 1. Vocabulary and round-trip check

```python
text = "to be or not to be that is the question whether tis nobler in the mind to suffer"
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
vocab_size = len(chars)

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

assert decode(encode(text)) == text
print(vocab_size)   # 18
```

**Actual output: round-trip passes exactly, `vocab_size = 18`** (letters
used plus the space character in this sample text).

## 2. Train/val split by position

```python
data = encode(text)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(len(train_data), len(val_data))   # 72, 8
```

## 3. `get_batch` in pure Python, with the shift verified

```python
import random

def get_batch(data, block_size, batch_size, seed=0):
    rng = random.Random(seed)
    ix = [rng.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    xs = [data[i:i+block_size] for i in ix]
    ys = [data[i+1:i+block_size+1] for i in ix]
    return xs, ys

xs, ys = get_batch(train_data, block_size=8, batch_size=4)
for x, y in zip(xs, ys):
    print(x, y, "shift check:", y[:-1] == x[1:])
```

**Actual output** (4 sampled chunks, all pass the shift check):

```text
[7,14,0,10,11,2,8,4]  [14,0,10,11,2,8,4,13]  shift check: True
[11,2,8,4,13,0,7,10]  [2,8,4,13,0,7,10,0]    shift check: True
[0,11,13,0,10,11,15,0] [11,13,0,10,11,15,0,15] shift check: True
[4,14,15,7,11,10,0,17] [14,15,7,11,10,0,17,6]  shift check: True
```

Every `y`'s first `block_size-1` values exactly match `x`'s last
`block_size-1` values — `y` is genuinely `x` shifted by one position, for
every randomly sampled chunk.

## 4. PyTorch version

```python
import torch

def get_batch_torch(data, block_size, batch_size):
    data_t = torch.tensor(data, dtype=torch.long)
    ix = torch.randint(len(data_t) - block_size, (batch_size,))
    x = torch.stack([data_t[i:i+block_size] for i in ix])
    y = torch.stack([data_t[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch_torch(train_data, block_size=8, batch_size=4)
print(xb.shape, yb.shape)   # torch.Size([4, 8]) torch.Size([4, 8]) -- matches pure-Python shapes
```

## 5. Model shape check

```python
model = GPT(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=4, d_ff=256, max_len=8)
logits = model(xb)
print(logits.shape)   # torch.Size([4, 8, 18])
```

## 6. Expected initial loss

```python
import math
print(-math.log(1 / vocab_size))
```

**Actual output: ≈ 2.890.** A completely untrained model's weights are
essentially random, so its output distribution over the vocabulary should
be close to uniform — cross-entropy loss for a uniform guess over 18
options is exactly `-log(1/18) ≈ 2.890` (Lesson 063's "uniform-random
model" perplexity calculation, applied here as a loss instead). If you
compute your actual untrained model's loss on a real batch and it comes
out close to this number (typically within a reasonable range, not exactly
identical due to the specific random initialization), that's a strong
sanity check the model and loss computation are wired correctly *before*
you invest any time training it — if the initial loss were wildly
different (e.g. much higher), that would flag a bug worth finding before
Lesson 065's training loop.
