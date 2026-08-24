# 03 — Solutions: Python Fundamentals & Data Structures

## 1–2. Vocabulary + id encoding

```python
words = ["the", "cat", "sat", "on", "the", "mat"]

vocab = {}
for w in words:
    if w not in vocab:
        vocab[w] = len(vocab)
print(vocab)  # {'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4}

ids = [vocab[w] for w in words]
print(ids)    # [0, 1, 2, 3, 0, 4]
```

## 3. `normalize`

```python
def normalize(values, low=0.0, high=1.0):
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    if span == 0:
        return [low for _ in values]
    return [low + (v - vmin) / span * (high - low) for v in values]

print(normalize([10, 20, 30, 40]))  # [0.0, 0.333..., 0.666..., 1.0]
```

## 4. `batch` generator

```python
def batch(iterable, size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

for b in batch(range(23), 5):
    print(b)
# [0,1,2,3,4] [5,6,7,8,9] [10,11,12,13,14] [15,16,17,18,19] [20,21,22]
```

## 5. `Dataset` class

```python
class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

ds = Dataset(["a", "b", "c"], [0, 1, 0])
print(len(ds))   # 3
print(ds[1])     # ('b', 1)
```

## 6. Degenerate `normalize` input

Already handled above: when `span == 0` every input value is identical, so there's
no meaningful position within a range — returning `low` for every element is a
reasonable, crash-free convention. The bug in the naive version is a
`ZeroDivisionError` on `(v - vmin) / span`; guarding on `span == 0` before dividing
fixes it.
