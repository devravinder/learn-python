# 01 — Concepts: Python Fundamentals & Data Structures

## Core built-in data structures

| Structure | Ordered | Mutable | Duplicates | Typical ML use |
|---|---|---|---|---|
| `list` | yes | yes | yes | a batch of samples, a training-loss history |
| `tuple` | yes | no | yes | a fixed shape like `(height, width, channels)` |
| `dict` | yes (3.7+) | yes | keys unique | a config, a `word -> index` vocabulary |
| `set` | no | yes | no | deduplicating tokens/labels, fast membership tests |

```python
losses = []                      # list: grows every epoch
image_shape = (224, 224, 3)      # tuple: shape never changes
vocab = {"the": 0, "cat": 1}     # dict: token -> id
seen_labels = {"cat", "dog"}     # set: unique labels
```

## Comprehensions

The idiomatic way to build a collection from another, and something you'll read
constantly in ML codebases:

```python
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
id_to_word = {idx: word for word, idx in vocab.items()}   # invert a dict
```

## Functions

```python
def normalize(values, low=0.0, high=1.0):
    """Rescale values to [low, high]."""
    vmin, vmax = min(values), max(values)
    span = vmax - vmin
    return [low + (v - vmin) / span * (high - low) for v in values]
```

- Default arguments (`low=0.0`) let callers override only what they need.
- `*args`/`**kwargs` let a function accept a variable number of positional/keyword
  arguments — you'll see this constantly in library code (e.g. `def forward(self, *inputs, **kwargs)`).

## Classes — the shape almost every model/dataset class follows

```python
class Dataset:
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
```

`__len__` and `__getitem__` are "dunder" (double-underscore) methods that hook into
Python's built-in protocols — implementing them is exactly what PyTorch's
`torch.utils.data.Dataset` expects later on.

## Iterators & generators

A generator produces values lazily, one at a time, instead of building a whole list
in memory — essential once datasets are too big to fit in RAM.

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

for b in batch(range(10), 3):
    print(b)   # [0,1,2] [3,4,5] [6,7,8] [9]
```

## Exceptions

```python
try:
    result = 1 / divisor
except ZeroDivisionError:
    result = float("inf")
```

Use exceptions for genuinely exceptional conditions (bad input, missing file) —
not as a substitute for a normal `if`/`else`.
