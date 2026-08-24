# 01 — Concepts: Bigram & MLP Character-Level Language Models

## The simplest possible language model: bigram counting

Given a corpus (e.g. a list of names, one per line), count how often each
character follows each other character, then normalize into
probabilities:

```python
from collections import defaultdict

counts = [[1]*V for _ in range(V)]   # start at 1: Laplace smoothing (Lesson 006)
for word in words:
    s = "." + word + "."             # '.' marks start/end
    for c1, c2 in zip(s, s[1:]):
        counts[stoi[c1]][stoi[c2]] += 1

probs = [[c / sum(row) for c in row] for row in counts]
```

Sampling a new "word": start at `.`, repeatedly sample the next character
from `probs[current]`, stop when `.` is sampled again. **This is a complete,
if very limited, language model** — it has learned real statistical
structure of the training data (e.g. which letters commonly start names,
which letter pairs are common) purely by counting.

## The exact same model, as a neural network

Represent each character as a **one-hot vector**, and replace the lookup
table with a single linear layer + softmax:

```python
def neural_bigram_step(x_idx, W):
    logits = W[x_idx]          # equivalent to one_hot(x) @ W, but indexing is simpler
    probs = softmax(logits)
    return probs
```

Train `W` via ordinary gradient descent (Lesson 015) on cross-entropy loss
(Lesson 016/063) between predicted and actual next character — exactly
Lesson 038's training loop, applied to text instead of XOR.

**The remarkable, verifiable result**: after training with no
regularization, this neural network's learned probabilities converge to
almost **exactly** the *raw, unsmoothed* frequency counts (`count / total`,
no Laplace `+1`) — because minimizing cross-entropy loss via gradient
descent **is** maximum likelihood estimation, and the raw frequency table
*is* the maximum likelihood solution for this exact model class. Counting
and gradient descent are two different *algorithms* arriving at the *same
answer*, for this simple case — a genuinely clarifying result once you've
verified it yourself in the practicals.

## Where smoothing fits into the neural picture

The counting model's `+1` Laplace smoothing (Lesson 006) has a direct
neural analogue: **L2 weight regularization** (Lesson 022/042,
`weight_decay` in the optimizer) pulls weights toward zero, which pulls
softmax outputs toward *uniform* — the same "don't be overconfident about
rarely-seen patterns" effect smoothing provides, achieved via a completely
different mechanism (a training-time penalty instead of an
arithmetic adjustment to counts).

## Why bigrams aren't enough, and what MLP fixes

A bigram model's prediction depends on **exactly one** previous character —
it has no way to use more context, however useful that context might be.
The `makemore` MLP (following Bengio et al. 2003, a direct historical
ancestor of every neural language model since) fixes this:

```
context: last `n` characters (e.g. n=3)
  -> look up each character's embedding (Lesson 057's nn.Embedding)
  -> concatenate the n embedding vectors
  -> hidden layer (Linear + tanh, Lesson 035)
  -> output layer (Linear + softmax over vocabulary)
  -> predicted distribution for the NEXT character
```

```python
import torch.nn as nn

class CharMLP(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_len * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context_ids):
        # context_ids: (batch, context_len)
        emb = self.embed(context_ids).flatten(1)   # (batch, context_len * embed_dim)
        hidden = torch.tanh(self.fc1(emb))
        return self.fc2(hidden)                      # (batch, vocab_size) logits
```

This is now using **actual context** (multiple previous characters) rather
than just one — a direct, meaningful step up in modeling power, and
structurally the same "embed -> hidden layer(s) -> output" shape every
neural network in this curriculum has used since Lesson 035, just applied
to a sliding window of character context.

## The direct line from here to Lesson 064

The MLP above has a **fixed** context window (`n` characters) — extending
it to see arbitrarily far back requires either a much larger fixed window
(wasteful, still capped) or a fundamentally different mechanism. That
mechanism is exactly attention (Lesson 058): instead of a fixed-size
concatenated context, let the model attend over the *entire* preceding
sequence, with a learned notion of relevance rather than a hard cutoff.
Lesson 064 replaces this MLP's fixed context window with exactly that —
the first real GPT block.
