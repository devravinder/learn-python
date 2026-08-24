# 01 — Concepts: The Language Modeling Objective

## The task, stated precisely

Given a sequence of tokens `x_1, x_2, ..., x_t`, predict a probability
distribution over what `x_(t+1)` will be. That's the **entire** training
objective for GPT-style models — nothing about "understanding," "reasoning,"
or "knowledge" is explicitly programmed in; all of that emerges as a side
effect of getting extremely good at this one task across a massive and
varied training corpus.

## The loss: cross-entropy, applied per token (Lesson 016, revisited)

At each position, the model outputs a probability distribution over the
entire vocabulary (via softmax over the final linear layer's logits,
Lesson 060). The loss is cross-entropy (Lesson 016) between that
distribution and the actual next token (a one-hot target):

```
Loss = -(1/T) * Σ_t log P(x_(t+1) | x_1, ..., x_t)
```

Averaged over every position in every sequence in a batch. In PyTorch:

```python
import torch.nn.functional as F

logits = model(input_ids)          # (batch, seq_len, vocab_size)
targets = input_ids[:, 1:]          # next-token targets: shift by one position
logits = logits[:, :-1, :]           # align: predict position t+1 from position t

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
)
```

**The shift-by-one is the entire trick**: the target for the model's
prediction at position `t` is simply the token at position `t+1` in the
same sequence — no separate labels needed at all, since the text itself
already contains "the correct answer" for next-token prediction. This is
why language modeling is called **self-supervised**: labels come free from
raw text, unlike Module 4's labeled datasets.

## Why this single objective produces such broad capability

To predict the next token well across a massive, diverse training corpus,
the model is implicitly forced to learn grammar (to predict grammatically
valid continuations), facts (to predict "Paris" after "The capital of
France is"), reasoning patterns (to continue a logical argument
correctly), and style (to match the register of the text so far) — not
because any of these were separately labeled, but because **getting next-
token prediction right, across enough varied real text, requires all of
them**. This is the central, somewhat surprising empirical finding behind
the entire modern LLM paradigm.

## Perplexity: cross-entropy loss, made interpretable

```
Perplexity = exp(cross-entropy loss)
```

Interpretation: perplexity is (roughly) "the effective number of equally-
likely choices the model is choosing between, on average, at each
position." A perplexity of 1 means perfect, certain prediction; a
perplexity of `V` (vocabulary size) means the model is no better than
uniform random guessing. Lower perplexity = better language model — this
is the standard way LLM quality gets reported and compared (Lesson 073
covers this fully for evaluation purposes).

```python
import math
perplexity = math.exp(loss.item())
```

## Teacher forcing, again (recalling Lesson 047)

During training, every position's prediction is conditioned on the
**true** preceding tokens (not the model's own possibly-wrong previous
predictions) — exactly Lesson 047's teacher forcing, now applied at LLM
scale. This is why training can process an entire sequence in **one
parallel forward pass** (compute all positions' predictions
simultaneously, since every position's *input* context is already known
ground truth) rather than needing to generate token-by-token during
training — a huge practical efficiency advantage, and part of why
Transformers train so much faster than they generate (generation, Lesson
066, *is* sequential, one token fed back in at a time, unlike training).

## Batching sequences of different lengths

Real training data has documents of varying length. Two common approaches:
- **Padding**: pad shorter sequences to a common length with a `[PAD]`
  token (Lesson 055), mask its loss contribution (don't train on
  predicting padding).
- **Packing/concatenation**: concatenate many documents together
  (separated by an end-of-text token, Lesson 062) into one long stream,
  then chop into fixed-length chunks — no padding waste at all, the
  standard approach for large-scale pretraining, and what Lesson 064's
  data loading will use.

## What "training an LLM" actually means, mechanically

Given everything above: an LLM training loop is Lesson 040's exact
training loop (`forward -> loss -> backward -> optimizer.step()`),
applied to a Transformer (Lesson 060), with cross-entropy loss on
shifted-by-one targets, at a much larger scale (more data, more parameters,
more compute) than anything else in this curriculum. There's no additional
conceptual machinery beyond what you already know — Lessons 064-066 build
this up concretely, one piece at a time, following Karpathy's "Let's build
GPT" progression.
