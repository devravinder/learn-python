# 03 — Solutions: LLM Evaluation

*(This code was actually run to produce the numbers below.)*

## 1. Train vs held-out perplexity

```python
import math

def perplexity(probs):
    nll = sum(-math.log(p) for p in probs) / len(probs)
    return math.exp(nll)

print(perplexity([0.9, 0.85, 0.95, 0.9]))     # "train"
print(perplexity([0.4, 0.3, 0.5, 0.35]))       # "held-out"
```

**Actual output: train perplexity ≈ 1.112, held-out perplexity ≈ 2.627** —
more than double, directly reflecting that the model is far less confident
(and presumably less accurate) on data it hasn't been trained on, exactly
the training/validation gap Lesson 017 warned about, now expressed in
perplexity terms instead of raw loss.

## 2. Why cross-tokenizer perplexity comparison is invalid

Perplexity is computed **per token**, and different tokenizers segment the
*same* text into different numbers of tokens (Lesson 062 — a larger
vocabulary tends to produce fewer, "larger" tokens per unit of text). A
model using a 50,000-token vocabulary might report lower perplexity
partly because each of its tokens already encodes more text (more
context "for free" per prediction), not necessarily because it's a better
language model in any deeper sense. A fair comparison requires either the
same tokenizer for both models, or a tokenizer-independent normalization
(e.g. bits-per-character/byte instead of per-token perplexity) — worth
checking explicitly whenever comparing reported perplexity numbers across
different papers/models.

## 3. Multiple-choice scoring

```python
choices_logprobs = [-5.2, -3.1, -8.7, -4.0]
best = choices_logprobs.index(max(choices_logprobs))
print(best)   # 1
```

**Actual output: index `1`** (log-probability `-3.1`, the least negative
i.e. highest probability) — matches taking `argmax` directly, confirming
"pick the choice the model finds least surprising" is mechanically
identical to standard `argmax` classification (Lesson 023-024).

## 4. Benchmark accuracy as held-out accuracy

```python
correct, total = 15, 20
print(100 * correct / total)   # 75.0
```

**Actual output: 75.0%.** This is computed with **exactly** the same
formula as any classification accuracy metric from Lesson 018/024
(`correct predictions / total predictions`) — a benchmark score is
fundamentally a held-out accuracy measurement on a specific, curated
dataset of questions, using the multiple-choice-via-log-probability
scoring mechanism from Q3 in place of a typical classifier's `predict()`
call.

## 5. Why benchmark contamination is harder to detect than overfitting

Ordinary overfitting (Lesson 017) is caught by comparing performance on a
train split vs. a *deliberately held-out* validation split you control —
if they diverge, you know something's wrong. **Benchmark contamination**
is different: the "held-out" benchmark test set may have been scraped
into the training corpus **without your knowledge**, often indirectly (a
web page quoting benchmark questions, a forum discussing them, a
paraphrase in some other document) — there's no clean internal
train/val split to compare against, because the contamination happened
*inside* what you thought was independent training data. Detecting it
requires actively searching the training corpus for near-duplicates of
benchmark content (a real, ongoing practice at labs training on
large-scale scraped web data) rather than anything a standard train/val
split would reveal on its own.

## 6. A practical evaluation checklist

Example, for a fine-tuned text classifier (Project 012-style):
1. Held-out test accuracy/F1 (Lesson 024) on a split never touched during
   training or hyperparameter tuning.
2. A stress test on hand-written, out-of-template examples (exactly
   Project 011/012's approach) — specifically targeting known failure
   modes, not just random new examples.
3. Confusion matrix review — which specific classes/cases does it get
   wrong, and does that pattern make business sense?
4. Manual read-through of 20-30 real predictions (correct and incorrect)
   before deploying — no automatic metric substitutes for actually looking
   at outputs.
5. If comparing to a previous model version: same test set, same metric,
   report both numbers side by side, not just "accuracy improved."
