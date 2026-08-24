# 01 — Concepts: Naive Bayes

## From Bayes' theorem to a classifier

Lesson 006 used Bayes' theorem to compute `P(spam | contains "free")` from
`P(contains "free" | spam)`. A classifier just generalizes this to *many*
features and picks the class with the highest posterior probability:

```
P(class | features) ∝ P(features | class) * P(class)
```

(`∝` because the denominator `P(features)` is the same across all classes,
so you can skip computing it and just compare the numerators.)

## The "naive" assumption

Computing `P(features | class)` directly for many features would need
exponentially much data (every possible feature *combination*, per class —
the curse of dimensionality, Lesson 019, in its rawest form). Naive Bayes
sidesteps this by **assuming all features are conditionally independent
given the class** (Lesson 006):

```
P(x1, x2, ..., xn | class) = P(x1|class) * P(x2|class) * ... * P(xn|class)
```

This is almost always technically false (word occurrences in a real email
aren't truly independent), but the classifier is fast, needs little data,
and works surprisingly well in practice regardless — especially for text.

## Working in log-space (numerical stability, familiar from Lesson 007)

Multiplying many small probabilities underflows to 0 in floating point.
Take logs instead, turning products into sums:

```
log P(class|x) ∝ log P(class) + Σ log P(x_i | class)
```

Then just pick the class with the highest sum — same idea as Lesson 007's
softmax numerical-stability trick, applied here to avoid underflow instead
of overflow.

## Variants, matched to feature type

- **GaussianNB**: assumes each continuous feature is Normally distributed
  within each class (Lesson 007) — parameters are just each class's
  per-feature mean/variance.
- **MultinomialNB**: for count data (e.g. word counts in a document) — the
  standard choice for text classification with bag-of-words/TF-IDF features
  (Lesson 049).
- **BernoulliNB**: for binary features (e.g. "does this word appear at all,
  yes/no") — also common for text, using presence/absence rather than
  counts.

```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_counts, y_train)   # X_train_counts: word-count vectors
```

## Laplace (additive) smoothing

If a word never appeared with a given class in training data,
`P(word | class) = 0` — which would zero out the *entire* product for any
new document containing that word, no matter how much other evidence
supports that class. **Laplace smoothing** adds a small constant (`alpha`,
default 1.0 in sklearn) to every count before computing probabilities,
ensuring no probability is ever exactly zero:

```
P(word | class) = (count(word, class) + alpha) / (total_words(class) + alpha * vocab_size)
```

## Why Naive Bayes is still a strong baseline for text

Despite the false independence assumption, text classification (spam
filtering, sentiment, topic classification) is one of Naive Bayes' best use
cases: text is high-dimensional (large vocabulary) with limited data per
class, exactly the regime where a low-variance, high-bias model with a
strong simplifying assumption tends to outperform more flexible models that
would need more data to avoid overfitting (Lesson 017). It's the standard
first baseline to try before reaching for logistic regression or a
transformer-based classifier (Lesson 048+).
