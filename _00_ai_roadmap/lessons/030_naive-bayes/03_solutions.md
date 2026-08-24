# 03 — Solutions: Naive Bayes

## 1. Two-feature Bayes by hand

```
P(spam) * P(free|spam) * P(winner|spam) = 0.4 * 0.9 * 0.7 = 0.252
P(not spam) * P(free|not) * P(winner|not) = 0.6 * 0.05 * 0.02 = 0.0006
```

`0.252 >> 0.0006`, so normalizing (dividing both by their sum ≈ 0.2526):
`P(spam|free,winner) ≈ 0.9976`, `P(not spam|...) ≈ 0.0024`. Classify as
**spam** with very high confidence — two independently-suspicious words
compound multiplicatively under the naive assumption.

## 2–3. sklearn Naive Bayes on toy text

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "win a free prize now", "meeting scheduled for tomorrow",
    "free money click here", "project deadline next week",
    "you are a winner claim now", "lunch with the team today",
]
labels = [1, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB().fit(X, labels)

new_text = vectorizer.transform(["free winner claim your prize"])
print(model.predict(new_text))   # [1] -> spam

vocab = vectorizer.get_feature_names_out()
free_idx = list(vocab).index("free")
print("log P(free|not spam), log P(free|spam):", model.feature_log_prob_[:, free_idx])
```

The log-probability for "free" (and similar words) should be visibly higher
under class 1 (spam) than class 0.

## 4. From-scratch Naive Bayes

```python
import numpy as np
from collections import Counter

def tokenize(text):
    return text.lower().split()

docs = [tokenize(t) for t in texts]
vocab = sorted(set(w for d in docs for w in d))

def train_naive_bayes(docs, labels, vocab, alpha=1.0):
    classes = set(labels)
    priors = {c: sum(1 for l in labels if l == c) / len(labels) for c in classes}
    word_probs = {}
    for c in classes:
        class_docs = [d for d, l in zip(docs, labels) if l == c]
        word_counts = Counter(w for d in class_docs for w in d)
        total_words = sum(word_counts.values())
        word_probs[c] = {
            w: (word_counts.get(w, 0) + alpha) / (total_words + alpha * len(vocab))
            for w in vocab
        }
    return priors, word_probs

priors, word_probs = train_naive_bayes(docs, labels, vocab)

def classify(text, priors, word_probs, vocab):
    words = tokenize(text)
    scores = {}
    for c in priors:
        score = np.log(priors[c])
        for w in words:
            if w in vocab:
                score += np.log(word_probs[c][w])
        scores[c] = score
    return max(scores, key=scores.get), scores

pred, scores = classify("free winner claim your prize", priors, word_probs, vocab)
print(pred, scores)   # should predict 1 (spam), matching sklearn
```

## 5. GaussianNB vs MultinomialNB on continuous data

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB

X, y = make_classification(n_samples=300, n_features=5, random_state=0)
gnb = GaussianNB().fit(X, y)
print("GaussianNB accuracy:", gnb.score(X, y))

try:
    mnb = MultinomialNB().fit(X, y)
except ValueError as e:
    print("MultinomialNB error on negative features:", e)
```

`GaussianNB` works fine since it's designed for continuous, potentially
negative features. `MultinomialNB` raises an error on negative values (it
assumes non-negative counts/frequencies) — a direct illustration that the
Naive Bayes *variant* must match your feature type, not just "Naive Bayes"
generically.

## 6. Naive Bayes vs Logistic Regression on text (preview of Project 003)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# using a larger synthetic spam/ham set for a meaningful split
more_texts = texts * 10  # toy repetition for a runnable example; Project 003 uses a real-sized set
more_labels = labels * 10

Xtr, Xte, ytr, yte = train_test_split(more_texts, more_labels, test_size=0.3, random_state=0)
tfidf = TfidfVectorizer()
Xtr_vec, Xte_vec = tfidf.fit_transform(Xtr), tfidf.transform(Xte)

nb = MultinomialNB().fit(Xtr_vec, ytr)
lr = LogisticRegression().fit(Xtr_vec, ytr)

print("NaiveBayes:", nb.score(Xte_vec, yte))
print("LogisticRegression:", lr.score(Xte_vec, yte))
```

On genuinely larger, more varied text data, the two often perform
comparably, with Naive Bayes typically training faster and Logistic
Regression sometimes edging ahead with enough data — exactly the tradeoff
Project 003 asks you to explore for real.
