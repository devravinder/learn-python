# 02 — Practicals: Naive Bayes

## Pen-and-paper (extends Lesson 006 Q3)

1. Two features now: an email contains "free" and/or "winner". Given:
   `P(spam)=0.4`, `P(free|spam)=0.9`, `P(free|not spam)=0.05`,
   `P(winner|spam)=0.7`, `P(winner|not spam)=0.02`. Using the naive
   independence assumption, compute (up to the shared denominator)
   `P(spam | free, winner)` vs `P(not spam | free, winner)` and decide the
   classification.

## Code

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

texts = [
    "win a free prize now", "meeting scheduled for tomorrow",
    "free money click here", "project deadline next week",
    "you are a winner claim now", "lunch with the team today",
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = spam
```

2. Vectorize `texts` with `CountVectorizer`, fit `MultinomialNB`, and predict
   the class for a new message: `"free winner claim your prize"`.

3. Print `model.feature_log_prob_` for a few words (use
   `vectorizer.get_feature_names_out()` to find their indices) — confirm
   words like "free"/"winner" have higher log-probability under the spam
   class than the non-spam class.

4. Implement Naive Bayes classification from scratch for the toy dataset
   above: compute `P(class)` priors, per-word `P(word|class)` with Laplace
   smoothing (`alpha=1`), and classify the same new message by comparing
   summed log-probabilities. Confirm it agrees with `sklearn`'s prediction.

5. Compare `MultinomialNB` and `GaussianNB` on a purely continuous dataset
   (`sklearn.datasets.make_classification`) — `GaussianNB` should work
   reasonably (matches its assumption), while `MultinomialNB` requires
   non-negative features and isn't really designed for this data
   (demonstrate what happens if you feed it negative-valued features:
   it should error or behave poorly).

6. On a larger real-ish text dataset (e.g.
   `sklearn.datasets.fetch_20newsgroups` with 2-3 categories, if you have
   internet access; otherwise generate a bigger synthetic version of the
   toy spam data above with ~50 examples), compare `MultinomialNB` to
   `LogisticRegression` (Lesson 023) on TF-IDF features
   (`TfidfVectorizer`). Report accuracy for both — this is a preview of
   Project 003.
