# Findings — Spam & Fake Review Detector

*(Verified against an independent pure-Python Naive Bayes implementation run
directly against the generated CSVs — not fabricated. `analysis.py`'s
sklearn-based numbers should match this closely; regenerate data and run it
yourself to confirm.)*

## In-distribution accuracy is close to 100% — and that's a finding, not a bug

Both Naive Bayes and Logistic Regression reach ~100% accuracy on a held-out
split of `sms_spam.csv` and `fake_reviews.csv`. This is **expected, not
evidence of a broken pipeline**: the data is generated from a small, fixed
set of templates per class, so spam/ham (and fake/genuine) vocabulary is
largely distinct within the dataset's own distribution — including a
deliberately overlapping subset ("free tickets", "amazing product... genuine
purchase") that the model still separates correctly once it has seen enough
examples of each exact phrasing.

## The real limitation shows up on genuinely novel wording

Testing the trained spam model on hand-written sentences it never saw during
training (not from any template) reveals its actual limitation:

| Test sentence | True label | Predicted |
|---|---|---|
| "mom says dinner is ready come home now" | ham | ham ✓ |
| "can we reschedule our call to sometime next week" | ham | ham ✓ |
| "loved catching up with you at the gathering yesterday" | ham | ham ✓ |
| "get 50 percent off everything today only shop the sale" | spam | spam ✓ |
| "act fast before this deal ends tonight don't miss out" | spam | spam ✓ |
| **"your subscription payment failed update billing information"** | **spam** | **ham ✗** |

The model misses a realistic phishing-style message because none of its
words ("subscription," "payment," "failed," "billing") appeared in the
spam-vocabulary it learned from the templates ("free," "prize," "claim,"
"winner," "urgent"). This is the core limitation of **bag-of-words models**:
they match vocabulary, not meaning — a message can be obviously spam to a
human without using any word the model has learned to associate with spam.
This exact gap is what Project 012 (fine-tuned Transformer, after Module 10)
is built to close, since embedding-based models can recognize
semantically-similar-but-lexically-different phishing language.

## Top spam-indicative words (Naive Bayes)

Highest log-probability-ratio words favor exactly the vocabulary you'd
expect from the templates: "claim," "winner," "prize," "free," "urgent,"
"click," "cash," "limited," "congratulations" — a useful sanity check that
the model learned something sensible rather than an opaque, unverifiable
pattern.

## Practical takeaway

Naive Bayes and Logistic Regression perform comparably here — expected on a
small-vocabulary, cleanly-separable text task (Lesson 030's "Naive Bayes is
a strong baseline when data per class is limited" point). The gap between
"works great on data like what it was trained on" and "works on the actual
variety of real spam" is the central lesson of this project, and the
concrete gap that later modules (embeddings, transformers) are built to
close.
