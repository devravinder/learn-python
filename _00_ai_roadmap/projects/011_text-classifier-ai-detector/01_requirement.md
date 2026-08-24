# 01 — Requirement: Human vs AI-Generated Text Detector

## The brief

> "We're getting a mix of human-written and AI-generated submissions and
> need a first-pass filter to flag likely-AI-generated text for review."

## What to produce

1. **EDA**: look at average sentence length, vocabulary diversity
   (unique words / total words), and frequency of a few hallmark phrases
   ("in conclusion," "furthermore," "it is important to note") between the
   two classes. Do these simple statistics alone already separate the
   classes somewhat?

2. **Baseline pipeline (TF-IDF)**: reuse Project 003's approach — TF-IDF
   features (Lesson 056) + Logistic Regression and Naive Bayes. Report
   `classification_report` (Lesson 024) for both.

3. **Dense embedding pipeline**: build a dense document representation
   instead of sparse TF-IDF — either (a) train your own word embeddings on
   this corpus (Lesson 057's skip-gram, or `TfidfVectorizer` +
   `TruncatedSVD` for an SVD-factorized dense embedding, conceptually
   related to GloVe) and average word vectors per document, or (b) use
   pretrained embeddings if you have internet access. Train the same two
   classifiers on these dense features.

4. **Compare**: TF-IDF vs dense embeddings — which representation gives
   better classification accuracy here? Given Lesson 056/057's discussion
   of what each representation captures (exact vocabulary overlap vs
   semantic similarity), does the result make sense for *this specific
   task* (which may hinge more on surface-level phrasing patterns than
   deep semantic meaning)?

5. **Error analysis**: print 5 misclassified examples from your best
   model. Do the errors look like genuinely ambiguous writing style, or a
   pattern your feature representation can't capture at all?

6. **Generalization stress test**: hand-write 3 new sentences yourself (not
   from any template) — a couple of clearly-human ones and a
   deliberately "AI-sounding" one — and test your best model on them.
   Does it generalize past the exact templates it was trained on, or does
   it clearly overfit to the synthetic patterns?

## Constraints

- Classical NLP + `sklearn` only (dense embeddings via your own training or
  SVD factorization, not a pretrained Transformer) — that upgrade is
  Project 012.
- Don't peek at `02_solutions/` before you have both pipelines compared
  yourself.
