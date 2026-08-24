# 01 — Requirement: Spam & Fake Review Detector

## The brief

Two related but separate text classification tasks, using the same toolkit:

1. **SMS spam detection** (`sms_spam.csv`): classify messages as spam or ham
   (legitimate).
2. **Fake review detection** (`fake_reviews.csv`): classify product reviews
   as fake (likely bot/incentivized) or genuine.

## What to produce

1. **Text preprocessing + vectorization**: build a pipeline using
   `CountVectorizer` or `TfidfVectorizer` (Lesson 030) to turn raw text into
   numeric features. Justify your choice of vectorizer in a comment.

2. **Two models, compared**: train both `MultinomialNB` and
   `LogisticRegression` on the same TF-IDF features for the spam task.
   Report `classification_report` (Lesson 024) for both. Which performs
   better, and speculate why given what you know about each model's
   assumptions.

3. **Threshold tuning**: spam detection has an asymmetric cost (Lesson 018 —
   a false positive means a real message gets buried in spam). Tune the
   decision threshold on `predict_proba` to prioritize precision over the
   default 0.5, and report the tradeoff.

4. **Error analysis**: print 5 misclassified messages for your best spam
   model. Do the errors make sense (ambiguous wording, unusual real
   messages that look spammy, etc.) or do they reveal a bug?

5. **Generalize to fake reviews**: apply the *same* pipeline (vectorizer +
   both models) to `fake_reviews.csv`. Report results. Is this task
   easier or harder than spam detection based on your metrics — and can you
   tell from an error analysis why?

6. **Feature inspection**: for the Naive Bayes spam model, print the top 15
   words most indicative of spam (highest `feature_log_prob_` difference
   between classes) — do they match your intuition about what spam sounds
   like?

## Constraints

- Use only classical NLP + `sklearn` — no transformers/embeddings yet
  (that upgrade is Project 012, once Module 10 is done).
- Don't peek at `02_solutions/` until you have your own working version.
