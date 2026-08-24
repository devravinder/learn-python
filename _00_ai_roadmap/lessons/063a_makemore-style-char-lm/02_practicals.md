# 02 — Practicals: Bigram & MLP Character-Level Language Models

A small names corpus (or substitute any list of ~20+ short words/names you
like):

```python
words = ["emma","olivia","ava","isabella","sophia","charlotte","mia","amelia",
         "harper","evelyn","abigail","emily","ella","elizabeth","camila","luna",
         "sofia","avery","mila","aria"]
```

## Counting-based bigram model (pure Python)

1. Build the vocabulary (`.` as start/end token, plus every unique
   character), and the Laplace-smoothed bigram count table from
   `01_concepts.md`. Normalize into `probs`.

2. Implement `sample(rng)`: starting from `.`, repeatedly sample the next
   character from the current character's probability row, stop at `.`.
   Generate 10 names. Do they look name-*ish* even when nonsensical?

3. Compute the average negative log-likelihood (cross-entropy loss,
   Lesson 063) of the training data under your bigram model, and its
   perplexity.

## Neural net equivalence (pure Python — no torch needed at this scale)

4. Implement the neural bigram model from `01_concepts.md`: a `V x V`
   weight matrix `W`, softmax, and a training loop (plain gradient
   descent, no smoothing/regularization at all). Train for 200 epochs on
   the same bigram pairs from your corpus.

5. Compare one row of the trained `W` (softmaxed) to the corresponding row
   of the **unsmoothed** (no `+1`) raw-frequency count table (rebuild
   counts without the Laplace `+1` for this comparison). How close are
   they? This is the key result of the lesson — take the comparison
   seriously rather than skimming past it.

6. Compare the neural model's final training loss to the *smoothed*
   counting model's loss from Q3. Which is lower, and why (relate to
   maximum likelihood vs. Laplace smoothing's deliberate bias away from
   the raw MLE solution)?

## MLP with real context (PyTorch)

7. Build a training set of `(context, target)` pairs using a context
   length of 3 characters (pad the start of each word with `.` characters
   so the first few predictions still have a full-length context).
   Implement and train the `CharMLP` from `01_concepts.md`
   (`embed_dim=8, hidden_dim=32`) on this data with cross-entropy loss.

8. Generate 10 new names by sampling from the trained MLP (feed the last 3
   generated/context characters in at each step, exactly like the bigram
   sampler but with a sliding 3-character window instead of 1). Compare
   the perplexity of the MLP model to the bigram model's — does more
   context measurably help?
