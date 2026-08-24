# 02 — Practicals: Scaling Laws & Compute Budgeting

## Compute budgeting arithmetic (pure Python)

1. Using `C ≈ 6*N*D`, compute the approximate training compute (in FLOPs)
   for: (a) a 1M-parameter model trained on 20M tokens, (b) a 10M-parameter
   model trained on 200M tokens (both following the Chinchilla ~20
   tokens/parameter ratio). How many times more compute does (b) need than
   (a)?

2. Given a GPU that achieves (hypothetically) `50 * 10^12` FLOPs/second
   sustained during training, estimate the wall-clock training time (in
   minutes) for scenario (a) and (b) from Q1.

3. For a **fixed** compute budget of `10^17` FLOPs, and following the
   Chinchilla `D ≈ 20*N` ratio, solve for the compute-optimal `N` and `D`
   (two equations: `C = 6*N*D` and `D = 20*N`, two unknowns). Report both.

4. Suppose you have a fixed dataset of 5 million tokens (common at hobby
   project scale) and want to follow the Chinchilla ratio. What model
   size `N` would be "compute-optimal" for that amount of data? Does that
   suggest a model larger or smaller than you might naively have guessed
   for "a real LLM"?

## Reflection

5. Given Q4's result, explain why building a very large model (e.g. tens
   of millions of parameters) but only training it on a small hobby-scale
   corpus (a few MB of text) is likely to waste most of that model's
   capacity — connect this to Lesson 017's bias-variance framing (which
   regime does an oversized model on too little data fall into?).

6. Project 013 asks you to choose your own model size and training corpus
   size. Using this lesson's ratio (~20 tokens per parameter) as a
   starting point, and whatever corpus size you're realistically planning
   to use, calculate a suggested parameter count for your own model
   *before* you start Project 013 — this is exactly the budgeting exercise
   a real training run would do first.
