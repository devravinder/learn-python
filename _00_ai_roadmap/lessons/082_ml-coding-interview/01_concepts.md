# 01 — Concepts: ML/DS Coding Interview Question Categories

## Different from a LeetCode interview, in a specific way

A general SWE coding interview tests data structures/algorithms
(trees, graphs, DP). An ML/DS coding interview overlaps with that but
adds a distinct category: **implement a classic ML building block from
memory, correctly, including the numerically-tricky parts** — not
because anyone hand-rolls k-means in production (you'd import sklearn),
but because it's the fastest way to check you actually understand the
algorithm rather than having only ever called `.fit()` on it. This
mirrors exactly what this entire curriculum has been doing since Lesson
038 (hand-rolling backprop, k-means, Naive Bayes, attention) — you've
already practiced this skill throughout, this lesson just names it as an
interview category and adds timing pressure.

## The recurring categories

**1. Classic algorithm from scratch.** KNN, k-means, linear/logistic
regression via gradient descent, a decision tree stump, Naive Bayes.
Interviewers are checking: do you know the actual update rule/distance
metric, not just the sklearn import.

**2. Numerically-aware implementations.** Softmax and log-sum-exp done
naively overflow on large inputs — a *very* common question specifically
because it distinguishes "knows the formula" from "knows why the formula
breaks in floating point and how to fix it" (subtract the max before
exponentiating). Sigmoid has the same issue for very negative inputs.

**3. Data manipulation without a library.** "Group these records by key
and compute the mean" (pandas' `groupby` reimplemented with dicts), "split
this list into train/test with a fixed seed" (`sklearn.model_selection
.train_test_split` reimplemented) — testing whether you understand what
the library call is *doing*, since in some interview formats you're
explicitly barred from importing pandas/sklearn.

**4. Evaluation metrics from scratch.** Precision, recall, F1, confusion
matrix, given raw predictions and labels as plain lists — directly
reusing Lesson 025's definitions, just without a library computing them
for you.

**5. Probability/stats brainteasers.** Monty Hall, expected value
puzzles, birthday paradox — often solved two ways in an interview:
reason it analytically, *then* verify with a quick simulation, which
also demonstrates comfort writing a Monte Carlo check (a real skill, not
just trivia recall).

**6. Vectorization awareness, even without numpy in the room.** Even
when asked to write pure-Python, a strong answer identifies which parts
*would* vectorize (the distance computation in KNN, the dot products in
logistic regression) and says so — showing you'd write the numpy/PyTorch
version in production, not that you don't know it exists.

## How to run the clock

- **Restate the algorithm in one sentence before coding** — same
  discipline as step 2 of the system-design framework (Lesson 081):
  naming the approach before writing code catches misunderstandings
  before they're embedded in code.
- **Write the naive/correct version first, optimize only if time
  remains** — a working O(n²) KNN beats a half-finished attempt at a
  k-d tree.
- **State the edge cases out loud even if you don't fully handle them**
  ("k-means can produce an empty cluster if a centroid never gets
  assigned a point — I'd re-initialize it randomly") — this is often
  worth as much as handling it in code, since it proves you know the
  failure mode exists.
- **Test with a tiny hand-checkable example**, the same verification-
  first habit this entire curriculum has modeled — pick inputs where you
  can compute the expected answer in your head, and check your code
  against it before declaring done.
