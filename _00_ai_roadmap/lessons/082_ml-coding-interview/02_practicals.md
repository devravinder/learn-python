# 02 — Practicals: ML/DS Coding Interview Prep

Pure Python only for all of these (no numpy/pandas/sklearn) — the
constraint an interview whiteboard/shared-doc format usually imposes.
Time-box each to 20 minutes before checking `03_solutions.md`.

1. **Numerically stable softmax.** Implement `softmax(x: list[float]) ->
   list[float]`. Test it on `[1, 2, 3]` (should match the textbook
   formula) and on `[1000, 1001, 1002]` — a naive `exp(x)` implementation
   should overflow/error on the second input; yours shouldn't.

2. **K-means from scratch.** Implement Lloyd's algorithm for 1D or 2D
   points: random init, assign-to-nearest-centroid, recompute centroids,
   repeat until convergence or a max iteration count. Handle the empty-
   cluster edge case explicitly (state your chosen fix even if you don't
   fully implement it).

3. **KNN classifier from scratch.** Implement `predict(train_X, train_y,
   query, k)` using Euclidean distance and majority vote. Test on a tiny
   hand-checkable dataset (e.g. 6 points in 2D, 2 classes) where you can
   verify the nearest neighbors by eye.

4. **Logistic regression via gradient descent, from scratch.** Implement
   the sigmoid, the gradient of binary cross-entropy w.r.t. weights, and
   a training loop. Verify it converges to near-perfect classification on
   a small linearly separable synthetic dataset.

5. **Precision/recall/F1 from scratch.** Given two plain lists
   (`y_true`, `y_pred`) of 0/1 values, compute the confusion matrix
   components and derive precision, recall, and F1 — no library calls.

6. **`train_test_split` from scratch.** Given a list of `n` items and a
   `test_size` fraction and a `seed`, return train/test index lists.
   Confirm the same seed always produces the same split, and that
   train+test sizes sum to `n` for a few different `test_size` values.

7. **Monty Hall simulation.** Reason analytically first (switching wins
   2/3 of the time — write your reasoning in a comment), then write a
   Monte Carlo simulation over 10,000+ trials and confirm the simulated
   win rate for "always switch" matches your analytical answer.

8. **`groupby`-mean from scratch.** Given a list of `(key, value)` pairs,
   compute the mean value per key using only a dict — no pandas.
