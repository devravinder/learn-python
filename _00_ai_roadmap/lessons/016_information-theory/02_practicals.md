# 02 — Practicals: Entropy, Cross-Entropy & KL Divergence

1. Compute the entropy of a fair coin (`[0.5, 0.5]`) and a biased coin
   (`[0.9, 0.1]`). Which has higher entropy? Does that match the intuition
   that the biased coin's outcome is easier to predict?

2. Compute the entropy of a fair six-sided die (`[1/6]*6`) vs a loaded die
   `[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]`. Confirm the uniform distribution has the
   *maximum possible* entropy for 6 outcomes (`log(6)`).

3. A classifier predicts `q = [0.7, 0.2, 0.1]` for 3 classes; the true label
   is class 0 (`p = [1, 0, 0]`). Compute the cross-entropy loss. Now suppose
   it had predicted `q = [0.1, 0.2, 0.7]` (confidently wrong) — compute the
   loss again and compare.

4. Implement `kl_divergence(p, q)` from scratch. Compute
   `KL(p||q)` and `KL(q||p)` for `p = [0.5, 0.5]`, `q = [0.9, 0.1]`, and
   confirm they're different (asymmetry).

5. Verify the identity `cross_entropy(p, q) == entropy(p) + kl_divergence(p, q)`
   numerically for a few random probability distributions (normalize random
   positive vectors to sum to 1).

6. **Loss-vs-confidence curve**: for a true label with `p = [1, 0]` (binary),
   plot cross-entropy loss as the predicted probability of the correct class
   ranges from 0.01 to 0.99. Confirm the loss blows up as the prediction
   approaches 0 for the correct class — explain why this is a *harsher*
   penalty for confident wrong predictions than, say, mean squared error
   would give.
