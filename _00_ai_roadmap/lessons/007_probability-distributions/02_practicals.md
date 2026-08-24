# 02 — Practicals: Probability Distributions

1. Using `np.random.default_rng(0).binomial(n=20, p=0.5, size=10000)`, plot a
   histogram of the results. What shape does it approximate as `n` grows large
   (hint: Central Limit Theorem)?

2. Draw 10,000 samples from a standard normal distribution
   (`rng.normal(0, 1, 10000)`). Compute the fraction of samples within 1, 2,
   and 3 standard deviations of the mean, and compare to the well-known
   "68-95-99.7" rule.

3. Given raw exam scores `[55, 60, 62, 70, 72, 75, 80, 85, 90, 95]` with mean
   `μ` and standard deviation `σ` computed from the data, convert every score
   to a z-score. Which score is the most unusually high or low relative to the
   group?

4. Implement `softmax(z)` from scratch using only NumPy (no `scipy.special`).
   Test it on `z = np.array([2.0, 1.0, 0.1])` and confirm the outputs sum to 1.

5. **Numerical stability**: try your `softmax` on `z = np.array([1000, 1001, 1002])`.
   What happens (look at `exp(1000)`)? Fix `softmax` by subtracting `max(z)`
   from `z` before exponentiating, and explain why this doesn't change the
   mathematical result but does fix the numerics.

6. Simulate rolling two fair dice 100,000 times and plot a histogram of the
   sum. Compare its shape to a Normal distribution's — this is the Central
   Limit Theorem in action, from summing just two uniform variables.
