# 02 — Practicals: Inferential Statistics & Hypothesis Testing

1. Simulate a population: `population = np.random.default_rng(0).normal(50, 10, 1_000_000)`.
   Draw 1,000 different random samples of size 30 from it, compute each
   sample's mean, and plot a histogram of those 1,000 means. Compare its
   shape and spread to the population's own distribution — this is the CLT,
   directly visualized.

2. From a *single* sample of size 30 (`rng.choice(population, 30)`), compute a
   95% confidence interval for the mean using the formula in `01_concepts.md`.
   Does it contain the true population mean (50)? Repeat with 100 different
   samples of size 30 and count what fraction of the resulting intervals
   contain 50 — it should land close to 95%.

3. **Model comparison via t-test**: two models' per-example errors on the same
   50 test cases:
   ```python
   rng = np.random.default_rng(1)
   errors_a = rng.normal(0.20, 0.05, 50)
   errors_b = rng.normal(0.17, 0.05, 50)
   ```
   Use `scipy.stats.ttest_rel` (paired, since it's the same test cases) to
   test whether model B's mean error is significantly lower than model A's.
   Report the p-value and your conclusion at `α = 0.05`.

4. **Type I error rate, empirically**: generate 1,000 pairs of samples from
   the *same* distribution (so `H0` is actually true by construction), run a
   two-sample t-test on each pair, and count what fraction produce
   `p < 0.05`. It should land close to 5% — confirming `α` really is a
   false-positive rate, not an arbitrary threshold.

5. **Chi-squared test**: given this contingency table of (UI variant) x
   (clicked or not):
   ```python
   import numpy as np
   table = np.array([
       [45, 55],   # variant A: 45 clicked, 55 did not
       [60, 40],   # variant B: 60 clicked, 40 did not
   ])
   ```
   Use `scipy.stats.chi2_contingency` to test whether click behavior is
   associated with variant. Report the p-value and conclusion.

6. Explain in your own words why "the test wasn't statistically significant"
   is not the same claim as "there is no real difference."
