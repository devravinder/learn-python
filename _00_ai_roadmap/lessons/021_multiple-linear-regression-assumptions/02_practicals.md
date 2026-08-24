# 02 — Practicals: Multiple Linear Regression & Assumptions

1. Create a dataset with a deliberately redundant feature:
   ```python
   rng = np.random.default_rng(0)
   x1 = rng.normal(0, 1, 200)
   x2 = x1 * 0.98 + rng.normal(0, 0.05, 200)   # nearly identical to x1
   x3 = rng.normal(0, 1, 200)
   y = 3*x1 + 2*x3 + rng.normal(0, 1, 200)
   X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
   ```
   Compute VIF for each feature. Which one(s) get flagged?

2. Fit a linear regression on `X, y`. Then refit after dropping `x2`. Compare
   the coefficient on `x1` between the two fits — does it change
   substantially? What does that tell you about the reliability of
   individual coefficients under multicollinearity?

3. Using Lesson 020's residuals, run a Breusch-Pagan test
   (`statsmodels`). Report the p-value and whether you'd conclude
   heteroscedasticity is present.

4. Create a categorical feature `region` with 3 categories, one-hot encode it
   with `drop_first=True`, and fit a regression including it alongside a
   numeric feature. Explain what the reference category's role is in
   interpreting the fitted coefficients.

5. Build a small dataset where an interaction genuinely matters (e.g. the
   effect of a discount on sales differs by whether it's a holiday). Fit a
   model without an interaction term and one with `discount * is_holiday`
   added as a feature. Compare R² between them.
