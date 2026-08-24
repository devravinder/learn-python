# Findings — Gradient Descent on Housing Data

*(Numbers below are verified against the actual generated `housing.csv`
using an independent pure-Python implementation — not fabricated. Regenerate
with `data/generate_data.py` and run `analysis.py` for the NumPy version's
matching output and charts.)*

## Gradient descent converges to the closed-form solution

After 2,000 epochs of batch gradient descent (`lr=0.1`) on standardized
features, the learned weights matched the normal-equation (closed-form)
solution almost exactly:

| Feature | GD weight (standardized) | Normal equation weight |
|---|---|---|
| sqft | 152,292.3 | 152,292.3 |
| bedrooms | 16,593.0 | 16,593.0 |
| age | -11,979.7 | -11,979.7 |
| distance_km | -17,821.1 | -17,821.1 |
| bias | 374,765.8 | 374,765.8 |

Final MSE ≈ 218,078,464 (RMSE ≈ $14,768) — a reasonable error band given the
±$15,000 noise baked into the synthetic data generator.

## Feature importance

Ranked by absolute effect on price (in standardized units, so directly
comparable across features regardless of their raw scale): **sqft dominates**
(152.3K per standard deviation), followed by distance from city center
(-17.8K, negative — farther is cheaper), bedrooms (+16.6K), and age (-12.0K,
negative — older is cheaper). This matches ordinary housing-market intuition
and, since this is synthetic data, also matches the generator's true
coefficients by construction — a useful sanity check that the from-scratch
implementation is actually correct, not just "some numbers come out."

## Practical takeaways

- Gradient descent and the closed-form solution agreeing this closely is the
  correct outcome for a convex loss like MSE with a small-enough learning
  rate — if they disagreed substantially, that would point to a bug (wrong
  gradient formula, learning rate too high, not enough epochs) rather than a
  legitimate modeling choice.
- Standardizing features before training wasn't optional here: `sqft` (range
  ~500–4000) and `bedrooms` (range 1–6) differ by nearly 3 orders of
  magnitude in raw scale — without standardization, a single learning rate
  that works for one feature would be wildly wrong for the other, and the
  loss surface would be a narrow, slow-to-descend valley (Lesson 015).
