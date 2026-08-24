# 01 — Requirement: Gradient Descent From Scratch on Housing Data

## The brief

You're given `housing.csv` (square footage, bedrooms, age, distance to city
center, sale price for 500 houses). Build a linear model
`price = w1*sqft + w2*bedrooms + w3*age + w4*distance + b` that predicts
price from the other four columns — **without using `sklearn.linear_model`**.
The point of this project is the training loop itself, not the answer.

## What to produce

1. **From-scratch multiple linear regression via batch gradient descent**:
   - Implement the vectorized prediction `ŷ = X @ w + b` and MSE loss using
     NumPy only (Lessons 010–011's matrix operations).
   - Implement the gradient of MSE with respect to `w` and `b`
     (Lessons 013–014) and the update rule (Lesson 015).
   - Standardize features first (Lesson 008) and explain in a comment why
     this dataset in particular needs it (look at the features' raw scales).

2. **Closed-form validation**: solve the same regression via the normal
   equation `w = (X^T X)^-1 X^T y` (Lesson 011's matrix inverse) on the
   standardized features. Confirm your gradient-descent weights converge
   close to the closed-form solution.

3. **Learning rate experiment**: plot loss curves for at least 3 learning
   rates and report which is best, referencing what you learned in Lesson
   015 about too-small/too-large learning rates.

4. **Batch vs mini-batch**: implement both and compare convergence speed
   (number of epochs to reach a fixed loss threshold) on this dataset.

5. **Report**: a short write-up of your final weights (in standardized-feature
   units), which feature has the largest effect on price, and whether that
   matches your intuition about housing prices.

## Constraints

- Use only NumPy for the numerical code — no `sklearn`, no `statsmodels`.
  It's fine (and expected) to use `sklearn` later to double-check your
  answer once you've built it yourself, but the delivered solution should be
  from-scratch.
- Don't peek at `02_solutions/` until you've got your own version working.
