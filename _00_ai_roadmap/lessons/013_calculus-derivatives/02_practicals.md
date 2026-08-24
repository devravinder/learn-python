# 02 — Practicals: Derivatives & Chain Rule

## Pen-and-paper

1. Compute the derivative of `f(x) = 3x^2 + 2x - 5`.

2. Compute the derivative of `f(x) = e^(2x)` using the chain rule (let
   `u = 2x`).

3. Compute the derivative of `f(x) = (x^2 + 1)^3` using the chain rule.

4. Derive the sigmoid derivative formula
   `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))` starting from
   `sigmoid(x) = 1 / (1 + e^-x)` and the quotient/chain rule. (Hint: write it
   as `(1 + e^-x)^-1` and use the chain rule directly.)

## Verify numerically in code

5. Implement `numerical_derivative(f, x)` (central difference) and use it to
   check your answers to Q1–Q3 at `x = 2`.

6. Implement `sigmoid(x)` and its analytical derivative, then compare against
   `numerical_derivative(sigmoid, x)` for a few values of `x` (e.g. -2, 0, 1,
   5). Confirm they match closely.

7. Implement a tiny gradient-check utility: given a function `f` and a
   hand-written derivative function `f_prime`, sample 5 random points and
   assert `abs(f_prime(x) - numerical_derivative(f, x)) < 1e-4` for all of
   them. Run it on `f(x) = x**3`, `f_prime(x) = 3*x**2`.
