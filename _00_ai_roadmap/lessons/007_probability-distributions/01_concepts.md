# 01 — Concepts: Probability Distributions

## PMF vs PDF vs CDF

- **PMF** (probability mass function) — for discrete random variables;
  `P(X = x)` directly.
- **PDF** (probability density function) — for continuous random variables;
  `P(X = x)` is technically 0 for any exact point, so the PDF describes
  *density*, and you integrate it over a range to get a probability.
- **CDF** (cumulative distribution function) — `P(X <= x)`, works for both
  discrete and continuous; always non-decreasing from 0 to 1.

## Bernoulli — a single coin flip

One trial, two outcomes (success/failure) with probability `p` of success.

```
P(X=1) = p,  P(X=0) = 1-p
E[X] = p,  Var(X) = p(1-p)
```

This is the distribution behind a single binary classification output (spam vs
not-spam), and the building block for the Binomial distribution below.

## Binomial — n independent Bernoulli trials

Number of successes in `n` independent trials, each with success probability
`p`.

```
P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
E[X] = n*p,  Var(X) = n*p*(1-p)
```

```python
import numpy as np
rng = np.random.default_rng(0)
samples = rng.binomial(n=10, p=0.3, size=1000)   # 1000 draws of "successes in 10 trials"
```

## Normal (Gaussian) — the default continuous distribution

```
PDF: f(x) = 1/(σ√(2π)) * exp(-(x-μ)²/(2σ²))
```

Parameterized by mean `μ` and standard deviation `σ`. Central to ML for two
reasons: (1) the Central Limit Theorem says sums/averages of many independent
random effects tend toward a Normal distribution, which is why measurement
noise, weight initializations, and residuals are often modeled as Gaussian;
(2) many loss functions (mean squared error) implicitly assume Gaussian noise.

```python
samples = rng.normal(loc=0.0, scale=1.0, size=1000)   # standard normal
```

The **standard normal** has `μ=0, σ=1`. Any normal variable can be converted to
standard normal via **z-score**: `z = (x - μ) / σ` — this is literally what
feature standardization does before training many models.

## Uniform

Every value in `[a, b]` equally likely. Used for random initialization ranges
and as the "no information" default prior.

```python
samples = rng.uniform(low=0.0, high=1.0, size=1000)
```

## Categorical / Multinomial — the classifier's output distribution

A Categorical distribution generalizes Bernoulli to `k` outcomes instead of 2
(e.g. "which of 10 digit classes"). A neural network classifier's final layer
produces a vector of probabilities over classes via **softmax**:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

which is exactly a Categorical distribution's parameters — this is why the
output layer of a classifier is a probability distribution, not just a score.

## Sampling vs computing exactly

Sometimes you can compute a probability exactly (small discrete cases via the
formulas above); often — especially once distributions combine or interact
non-trivially — it's easier to draw many random samples and estimate the
quantity you want (a probability, a mean, a variance) from the sample. This
"simulate instead of solve" approach is used constantly in ML (e.g. dropout,
Monte Carlo estimates, sampling from a language model's output distribution).
