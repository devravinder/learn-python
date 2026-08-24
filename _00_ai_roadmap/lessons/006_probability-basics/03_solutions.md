# 03 — Solutions: Probability Basics

## 1. Die roll

`Ω = {1,2,3,4,5,6}`, `A = {2,4,6}`, `B = {4,5,6}`.

- `P(A) = 3/6 = 0.5`
- `P(B) = 3/6 = 0.5`
- `A ∩ B = {4,6}` → `P(A, B) = 2/6 = 1/3`
- `P(A | B) = P(A,B)/P(B) = (1/3)/(1/2) = 2/3`

Since `P(A | B) = 2/3 ≠ P(A) = 1/2`, **A and B are not independent** — knowing
the roll is > 3 makes it more likely to be even (because 4 and 6 are both even
and > 3, while only 5 is odd and > 3).

## 2. At least one heads

Outcomes: `{HH, HT, TH, TT}`, each with probability 1/4.
`P(at least one heads) = 3/4` (all except `TT`).
Check: `P(no heads) = P(TT) = 1/4`, so `1 - 1/4 = 3/4`. ✓.

## 3. Spam filter via Bayes

```
P(spam) = 0.40, P(not spam) = 0.60
P(free | spam) = 0.90
P(free | not spam) = 0.05

P(free) = 0.90*0.40 + 0.05*0.60 = 0.36 + 0.03 = 0.39

P(spam | free) = P(free|spam)*P(spam) / P(free) = 0.36 / 0.39 ≈ 0.923
```

About **92.3%** chance the email is spam given it contains "free".

## 4. Why the disease-test answer feels surprising

Accuracy describes how the test behaves *given* the true condition
(`P(positive | disease)`), but the question asks the reverse
(`P(disease | positive)`). When the condition is rare, the pool of healthy
people is enormous compared to the pool of sick people, so even a small false
positive rate among the huge healthy group produces more false positives in
absolute terms than true positives from the tiny sick group. Bayes' theorem is
exactly the tool for correctly inverting the conditional instead of trusting
intuition.

## 5. Simulating the die roll

```python
import numpy as np

rng = np.random.default_rng(0)
rolls = rng.integers(1, 7, size=100_000)

A = (rolls % 2 == 0)
B = (rolls > 3)

p_a = A.mean()
p_b = B.mean()
p_a_given_b = A[B].mean()   # restrict to rolls where B is True, then check A

print(p_a, p_b, p_a_given_b)   # ~0.5, ~0.5, ~0.667
```

## 6. Simulating the spam scenario

```python
import numpy as np

rng = np.random.default_rng(0)
n = 100_000

is_spam = rng.random(n) < 0.40
has_free = np.where(
    is_spam,
    rng.random(n) < 0.90,
    rng.random(n) < 0.05,
)

p_spam_given_free = is_spam[has_free].mean()
print(p_spam_given_free)   # should land close to ~0.923
```

`np.where(condition, a, b)` picks from `a` where `condition` is True and `b`
elsewhere — here, generating "contains free" with a different probability
depending on whether the email is spam.
