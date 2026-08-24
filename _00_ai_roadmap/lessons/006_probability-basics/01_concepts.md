# 01 — Concepts: Probability Basics

## Sample space, events, and probability

The **sample space** `Ω` is the set of all possible outcomes of an experiment
(e.g. rolling a die: `{1,2,3,4,5,6}`). An **event** is any subset of the sample
space (e.g. "rolling an even number" = `{2,4,6}`). A probability function `P`
assigns each event a number in `[0, 1]`, with `P(Ω) = 1`.

**Axioms** (Kolmogorov):
1. `P(A) >= 0` for any event `A`
2. `P(Ω) = 1`
3. If `A` and `B` are mutually exclusive (can't both happen), `P(A ∪ B) = P(A) + P(B)`

## Random variables

A **random variable** `X` maps outcomes to numbers, so we can do math with
"the result of a random process" directly (e.g. `X` = number of heads in 10
coin flips). Random variables are **discrete** (countable outcomes: die roll,
word chosen from a vocabulary) or **continuous** (uncountable: a person's
height, a model's output logit).

## Joint, marginal, and conditional probability

- **Joint**: `P(A, B)` — probability both `A` and `B` happen.
- **Marginal**: `P(A) = Σ_b P(A, B=b)` — probability of `A` regardless of `B`
  ("summing out" `B`).
- **Conditional**: `P(A | B) = P(A, B) / P(B)` — probability of `A` *given*
  we already know `B` happened. This is the object a classifier actually
  outputs: `P(label | features)`.

## Independence

`A` and `B` are **independent** if knowing one tells you nothing about the
other: `P(A | B) = P(A)`, equivalently `P(A, B) = P(A) * P(B)`. Naive Bayes is
"naive" precisely because it assumes all features are independent given the
class — an assumption that's usually false but works well enough in practice.

## Bayes' theorem

```
P(A | B) = P(B | A) * P(A) / P(B)
```

Read as: "how to flip a conditional probability around." This is the single
most-reused equation in ML:

- **Naive Bayes classifiers** compute `P(class | features)` from
  `P(features | class)` and `P(class)` directly via Bayes.
- **`P(A)`** is called the **prior** (what you believed before seeing evidence),
  **`P(A|B)`** the **posterior** (updated belief after evidence `B`), and
  **`P(B|A)`** the **likelihood** (how well `A` explains the evidence).

### Worked example

A test for a rare disease (1 in 1,000 people have it) is 99% accurate (for
both people who have it and people who don't). Given a positive test, what's
the probability you actually have the disease?

```
P(disease) = 0.001
P(positive | disease) = 0.99
P(positive | no disease) = 0.01   (false positive rate)

P(positive) = P(positive|disease)*P(disease) + P(positive|no disease)*P(no disease)
            = 0.99*0.001 + 0.01*0.999
            = 0.00099 + 0.00999
            = 0.01098

P(disease | positive) = 0.99 * 0.001 / 0.01098 ≈ 0.0902  (~9%)
```

Despite a 99%-accurate test, a positive result only means ~9% chance of
actually having the disease — because the disease is so rare, false positives
from the healthy majority outnumber true positives from the small infected
group. This exact reasoning (base rates dominating naive intuition) is why
accuracy alone is a misleading metric on imbalanced datasets (Lesson 018).

## Expectation and variance (preview — full treatment in Lesson 008)

`E[X]` (expected value) is the long-run average of a random variable, weighted
by probability: `E[X] = Σ_x x * P(X=x)`. This is the mathematical definition
behind "loss function" — a loss is an expectation over the data distribution
that training tries to minimize.
