# 02 — Practicals: Probability Basics

## Pen-and-paper

1. A fair six-sided die is rolled once. Let `A` = "roll is even",
   `B` = "roll is greater than 3". Compute `P(A)`, `P(B)`, `P(A, B)`, and
   `P(A | B)`. Are `A` and `B` independent?

2. Two fair coins are flipped. What is `P(at least one heads)`? (Compute it
   directly by enumerating outcomes, then double check via
   `1 - P(no heads)`.)

3. A spam filter: 40% of emails are spam. 90% of spam emails contain the word
   "free"; 5% of non-spam emails contain "free". An email contains "free".
   Using Bayes' theorem, what's the probability it's spam?

4. Explain in your own words why, in the disease-testing example from
   `01_concepts.md`, the answer is so much lower than the test's "99% accurate"
   headline number suggests.

## Simulate to check your answers (NumPy)

5. Simulate 100,000 rolls of a fair die with `np.random.default_rng`. Estimate
   `P(A)`, `P(B)`, and `P(A | B)` from the simulation (count occurrences) and
   compare to your pen-and-paper answers from Q1.

6. Simulate the spam scenario from Q3: generate 100,000 emails, 40% spam per a
   random draw; for spam emails include "free" with 90% probability, for
   non-spam with 5% probability. From the simulated data, estimate
   `P(spam | contains "free")` and compare to your Bayes' theorem answer.
