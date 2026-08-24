# 03 — Solutions: Scaling Laws & Compute Budgeting

*(This code was actually run to produce the numbers below.)*

## 1. Compute for two Chinchilla-ratio scenarios

```python
N_a, D_a = 1_000_000, 20_000_000
N_b, D_b = 10_000_000, 200_000_000
C_a, C_b = 6*N_a*D_a, 6*N_b*D_b
print(C_a, C_b, C_b / C_a)
```

**Actual output: `C_a = 1.2e14`, `C_b = 1.2e16`, ratio = exactly `100`** —
a 10x increase in both parameters and data (following the fixed ratio)
compounds to a 100x compute requirement, since `C` scales with the
*product* `N*D`, not either alone.

## 2. Wall-clock time estimates

```python
flops_per_sec = 50e12
print(C_a / flops_per_sec / 60, C_b / flops_per_sec / 60)   # minutes
```

**Actual output: scenario (a) ≈ 0.04 minutes (2.4 seconds), scenario (b) ≈
4.0 minutes.** At this hypothetical sustained throughput, even the larger
scenario trains in a few minutes — a reminder that hobby-scale LLM
training (as opposed to real frontier-model training) is genuinely
fast, precisely because the parameter/data counts are so many orders of
magnitude smaller.

## 3. Compute-optimal N and D for a fixed budget

```python
C = 1e17
N_opt = (C / 120) ** 0.5   # from C = 6*N*D and D = 20*N => C = 120*N^2
D_opt = 20 * N_opt
print(N_opt, D_opt, 6*N_opt*D_opt)   # verify against C
```

**Actual output: `N_opt ≈ 28.87 million`, `D_opt ≈ 577.35 million` tokens,
and the check `6*N_opt*D_opt` reproduces `1e17` exactly** — confirming the
algebra (`C = 6*N*D`, `D = 20*N` → `C = 120*N²` → `N = sqrt(C/120)`) is
correct and self-consistent.

## 4. Compute-optimal model size for a fixed 5M-token dataset

```python
D_fixed = 5_000_000
N_suggested = D_fixed / 20
print(N_suggested)
```

**Actual output: `250,000` parameters.** For a hobby-scale corpus of 5
million tokens, the Chinchilla ratio suggests a model with only a **quarter
million** parameters — dramatically smaller than "a real LLM" might
naively suggest, and almost certainly smaller than many first-time
from-scratch GPT projects default to building.

## 5. Why an oversized model on too little data wastes capacity

An oversized model relative to its training data has enormous capacity to
memorize the specific training corpus rather than learn generalizable
statistical structure — exactly Lesson 017's **high-variance/overfitting**
regime, just reached via "too many parameters for the data" instead of
"too flexible a function class." You'd expect training loss to keep
dropping (the model has more than enough capacity to fit the small
corpus closely, even exactly) while validation loss plateaus or worsens —
the same diagnostic signature from Lesson 017, now identified through the
lens of scaling laws rather than a generic capacity argument.

## 6. Sizing Project 013's own model

Using this lesson's ratio as a starting point: for a corpus of, say, 2MB
of plain text (roughly 400,000-500,000 characters, so a comparable order
of magnitude in tokens at character-level tokenization), the Chinchilla
ratio suggests a model in the tens-of-thousands to low-hundred-thousands
of parameters range — much smaller than Lesson 060's example
configurations might suggest by default. **Do this calculation for your
own actual planned corpus size before starting Project 013's training
run** — it's a genuinely useful, realistic sizing exercise, not just a
theoretical lesson.
