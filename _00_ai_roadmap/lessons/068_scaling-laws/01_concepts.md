# 01 — Concepts: Scaling Laws & Compute Budgeting

## The empirical finding: loss follows predictable power laws

Kaplan et al. (2020) found that language model test loss decreases
smoothly and predictably as a **power law** in three quantities: model
size (parameters `N`), dataset size (tokens `D`), and compute (`C`) — as
long as the other two aren't the bottleneck:

```
Loss(N) ≈ (N_c / N) ^ alpha_N     (holding data/compute effectively unlimited)
```

Practically: you can often **predict** how much a larger model or more
data will improve loss *before* actually training it, by fitting this
power law to a handful of smaller, cheaper training runs — a genuinely
useful practical tool used by real labs to plan expensive training runs in
advance rather than guessing.

## Compute-optimal training: the Chinchilla result

Given a **fixed compute budget** `C`, there's an optimal split between
model size and data size. Kaplan et al.'s original work suggested scaling
model size faster than data; Hoffmann et al. (2022, "Chinchilla") found
that many early large models (including GPT-3) were actually
**undertrained relative to their size** — a *smaller* model trained on
*more* data would have achieved better loss for the same compute budget.

**The Chinchilla rule of thumb**: for compute-optimal training, scale
model parameters and training tokens **roughly proportionally** —
approximately **20 tokens of training data per parameter** (e.g. a 1
billion parameter model wants roughly 20 billion training tokens for
compute-optimal use of a given budget).

```
C ≈ 6 * N * D    (a standard approximation: ~6 FLOPs per parameter per token, forward+backward)
```

## Why this matters directly for Project 013

Given your actual compute budget (a laptop, a free-tier GPU notebook,
whatever you have), scaling laws suggest a **right-sized** model rather
than "as big as possible": a tiny model trained on far more data than the
Chinchilla ratio suggests is data-underutilizing that model's capacity; a
large model trained on too little data (very likely at hobby scale) leaves
performance on the table relative to a smaller model that could be trained
to convergence on the same token budget. **Practical implication**: at
small scale, prioritize getting enough training tokens relative to your
model size over maximizing parameter count — a common mistake for a first
from-scratch LLM project is building too large a model for too little
data, which trains slowly and undertrains badly.

## Overfitting at LLM scale looks different than Module 4-7's version

With a properly-sized model and enough diverse data, LLMs are remarkably
resistant to Lesson 017's classical overfitting (huge datasets, single-pass
training) — the more common problem at hobby/project scale is the
opposite: **too little data for the model size**, causing the model to
memorize the (small) training corpus rather than learning generalizable
language structure. Watching train/val loss diverge (Lesson 065) is still
the right diagnostic; the fix at LLM scale is usually "get more/more
diverse data" or "use a smaller model," not primarily dropout/weight decay
as in Module 6.

## Estimating training time and cost, roughly

```
training_time ≈ C / (achieved FLOPs/second)
```

Where achieved FLOPs/second depends heavily on hardware (GPU model),
precision (Lesson 067's mixed precision — often 2x+ throughput),
implementation efficiency (are you actually saturating the GPU, or
bottlenecked on data loading), and batch size. In practice: **benchmark a
handful of training steps first**, measure actual wall-clock time per
step, then extrapolate to your target step count — far more reliable than
theoretical FLOP calculations alone, which routinely overestimate real
achieved throughput.

## What "big" actually means, put in perspective

| Model | Approx. parameters | Approx. training tokens |
|---|---|---|
| Your Lesson 060 toy example | ~50K | (didn't train on real data) |
| Project 013's likely scale | 1-10M | Hundreds of thousands to low millions |
| GPT-2 small | 124M | ~10 billion (arguably undertrained by Chinchilla standards) |
| GPT-3 | 175B | ~300 billion |
| Modern frontier models | Not fully public, widely estimated in the hundreds of billions to low trillions | Trillions+ |

Project 013 sits many orders of magnitude below even GPT-2 — the value is
in correctly implementing and understanding every piece (which you now
have, across Lessons 058-068), not in approaching real-world scale. Scaling
laws are exactly what tell you, honestly, how much of the gap between your
project and a frontier model is "different algorithm" (none — same
architecture) versus "different scale" (essentially all of it).
