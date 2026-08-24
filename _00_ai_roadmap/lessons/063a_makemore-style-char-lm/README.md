# Lesson 063a — Bigram & MLP Character-Level Language Models (makemore-style)

**Module:** 11 — Building Your Own LLM (Karpathy-style)
**Prerequisites:** [038](../038_nn-from-scratch/README.md), [063](../063_language-modeling-objective/README.md)
**Estimated time:** 2.5–3 hours

## Objective

Following [Andrej Karpathy's `makemore`](https://github.com/karpathy/makemore)
directly: build the simplest possible character-level language model (a
bigram counting table), then show a single-layer neural network trained
with gradient descent converges to the **exact same thing** — the clearest
possible proof that "training a language model" is not a different kind of
activity from anything else in this curriculum, just cross-entropy loss +
gradient descent, applied to text. Then extend to a small MLP with real
context, one step closer to Lesson 064's full GPT.

## Contents

1. [01_concepts.md](01_concepts.md)
2. [02_practicals.md](02_practicals.md)
3. [03_solutions.md](03_solutions.md)

## Resources

- [Andrej Karpathy — The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)
- [Andrej Karpathy — Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)
