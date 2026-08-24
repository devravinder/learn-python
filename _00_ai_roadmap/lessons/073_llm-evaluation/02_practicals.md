# 02 — Practicals: LLM Evaluation

## Held-out perplexity (pure Python, extending Lesson 063)

1. Given per-token predicted probabilities-of-correct-token for a
   "training" sequence (`[0.9, 0.85, 0.95, 0.9]`) and a "held-out" sequence
   the model has never seen (`[0.4, 0.3, 0.5, 0.35]`), compute perplexity
   for both. Confirm held-out perplexity is much higher — exactly the gap
   Lesson 017's overfitting framing predicts between training and
   validation performance.

2. Explain why comparing a perplexity of `15` (computed with a 5,000-token
   vocabulary tokenizer) to a perplexity of `12` (computed with a
   50,000-token vocabulary tokenizer) is **not** a valid "model B is
   better" conclusion on its own — connect this to `01_concepts.md`'s
   point about tokenizer-dependence.

## Multiple-choice scoring mechanics (pure Python simulation)

3. Simulate `score_choice` from `01_concepts.md` using toy log-probabilities
   instead of a real model: for a question with 4 choices, given
   (fabricated) total log-probabilities `[-5.2, -3.1, -8.7, -4.0]`, pick
   the model's predicted answer (highest log-probability = least
   "surprising" continuation). Confirm this matches taking the `argmax`.

4. Simulate scoring 20 toy multiple-choice questions where the correct
   answer's fabricated log-probability is the highest 15 times out of 20.
   Compute the resulting benchmark accuracy (%). Does this match how you'd
   compute accuracy for any classifier (Lesson 024) — is a benchmark score
   fundamentally just held-out accuracy on a specific, curated dataset?

## Reflection

5. Look up (or recall from `01_concepts.md`) what "benchmark contamination"
   means, and explain in your own words why it's a fundamentally different
   problem from ordinary overfitting (Lesson 017) — what makes it hard to
   even detect, compared to overfitting, which a simple train/val split
   reveals directly?

6. For a model you might build (e.g. Project 013's own GPT, or a
   fine-tuned classifier), design a **practical evaluation checklist**
   (3-5 items) following `01_concepts.md`'s guidance — be specific about
   what held-out data, what metric(s), and what manual review step you'd
   actually use, not just "evaluate the model" generically.
