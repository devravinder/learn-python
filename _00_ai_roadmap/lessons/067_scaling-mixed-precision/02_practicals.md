# 02 — Practicals: Scaling Techniques

## Gradient accumulation equivalence (pure Python — no dependencies)

1. For a toy "loss function" `loss(w, x) = (w*x - 1)**2` and a batch
   `xs = [1.0, 2.0, 3.0, 4.0]`, compute the gradient `d(loss)/dw` at
   `w=0.5` for the **full batch at once** (average gradient over all 4
   samples).

2. Now compute it as **2 micro-batches of size 2** (`[1.0,2.0]` and
   `[3.0,4.0]`), remembering to divide each micro-batch's average gradient
   contribution appropriately before summing, matching
   `01_concepts.md`'s `loss / accumulation_steps` step. Confirm the summed
   micro-batch gradient exactly equals the full-batch gradient from Q1.

3. Deliberately **forget** to divide by `accumulation_steps` in Q2 and
   recompute. Confirm the result is now `accumulation_steps` times too
   large — a concrete demonstration of the bug `01_concepts.md` warns
   about.

## Memory estimation (pure arithmetic)

4. Estimate the memory (in GB) needed just to **store the weights** of a
   124M-parameter model (GPT-2 small's size) in `float32` (4 bytes/param)
   vs `float16`/`bfloat16` (2 bytes/param). How much memory does switching
   precision save, just for the weights?

5. Adam-family optimizers roughly need to store: the `float32` weights,
   gradients (same size as weights), and 2 moving-average buffers (`m` and
   `v`, each also the same size as weights) — roughly **4x the raw weight
   memory** in total, before even accounting for activations. Estimate
   total optimizer-related memory for the same 124M-parameter model.
   Does it fit comfortably in a consumer GPU with 8GB, 12GB, or 24GB of
   memory?

6. Given your Q5 estimate, if you only had an 8GB GPU, name at least two
   of `01_concepts.md`'s techniques you could combine to make training
   this model size more feasible, and explain briefly what each one
   actually trades away to buy that feasibility.
