# 03 — Solutions: Scaling Techniques

*(This code was actually run to produce the numbers below.)*

## 1–3. Gradient accumulation equivalence

```python
def grad(w, x):
    return 2 * (w*x - 1) * x   # d/dw of (w*x - 1)^2

xs = [1.0, 2.0, 3.0, 4.0]
w = 0.5
full_grad = sum(grad(w, x) for x in xs) / len(xs)
print(full_grad)   # 2.5

micro1, micro2 = [1.0, 2.0], [3.0, 4.0]
accum_steps = 2
g1 = sum(grad(w, x) for x in micro1) / len(micro1) / accum_steps
g2 = sum(grad(w, x) for x in micro2) / len(micro2) / accum_steps
print(g1 + g2)   # 2.5 -- exact match

g1_wrong = sum(grad(w, x) for x in micro1) / len(micro1)
g2_wrong = sum(grad(w, x) for x in micro2) / len(micro2)
print(g1_wrong + g2_wrong, (g1_wrong + g2_wrong) / full_grad)
```

**Actual output:** full-batch gradient `2.5`; correctly-accumulated
gradient `2.5` (**exact match**); without dividing by `accumulation_steps`,
the result is `5.0` — **exactly 2x too large**, matching `accumulation_steps=2`
precisely. This confirms both that gradient accumulation is mathematically
exact when done correctly, and exactly how the forgotten-division bug
manifests (a scaling error proportional to the number of accumulation
steps, which would silently act like an unintended learning-rate increase).

## 4–5. Memory estimation

```python
params = 124_000_000
fp32_gb = params * 4 / 1e9
fp16_gb = params * 2 / 1e9
adam_total_gb = fp32_gb * 4   # weights + grad + m + v, all same size

print(fp32_gb, fp16_gb, adam_total_gb)
```

**Actual output: fp32 weights ≈ 0.496 GB, fp16 weights ≈ 0.248 GB
(exactly half); total Adam-related memory ≈ 1.984 GB.** Roughly 2GB for
optimizer state comfortably fits within an 8GB GPU, **before** accounting
for activation memory (which scales with batch size and sequence length,
and is often the larger practical constraint in real training runs) — a
useful reminder that "does the model fit" is really "does the model +
activations + optimizer state fit," not just the weight count alone.

## 6. Techniques for an 8GB GPU

Two natural choices, given the memory budget in Q5 leaves headroom mainly
for activations:

- **Mixed precision (fp16/bf16)**: roughly halves both weight *and*
  activation memory, at the cost of needing gradient scaling (Lesson
  067's `GradScaler`) to avoid `float16`'s underflow risk — or using
  `bfloat16` if the hardware supports it, sidestepping that risk entirely
  at a small precision cost.
- **Gradient checkpointing**: trades roughly 20-30% more compute time for
  a substantial reduction in activation memory specifically (the part not
  addressed by precision changes to weights/optimizer state) — the two
  techniques target different memory categories (weights/optimizer vs.
  activations) and combine naturally rather than being redundant with
  each other.

Gradient accumulation could also help *indirectly*: if activation memory
from a large batch is the bottleneck, using a smaller physical batch size
with accumulation reaches the same effective batch size at lower peak
memory, trading more forward/backward passes (more wall-clock time) for
that memory reduction.
