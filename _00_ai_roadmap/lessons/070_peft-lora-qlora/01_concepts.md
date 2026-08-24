# 01 — Concepts: LoRA & QLoRA

## The core idea, connecting directly back to Lesson 012

Lesson 012 showed that a weight update during training is often
well-approximated by a **low-rank** matrix — most of the useful change
lives in a lower-dimensional subspace, not spread evenly across every
possible direction. **LoRA (Low-Rank Adaptation)** takes this literally:
instead of fine-tuning a full weight matrix `W` (shape `d_out x d_in`,
potentially millions of parameters), **freeze `W` entirely** and learn a
low-rank update:

```
ΔW = B @ A
```

where `A` is `r x d_in` and `B` is `d_out x r`, with rank `r` chosen small
(commonly 4-64) — far fewer total parameters than `d_out * d_in` when `r`
is small relative to both dimensions.

```python
new_output = x @ W.T + x @ A.T @ B.T   # W frozen; only A, B are trained
```

## Why this works: fine-tuning updates tend to be low-rank in practice

Empirically (and somewhat theoretically motivated), the *useful* update
needed to adapt a pretrained model to a new task doesn't require full-rank
freedom — LoRA's authors found rank 4-8 sufficient for many tasks, meaning
the number of trainable parameters can be **orders of magnitude** smaller
than full fine-tuning while recovering most of its benefit.

```python
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False   # freeze the pretrained weight

        d_out, d_in = original_linear.weight.shape
        self.A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, rank))   # B starts at 0: no change at init
        self.scale = alpha / rank

    def forward(self, x):
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.scale
```

**`B` initialized to zero is deliberate**: at the start of fine-tuning,
`ΔW = B @ A = 0`, so the model behaves *exactly* like the original
pretrained model — fine-tuning then gradually introduces the adaptation,
rather than starting from a randomly-perturbed (and likely worse) point.

## Parameter count comparison, concretely

For a `d_model=4096` attention projection matrix (a real LLM scale):
`d_out * d_in = 4096 * 4096 ≈ 16.8M` parameters for full fine-tuning of
just that one matrix. LoRA at `rank=8`: `A` is `8 * 4096 = 32,768`, `B` is
`4096 * 8 = 32,768` — **~65K parameters, roughly 250x fewer** for an
equivalent-shaped update, applied per matrix you choose to adapt (commonly
just the attention Q/K/V/O projections, not every matrix in the model).

## Why LoRA dramatically reduces memory, not just parameter count

Since `W` is frozen (`requires_grad=False`), it needs **no gradient, no
optimizer state** (Lesson 067's ~4x-per-parameter accounting only applies
to the *trainable* parameters — `A` and `B`, not the frozen `W`). This is
why LoRA fine-tuning of even a multi-billion-parameter model can fit on a
single consumer GPU: the frozen base model needs only inference-level
memory, and the trainable LoRA parameters are a tiny addition on top.

## QLoRA: LoRA + quantization

**QLoRA** goes further: **quantize** the frozen base model's weights to
very low precision (commonly 4-bit, vs. `float16`/`float32`) before adding
LoRA adapters on top. The frozen weights only need to support inference
(and their memory footprint shrinks dramatically at 4-bit), while the
small trainable LoRA parameters stay at higher precision for stable
training. This combination is what makes fine-tuning models with tens of
billions of parameters feasible on a single consumer GPU — a technique
directly enabling widespread open-model fine-tuning that would otherwise
require data-center-scale hardware.

## Merging LoRA weights back in (for deployment)

After training, `ΔW = B @ A` can be **merged directly into `W`**
(`W_new = W + scale * B @ A`) — producing a model with the exact same
architecture and inference cost as the original, no separate adapter
computation needed at serving time. This is why LoRA is popular not just
for training efficiency but also deployment simplicity — you can ship
either the merged full model or the tiny adapter weights separately (to
apply on top of a shared base model for multiple different fine-tuned
"personalities" without duplicating the whole base model per variant).

## When to reach for LoRA vs. full fine-tuning

- **LoRA/QLoRA**: the default choice for adapting a model you didn't train
  yourself, especially on limited hardware — which describes the vast
  majority of practical fine-tuning work, including anything you'll do
  with a model larger than Project 013's from-scratch GPT.
- **Full fine-tuning**: still used when you have the compute budget and
  need maximum adaptation capacity (e.g. teaching genuinely new
  capabilities far from the base model's pretraining distribution) —
  increasingly rare outside large labs with dedicated infrastructure.
