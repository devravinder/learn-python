# 01 — Concepts: Mixed Precision, Gradient Accumulation, Checkpointing

## Numeric precision: what `float32` vs `float16`/`bfloat16` actually means

Every number in a neural network is stored with finite precision.
`float32` (the default) uses 32 bits per number; `float16`/`bfloat16` use
16 — half the memory, and typically 2x+ faster matrix multiplication on
modern GPUs, which have dedicated hardware for lower-precision math.

## Mixed precision training: fast where safe, precise where needed

Training entirely in `float16` can cause numerical issues — gradients can
underflow to exactly 0 in `float16`'s narrower range, silently halting
learning for affected parameters. **Automatic Mixed Precision (AMP)** runs
the forward pass and most computation in half precision (fast, low
memory), but keeps a `float32` master copy of weights and uses **gradient
scaling** (temporarily multiplying the loss by a large constant before
`.backward()`, then unscaling gradients before the optimizer step) to
prevent small gradients from underflowing to zero.

```python
scaler = torch.cuda.amp.GradScaler()

for xb, yb in train_loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

`bfloat16` (used by most modern LLM training, on hardware that supports
it) has the same exponent range as `float32` (avoiding the underflow
problem entirely) at the cost of less precision per number — often used
*without* a gradient scaler at all, since the underflow risk that
motivates scaling is specific to `float16`'s narrower range.

## Gradient accumulation: simulating a larger batch than fits in memory

Bigger batches (Lesson 015) generally give more stable gradient estimates,
but a batch that's too large simply won't fit in GPU memory. **Gradient
accumulation** runs several smaller "micro-batches," accumulating
gradients (not resetting between them) before finally calling
`optimizer.step()` once — mathematically equivalent to one large batch,
achieved with several small ones.

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (xb, yb) in enumerate(train_loader):
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1)) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

Dividing the loss by `accumulation_steps` before `.backward()` is
essential — without it, gradients from `accumulation_steps` micro-batches
would sum to `accumulation_steps` times too large, effectively multiplying
your learning rate unintentionally.

## Gradient checkpointing: trading compute for memory

Normal backpropagation (Lesson 037) requires storing every layer's
intermediate activations from the forward pass, to reuse during the
backward pass — memory cost that grows with model depth. **Gradient
checkpointing** discards most intermediate activations after the forward
pass and **recomputes** them during the backward pass instead, trading
roughly 20-30% more compute time for a substantial memory reduction —
often the difference between "the model fits on your GPU" and "it
doesn't," for a given model size.

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        return checkpoint(self._forward, x, mask, use_reentrant=False)
    def _forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x
```

## Putting it together: the practical scaling toolkit

| Technique | Solves | Cost |
|---|---|---|
| Mixed precision (fp16/bf16) | Speed + memory | Slight precision loss, occasional instability if not careful |
| Gradient accumulation | "My desired batch size doesn't fit in memory" | More forward/backward passes per optimizer step (more wall-clock time per step, but stable training) |
| Gradient checkpointing | "My model doesn't fit in memory" | ~20-30% more compute time |

None of these change the *model* or the *math* being learned — they're
purely about making training feasible within real hardware constraints,
which is why they matter increasingly as model size grows toward what
Project 013 will use, and are standard practice at every real LLM training
lab regardless of scale.

## A concrete budgeting exercise (setting up Lesson 068)

Given a GPU with `X` GB of memory, a model with `P` parameters (each
needing ~4 bytes in fp32 for weights, plus similar for gradients and
optimizer state — Adam roughly triples the effective memory per parameter
beyond just the weights themselves), you can estimate whether a given
model/batch-size combination fits *before* running it, avoiding wasted
time on an out-of-memory crash partway through a training run. Lesson 068
covers this kind of compute/scale estimation formally.
