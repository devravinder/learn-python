# 01 — Concepts: Build a GPT, Part 2 — Training Loop

## The training loop itself — nothing new

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

for step in range(max_steps):
    xb, yb = get_batch(train_data, block_size, batch_size)   # Lesson 064
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))   # Lesson 063

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}: train loss {loss.item():.4f}")
```

This **is** Lesson 040's training loop, unchanged in shape. `AdamW`
(Lesson 041) is the standard optimizer choice for Transformer training —
its decoupled weight decay tends to work better for this architecture
family than plain Adam or SGD.

## Evaluating on both train and validation loss

A single training batch's loss is noisy (one random sample). Track a more
stable **estimate** by averaging loss over several batches, for both
train and validation data:

```python
@torch.no_grad()
def estimate_loss(model, data_dict, eval_iters=50):
    model.eval()
    out = {}
    for split, data in data_dict.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size)
            logits = model(xb)
            losses[k] = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
        out[split] = losses.mean()
    model.train()
    return out
```

`model.eval()`/`model.train()` (Lesson 042) matter here specifically
because of dropout — you don't want dropout's randomness contaminating
your evaluation loss estimate. Watching **both** train and val loss
together is your primary tool for catching overfitting (Lesson 017) during
training — if val loss stops decreasing (or starts rising) while train
loss keeps falling, that's the signal to stop, add regularization, or get
more data.

## Learning rate warmup and decay (Lesson 041, applied to Transformers specifically)

Transformer training is well known to benefit from a **warmup** period
(small learning rate ramping up over the first few hundred steps) before
switching to a **decay** schedule (commonly cosine decay) for the rest of
training:

```python
def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff
```

Warmup specifically matters for Adam-family optimizers early in training:
their moving-average gradient statistics (Lesson 041) are unreliable when
there's been almost no data to average yet, and starting at full learning
rate immediately can cause early instability that a brief warmup avoids.

## Gradient clipping (Lesson 045, revisited)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Same tool, same purpose as Lesson 045's RNN training — prevents rare,
unusually large gradients (which do happen in Transformer training too)
from causing a destabilizing parameter update. Standard, cheap insurance
almost universally included in real LLM training code.

## Checkpointing (Lesson 040, revisited)

```python
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "step": step,
}, f"checkpoint_step{step}.pt")
```

Saving the **optimizer** state alongside the model (not just the weights)
matters for LLM training specifically because training runs are often
long enough that you genuinely need to resume from a crash or intentional
pause — resuming Adam-family optimizers without their saved moving-average
state effectively restarts that part of training from scratch, discarding
useful accumulated statistics.

## What to actually watch during your first real training run

- **Loss should decrease steadily** from the near-`log(vocab_size)`
  starting point (Lesson 064 Q6) toward something meaningfully lower.
- **If loss becomes `nan`**: almost always learning rate too high, or a
  missing gradient clip — check both first (Lesson 015/045's exact
  diagnosis, still true here).
- **If val loss plateaus far above train loss**: overfitting (Lesson 017)
  — for a from-scratch small GPT on a small corpus, this is common and
  expected; more data or a smaller model are the standard fixes.
- **Sample generated text periodically** (Lesson 066 covers generation
  properly) — watching the *qualitative* output evolve from random
  characters to recognizable word-fragments to real (if often nonsensical)
  sentences is one of the most satisfying and diagnostic things you can do
  during training, and often catches bugs loss curves alone miss.
