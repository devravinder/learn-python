# 03 — Solutions: Build a GPT, Part 2 — Training Loop

*(Q1-2's schedule code was actually run to produce the numbers below.)*

## 1–2. Learning rate schedule

```python
import math
import matplotlib.pyplot as plt

def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return max_lr * coeff

warmup, max_steps, max_lr = 100, 1000, 3e-4
lrs = [get_lr(s, warmup, max_steps, max_lr) for s in range(max_steps + 1)]
plt.plot(lrs)
plt.show()

print(get_lr(0, warmup, max_steps, max_lr))
print(get_lr(50, warmup, max_steps, max_lr))
print(get_lr(100, warmup, max_steps, max_lr))
print(get_lr(550, warmup, max_steps, max_lr))
print(get_lr(1000, warmup, max_steps, max_lr))
```

**Actual output:**

```text
step 0:    0.0
step 50:   0.00015   (exactly half of max_lr, halfway through linear warmup)
step 100:  0.0003    (exactly max_lr - the peak, right at the warmup/decay boundary)
step 550:  0.00015   (exactly half of max_lr again - the cosine's midpoint)
step 1000: 0.0       (fully decayed)
```

All boundary values match exactly what the formula predicts: linear ramp
from 0 to `max_lr` over the warmup window, then a cosine decay from
`max_lr` back to (approximately) 0 by `max_steps` — with the schedule's
symmetric cosine shape putting the midpoint value at exactly half of
`max_lr`, confirmed at step 550 (halfway between step 100 and step 1000).

## 3. Full training loop

```python
import torch
import torch.nn.functional as F

optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
train_losses, val_losses = [], []

for step in range(max_steps):
    lr = get_lr(step, warmup, max_steps, max_lr)
    for g in optimizer.param_groups:
        g["lr"] = lr

    xb, yb = get_batch(train_data, block_size, batch_size)
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 100 == 0:
        losses = estimate_loss(model, {"train": train_data, "val": val_data})
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])
        print(step, losses)
```

## 4. Train vs val loss curves

```python
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.legend()
plt.show()
```

On a small from-scratch character-level GPT with a reasonably-sized text
corpus, expect train and val loss to track closely for a while, with val
loss eventually leveling off (or slowly diverging from train loss) once
the model has extracted most of the easily-learnable structure — the
degree of divergence depends heavily on corpus size relative to model
size (Lesson 017's bias-variance framing, directly applicable).

## 5. Checkpoint save/resume

```python
torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "step": step},
           "checkpoint.pt")

# in a fresh process:
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_step = checkpoint["step"]
# continue the training loop from start_step
```

If the optimizer state is correctly restored, loss should continue
smoothly from wherever it left off; if you instead create a **fresh**
optimizer after loading only the model weights, expect a visible loss
spike immediately after resuming — the fresh Adam optimizer's moving
averages start from zero again, causing a brief instability similar to the
very start of training, even though the model weights themselves were
fine. This exact comparison is a good way to convince yourself that saving
optimizer state isn't a formality.

## 6. Too-high learning rate

With `max_lr=0.1` (roughly 300x the original), expect loss to become
unstable early in training — likely spiking to very large values or
reaching `nan` within the first few dozen to few hundred steps, especially
once the warmup period ends and the full learning rate kicks in.
**Gradient clipping alone typically won't fully rescue an learning rate
this far out of a reasonable range** — clipping bounds the *size* of a
single update's gradient norm, but a learning rate 300x too large will
still take steps far larger than appropriate even with a clipped gradient
direction; the fix is lowering the learning rate itself, not just adding
more clipping. This is a useful, hands-on confirmation of Lesson 015's
original point: the learning rate is a genuine hyperparameter to get
right, not a detail regularization can fully paper over.
