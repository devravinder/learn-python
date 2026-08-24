# 02 — Practicals: Build a GPT, Part 2 — Training Loop

## Learning rate schedule (pure Python — no dependencies)

1. Implement `get_lr(step, warmup_steps, max_steps, max_lr)` from
   `01_concepts.md`. For `warmup_steps=100, max_steps=1000, max_lr=3e-4`,
   compute and plot the learning rate at every step from 0 to 1000.
   Confirm it ramps up linearly for the first 100 steps, then decays
   smoothly (cosine shape) to near 0 by step 1000.

2. Confirm the schedule's exact boundary values: `get_lr(0, ...)` should
   be `0` (or very close), `get_lr(100, ...)` should be exactly `max_lr`
   (the peak, right at the end of warmup), and `get_lr(1000, ...)` should
   be very close to `0` again (full decay).

## Full training run (PyTorch — needs real text + time)

3. Using Lesson 064's data pipeline and GPT model, implement the full
   training loop from `01_concepts.md`: `AdamW` optimizer, the LR schedule
   from Q1 (call `optimizer.param_groups[0]["lr"] = get_lr(step, ...)`
   each step), gradient clipping, and periodic `estimate_loss` calls.
   Train for a few thousand steps on a real text file.

4. Plot train and validation loss (from your periodic `estimate_loss`
   calls) over training. Does validation loss track training loss
   closely, or diverge (Lesson 017's overfitting signature)?

5. Save a checkpoint partway through training, then write a small script
   that loads it back (`model.load_state_dict(...)`,
   `optimizer.load_state_dict(...)`) and confirms training loss continues
   smoothly from where it left off (not spiking back up, which would
   indicate the optimizer state wasn't actually restored).

6. Deliberately set the learning rate very high (e.g. `max_lr=0.1` instead
   of `3e-4`) and retrain from scratch for a few hundred steps. Does loss
   become unstable or `nan`? Add gradient clipping back if you removed it
   for this test — does clipping alone prevent the `nan`, or is the
   learning rate itself simply too high regardless?
