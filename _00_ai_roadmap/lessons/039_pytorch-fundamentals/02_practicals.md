# 02 — Practicals: PyTorch Fundamentals

1. Create a `(3, 4)` tensor of random normal values. Reshape it to `(4, 3)`
   and to `(2, 6)`. Confirm `.numpy()` gives a NumPy array with matching
   values (Lesson 003's operations should all feel familiar).

2. Recreate Lesson 038's tiny hand-derived network in PyTorch:
   `x=2, w1=0.5 (requires_grad), b1=0 (requires_grad), w2=1.0 (requires_grad),
   b2=0 (requires_grad), y=3`, with `relu` between layers. Compute the loss
   and call `.backward()`. Confirm `w1.grad, b1.grad, w2.grad, b2.grad`
   exactly match Lesson 037/038's values (`-8.0, -4.0, -4.0, -4.0`).

3. Forget to zero gradients: call `.backward()` twice in a row on the same
   computation (rebuild the graph each time, e.g. inside a loop) without
   resetting `.grad`. Observe that the gradient values are *not* what you'd
   expect the second time (they've accumulated) — then fix it with
   `w1.grad = None` (or rebuild fresh tensors) between iterations.

4. Build the `MLP` class from `01_concepts.md` using `nn.Module`. Train it
   on Lesson 035's XOR data (`torch.tensor` inputs, MSE loss,
   `torch.optim.SGD`) for 500 epochs. Print the loss every 100 epochs and
   confirm it converges near 0.

5. Compare training the same XOR MLP with `torch.optim.SGD(lr=0.1)` vs
   `torch.optim.Adam(lr=0.1)` (Lesson 041 covers optimizers properly, but
   try it now) — does one converge faster on this tiny problem?

6. Time (`time.time()`) 1000 forward+backward passes of a `(1000, 1000) @
   (1000, 1000)` matrix multiply chain on CPU. If you have GPU access, move
   everything to `"cuda"` and time again — report the speedup (if no GPU is
   available, note that and skip the comparison, but read about typical
   speedups in the PyTorch docs).
