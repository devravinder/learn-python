# 02 — Practicals: LoRA & QLoRA

## Parameter count arithmetic (pure Python)

1. For a weight matrix of shape `(d_out=4096, d_in=4096)`, compute the
   full fine-tuning parameter count vs. LoRA parameter count
   (`rank * d_in + d_out * rank`) at `rank=4, 8, 16, 64`. At what rank does
   LoRA's parameter count reach 1% of full fine-tuning's?

2. Repeat Q1 for a smaller matrix, `(d_out=768, d_in=768)` (roughly
   GPT-2-small attention-projection scale). Does the *relative* savings
   (LoRA params / full params) stay similar, or change, as the base matrix
   shrinks?

## LoRA mechanics (pure Python — no dependencies)

3. Implement a tiny `LoRALinear`-equivalent using plain Python lists: a
   frozen weight matrix `W`, and trainable low-rank matrices `A` (rank x
   d_in) and `B` (d_out x rank), with `B` initialized to all zeros.
   Compute `output = x @ W.T + scale * (x @ A.T @ B.T)` for a random input
   `x` and confirm the output **exactly equals** `x @ W.T` alone (since
   `B` is all zeros, `ΔW` contributes nothing at initialization).

4. "Train" `A` and `B` by hand for a few steps (pick simple gradient
   values, or use a tiny toy loss and compute gradients manually) and
   confirm the output **changes** once `B` is no longer all zeros — a
   direct, mechanical confirmation of "no effect at init, effect grows as
   training proceeds."

## PyTorch: LoRA on a real (tiny) model

5. Implement the full `LoRALinear` class from `01_concepts.md`. Wrap one
   `nn.Linear` layer from Project 013's small GPT (or any small model) in
   it, freeze the original, and confirm
   `sum(p.numel() for p in model.parameters() if p.requires_grad)` is much
   smaller than the total parameter count.

6. Fine-tune the LoRA-wrapped model on a small new dataset (any small text
   sample) for a few steps, then **merge** the LoRA weights back into the
   original (`W_new = W + scale * B @ A`) and confirm a forward pass
   through the merged single matrix produces the same output as the
   original two-part (frozen `W` + LoRA adapter) computation, up to
   floating-point precision — verifying the merge operation is
   mathematically correct before you'd ever rely on it for deployment.
