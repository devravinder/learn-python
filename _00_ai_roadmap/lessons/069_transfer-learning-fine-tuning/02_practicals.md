# 02 — Practicals: Transfer Learning & Fine-Tuning

1. Load a small pretrained model (`gpt2`, the smallest variant) and
   generate text from a prompt **before** any fine-tuning
   (`model.generate(...)`) — establish a baseline for comparison.

2. Fine-tune it (full fine-tuning, all parameters) on a small custom text
   sample (a few hundred lines of a specific style — e.g. recipes, poems,
   or your own writing) for a few epochs at a small learning rate
   (`lr=5e-5`). Generate from the same prompt as Q1 afterward — does the
   style visibly shift toward your fine-tuning data?

3. **Deliberately induce catastrophic forgetting**: fine-tune for many more
   epochs than needed (e.g. 20 instead of 2-3) at a higher learning rate
   (e.g. `lr=5e-4`). Test the model on a completely unrelated, general
   prompt it should have handled well *before* fine-tuning (e.g. "The
   capital of France is"). Has quality on this unrelated prompt visibly
   degraded compared to the original pretrained model?

4. Compare memory usage (`torch.cuda.memory_allocated()` if on GPU, or
   estimate via Lesson 067's parameter-counting approach) between just
   loading the model for inference vs. loading it plus an `AdamW`
   optimizer ready for full fine-tuning. Confirm the fine-tuning setup
   needs noticeably more memory, matching Lesson 067/069's "4x parameter
   memory for Adam" estimate.

5. Repeat Q2's fine-tuning but freeze all parameters except the final
   `lm_head` layer (`for p in model.transformer.parameters(): p.requires_grad = False`).
   Compare the fine-tuned output quality/style-adaptation to Q2's full
   fine-tuning — is frozen-except-head noticeably more limited in how much
   it can adapt style, consistent with `01_concepts.md`'s feature-
   extraction-vs-full-fine-tuning tradeoff?

6. Write 3-4 sentences: given what you observed in Q3 (forgetting) and Q4
   (memory cost), explain why parameter-efficient fine-tuning (Lesson 070)
   is appealing even before considering its cost savings — what does
   *freezing* the original weights buy you directly, independent of the
   memory argument?
