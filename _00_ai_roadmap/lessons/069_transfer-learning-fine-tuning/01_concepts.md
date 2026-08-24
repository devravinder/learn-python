# 01 — Concepts: Transfer Learning & Fine-Tuning

## Why start from a pretrained model at all

Project 013 trained on (at best) a few million tokens. GPT-3-class models
train on hundreds of billions to trillions of tokens, learning general
language structure, world knowledge, and broad capability no hobby-scale
run can replicate. **Fine-tuning** takes that pretrained capability and
adapts it to a specific task/domain/style with vastly less additional data
and compute than pretraining from scratch — exactly Lesson 044's CNN
transfer-learning idea (reuse pretrained general features, retrain only
what's task-specific), applied to language models.

## Full fine-tuning: update every weight

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# continue training on YOUR data, same objective as Lesson 063
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)   # note: much smaller LR than pretraining
```

Mechanically identical to Lesson 065's training loop — same cross-entropy
loss (Lesson 063), same optimizer family, just initialized from pretrained
weights instead of random ones, and a **much smaller learning rate** (full
fine-tuning risks catastrophically overwriting useful pretrained
knowledge if the learning rate is too aggressive — precisely the
"catastrophic forgetting" risk flagged back in Project 012).

## Full fine-tuning's practical problem: memory

Fine-tuning **all** of a large model's parameters requires storing
gradients and optimizer state for **every** parameter (Lesson 067's memory
accounting: roughly 4x the raw parameter memory for Adam) — for a
billion-plus parameter model, this can easily exceed what a single
consumer GPU can hold, even though the *pretrained model itself* might fit
for inference alone. This is the direct motivation for Lesson 070's
parameter-efficient methods.

## Catastrophic forgetting, precisely

Fine-tuning on a narrow dataset can degrade the model's broader pretrained
capability — improving performance on your specific task while making it
measurably worse at things it could do well before fine-tuning at all.
Mitigations:
- **Lower learning rate** than pretraining (smaller updates, less
  disruption).
- **Fewer epochs** — stop as soon as the target task is learned, don't
  keep training past that point.
- **Mixing in some general-purpose data** alongside your fine-tuning data,
  so the model doesn't forget general capability while specializing.
- **Parameter-efficient fine-tuning** (Lesson 070) — freezing most of the
  original weights entirely sidesteps most forgetting risk by
  construction, since the original weights literally don't change.

## Full fine-tuning vs. feature extraction (the CNN analogy, made explicit)

- **Feature extraction** (Lesson 044's "freeze everything except the final
  layer"): freeze the pretrained model, train only a small new head — fast,
  cheap, but limited adaptation capacity.
- **Full fine-tuning**: update every weight — maximum adaptation capacity,
  maximum cost and forgetting risk.
- **Parameter-efficient fine-tuning** (Lesson 070): a middle ground —
  freeze the original weights, add a small number of new trainable
  parameters that adapt the model's behavior — most of full fine-tuning's
  adaptation benefit, a small fraction of its cost.

## Choosing a base model to fine-tune

Practical considerations: model size (bigger = more capable but more
compute to fine-tune and serve), license (some open-weight models restrict
commercial use), and whether it's a **base** (raw pretrained, completes
text) or **instruction-tuned** (already fine-tuned to follow instructions,
Lesson 071) checkpoint — fine-tuning further on top of an already
instruction-tuned model is common and often the more practical starting
point for building an assistant-style application specifically.

## What "fine-tuning" means for Module 12 as a whole

This lesson: adapt a pretrained LM's *raw completion* behavior toward a
narrower domain/style (still just "predict next token," same objective as
Lesson 063). Lesson 071 changes the *task shape* itself (instruction
following: input is a prompt, target is a response, not simply "continue
this text"). Lesson 072 changes the *training signal* entirely (human
preference rather than next-token likelihood). Each lesson is a
genuinely different kind of adaptation, not just "more of the same
fine-tuning" — worth keeping the distinctions clear.
