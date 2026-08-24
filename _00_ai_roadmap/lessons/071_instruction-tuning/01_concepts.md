# 01 — Concepts: Instruction Tuning

## Base models complete text; they don't inherently follow instructions

A pretrained base model (Module 11's training objective, at real scale)
given the prompt `"Write a haiku about autumn"` might just as easily
continue with *more prompts like it* (since that's a plausible continuation
of text that looks like a list of writing exercises) rather than actually
writing a haiku. It has no inherent notion of "the user wants me to *do*
this," only "what text plausibly comes next." **Instruction tuning**
(a form of **Supervised Fine-Tuning**, SFT) fixes this via training data
shaped explicitly like the desired behavior.

## The data: (instruction, response) pairs

```python
examples = [
    {"instruction": "Write a haiku about autumn.",
     "response": "Leaves drift silently\nGolden hues paint the cold ground\nWinter waits nearby"},
    {"instruction": "Explain photosynthesis in one sentence.",
     "response": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."},
    # ... thousands more, ideally covering diverse instruction types
]
```

Same underlying training mechanism as Module 11 (cross-entropy on
next-token prediction, Lesson 063) — what's different is **what the model
is trained to predict from what**.

## Chat templates: formatting instructions consistently

A **chat template** wraps instruction/response pairs (and, for multi-turn
data, full conversation histories) in a consistent format with special
tokens (Lesson 062) marking roles:

```
<|user|>
Write a haiku about autumn.
<|assistant|>
Leaves drift silently
Golden hues paint the cold ground
Winter waits nearby
<|endoftext|>
```

The model is trained on this **entire formatted sequence** using the exact
same next-token objective — but crucially, **loss is typically masked
(zeroed out) on the instruction/prompt portion**, only computed on the
response tokens:

```python
labels = input_ids.clone()
labels[:prompt_length] = -100   # -100 is PyTorch's convention for "ignore this position"
loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
```

**Why mask the prompt**: you don't want the model being trained to predict
the *user's* instruction text (that's not a skill you're trying to teach —
the model isn't supposed to predict what users will type) — only to
generate a good *response*, given the instruction as fixed context.

## Where the training data comes from

- **Human-written**: instructions and responses written/curated by people
  — high quality, expensive to scale.
- **Model-generated + filtered**: use a strong existing model to generate
  candidate instruction/response pairs, filter/edit for quality —
  much cheaper to scale, introduces some risk of propagating that model's
  own quirks/errors into the new model.
- **Public instruction-tuning datasets**: many exist openly (e.g. various
  "Alpaca-style" datasets) — a practical starting point for Project-scale
  experimentation without building a data pipeline from scratch.

## Instruction tuning changes behavior, not (mainly) knowledge

A common, important nuance: instruction tuning mostly teaches the model
**how to use** the knowledge/capability it already has from pretraining
(format, tone, task-following behavior), rather than adding substantial
new factual knowledge — the pretraining corpus (Module 11's massive-scale
version) is where most factual/world knowledge comes from. This is part of
why instruction-tuning datasets can be comparatively small (thousands to
low millions of examples) relative to pretraining corpora (billions to
trillions of tokens) and still produce a dramatic behavioral shift.

## LoRA + instruction tuning: the practical combination

Lesson 070's LoRA applies directly here — instruction-tune a pretrained
base model efficiently by freezing it and training small LoRA adapters on
instruction data, rather than full fine-tuning. This combination (base
model + LoRA instruction-tuning) is exactly how most practical,
resource-constrained fine-tuning of open models is actually done.

## Setting up Lesson 072

Instruction tuning teaches a model to produce *plausible, well-formatted*
responses. It doesn't have any explicit mechanism for learning that some
correct-format responses are **better** than other correct-format
responses (more helpful, safer, more aligned with what a human would
actually prefer) — that's exactly the gap Lesson 072's RLHF/DPO closes,
using human *preference* data rather than fixed target responses.
