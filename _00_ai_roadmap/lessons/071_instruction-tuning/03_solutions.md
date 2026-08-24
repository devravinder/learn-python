# 03 — Solutions: Instruction Tuning

*(Q1-2's code was actually run to produce the numbers below.)*

## 1–2. Loss masking, verified

```python
import math

def masked_cross_entropy(probs, masked):
    losses = [-math.log(p) for p, m in zip(probs, masked) if not m]
    return sum(losses) / len(losses)

probs = [0.9, 0.9, 0.9, 0.2, 0.1]     # positions 0-2 = prompt (high confidence, easy), 3-4 = response
masked = [True, True, True, False, False]

loss_masked = masked_cross_entropy(probs, masked)
manual = (-math.log(0.2) + -math.log(0.1)) / 2
print(loss_masked, manual)   # 1.956, 1.956 -- exact match

loss_unmasked = masked_cross_entropy(probs, [False]*5)
print(loss_unmasked)   # 0.846
```

**Actual output: masked loss ≈ 1.956 (exactly matching the manual
calculation using only positions 3-4); unmasked loss ≈ 0.846 — noticeably
*lower*.**

This is the crux of why masking matters, and why forgetting it is a
*silent* bug rather than an obvious crash: the unmasked version looks
**better** (lower loss) purely because the easy, high-confidence prompt
positions (0.9 probability each) drag the average down favorably, hiding
how poorly the model is actually doing on the response tokens (0.2, 0.1)
that are the entire point of the training signal. A training run with this
bug would report an artificially rosy loss curve while the model learns
essentially nothing useful about generating good responses.

## 3. Chat template formatting

```python
def format_example(instruction, response):
    return f"<|user|>\n{instruction}\n<|assistant|>\n{response}<|endoftext|>"

examples = [
    ("Write a haiku about autumn.", "Leaves drift silently\nGolden hues paint the cold ground\nWinter waits nearby"),
    ("Explain photosynthesis in one sentence.", "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("List three primary colors.", "Red, blue, and yellow are the three primary colors."),
]
for instr, resp in examples:
    print(format_example(instr, resp))
    print("---")
```

## 4. Finding the mask boundary

```python
formatted = format_example(*examples[0])
tokens = formatted.split()   # word-level stand-in tokenizer for this exercise

marker = "<|assistant|>"
prompt_length = tokens.index(marker) + 1   # everything up to and including the marker is "prompt"
print(prompt_length, tokens[:prompt_length])
```

This index is exactly the boundary you'd pass to `labels[:prompt_length] = -100`
in Q5 — everything up to and including the `<|assistant|>` marker is
masked (the model isn't trained to predict it), everything after
(the actual response) contributes to the loss.

## 5. HF-style masked training step (PyTorch)

```python
import torch
import torch.nn.functional as F

input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"]
labels = input_ids.clone()
labels[:, :prompt_length] = -100

logits = model(input_ids).logits
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

To confirm masking is working: construct a controlled test where the
"model's" logits for the prompt positions are deliberately terrible
(e.g. all zeros, definitely not predicting the right prompt tokens) while
the response-position logits are set up to predict correctly. If `loss`
comes out low (reflecting only the good response-position predictions),
masking is working as intended; if it's high (penalized by the
deliberately-bad prompt predictions), the `ignore_index`/masking isn't
actually being applied.

## 6. Fine-tuning and testing generalization

Expect the small instruction-tuned model, even on a handful of examples,
to at least attempt the **format** of an instruction-following response
(structured, addressed to the instruction, ending appropriately) on a
new instruction it wasn't trained on — even if the *content* quality is
limited by how little training data and model capacity were used. This
distinction (learned the *shape* of the desired behavior quickly; content
quality scales with data/model size) is exactly `01_concepts.md`'s point
about instruction tuning teaching *how to use* existing capability rather
than injecting large amounts of new capability.
