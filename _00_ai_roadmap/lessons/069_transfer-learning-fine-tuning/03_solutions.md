# 03 — Solutions: Transfer Learning & Fine-Tuning

*(Requires `transformers`/PyTorch — not executable in the authoring
sandbox; code is carefully written but not independently run. Verify by
actually running it.)*

## 1. Baseline generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "The best way to spend a weekend is"
inputs = tokenizer(prompt, return_tensors="pt")
baseline_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.8)
print(tokenizer.decode(baseline_output[0]))
```

## 2. Fine-tuning on a specific style

```python
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=64):
        ids = tokenizer("\n".join(texts), return_tensors="pt")["input_ids"][0]
        self.examples = [ids[i:i+block_size] for i in range(0, len(ids) - block_size, block_size)]
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]

# fine_tune_texts: your small custom-style corpus, list of strings
dataset = TextDataset(fine_tune_texts, tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in dataset:
        batch = batch.unsqueeze(0)
        outputs = model(batch, labels=batch)   # HF models compute shifted CE loss internally when labels given
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

finetuned_output = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.8)
print(tokenizer.decode(finetuned_output[0]))
```

Compare `baseline_output` and `finetuned_output` on the same prompt —
expect the fine-tuned version's vocabulary/style to shift noticeably
toward the fine-tuning corpus's characteristic phrasing after even a few
epochs on a small dataset, since the base model already has strong general
language ability and only needs a nudge toward the target style.

## 3. Inducing catastrophic forgetting

```python
optimizer_aggressive = torch.optim.AdamW(model.parameters(), lr=5e-4)
for epoch in range(20):
    for batch in dataset:
        batch = batch.unsqueeze(0)
        outputs = model(batch, labels=batch)
        outputs.loss.backward()
        optimizer_aggressive.step()
        optimizer_aggressive.zero_grad()

general_prompt = "The capital of France is"
general_inputs = tokenizer(general_prompt, return_tensors="pt")
degraded_output = model.generate(**general_inputs, max_new_tokens=20)
print(tokenizer.decode(degraded_output[0]))
```

With this much more aggressive fine-tuning (more epochs, 10x the learning
rate), expect the model's output on the unrelated general-knowledge prompt
to become noticeably worse or stranger than the original pretrained
model's — often veering into the fine-tuning corpus's style/vocabulary
even for unrelated prompts, a direct, observable demonstration of
catastrophic forgetting rather than just a theoretical risk.

## 4. Memory comparison

```python
import torch

model_inference_only = AutoModelForCausalLM.from_pretrained("gpt2")
n_params = sum(p.numel() for p in model_inference_only.parameters())
inference_memory_gb = n_params * 4 / 1e9   # fp32 weights only

finetuning_memory_gb = inference_memory_gb * 4   # weights + grad + m + v (Lesson 067)
print(f"inference-only: ~{inference_memory_gb:.2f} GB")
print(f"full fine-tuning setup: ~{finetuning_memory_gb:.2f} GB")
```

For GPT-2 small (~124M params), expect roughly ~0.5GB for inference-only
weights vs. ~2GB for a full fine-tuning setup — matching Lesson 067's
"~4x" rule of thumb directly, now applied to a real pretrained model
rather than a hypothetical one.

## 5. Frozen-except-head fine-tuning

```python
for p in model.transformer.parameters():
    p.requires_grad = False
# only model.lm_head remains trainable

optimizer_head_only = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=5e-5
)
# ... same training loop as Q2 ...
```

Expect noticeably **less** stylistic adaptation than full fine-tuning
(Q2) — the frozen transformer body can't adjust *how* it represents
language internally, only the final projection to vocabulary logits can
change, a much more limited lever than full fine-tuning's ability to
reshape internal representations throughout the network.

## 6. Why freezing weights matters beyond memory savings

Freezing the original pretrained weights means the model's general
language capability is **structurally protected** — there's no mechanism
by which fine-tuning can degrade it, since those weights simply don't
change at all. This directly eliminates catastrophic forgetting risk (Q3)
for whatever remains frozen, independent of how carefully you tune the
learning rate or epoch count for the fine-tuned portion — a qualitatively
different (and more reliable) safety property than "be careful with
hyperparameters," which is exactly why parameter-efficient methods like
LoRA (Lesson 070) are often preferred even when memory isn't the binding
constraint.
