# 01 — Questions

Use `distilgpt2` or `gpt2` (smallest variants) as your base model —
small enough to fine-tune on modest hardware.

1. Write your own custom instruction dataset: **at least 20** examples, all
   sharing a consistent, specific "persona" or task (e.g. always responds
   in pirate speak, always formats answers as bullet points, always
   answers as a specific fictional character — your choice, but make it
   something whose effect will be *obvious* in generated text). Format
   each with Lesson 071's chat template.

2. Load the base model and generate from 3 test prompts **before** any
   fine-tuning — save these baseline outputs for comparison.

3. Implement LoRA (Lesson 070) on the model's attention projection layers
   (`q_proj`/`v_proj`, or use the `peft` library's `LoraConfig` +
   `get_peft_model` if you prefer not to hand-write the wrapping). Report
   `trainable_params / total_params` — confirm it's a small fraction.

4. Fine-tune using Lesson 071's masked-loss approach (loss computed only
   on response tokens, not the instruction/prompt portion) on your Q1
   dataset for a handful of epochs.

5. Generate from the **same 3 test prompts** from Q2 using the fine-tuned
   model. Compare side by side with the baseline outputs — does the
   persona/behavior you targeted show up clearly?

6. Test on **one prompt style very different from your training examples**
   (a different topic/domain than anything in your 20 training examples).
   Does the persona/behavior still show up, or does it only apply to
   prompts resembling training data closely? Report honestly either way —
   this is a real generalization check, not a formality.
