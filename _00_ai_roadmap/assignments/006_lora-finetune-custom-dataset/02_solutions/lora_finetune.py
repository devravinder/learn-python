"""LoRA instruction fine-tuning of a small open model (distilgpt2), using
the `peft` library for LoRA (Lesson 070) and manual loss masking
(Lesson 071).

Usage:
    pip install transformers peft
    python lora_finetune.py
"""
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"

# --- Q1: a toy "pirate persona" instruction dataset ---
TRAINING_EXAMPLES = [
    ("What is the capital of France?", "Arrr, 'tis Paris, ye landlubber, the city o' lights!"),
    ("How do I make tea?", "Boil yer water, matey, steep the leaves, and drink up before the storm hits!"),
    ("What's 2 plus 2?", "Arrr, tis four, as sure as the tide comes in!"),
    ("Tell me about the weather.", "The winds be fierce today, cap'n, best batten down the hatches!"),
    ("What is a computer?", "A magical chest o' thinkin' metal, arrr, used by landlubbers for their business!"),
    # ... (add 15+ more for a real run; kept short here for readability)
]
TEST_PROMPTS = ["What is the capital of Japan?", "How do I bake bread?", "What is gravity?"]


def format_example(instruction, response=""):
    return f"<|user|>\n{instruction}\n<|assistant|>\n{response}"


def generate(model, tokenizer, prompt, max_new_tokens=40):
    inputs = tokenizer(format_example(prompt), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                          temperature=0.8, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # --- Q2: baseline generations ---
    print("=== BEFORE fine-tuning ===")
    baseline = {p: generate(model, tokenizer, p) for p in TEST_PROMPTS}
    for p, r in baseline.items():
        print(f"{p!r} -> {r!r}")

    # --- Q3: wrap with LoRA ---
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["c_attn"],   # distilgpt2's combined qkv projection
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # reports trainable/total directly

    # --- Q4: masked-loss fine-tuning ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(10):
        total_loss = 0.0
        for instruction, response in TRAINING_EXAMPLES:
            prompt_text = format_example(instruction) + "\n"
            full_text = prompt_text + response + tokenizer.eos_token

            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
            prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

            labels = full_ids.clone()
            labels[:, :prompt_len] = -100

            outputs = model(full_ids, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
        if epoch % 2 == 0:
            print(f"epoch {epoch}: avg loss {total_loss / len(TRAINING_EXAMPLES):.4f}")

    # --- Q5: compare after fine-tuning ---
    print("\n=== AFTER fine-tuning ===")
    for p in TEST_PROMPTS:
        print(f"{p!r} -> {generate(model, tokenizer, p)!r}")

    # --- Q6: generalization test on an out-of-domain prompt ---
    novel_prompt = "Explain how photosynthesis works."
    print("\n=== Generalization test ===")
    print(f"{novel_prompt!r} -> {generate(model, tokenizer, novel_prompt)!r}")


if __name__ == "__main__":
    main()
