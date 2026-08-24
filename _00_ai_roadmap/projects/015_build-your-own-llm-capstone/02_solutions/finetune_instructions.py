"""Stage 3: instruction fine-tune Project 013's own pretrained GPT, using
Lesson 071's masked-loss approach - no external HF model involved.

Usage:
    python finetune_instructions.py --checkpoint path/to/gpt_checkpoint.pt --data path/to/corpus.txt
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_013 = Path(__file__).parents[2] / "013_train-your-own-gpt" / "02_solutions"
sys.path.insert(0, str(PROJECT_013))
from bpe_tokenizer import BPETokenizer  # noqa: E402
from model import GPT  # noqa: E402

# --- Stage 3, Q2: a small instruction dataset in the model's own domain ---
INSTRUCTION_EXAMPLES = [
    ("Continue this story:", "The forest spirit finally reached the misty forest, wondering what came next."),
    ("Write a short tale about a knight.", "The last knight bravely entered the crumbling castle, and found something unexpected."),
    ("Describe a scene in a forest.", "The wandering scholar quietly watched the silent river, as the sun began to set."),
    # ... add 20-50 real examples matched to your own pretraining corpus's domain
]


def format_example(instruction, response=""):
    return f"<|user|>\n{instruction}\n<|assistant|>\n{response}"


def finetune(model, tokenizer, examples, epochs=10, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    for epoch in range(epochs):
        total_loss = 0.0
        for instruction, response in examples:
            prompt_text = format_example(instruction) + "\n"
            full_text = prompt_text + response

            prompt_ids = tokenizer.encode(prompt_text)
            full_ids = tokenizer.encode(full_text)
            prompt_len = len(prompt_ids)

            x = torch.tensor([full_ids[:-1]], dtype=torch.long)
            y = torch.tensor([full_ids[1:]], dtype=torch.long)

            # mask the prompt portion of the TARGETS (Lesson 071)
            mask_len = max(0, prompt_len - 1)
            y_masked = y.clone()
            y_masked[:, :mask_len] = -100

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y_masked.reshape(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 2 == 0:
            print(f"epoch {epoch}: avg masked loss {total_loss / len(examples):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True, help="original pretraining corpus (for tokenizer path)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]

    tokenizer = BPETokenizer()
    tokenizer.load(Path(args.data).with_suffix(".tokenizer.json"))

    model = GPT(
        vocab_size=len(tokenizer.vocab), d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], d_ff=config["d_ff"], max_len=config["block_size"],
    )
    model.load_state_dict(checkpoint["model_state"])

    print("=== BEFORE instruction fine-tuning ===")
    test_instructions = ["Continue this story:", "Write a short tale about a dragon.", "Describe a quiet morning."]
    for instr in test_instructions:
        prompt_ids = tokenizer.encode(format_example(instr) + "\n")
        idx = torch.tensor([prompt_ids])
        with torch.no_grad():
            for _ in range(30):
                logits = model(idx[:, -config["block_size"]:])[:, -1, :]
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, next_id], dim=1)
        print(f"{instr!r} -> {tokenizer.decode(idx[0].tolist())!r}")

    finetune(model, tokenizer, INSTRUCTION_EXAMPLES, epochs=args.epochs)

    torch.save({"model_state": model.state_dict(), "config": config}, "gpt_instruction_tuned.pt")

    print("\n=== AFTER instruction fine-tuning ===")
    for instr in test_instructions:
        prompt_ids = tokenizer.encode(format_example(instr) + "\n")
        idx = torch.tensor([prompt_ids])
        with torch.no_grad():
            for _ in range(30):
                logits = model(idx[:, -config["block_size"]:])[:, -1, :]
                next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                idx = torch.cat([idx, next_id], dim=1)
        print(f"{instr!r} -> {tokenizer.decode(idx[0].tolist())!r}")


if __name__ == "__main__":
    main()
