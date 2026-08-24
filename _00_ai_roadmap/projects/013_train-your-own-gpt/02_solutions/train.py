"""Train a small GPT on a custom text corpus.

Usage:
    python data/generate_fallback_corpus.py        # or supply your own text file
    python train.py --data data/fallback_corpus.txt --vocab_size 512 --max_steps 3000
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer
from model import GPT


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            logits = model(xb)
            losses[k] = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        out[name] = losses.mean().item()
    model.train()
    return out


def get_lr(step, warmup_steps, max_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fallback_corpus.txt")
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = Path(args.data).read_text()

    tokenizer_path = Path(args.data).with_suffix(".tokenizer.json")
    tokenizer = BPETokenizer()
    if tokenizer_path.exists():
        tokenizer.load(tokenizer_path)
        print(f"loaded existing tokenizer from {tokenizer_path}")
    else:
        print(f"training tokenizer (vocab_size={args.vocab_size})...")
        tokenizer.train(text, vocab_size=args.vocab_size, verbose=True)
        tokenizer.save(tokenizer_path)
        print(f"saved tokenizer to {tokenizer_path}")

    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(text)} chars -> {len(data)} tokens (train {len(train_data)}, val {len(val_data)})")

    model = GPT(
        vocab_size=len(tokenizer.vocab), d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff, max_len=args.block_size,
    ).to(device)
    print(f"model parameters: {model.num_parameters():,}")

    # sizing sanity check against Lesson 068's Chinchilla-ish ratio
    suggested_n = len(train_data) / 20
    print(f"Chinchilla-suggested parameter count for this corpus: ~{suggested_n:,.0f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1)

    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, args.block_size, args.batch_size, 20, device)
            ppl = {k: math.exp(v) for k, v in losses.items()}
            print(f"step {step}: train_loss={losses['train']:.4f} val_loss={losses['val']:.4f} "
                  f"train_ppl={ppl['train']:.2f} val_ppl={ppl['val']:.2f} lr={lr:.2e}")

    torch.save({"model_state": model.state_dict(), "config": vars(args)}, "gpt_checkpoint.pt")
    print("saved checkpoint to gpt_checkpoint.pt")


if __name__ == "__main__":
    main()
