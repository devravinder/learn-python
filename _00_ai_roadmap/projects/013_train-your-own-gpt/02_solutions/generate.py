"""Generate text from a trained GPT checkpoint.

Usage:
    python generate.py --prompt "The forest spirit" --temperature 0.8 --max_new_tokens 100
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from bpe_tokenizer import BPETokenizer
from model import GPT


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_value = values[..., -1, None]
        logits = torch.where(logits < min_value, torch.full_like(logits, float("-inf")), logits)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits[sorted_mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_indices, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temperature=1.0, top_k=0, top_p=0.0):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)[:, -1, :] / temperature
        logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="gpt_checkpoint.pt")
    parser.add_argument("--data", default="data/fallback_corpus.txt")
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.0)
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

    start_ids = tokenizer.encode(args.prompt)
    idx = torch.tensor([start_ids], dtype=torch.long)

    out = generate(model, idx, args.max_new_tokens, config["block_size"],
                    temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
