# 01 — Concepts: Sampling & Generation

## The autoregressive generation loop (Lesson 047, at GPT scale)

Exactly Lesson 047's Seq2Seq decoder loop: predict one token, append it to
the input, repeat.

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]           # crop to the last block_size tokens
        logits = model(idx_cond)
        logits = logits[:, -1, :]                    # only the LAST position's prediction matters
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # SAMPLE, don't just take argmax
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

`idx[:, -block_size:]`: since the model only ever learned to handle
sequences up to `block_size` long (Lesson 064's positional embedding is
fixed to that length), once generated text exceeds `block_size`, you must
crop to the most recent `block_size` tokens — the model has no memory
beyond what it can currently attend to.

## Greedy decoding vs sampling

- **Greedy**: always pick `argmax(probs)` — deterministic, but tends to
  produce repetitive, "safe" text (a known failure mode: greedy decoding
  can get stuck in loops, repeating the same phrase).
- **Sampling** (`torch.multinomial`): draw from the actual probability
  distribution — introduces controlled randomness, generally produces more
  natural, varied text, at the cost of occasional lower-probability (and
  occasionally nonsensical) choices.

## Temperature: controlling how "sharp" the distribution is (Lesson 036, revisited)

```python
logits = logits / temperature
probs = F.softmax(logits, dim=-1)
```

- `temperature < 1`: sharpens the distribution — closer to greedy, more
  conservative/repetitive, fewer surprises.
- `temperature = 1`: unmodified model distribution.
- `temperature > 1`: flattens the distribution — more random/diverse,
  higher risk of incoherent output.

This is exactly Lesson 036's temperature-scaled softmax, now controlling
actual text generation instead of a toy example.

## Top-k sampling: only consider the k most likely tokens

```python
def top_k_filter(logits, k):
    values, indices = torch.topk(logits, k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, indices, values)
    return filtered
```

Zero out (set to `-inf`, then softmax) every token outside the top `k` —
prevents sampling from the "long tail" of very-low-probability, often
nonsensical tokens, while still allowing controlled randomness among the
genuinely plausible candidates.

## Top-p (nucleus) sampling: a dynamic version of top-k

Instead of a fixed count `k`, keep the smallest set of tokens whose
cumulative probability exceeds threshold `p` (e.g. 0.9) — adapts
automatically to how "confident" the distribution is at each step: a very
peaked distribution keeps very few tokens, a very flat one keeps more.
Generally considered a more robust default than fixed-`k` top-k, and used
by most production LLM APIs.

```python
def top_p_filter(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()   # keep the first token that crosses p
    sorted_mask[..., 0] = False
    sorted_logits[sorted_mask] = float("-inf")
    logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits
```

## Combining strategies (what real systems actually do)

Most production text-generation setups combine **temperature + top-k (or
top-p)** together — temperature adjusts overall randomness, top-k/top-p
prevents that randomness from ever reaching into clearly-bad tail tokens.
There's no single "correct" setting — it's a genuine product/UX decision
depending on whether you want more creative or more conservative output.

## KV-caching (a preview — full treatment in Lesson 067/074)

The naive `generate` loop above **recomputes attention over the entire
growing sequence at every single step** — wasteful, since most of that
computation (keys and values for all the already-generated tokens) doesn't
actually change between steps. **KV-caching** stores previously-computed
keys/values and only computes the new token's contribution each step — a
major inference speedup used by every real LLM serving system, covered
properly once you reach Lesson 067's scaling techniques and Lesson 074's
inference optimization.

## What good generated text tells you (and what it doesn't)

Watching your own trained model generate increasingly coherent text over
training (random characters → word-fragments → real words → short
plausible phrases) is one of the most rewarding parts of this whole
curriculum — genuinely worth pausing to enjoy once you get there. But
remember: even a very fluent small model has no real "understanding" in
any deep sense — it has learned to model the statistical structure of its
training text extremely well, at whatever scale it was trained. Module 12
onward is about steering that raw capability toward being useful,
instructable, and aligned with what you actually want from it.
