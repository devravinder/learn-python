# 01 — Concepts: Build a GPT, Part 1 — Architecture & Data

## Character-level tokenization (the simplest choice, deliberately)

Following Lesson 063a's warm-up and Karpathy's own progression: start with
one-token-per-character (Lesson 055's simplest option), not BPE (Lesson
062) — fewer moving parts while you're getting the training loop right for
the first time. Lesson 068a upgrades to a real BPE tokenizer once the core
model works.

```python
text = open("input.txt").read()   # any plain text file - a book, a corpus, your own writing
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
```

## Train/validation split

```python
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
```

A held-out validation set (Lesson 017/024) is what tells you honestly
whether the model is learning generalizable structure or just memorizing
the training text — track validation loss throughout Lesson 065's training.

## Block size and the random-chunk batching scheme

`block_size` (also called context length or `max_len`, matching Lesson
060's GPT class) is the maximum sequence length the model trains on and can
attend across. Training data is far longer than one `block_size` chunk, so
every training step samples **random contiguous chunks**:

```python
def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # shifted by 1 - Lesson 063's targets
    return x, y
```

Notice `y` is exactly `x` shifted by one position — Lesson 063's
shift-by-one target, applied to a whole batch of random chunks at once.
Within one `block_size`-length chunk, the model is simultaneously trained
on **every** sub-length prediction task (predict char 2 from char 1;
predict char 5 from chars 1-4; etc.) — one chunk yields `block_size`
separate training signals via the causal mask (Lesson 059), not just one.

## Assembling the GPT (reusing Lesson 060 directly)

```python
model = GPT(
    vocab_size=vocab_size,
    d_model=64,       # small - deliberately modest for a fast first training run
    n_heads=4,
    n_layers=4,
    d_ff=256,
    max_len=block_size,
)
```

**Nothing new architecturally** — this is exactly Lesson 060's `GPT` class.
The only genuinely new things in this lesson are (1) real data instead of
random tensors, and (2) the random-chunk batching scheme. Keeping the
architecture unchanged and only wiring up data is a deliberate way to
isolate what's actually new to learn at each step.

## Choosing a first training corpus

Any plain text works, but a few practical notes for a first run:
- **Size**: even a few hundred KB (e.g. a single book's text) is enough to
  see a small character-level GPT learn recognizable structure (words,
  some grammar) within a reasonable training time on a laptop/free-tier
  GPU.
- **A single consistent style** (one author, one genre) tends to produce
  more visibly coherent-looking generated text at small model scale than a
  highly varied corpus, since the model has less stylistic variety to
  spread its limited capacity across.
- Public-domain text (e.g. Project Gutenberg books) is a common, easy,
  copyright-safe choice for this kind of experiment.

## Sanity-checking shapes before training anything

Before writing the training loop (Lesson 065), always verify:

```python
xb, yb = get_batch(train_data, block_size=32, batch_size=4)
print(xb.shape, yb.shape)   # (4, 32) both

logits = model(xb)
print(logits.shape)          # (4, 32, vocab_size)
```

This kind of shape check — cheap, fast, catches a large fraction of real
bugs immediately — is worth making a habit before ever starting a real
training run, especially once runs take minutes/hours rather than seconds.
