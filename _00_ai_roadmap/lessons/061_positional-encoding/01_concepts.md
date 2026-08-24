# 01 — Concepts: Positional Encoding

## Recall the problem (from Lesson 060)

Attention computes `Q @ K^T` — a set of pairwise similarities with no
inherent notion of "position 3 comes before position 7." Without adding
positional information somewhere, shuffling input tokens would produce the
same set of attention outputs, just correspondingly shuffled — clearly
wrong for language, where order carries meaning ("dog bites man" vs "man
bites dog").

## Sinusoidal positional encoding — the original scheme

For position `pos` and embedding dimension index `i` (out of `d_model`
total):

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each dimension oscillates at a different frequency — low dimensions
oscillate fast (change a lot between adjacent positions), high dimensions
oscillate slowly (change gradually over many positions) — together forming
a unique "fingerprint" vector per position, added directly to the token
embedding.

```python
import numpy as np

def sinusoidal_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** (2 * (i // 2) / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe
```

## Why sine/cosine specifically: a useful mathematical property

For any fixed offset `k`, `PE(pos+k)` can be expressed as a **linear
function** of `PE(pos)` (a consequence of the angle-addition trig
identities: `sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`, similarly for
`cos`). This means the *relative* position between two tokens is encoded
in a way the model can, in principle, learn to extract via a simple linear
operation — not just "where am I," but "how far is that other token from
me," which is usually what actually matters for language.

## RoPE (Rotary Position Embedding) — what most modern LLMs actually use

Instead of *adding* a positional vector to the embedding, RoPE **rotates**
the query and key vectors by an angle proportional to their position,
directly inside the attention computation:

```
q_rotated = R(pos_q) @ q
k_rotated = R(pos_k) @ k
```

where `R(pos)` is a rotation matrix (Lesson 011's rotation matrices,
literally the same object) with angle depending on position and a
per-dimension frequency (similar spirit to sinusoidal encoding's varying
frequencies). The key property: `(R(pos_q) @ q) · (R(pos_k) @ k)` depends
**only on `pos_q - pos_k`** (the relative distance), not on the absolute
positions — attention scores naturally become relative-position-aware,
directly in the dot product itself, without needing the model to learn to
extract that relationship the way sinusoidal encoding merely makes
*possible*.

## Why RoPE won out in practice

- **Better length extrapolation**: models trained on RoPE tend to
  generalize somewhat better to sequence lengths longer than seen in
  training compared to learned positional embeddings (though not
  unlimited — a very active research area, e.g. "YaRN" and other RoPE
  extensions specifically target this).
- **Relative position built into the dot product itself**, rather than
  merely learnable from an additive signal.
- **No extra parameters** (unlike learned positional embeddings) and no
  separate vector to add (unlike sinusoidal) — it modifies Q/K directly
  inside attention.

## Learned positional embeddings — the simplest option (what GPT-2 used)

```python
pos_embed = nn.Embedding(max_seq_len, d_model)   # exactly Lesson 060's GPT class
```

Simplest to implement (already used in Lesson 060's `GPT` class), but
fixed to `max_seq_len` — cannot handle sequences longer than whatever was
set at training time at all (no extrapolation, unlike sinusoidal/RoPE).
Fine for a first working GPT (Project 013 starts here for simplicity);
worth knowing RoPE is the more modern choice if you want to extend the
project further.

## Practical takeaway for Module 11

Project 013's first working GPT uses learned positional embeddings
(simplest, matches Lesson 060's code directly) to keep the initial
build's moving parts minimal. Understanding sinusoidal encoding and RoPE
here means you can recognize and swap in either upgrade later — the rest
of the architecture (attention, feedforward, residuals) doesn't change at
all when you change how position is encoded, since positional information
only ever touches the input embeddings or the Q/K vectors, never anything
downstream.
