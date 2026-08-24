# 01 — Concepts: Tokenization for LLMs

## Byte-level BPE: GPT's actual approach

Lesson 055's BPE trainer merged **characters**. Real GPT-2/GPT-3/GPT-4
tokenizers merge **raw UTF-8 bytes** instead. This has one huge practical
advantage: there are only **256 possible byte values**, so the base
vocabulary (before any merges) can represent **any possible text input in
any language, including emoji and malformed text** — with **zero true
out-of-vocabulary tokens, ever**. A character-level base vocabulary would
need a separate entry for every Unicode character that might appear
(tens of thousands), and would still fail on truly novel byte sequences;
byte-level BPE sidesteps this completely by starting from the smallest
possible universal alphabet.

```python
text = "café 🎉"
byte_sequence = text.encode("utf-8")
print(list(byte_sequence))   # a sequence of integers 0-255, always representable
```

## The BPE training algorithm, at byte scale (recap + what's different)

Exactly Lesson 055's algorithm — repeatedly find the most frequent adjacent
pair and merge it — just starting from bytes instead of characters, and run
for far more merges (GPT-2 uses ~50,000 merges, producing a ~50,257-token
vocabulary). Lesson 068a implements this exact algorithm, at byte scale,
completely from scratch.

## Vocabulary size: a real tradeoff, not a free parameter

- **Smaller vocabulary** (e.g. a few thousand tokens): shorter merge list,
  faster to train the tokenizer itself, but text encodes into **more**
  tokens (closer to character-level) — longer sequences for the model to
  process, more compute per unit of text, and the model has less "budget"
  per token to encode meaning.
- **Larger vocabulary** (e.g. 100,000+ tokens): text encodes into
  **fewer** tokens (more whole-word coverage), shorter sequences — but a
  bigger embedding table and output projection (both scale with vocabulary
  size, Lesson 060's `nn.Embedding`/`nn.Linear` at the vocab dimension),
  and rarer tokens get less training signal individually.

Real LLMs settle in the 30,000-100,000+ range as a practical middle ground;
the "right" choice depends on the target languages, domain, and model
scale.

## SentencePiece: language-agnostic tokenization

Many BPE implementations assume text is already split into words by
whitespace before merging begins — a reasonable assumption for English,
but not for languages without whitespace-delimited words (Japanese,
Chinese, Thai). **SentencePiece** treats the input as a raw stream of
characters/bytes (including whitespace, encoded as a special marker
character, commonly `▁`), and trains directly on that — no
language-specific pre-tokenization assumption baked in. Used by LLaMA,
T5, and many multilingual models.

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(input="corpus.txt", model_prefix="tok", vocab_size=8000)
sp = spm.SentencePieceProcessor(model_file="tok.model")
print(sp.encode("Hello, world!", out_type=str))
```

## Unigram Language Model tokenization (SentencePiece's other mode)

Beyond BPE, SentencePiece also supports a **unigram LM** algorithm: start
with a large candidate set of subword pieces, then iteratively **remove**
pieces that contribute least to the training corpus's likelihood under a
unigram (independence-assumption, Lesson 006) probability model — the
reverse direction from BPE's bottom-up merging (top-down pruning instead).
Produces broadly similar practical results to BPE; the specific algorithm
matters less than having *some* principled subword segmentation.

## Special tokens, revisited for LLMs specifically

- `<|endoftext|>` (GPT-style): marks document boundaries in the training
  corpus, so the model learns "a new, unrelated document is starting" —
  important since LLM pretraining corpora concatenate many separate
  documents.
- Chat/instruction-tuned models (Lesson 071) add further special tokens
  marking turn boundaries (e.g. `<|user|>`, `<|assistant|>`) — not needed
  for base pretraining (Module 11) but essential once you fine-tune for
  chat-style interaction.

## Tokenization artifacts worth knowing about

- **Numbers**: BPE often splits numbers inconsistently (e.g. "380" might
  be one token, "381" might split into "38"+"1") purely based on training-
  corpus frequency — a well-documented cause of LLMs being unexpectedly
  bad at basic arithmetic, since the same numeric value can look
  structurally different to the model depending on incidental tokenization.
- **Leading spaces matter**: GPT-2's byte-level BPE treats `" world"` (with
  a leading space) as a different token sequence than `"world"` — an
  easy-to-miss detail when debugging unexpected tokenization output.
- **Case sensitivity**: unlike classical NLP preprocessing (Lesson 055's
  lowercasing), LLM tokenizers typically preserve case — "The" and "the"
  are different tokens, since case can carry real meaning (start of
  sentence, proper nouns) that lowercasing would destroy.

## Why Lesson 068a builds this from scratch, not just `import tiktoken`

Using a library's tokenizer is instant and correct, but a from-scratch
implementation (Karpathy's `minbpe` approach) is what makes the previous
sections' claims fully concrete: you'll watch merges happen, choose your
own vocabulary size, and see firsthand how tokenization artifacts (like
the numeric splitting issue above) arise directly from training-corpus
statistics — completing the "build it yourself to really understand it"
approach this entire curriculum has followed since Lesson 038.
