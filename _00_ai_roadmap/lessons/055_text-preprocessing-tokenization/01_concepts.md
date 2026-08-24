# 01 — Concepts: Text Preprocessing & Tokenization

## The preprocessing pipeline (classical NLP, what Project 003 already used)

- **Lowercasing**: `"Free"` and `"free"` treated the same — reduces
  vocabulary size, though it discards information (e.g. proper nouns,
  acronyms) that sometimes matters.
- **Punctuation/whitespace handling**: strip or separate punctuation as its
  own token, depending on the task.
- **Stopword removal**: dropping very common, low-information words
  ("the," "a," "is") — standard for classical bag-of-words methods
  (Project 003), but **not** used for modern neural/Transformer models,
  which handle common words fine via learned representations and often
  benefit from keeping full sentence structure intact.
- **Stemming/lemmatization**: reducing words to a root form
  (`"running"/"ran"/"runs" -> "run"`). Stemming is a crude, fast, rule-based
  chop (can produce non-words); lemmatization uses a vocabulary/grammar to
  produce the true dictionary root — slower, more accurate.

```python
import re
def basic_preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # strip punctuation
    return text.split()
```

## Why word-level tokenization breaks down

Splitting on whitespace gives one token per word — simple, but has two
serious problems for a large-scale model:
1. **Unbounded vocabulary**: every misspelling, rare name, or new word
   (`"COVID-19"`, `"transformerXL"`) becomes a brand-new, never-seen token
   at inference time — the classic **out-of-vocabulary (OOV)** problem.
2. **No shared structure between related words**: `"run"`, `"running"`,
   `"runner"` become three completely unrelated tokens with no inherent
   connection, wasting model capacity re-learning similar meanings
   separately.

## Character-level tokenization: the opposite extreme

Split into individual characters — a small, fixed vocabulary (e.g. ~100
characters covers most English text), never has an OOV problem. Downside:
sequences become very long (a sentence becomes hundreds of tokens instead
of tens), and each character carries very little meaning on its own,
making the modeling problem harder in a different way (Lesson 045's RNN
long-range-dependency challenges, worse with longer sequences).

## Subword tokenization: the practical middle ground

Split words into frequently-occurring **sub-word pieces** — common words
stay as a single token (`"the"`), rare/complex words split into meaningful
pieces (`"tokenization" -> "token" + "ization"`). This is what
**every modern LLM actually uses** (GPT models use a BPE variant, Lesson
062 implements Byte-Pair Encoding from scratch, following Karpathy's
`minbpe`). Benefits:
- Bounded, fixed vocabulary size (commonly 30,000-100,000+ tokens) — no OOV
  problem, since any unseen word decomposes into known sub-pieces (down to
  individual bytes/characters in the worst case).
- Related words often share sub-tokens (`"run"`, `"running"` both contain
  `"run"`), giving the model a head start on related meanings.
- Sequences stay much shorter than character-level, longer than
  pure-word-level for rare content.

```python
# using Hugging Face's tokenizers (a *pretrained* subword tokenizer, for now)
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("Tokenization splits rare words into pieces.")
print(tokenizer.convert_ids_to_tokens(tokens))
```

## Vocabulary and special tokens

Every tokenizer maps tokens to integer IDs via a fixed **vocabulary**.
Special tokens handle specific roles:
- `[PAD]` / `<pad>`: pads shorter sequences in a batch to a common length
  (needed for the batched tensor operations from Lesson 040).
- `[UNK]` / `<unk>`: a fallback for truly unrepresentable input (rare with
  subword tokenizers, common with word-level ones).
- `[CLS]`, `[SEP]`: BERT-style markers for sequence start/separation.
- `<|endoftext|>`: GPT-style end-of-sequence marker.

## From tokens to numbers: this is where Lesson 050's embeddings come in

Tokenization produces a sequence of integer IDs — Lesson 050 (word
embeddings) covers how those IDs become the dense vectors a neural network
actually operates on (an `nn.Embedding` lookup table, Lesson 039's
territory revisited specifically for token IDs).

## Why this lesson matters directly for Module 11

Every step of building your own LLM (Module 11) starts with tokenization:
Lesson 063a's `makemore`-style character-level modeling uses the simplest
possible tokenizer (one token per character) deliberately, to keep the
focus on the model itself; Lesson 068a implements a real BPE tokenizer from
scratch, exactly the subword approach described here, so your own GPT
(Project 013) uses the same kind of tokenization real LLMs use.
