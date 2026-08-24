# 01 — Requirement: Train Your Own GPT

## The brief

Assemble a complete GPT training pipeline from Module 11's pieces, train
it on a real text corpus, and generate new text from it.

## What to produce

1. **Choose and size your model** (Lesson 068): given your corpus's token
   count, compute a Chinchilla-ratio-informed parameter count
   (`N ≈ D / 20`) *before* picking `d_model`/`n_layers`/`n_heads` — don't
   just guess a size. Document your reasoning.

2. **Tokenizer** (Lesson 068a): train a `BPETokenizer` on your corpus with
   a vocabulary size appropriate to corpus size (a few hundred to a couple
   thousand tokens for a small hobby corpus — not GPT-2's 50,257, which
   needs far more data to justify). Save the trained merges/vocab so you
   don't need to retrain the tokenizer every run.

3. **Data pipeline** (Lesson 064): encode your full corpus with the
   trained tokenizer, split train/val (90/10, by position), implement
   `get_batch`.

4. **Model** (Lesson 060): instantiate your sized `GPT` with your chosen
   config.

5. **Training loop** (Lessons 065, 067): `AdamW`, LR warmup+cosine decay,
   gradient clipping, periodic train/val loss estimation, checkpointing.
   Train until validation loss clearly plateaus (or you run out of
   patience/compute — report which).

6. **Generation** (Lesson 066): implement `generate` with temperature and
   top-k/top-p sampling. Generate at least 3 samples at different
   temperatures from a few different starting prompts.

7. **Report**:
   - Your sizing decision (Q1) and actual final train/val loss + perplexity.
   - Loss curves (train and val, over training).
   - 3-5 generated text samples, with the temperature/sampling settings
     used for each.
   - Honest qualitative assessment: does the generated text show
     recognizable words? Grammatically plausible short phrases? Any
     longer-range coherence? Compare this to what Lesson 063a's simple
     bigram/MLP models could produce — is there a clear qualitative step
     up from attention's added context?

## Stretch goals (optional, not required for a complete submission)

- Compare character-level tokenization (Lesson 064) vs your BPE tokenizer
  (Lesson 068a) on the same corpus/model size — does BPE's shorter token
  sequences let you fit more effective context into the same
  `block_size`, and does that show up in loss or generated-text quality?
- Implement KV-caching (Lesson 066's preview) for faster generation.
- Try RoPE (Lesson 061) instead of learned positional embeddings.

## Constraints

- Train a genuinely new model on your own chosen corpus — don't just run
  someone else's pretrained checkpoint.
- Don't peek at `02_solutions/` before you have your own working pipeline
  and at least one completed training run.
