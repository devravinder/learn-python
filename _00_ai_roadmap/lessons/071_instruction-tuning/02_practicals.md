# 02 — Practicals: Instruction Tuning

## Loss masking mechanics (pure Python — the key detail, verified directly)

1. Implement a from-scratch masked cross-entropy: given a list of
   predicted probabilities-of-correct-token per position and a parallel
   list of "is this position masked" booleans, compute the average
   cross-entropy loss **only over unmasked positions**. Test on
   `probs = [0.9, 0.9, 0.9, 0.2, 0.1]` with the first 3 positions masked
   (these represent prompt tokens) — confirm the loss only reflects
   positions 3-4 (the response tokens), and manually verify the number
   matches averaging `-log(0.2)` and `-log(0.1)` only.

2. Confirm what happens if you *forget* to mask: compute the same loss
   over **all 5** positions instead. Is the resulting loss number lower or
   higher than the correctly-masked version, and does that make sense
   given the first 3 (prompt) positions have very high "correctness"
   probabilities that would otherwise dominate the average favorably?

## Formatting and tokenizing instruction data (pure Python)

3. Write a `format_example(instruction, response)` function that produces
   the chat-template string from `01_concepts.md`
   (`<|user|>\n{instruction}\n<|assistant|>\n{response}<|endoftext|>`).
   Apply it to 3 example instruction/response pairs.

4. Using any tokenizer you like (even Lesson 068a's `BPETokenizer`, or
   simple `.split()` for a word-level stand-in), tokenize one formatted
   example and determine the index where the `<|assistant|>` marker ends
   (i.e. where the response begins) — this index is exactly what you'd use
   as the mask boundary in Q1's masking logic, applied to real tokenized
   data instead of a toy probability list.

## PyTorch: full masked training step

5. Implement a training step using Hugging Face's `-100` `ignore_index`
   convention (`01_concepts.md`): build `input_ids` for a formatted
   example, set `labels = input_ids.clone()`, then set
   `labels[:prompt_length] = -100`. Compute
   `F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)`
   and confirm (via a small controlled example, e.g. all-correct
   predictions on the prompt but random elsewhere) that the loss is
   unaffected by how well the model "predicts" the masked prompt tokens.

6. Fine-tune a small pretrained model (or Project 013's own GPT, formatted
   with a toy instruction dataset you write yourself, even 10-20 examples)
   using this masked-loss approach for a few epochs. Test it on a **new**
   instruction (not in your training set) — does it attempt to follow the
   instruction format (produce an "assistant-style" response) rather than
   just continuing the prompt text unstructured?
