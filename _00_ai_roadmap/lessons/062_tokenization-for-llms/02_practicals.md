# 02 — Practicals: Tokenization for LLMs

## Byte-level BPE (pure Python — a preview of Lesson 068a)

1. Encode a short paragraph to raw bytes: `list(text.encode("utf-8"))`.
   Print the number of bytes and the number of *unique* byte values.
   Confirm the unique count is at most 256, regardless of what text you
   use (even non-English text or emoji) — this is byte-level BPE's
   "universal base vocabulary" property from `01_concepts.md`.

2. Implement `get_pair_counts(ids)` (count adjacent-pair frequencies in a
   list of integer IDs) and `merge(ids, pair, new_id)` (replace every
   occurrence of `pair` with a new single ID) — the same two functions
   from Lesson 055's character-level BPE, now operating on byte IDs.

3. Run 15 merge steps on a paragraph of your choice (starting IDs 0-255
   for raw bytes, new merged tokens get IDs 256, 257, ...). Print each
   merge and its frequency. Do early merges correspond to extremely common
   short sequences (e.g. `"th"`, `"e "`, `" the "` for English text)?

4. Compute the **compression ratio**: original byte count / final token
   count after your 15 merges. This ratio is exactly what a real
   tokenizer's vocabulary size decision trades off (Lesson 062's
   vocabulary-size discussion) — more merges (bigger vocabulary) keeps
   pushing this ratio higher.

5. Test your trained merges on text **not** in the training paragraph
   (e.g. a sentence using different words but similar common substrings
   like "the", "ing"). Do your learned merges still apply and compress the
   new text somewhat, even though it wasn't part of training?

6. If you have `tiktoken` installed (`pip install tiktoken`), encode a
   sentence containing a number (e.g. `"There are 1234567 people."`) with
   GPT-2's encoding (`tiktoken.get_encoding("gpt2")`) and print the decoded
   pieces (`enc.decode([tok])` for each token). Confirm the number splits
   into multiple, not-obviously-meaningful chunks rather than one token —
   a live example of `01_concepts.md`'s "why LLMs are bad at arithmetic"
   tokenization artifact.
