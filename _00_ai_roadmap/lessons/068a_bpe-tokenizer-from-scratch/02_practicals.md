# 02 — Practicals: A Complete BPE Tokenizer

1. Implement the full `BPETokenizer` class from `01_concepts.md`
   (`train`, `encode`, `decode`, plus the two static helper methods).

2. Train it on a repetitive text sample with `vocab_size=280` (24 merges
   beyond the base 256 bytes). Print each merge as it happens, along with
   the actual bytes it represents (`self.vocab[new_id]`) — do later merges
   combine earlier merged chunks into recognizable multi-word phrases?

3. **Round-trip test on the training text**: encode the exact text you
   trained on, decode it back, and confirm it matches exactly. Also
   confirm `encode(training_text)` produces the *same* token IDs `train()`
   ended with internally — if it doesn't, your `encode` isn't applying
   merges in the same order `train()` learned them (re-read
   `01_concepts.md`'s note on merge ordering).

4. **Round-trip test on genuinely new text** (not seen during training,
   but sharing some vocabulary/phrases). Confirm exact round-trip
   correctness, and report the compression ratio (original bytes / token
   count).

5. **Unicode stress test**: encode and decode a string containing accented
   characters, emoji, and non-Latin script (e.g. `"café 🎉 日本語"`).
   Confirm the round trip is exact — this is byte-level BPE's core promise
   (Lesson 062) verified directly, not just claimed.

6. Compare final vocabulary sizes and compression ratios for
   `vocab_size=280` vs `vocab_size=320` (more merges) on the same training
   text. Does more training-time merging produce better compression on new
   text too, or mostly just on the training text itself (relate to
   Lesson 017's overfitting framing — can a tokenizer "overfit" in a
   loose sense to its training corpus)?
