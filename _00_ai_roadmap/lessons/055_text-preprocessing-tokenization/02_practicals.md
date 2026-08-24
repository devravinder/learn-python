# 02 — Practicals: Text Preprocessing & Tokenization

1. Implement `basic_preprocess(text)` (lowercase, strip punctuation, split
   on whitespace) and apply it to a paragraph of your choice. Report
   vocabulary size (unique tokens).

2. Implement simple stemming by hand for a small set of rules (strip
   trailing `"ing"`, `"ed"`, `"s"` — a crude approximation of a real
   stemmer). Apply it to `["running", "runs", "ran", "easily", "flies"]`
   and note which results are and aren't real words — this is the
   expected, known imprecision of rule-based stemming.

3. Tokenize a short paragraph at the **character level** and at the
   **word level**. Compare sequence lengths and vocabulary sizes between
   the two. For a 50-word paragraph, roughly how many characters are
   there per word on average, and how does that ratio show up in the two
   sequence lengths?

4. Simulate the out-of-vocabulary problem: build a word-level vocabulary
   from one paragraph of text, then tokenize a *different* paragraph
   (on a different topic) using only that vocabulary, mapping any
   unrecognized word to `<unk>`. Report what fraction of words in the new
   paragraph are OOV.

5. Implement a **minimal Byte-Pair Encoding (BPE) trainer** (a preview of
   Lesson 068a's full version): starting from character-level tokens,
   repeatedly find the most frequent adjacent pair of tokens in your
   corpus and merge it into a new single token, for a fixed number of
   merges (e.g. 10). Apply it to a small repetitive text sample and print
   the vocabulary after each merge — do frequent multi-character chunks
   (e.g. common suffixes) get merged first?

6. If you have `transformers` installed, tokenize the same sentence with
   GPT-2's tokenizer and with BERT's tokenizer
   (`AutoTokenizer.from_pretrained("bert-base-uncased")`). Compare the
   resulting tokens — do they split words differently? Look up (or infer
   from the output) what algorithm each one uses.
