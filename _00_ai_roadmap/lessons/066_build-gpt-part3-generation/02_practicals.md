# 02 — Practicals: Sampling & Generation

## Sampling strategies, pure Python first

1. Implement `softmax` and temperature scaling
   (`softmax([l/temperature for l in logits])`) from scratch. For
   `logits = [2.0, 1.0, 0.5, 0.1]`, compute and print the resulting
   distribution at `temperature = [0.1, 0.5, 1.0, 2.0]`. Confirm low
   temperature sharpens toward one-hot and high temperature flattens
   toward uniform (Lesson 036, revisited here for generation specifically).

2. Implement `top_k_filter(logits, k)` in pure Python: keep only the `k`
   largest logits, set the rest to `-inf`, then softmax. For the same
   logits and `k=2`, confirm exactly 2 tokens have nonzero probability
   after filtering.

3. Implement `top_p_filter(logits, p)` in pure Python: sort logits
   descending, softmax, take a cumulative sum, keep tokens up to (and
   including) the first one that pushes cumulative probability past `p`,
   set the rest to `-inf`. For `logits = [2.0, 1.8, 1.5, 1.0, 0.5, 0.1]`
   (a fairly spread-out, non-peaked distribution) and `p=0.9`, how many
   tokens survive? Compare to `p=0.5` — confirm a stricter `p` keeps
   noticeably fewer tokens here, unlike a very peaked distribution where
   the top 1-2 tokens alone might already exceed even a high `p`.

## PyTorch: the full generation loop

4. Implement `generate(model, idx, max_new_tokens, block_size)` from
   `01_concepts.md` using your trained (or even untrained, for a shape/
   mechanics check) GPT from Lessons 064-065. Generate 50 new tokens from
   a single-character starting prompt and decode the result to text.

5. Generate the same starting prompt 5 times with **greedy** decoding
   (`argmax` instead of sampling) and 5 times with **temperature=1.0
   sampling**. Confirm greedy always produces the *exact same* output
   (deterministic) while sampling produces 5 different outputs.

6. Generate with temperature values `[0.3, 0.8, 1.5]` from the same
   trained model and prompt. Read the outputs qualitatively — does low
   temperature look more repetitive/conservative and high temperature
   look more chaotic, matching `01_concepts.md`'s description?
