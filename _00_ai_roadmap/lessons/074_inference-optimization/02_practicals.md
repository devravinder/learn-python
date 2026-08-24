# 02 — Practicals: Inference Optimization

## Quantifying KV-cache savings (pure Python arithmetic)

1. Without caching, generating token `t` requires recomputing attention
   over all `t` positions (roughly `O(t)` work per token, ignoring
   constants). Compute the **total** work to generate `N=100` tokens one
   at a time without caching: `sum(t for t in range(1, 101))`. Compare to
   **with** caching, where generating token `t` only requires `O(1)` new
   work (attending the new query against the cached keys/values) —
   `sum(1 for t in range(1, 101))`. What's the speedup ratio?

2. Repeat Q1 for `N=1000` and `N=10000`. Does the speedup ratio grow, shrink,
   or stay constant as `N` grows? Explain why, relating the without-cache
   total to the closed-form sum `N*(N+1)/2` (quadratic in `N`) vs the
   with-cache total (linear in `N`).

## Quantization arithmetic

3. For a 7-billion-parameter model, compute memory footprint at
   `float32` (4 bytes/param), `float16` (2 bytes/param), `int8` (1
   byte/param), and `int4` (0.5 bytes/param). Which precisions fit
   comfortably on a 24GB consumer GPU (for inference only — no
   gradients/optimizer state needed)?

## Speculative decoding simulation (pure Python)

4. Simulate speculative decoding's acceptance mechanic: a "draft model"
   proposes 5 tokens; a "target model" (simulated here as a fixed ground-
   truth sequence you define) accepts a prefix of matching tokens and
   rejects at the first mismatch. For draft `[5, 12, 7, 9, 3]` and true
   target continuation `[5, 12, 8, 9, 3]`, determine how many tokens get
   accepted before the first mismatch, and what the corrected next token
   should be.

5. Simulate this over 20 "rounds" of 5-token drafts each, with a
   configurable per-token draft-accuracy probability (e.g. 70% chance each
   draft token matches what the target model would have produced,
   independently per token). Compute the average number of tokens
   accepted per round. At what per-token accuracy does speculative
   decoding stop being worth it compared to just generating one token at a
   time with the target model alone (hint: think about when average
   accepted-tokens-per-round drops toward 1)?

6. Explain in your own words why speculative decoding **never** produces
   worse output quality than using the target model alone, even though a
   faster/weaker draft model is involved in the process — what exactly
   guarantees this (revisit `01_concepts.md`'s "large model determines
   every accepted token" point)?
