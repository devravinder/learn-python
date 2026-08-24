# 02 — Practicals: Serving LLMs

## Latency vs. throughput tradeoff (pure Python simulation)

1. Simulate serving requests **one at a time** (no batching): each request
   takes a fixed `0.1` seconds of "model compute" regardless of batch
   size. For 20 requests arriving, compute total time to serve all of
   them, and the average latency per request (should be straightforward
   with no batching: request `i` finishes at `(i+1)*0.1` seconds).

2. Simulate **batched** serving: group requests into batches of 5; a batch
   of 5 takes `0.15` seconds total (less than `5*0.1=0.5s` sequential,
   since batched GPU compute amortizes overhead — a simplification of the
   real efficiency gain). Compute total time to serve all 20 requests
   (4 batches of 5) and compare **throughput** (requests/second) to Q1's
   unbatched throughput.

3. Now compute **individual latency** under batching: the *last* request
   to arrive in a batch of 5 might have to wait for the other 4 to also
   arrive before the batch starts (worst case, a request arrives just
   after a batch was dispatched and must wait for the next one to fill).
   Estimate worst-case added latency for an unlucky request under this
   batching scheme, and compare it to Q1's unbatched per-request latency —
   confirm the tradeoff `01_concepts.md` describes (better throughput,
   worse worst-case individual latency) shows up numerically, not just
   conceptually.

## Toy local serving (Python, no GPU cluster needed)

4. Write a minimal script that loads a small local model (Project 013's
   own trained GPT works, or any small HF model) and exposes a plain
   Python function `generate_response(prompt) -> str` — this is the "model
   serving" logic itself, before any web framework wraps it (Lesson 078
   adds the actual HTTP layer).

5. Implement `stream_generate` from `01_concepts.md`: a generator function
   that yields one decoded token at a time instead of returning the full
   response at once. Iterate over it and print tokens as they're
   produced, with a tiny `time.sleep(0.05)` between them to simulate
   real generation latency — does watching it print incrementally feel
   different from waiting for the whole thing at once, even though total
   time is unchanged?

6. If you have `llama-cpp-python` installed and a small GGUF model
   downloaded: load it and generate from a prompt, comparing generation
   speed (tokens/second) to loading the same-sized model via
   `transformers`/PyTorch on CPU. Does the quantized llama.cpp version
   generate noticeably faster on CPU, consistent with `01_concepts.md`'s
   description of what llama.cpp specifically optimizes for?
