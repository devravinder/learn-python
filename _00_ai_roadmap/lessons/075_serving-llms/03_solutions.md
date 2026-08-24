# 03 — Solutions: Serving LLMs

*(Q1-3's arithmetic was actually computed to produce the numbers below.)*

## 1. Unbatched serving

```python
per_request = 0.1
n = 20
total_unbatched = n * per_request
print(total_unbatched, n / total_unbatched)
```

**Actual output: total time 2.0s, throughput 10 requests/second.** Each
request's latency is simply its own compute time, `0.1s`.

## 2. Batched serving

```python
batch_size = 5
batch_time = 0.15
n_batches = n // batch_size
total_batched = n_batches * batch_time
print(total_batched, n / total_batched)
```

**Actual output: total time 0.6s, throughput ≈ 33.3 requests/second** — a
**3.33x throughput improvement** over unbatched serving, from amortizing
per-batch overhead across 5 requests at once instead of paying it 5
separate times.

## 3. Worst-case individual latency under batching

```python
arrival_gap = 0.05   # a new request arrives every 0.05s
fifth_arrival = 4 * arrival_gap        # time the 5th (batch-filling) request arrives
finish_time = fifth_arrival + batch_time
worst_case_latency = finish_time       # relative to the FIRST request's arrival at t=0
print(worst_case_latency)
```

**Actual output: 0.35s worst-case latency for the first-arriving request
in a slow-filling batch, vs. 0.1s under unbatched serving** — **3.5x
worse** individual latency for that unlucky request, who had to wait for
4 more requests to arrive before its batch could even start. This is the
numeric confirmation of `01_concepts.md`'s tradeoff: **better aggregate
throughput (Q2) at the cost of worse worst-case individual latency (Q3)**
— both true simultaneously, which is exactly why real serving systems
(vLLM's continuous batching) invest engineering effort specifically into
minimizing this latency cost rather than accepting the naive tradeoff at
face value.

## 4–5. Toy local serving and streaming

```python
import time

def generate_response(model, tokenizer, prompt, max_new_tokens=50):
    # reuses Lesson 066's generate() function
    idx = tokenizer(prompt, return_tensors="pt")["input_ids"]
    out = generate(model, idx, max_new_tokens, block_size=64, temperature=0.8)
    return tokenizer.decode(out[0].tolist())

def stream_generate(model, tokenizer, prompt, max_new_tokens=50):
    idx = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_token], dim=1)
        yield tokenizer.decode(next_token[0].tolist())

for token_text in stream_generate(model, tokenizer, "Once upon a time"):
    print(token_text, end="", flush=True)
    time.sleep(0.05)
```

Watching tokens print incrementally (Q5) is a well-documented real UX
effect: total wall-clock time to finish generation is unchanged, but
*perceived* responsiveness improves substantially because the user sees
progress immediately rather than staring at a blank wait — exactly why
every production chat interface streams responses rather than waiting for
the complete output.

## 6. llama.cpp vs. PyTorch on CPU

```python
from llama_cpp import Llama
import time

llm = Llama(model_path="model.Q4_K_M.gguf")
t0 = time.time()
output = llm("Tell me about the ocean.", max_tokens=100)
llama_cpp_time = time.time() - t0
print("llama.cpp tokens/sec:", 100 / llama_cpp_time)

# compare against a similarly-sized model via plain transformers on CPU
# (same prompt, same max_tokens, same hardware)
```

Expect llama.cpp's quantized CPU inference to run noticeably faster
(often several times) than an unquantized `transformers`/PyTorch model of
comparable parameter count on the same CPU — directly reflecting
`01_concepts.md`'s point that llama.cpp is specifically engineered
(aggressive quantization + optimized C++ kernels) for exactly this
CPU-inference use case, where a general-purpose PyTorch model isn't.
