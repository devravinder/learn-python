# 01 — Concepts: Serving LLMs

## Two different deployment stories, two different tools

- **vLLM**: a high-throughput serving engine for running models on GPU
  servers, built around **PagedAttention** (a memory-management technique
  for the KV-cache, Lesson 074, that reduces fragmentation and enables
  efficient continuous batching across many simultaneous requests) — the
  standard choice for serving an LLM to many users at once.
- **llama.cpp**: a CPU-friendly (also supports GPU) inference engine
  written in C++, built around aggressive **quantization** (Lesson 074),
  designed to run models on ordinary laptops/phones/edge devices without
  a datacenter GPU at all — the standard choice for local, single-user
  inference.

Neither introduces new *algorithms* beyond what Lesson 074 covered — both
are engineering-optimized implementations of KV-caching, quantization, and
batching, tuned for different deployment targets (many-user server vs.
single-user local).

## Serving with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=100)

outputs = llm.generate(["Tell me about the ocean."], sampling_params)
print(outputs[0].outputs[0].text)
```

vLLM also exposes an **OpenAI-compatible API server**
(`python -m vllm.entrypoints.openai.api_server --model ...`) — meaning
existing code written against OpenAI's API (Lesson claude-api reference
pattern, or any OpenAI SDK usage) can point at your self-hosted vLLM
server with minimal changes, a deliberate compatibility design choice that
makes self-hosting easier to adopt.

## Serving with llama.cpp

llama.cpp works with **GGUF**-format quantized model files (a file format
specifically designed for this ecosystem):

```bash
./llama-cli -m model.Q4_K_M.gguf -p "Tell me about the ocean." -n 100
# or run its own local OpenAI-compatible server:
./llama-server -m model.Q4_K_M.gguf --port 8080
```

The `Q4_K_M` naming reflects the quantization scheme (roughly 4-bit,
mixed precision for different weight groups within the model) — different
quantization levels trade file size/speed against output quality, and
llama.cpp's ecosystem offers many pre-quantized variants of popular open
models.

## Latency vs. throughput: two different optimization targets

- **Latency**: how long until *one* request gets its response — matters
  most for interactive, single-user applications (a chat interface
  someone is actively waiting on).
- **Throughput**: how many total tokens/requests can be served per
  second across *many* concurrent users — matters most for a
  high-traffic service.

Batching (Lesson 074) generally **improves throughput** (more work done
per unit of GPU time) but can **hurt individual latency** slightly (a
request might wait briefly for a batch to fill) — a genuine tradeoff
real serving systems tune explicitly (e.g. vLLM's continuous batching is
specifically designed to minimize this tradeoff's cost).

## Streaming responses

Real chat interfaces show tokens appearing one at a time rather than
waiting for the entire response to generate — implemented by yielding each
newly-generated token to the client as soon as it's produced (Lesson 066's
generation loop, with each iteration's new token sent immediately instead
of accumulated silently):

```python
def stream_generate(model, tokenizer, prompt):
    idx = tokenizer(prompt, return_tensors="pt")["input_ids"]
    for _ in range(100):
        next_token = generate_one_step(model, idx)   # Lesson 066's per-step logic
        idx = torch.cat([idx, next_token], dim=1)
        yield tokenizer.decode(next_token[0])
```

This dramatically improves *perceived* latency (the user sees output
starting almost immediately) even though total generation time is
unchanged — a UX-level optimization layered on top of the actual
inference-speed optimizations from Lesson 074.

## Where this leaves you heading into Module 14-15

Every application in Module 14 (RAG, agents) and every productionization
concern in Module 15 (Project 016's full-stack app) assumes an LLM
reachable via some kind of API — whether that's a commercial provider's
API, or your own self-hosted vLLM/llama.cpp server. This lesson is what
makes "just call the model" concrete rather than hand-wavy for the
remainder of the curriculum.
