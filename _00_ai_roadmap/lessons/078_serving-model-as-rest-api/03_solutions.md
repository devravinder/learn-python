# 03 — Solutions: Serving a Model as a REST API

*(`fastapi` isn't available in the authoring sandbox — code below follows
the documented FastAPI API precisely; run it yourself to verify timing
numbers, which are hardware-dependent anyway.)*

## 1. Minimal serving endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "../../projects/013_train-your-own-gpt/02_solutions"))
from model import GPT
from bpe_tokenizer import BPETokenizer

app = FastAPI()

checkpoint = torch.load("gpt_checkpoint.pt", map_location="cpu")
config = checkpoint["config"]
tokenizer = BPETokenizer()
tokenizer.load("data/fallback_corpus.tokenizer.json")
model = GPT(vocab_size=len(tokenizer.vocab), d_model=config["d_model"], n_heads=config["n_heads"],
            n_layers=config["n_layers"], d_ff=config["d_ff"], max_len=config["block_size"])
model.load_state_dict(checkpoint["model_state"])
model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    completion: str

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    idx = torch.tensor([tokenizer.encode(request.text)])
    with torch.no_grad():
        for _ in range(30):
            logits = model(idx[:, -config["block_size"]:])[:, -1, :]
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_id], dim=1)
    return PredictResponse(completion=tokenizer.decode(idx[0].tolist()))
```

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "The forest"}'
```

## 2. Loading per-request (the mistake, demonstrated)

```python
@app.post("/predict_slow")
def predict_slow(request: PredictRequest):
    checkpoint = torch.load("gpt_checkpoint.pt", map_location="cpu")   # BUG: reloaded every request
    model = GPT(...)
    model.load_state_dict(checkpoint["model_state"])
    # ... rest of inference
```

Timing 3 consecutive requests to `/predict_slow` should show each one
taking roughly the same, noticeably large amount of time (model file I/O
+ deserialization + weight loading, every single call) — vs. `/predict`
where only the first server startup pays that cost and every request
after is fast. The comparison should make the bug's cost impossible to
miss, exactly like accidentally reconnecting to a database per-request in
any backend framework.

## 3. Input validation matching model constraints

```python
from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 500:   # a proxy limit; real check would tokenize and compare to block_size
            raise ValueError("text too long")
        return v
```

A request exceeding the limit gets a clean `422 Unprocessable Entity`
response with a clear error message from FastAPI automatically — no need
to let it reach the model and fail confusingly inside a tensor shape
mismatch deep in `forward()`.

## 4. Health check with failure simulation

```python
model_state = {"loaded": False, "model": None}

@app.get("/health")
def health():
    return {"status": "ok" if model_state["loaded"] else "degraded", "model_loaded": model_state["loaded"]}

try:
    checkpoint = torch.load("nonexistent_checkpoint.pt")
    model_state["loaded"] = True
except FileNotFoundError:
    print("WARNING: model checkpoint not found, starting in degraded mode")
    model_state["loaded"] = False
```

With a bad checkpoint path, `/health` should report `model_loaded: false`
and status `degraded` rather than the whole server failing to start or
crashing on the first request — a real operational pattern (start
degraded and report it, rather than hard-crash) worth using deliberately.

## 5. Async vs sync for CPU-bound inference

```python
import time

@app.post("/predict_async")
async def predict_async(request: PredictRequest):
    # same synchronous, CPU-bound model call as before, just inside `async def`
    ...
```

Timing sequential requests to a purely `async def`-wrapped but still
synchronous, CPU-bound model call typically shows **no improvement** over
the plain `def` version — `async def` only helps concurrency for I/O-bound
work (waiting on network/disk); it doesn't parallelize CPU-bound
computation just by being declared `async`. Real concurrency for CPU-bound
inference needs a thread/process pool (`fastapi.concurrency.run_in_threadpool`)
or multiple worker processes (`uvicorn --workers N`) — confirming
`01_concepts.md`'s warning empirically rather than taking it on faith.

## 6. Comparison to Express/Spring

Route declaration, request/response typing, and the overall project
structure feel essentially identical to an Express or Spring REST API —
the same mental model of "declare a route, validate the input, return
JSON" applies directly, and existing instincts about clean API design
transfer without modification. The genuinely different part is
operational: the "business logic" inside the handler is a model forward
pass rather than a database query, which changes the *resource profile*
of a request (CPU/GPU-bound and comparatively heavy, rather than I/O-bound
and typically light) — meaning concurrency, scaling, and startup-cost
assumptions that work fine for a typical CRUD service need rethinking
specifically around model loading and inference cost, not around the web
framework itself.
