# 01 — Concepts: Serving a Model as a REST API

## Mapped directly to what you already know

| Express / Spring concept | FastAPI equivalent |
|---|---|
| `app.post("/route", handler)` | `@app.post("/route")` decorator on a function |
| Request body validation (Joi, `@Valid` + DTO) | Pydantic `BaseModel` — declare the shape, validation is automatic |
| Middleware | FastAPI middleware (`@app.middleware("http")`) — same concept, same placement in the request lifecycle |
| `npm install` / `package.json` | `pip install` / `requirements.txt` (Lesson 001, already familiar) |
| Route handler doing business logic then returning JSON | Route handler running model inference then returning JSON — **this is the only genuinely new part** |

If you've built a CRUD API in Express or Spring, you already know the
*shape* of this lesson — the novelty is entirely in what happens inside
the handler (a model forward pass, Lesson 040), not the web framework
itself.

## A minimal FastAPI model server

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch

app = FastAPI()
model = load_model()   # load ONCE at startup, not per-request (see below)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    with torch.no_grad():
        output = model(request.text)
    return PredictResponse(prediction=output.label, confidence=output.confidence)
```

Run with `uvicorn main:app --reload` (`--reload` for development, exactly
like `nodemon`/Spring DevTools' auto-restart-on-change during development —
**never** use `--reload` in production, same as never running a Node dev
server in production).

## Why load the model once, not per-request

Loading a model (especially an LLM) can take seconds — doing it inside the
route handler would mean every single request pays that cost, which is
obviously wrong the moment you think about it in Express/Spring terms
(you wouldn't reconnect to a database on every request either). FastAPI's
`lifespan` context manager (or simply loading at module level, as above)
loads once at server startup:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()   # startup
    yield
    # cleanup code here, if any, runs on shutdown

app = FastAPI(lifespan=lifespan)
```

## Async vs sync handlers — a genuine, new consideration

Unlike a typical CRUD handler (mostly I/O-bound: waiting on a database),
model inference is often **CPU/GPU-bound** — the request-handling thread is
genuinely busy computing, not waiting. FastAPI's `async def` handlers are
most beneficial for I/O-bound work; for CPU-bound model inference, running
in a thread pool (`run_in_threadpool`) or a separate worker process avoids
blocking the whole event loop on one slow request — worth knowing since
naively `async def`-ing a synchronous, CPU-heavy model call doesn't
automatically parallelize it the way it would for I/O-bound endpoints.

## Input validation matters more, not less, with a model behind it

A malformed request to a typical CRUD endpoint fails fast with a clear
error. A malformed or adversarial input to a model can silently produce a
nonsensical (but successfully-returned) prediction instead of an obvious
error — validate input shape/type/range explicitly (Pydantic handles
type/shape; you still need to think about *value* validation — e.g.
empty strings, absurdly long inputs exceeding your model's `max_len`,
Lesson 060) rather than assuming a model call failing loudly the way a
malformed SQL query would.

## Health checks and readiness — same as any service you'd operate

```python
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
```

Exactly the same pattern as any service health-check endpoint you've
built before (load balancers, container orchestrators, and uptime
monitors all expect this) — a model-serving API is still, fundamentally,
a service that needs the same operational basics as anything else you've
deployed.

## What's genuinely new vs. what transfers

**Transfers directly**: routing, request/response validation, middleware,
error handling, deployment basics, health checks — your existing
full-stack instincts are correct here.

**Genuinely new**: model loading/lifecycle (once, not per-request),
inference being CPU/GPU-bound rather than I/O-bound (changes concurrency
tradeoffs), input validation needing to account for a model's specific
constraints (sequence length, expected data distribution) rather than
just type-correctness, and — Lesson 079 next — the deployment artifact
itself needing to package model weights alongside code.
