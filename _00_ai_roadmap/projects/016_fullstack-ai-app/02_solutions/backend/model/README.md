# The model seam: toy model vs. a real trained model

`markovModel.js` is an order-2 word-level Markov chain — genuinely
runnable with zero dependencies, but a toy, and labeled as one everywhere
it's surfaced (the chat UI's first message says so explicitly). This file
documents exactly what changes to serve a real trained model instead,
without touching the frontend or the API contract at all.

## Option A: a Python FastAPI sidecar (recommended — reuses Project 013/015)

1. Run Project 015's `serve_api.py` (or Project 013's trained GPT wrapped
   the same way) as its own service, e.g. on port `8001`.
2. Replace the body of `POST /api/chat` in `server.js` with a proxy call:

```js
app.post("/api/chat", async (req, res) => {
  const { message } = req.body || {};
  const response = await fetch("http://localhost:8001/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: message }),
  });
  const data = await response.json();
  res.json({ reply: data.completion });
});
```

3. Add the FastAPI sidecar as a third service in `docker-compose.yml`
   (same shape as the `backend`/`frontend` services already there), and
   point the Express service at it via an internal Docker network URL
   (`http://model-server:8001/predict`) instead of `localhost`.

The React frontend needs **zero changes** — it only ever talks to
Express's `/api/chat`, which keeps the exact same request/response shape
regardless of what generates the reply behind it. This is the same
"backend for frontend" pattern you'd already use to shield a React app
from a backend implementation change in any other context.

## Option B: a hosted LLM API

Same proxy pattern as Option A, but the fetch target is the hosted
provider's chat-completions endpoint, with the API key read from an
environment variable (never hardcoded, never sent to the frontend) —
exactly the secrets-handling discipline from Lesson 080.

## Why the toy model is the right default for this project

The point of this capstone is verifying the *integration* — request
validation, health checks, loading states, Docker packaging, the
frontend/backend contract — all of which are identical in shape
regardless of what's behind `/api/chat`. Building the toy model in plain
JavaScript means every layer of this project actually runs and was
verified end-to-end (see `FINDINGS.md`), rather than the more common
pattern elsewhere in this curriculum of writing correct-but-unexecuted
PyTorch/FastAPI code because the sandbox lacks those libraries.
