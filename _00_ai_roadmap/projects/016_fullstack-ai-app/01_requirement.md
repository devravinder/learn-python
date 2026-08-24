# 01 — Requirement: Full-Stack AI App

## Brief

Build a small chat-style web app: a React frontend where a user types a
prompt, a Node/Express backend that generates a text continuation and
returns it, and a clear seam where a real trained language model (Project
013's GPT, or any Hugging Face model) would plug in for production use.

This project is deliberately scoped so **every layer of it actually runs
and is verifiable** in an environment without PyTorch/a GPU — including
the "model" itself — while documenting precisely how to swap in the real
trained model when that environment is available. The point of this
capstone is the full-stack integration and deployment pipeline, which are
identical in shape whether the box in the middle is a 5-line Markov chain
or a 100M-parameter Transformer.

## Functional requirements

1. **Backend** (`Node.js` + `Express`):
   - `POST /api/chat` — accepts `{ "message": string }`, returns
     `{ "reply": string }`.
   - `GET /api/health` — returns model-loaded status (Lesson 078's
     health-check pattern, in Express instead of FastAPI).
   - The "model" is an order-2 word-level Markov chain trained at server
     startup on a bundled text corpus — small, fast, fully inspectable,
     and **honestly labeled as a toy model**, not dressed up as more than
     it is.
   - A documented seam (`model/README.md`) showing exactly what changes
     to call a real model instead: either (a) a Python FastAPI sidecar
     running Project 013's trained GPT, called via HTTP from Express, or
     (b) a call to a hosted LLM API.

2. **Frontend** (`React`, via Vite):
   - A single chat view: message list, text input, submit button.
   - Calls the backend's `/api/chat` endpoint and renders the reply.
   - Shows a loading state while waiting for a response — a real UX
     concern given Lesson 080's discussion of inference latency.

3. **Containerization & deployment** (ties back to Lessons 079–080):
   - A `Dockerfile` for the backend and one for the frontend (or a single
     multi-stage build serving the built frontend as static files from
     Express).
   - A `docker-compose.yml` running both services together.

## Non-functional requirements

- The backend must start and respond correctly with **zero external
  network calls and zero API keys** — the toy Markov model needs nothing
  but the bundled corpus, so the whole app runs fully offline out of the
  box.
- Clear separation between the "toy model" code path and the "swap in a
  real model" documentation, so a reader doesn't mistake one for the
  other.

## Stretch goals

- Add conversation history (multi-turn context) to the Markov generator
  or to the real-model seam.
- Add streaming responses (token-by-token) from backend to frontend,
  using the same `text/event-stream` pattern a real LLM API would use —
  directly relevant if the swapped-in model is a hosted LLM API that
  supports streaming.
- Deploy it for real, following Lesson 080's target comparison.
