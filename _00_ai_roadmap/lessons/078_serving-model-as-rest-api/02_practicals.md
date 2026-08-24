# 02 — Practicals: Serving a Model as a REST API

1. Build a minimal FastAPI app with one `POST /predict` endpoint that
   loads Project 013's own trained GPT (or any small model you have) once
   at startup and generates a completion for a `text` field in the
   request body. Test it with `curl` or `requests` (Lesson 015's client
   pattern) — no different from testing any REST endpoint you've built
   before.

2. Deliberately load the model **inside** the route handler instead of at
   startup, and time 3 consecutive requests. Confirm every request pays
   the full model-loading cost (should be obviously, dramatically slower
   than loading once) — a concrete demonstration of why this "obvious in
   Express/Spring terms" mistake matters here too.

3. Add Pydantic validation for the request: `text` must be a non-empty
   string under some max length (matching your model's `block_size`,
   Lesson 060). Send a request exceeding that length and confirm you get
   a clean validation error rather than the model silently truncating or
   erroring confusingly deep inside a forward pass.

4. Add a `GET /health` endpoint reporting whether the model is loaded.
   Simulate a "model failed to load" scenario (e.g. point at a
   nonexistent checkpoint path) and confirm `/health` correctly reports
   the failure rather than the server crashing silently or hanging.

5. Time 10 sequential requests to your `/predict` endpoint using a
   CPU-bound model. Then try converting the handler to `async def`
   without changing anything else about how the model runs internally —
   does response time actually improve? (It likely won't, for genuinely
   CPU-bound work — confirm this yourself rather than assuming `async`
   is a free performance win, per `01_concepts.md`'s note.)

6. Write a short comparison (3-4 sentences) of this FastAPI server's
   structure to an Express or Spring REST API you've built before —
   what's identical, and what's the one part that's genuinely different?
