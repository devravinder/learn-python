# Findings — Full-Stack AI App

*(Unlike most other unverified-due-to-missing-PyTorch projects in this
curriculum, this one was chosen and scoped specifically so that **every
layer actually runs** in this sandbox — Node/npm were available even
though PyTorch/FastAPI were not. Everything below is real, executed
output, not predicted behavior. Docker builds are the one exception,
disclosed at the bottom — no Docker daemon was running in this sandbox.)*

## Backend (verified)

`npm install` in `backend/` succeeded (70 packages, 0 vulnerabilities).
Starting `node server.js` loads the Markov model from the bundled
107,899-word corpus (reused from Project 013's `fallback_corpus.txt`) and
serves on port 8000. Real responses from a running server:

```
GET /api/health
-> {"status":"ok","model_loaded":true}

POST /api/chat {"message": "the forest was quiet"}
-> {"reply":"The forest spirit bravely entered a sunlit meadow, as the
    rain began to set. A curious child finally reached the silent river,
    and remembered it for years to come. The village blacksmith walked
    through an old stone tower, with a growing sense"}

POST /api/chat {"message": ""}
-> HTTP 422 {"error":"message must be a non-empty string"}
```

The seed-matching logic (`_pickSeed` in `markovModel.js`) found an n-gram
overlapping the prompt's vocabulary ("forest") and generated a
continuation starting from it — the reply is thematically related to the
prompt, though it's a Markov chain, not a model that understands the
prompt; longer generations visibly wander as the chain loses any
connection to the seed, an honest limitation worth noting rather than
hiding.

## Frontend (verified)

`npm install` in `frontend/` succeeded. `npm run build` (Vite) compiled
successfully:

```
dist/index.html                   0.46 kB
dist/assets/index-D5RlXai7.css    2.78 kB
dist/assets/index-IEgmPdhD.js   191.99 kB
✓ built in 141ms
```

`npm run preview` served the built `dist/` output on port 4173 and
returned the expected `index.html` — confirming the static build is
actually servable, the same check Lesson 079's multi-stage Docker build
relies on implicitly.

## Full integration

Manually verified: the backend accepts `POST /api/chat` requests in
exactly the shape `App.jsx`'s `sendMessage` function sends them
(`{"message": string}` in, `{"reply": string}` or `{"error": string}`
out) — confirmed by reading both sides side by side, since running the
Vite dev server and clicking through the actual UI wasn't done in this
authoring pass. **If you're working through this project yourself, run
`npm run dev` in `frontend/` alongside `node server.js` in `backend/` and
actually click through the chat UI** — that manual click-through is the
one verification step not performed here, consistent with this
curriculum's rule to say so explicitly rather than claim UI testing that
didn't happen.

## Docker (not verified)

Same limitation as Lesson 079: the `docker` CLI is present in this
sandbox but the daemon isn't running, so `docker compose up` was not
executed. Both Dockerfiles follow standard, documented multi-stage build
syntax; the one genuinely tricky part — that Vite's `VITE_API_BASE` must
be passed as a build `ARG`, not a container-runtime `environment:` value,
since Vite inlines `import.meta.env.VITE_*` values at build time — is
handled correctly in `frontend/Dockerfile` and `docker-compose.yml`, but
run it yourself to confirm the built image actually serves correctly.
