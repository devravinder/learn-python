# 03 — Solutions: Cloud Deployment

*(No cloud account or internet access was available in the authoring
sandbox — these are worked through as documented platform behavior and
the reasoning `01_concepts.md` lays out; run them yourself to get real
numbers, which are account/region/hardware-dependent anyway.)*

## 1. Deploy to a PaaS

On Render, this is: connect the GitHub repo, point it at the
`Dockerfile` from Lesson 079, set the port to `8000`, deploy. The
resulting `https://<app-name>.onrender.com/predict` should behave
identically to the local `localhost:8000/predict` — the platform is just
running the same container image behind a load balancer and TLS
termination it manages for you, nothing about the app itself changes.

## 2. Cold start timing

Expected pattern: the first request after the service has scaled to zero
(or auto-slept, on a free tier) should take several seconds to tens of
seconds — dominated by container start + model checkpoint loading, per
`01_concepts.md`'s cold-start discussion — while a request 30 seconds
later (service still warm) should complete in the model's normal
inference latency (milliseconds to low seconds, depending on model size
and CPU vs. GPU). The gap between the two numbers **is** the cold-start
cost, and it should be dramatically larger than the equivalent gap for a
plain Express/Spring health-check endpoint, where cold start is typically
sub-second.

## 3. Secrets via platform config

```bash
# Render/Fly/Railway all expose an equivalent of:
MODEL_PATH=/app/models/gpt_checkpoint.pt
OPENAI_API_KEY=sk-...   # if used anywhere, e.g. for a comparison baseline
```

Read the same way you'd read any env var in the app (`os.environ["MODEL_PATH"]`
in Python, no different from `process.env.MODEL_PATH` in Node). A `git
log -p -- Dockerfile` and `git grep -i "sk-"` afterward should turn up
nothing — the same secret hygiene check you'd run on any repo before
pushing it.

## 4. Warm instance vs. scale-to-zero

With min-replicas set to 1, exercise 2's timing test should show **no
cold-start gap at all** — every request, first or hundredth, sees roughly
the same latency, because the container and loaded model are never torn
down between requests. The tradeoff, exactly as `01_concepts.md`
describes, is cost: the platform now bills for continuous uptime instead
of only the time actually spent serving requests, which is the core
scale-to-zero-vs-always-warm decision every real ML deployment has to
make explicitly.

## 5. Redeploy via git push

```bash
git commit -am "add model_version to /predict response"
git push origin main   # platform's webhook picks this up and redeploys
```

The new field should appear in the JSON response at the same public URL
within the platform's normal build-and-deploy window (typically under a
few minutes for a small image) — the exact same CI/CD mental model as
redeploying any other containerized service, with no ML-specific step
inserted anywhere in the flow.

## 6. GPU vs. CPU latency and cost comparison

Expected qualitative result: the GPU instance should show meaningfully
lower per-request inference latency (often 5–20x for a Transformer
forward pass, depending on model size and batch size) but at a
substantially higher hourly cost (a `g4dn.xlarge` runs roughly 5-10x the
hourly cost of a small CPU instance, prices vary by region/provider and
change over time). For a small model or low-traffic endpoint, the CPU
deployment can be the more cost-effective choice despite the latency
gap — the same "is the speedup worth the money" tradeoff that governs
instance-type choice for any compute-bound workload, just with a bigger
gap between the cheap and fast options than most web services ever
encounter.
