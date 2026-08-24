# 02 — Practicals: Cloud Deployment

*(These exercises require an actual cloud/PaaS account and internet
access, neither available in the authoring sandbox — work through them on
your own machine/account. Free tiers exist on Render, Fly.io, and AWS/GCP
for everything below.)*

1. Deploy Lesson 079's Docker image to a PaaS (Render, Railway, or
   Fly.io — pick one). Confirm the `/predict` endpoint is reachable at
   the public URL exactly as it was at `localhost:8000`.

2. Force a cold start: scale the service to zero (or let it idle out on a
   free tier that auto-sleeps), then time how long the *first* request
   after waking takes vs. a request 30 seconds later. Record both
   numbers.

3. Set `MODEL_PATH` (and any API keys your app uses) via the platform's
   environment-variable/secrets UI rather than hardcoding them — confirm
   the app reads them correctly, and confirm the values never appear in
   your Dockerfile or in your git history.

4. Configure a minimum of 1 warm instance (min-replicas ≥ 1, or the
   platform's equivalent "always on" setting) and repeat exercise 2's
   timing test. Confirm the cold-start penalty disappears, and note the
   cost difference the platform reports for always-on vs. scale-to-zero.

5. Push a code change (e.g. edit the `/predict` response to include a
   `model_version` field) and redeploy via the platform's normal
   git-push-to-deploy or CLI flow. Confirm the change is live at the
   public URL without any manual SSH/file-copy step — the same workflow
   you'd use redeploying any Node/Spring service to the same platform.

6. If you have access to a GPU-enabled cloud target (AWS `g4dn.xlarge`,
   Lambda Labs, or similar): deploy the same image there instead, and
   compare inference latency for one `/predict` call against the CPU-only
   deployment from exercise 1. Record both latencies and the hourly cost
   of each instance type.
