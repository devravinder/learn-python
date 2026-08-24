# 01 — Concepts: Deploying an Inference Endpoint to the Cloud

## The container is already the hard part — this is "where does it run"

You've already done the equivalent of `docker build` for a Node/Spring
service and pushed it somewhere before. The deployment-target landscape
for an ML container is the same landscape, with one added axis: does this
workload need a GPU, and if so, who is renting you one.

## Picking a target: the same decision you've made before, plus a GPU column

| Target | Like (your background) | GPU support | Cold start | Best for |
|---|---|---|---|---|
| **Render / Railway / Fly.io** | Heroku-style PaaS | CPU only (cheap tier) | Low | Small models, CPU inference, demos |
| **AWS ECS / Google Cloud Run** | Container orchestration you may have touched | Cloud Run: no GPU (as of most tiers); ECS: yes via GPU instance types | Cloud Run: can be near-zero if kept warm; ECS: depends on scaling policy | Production APIs with real traffic |
| **AWS EC2 / GCP Compute Engine + Docker** | A VM you SSH into and run `docker run` on, same as any VPS deploy you've done | Yes, any GPU instance type | None (always-on) if you don't stop the instance | Full control, batch jobs, cost-sensitive steady load |
| **Hugging Face Inference Endpoints / Replicate / Modal** | No direct analogue — ML-specific managed inference | Yes, first-class | Can be significant (model loading onto GPU) unless kept warm | Getting a model endpoint live fast without managing infra |
| **Kubernetes (EKS/GKE) + KServe/Seldon** | If you've run K8s for microservices, this generalizes directly | Yes, via node pools | Depends on config | Large-scale, multi-model production serving |

The decision tree is identical in spirit to choosing between Heroku,
raw EC2, and Kubernetes for a Node service — this table just adds the
GPU-availability column, which didn't matter before.

## Cold starts are a bigger deal for ML than for a typical API

A cold-started Express server needs to: start a Node process, maybe warm
a DB connection pool. Milliseconds to low seconds.

A cold-started model server needs to: start the process, **and load
gigabytes of weights from disk (or download them) into memory, and
(if GPU) transfer them onto the device** — this can take anywhere from a
few seconds (a small model, local SSD) to a minute or more (a large model,
weights fetched from remote storage). On a serverless/autoscale-to-zero
platform, this cold-start cost is paid on the *first request after
scaling to zero* — a UX problem you may not have had to think about with
a typical stateless API where cold starts are near-instant.

Mitigations, in increasing order of cost:
1. **Keep at least one instance warm** (min-replicas ≥ 1) — you pay for
   idle time, but eliminate cold starts entirely.
2. **Lazy-load lighter, load the model async at startup with a `/health`
   endpoint that reports `not ready` until loading completes** — doesn't
   eliminate the cold start, but at least fails fast/clearly instead of
   hanging a request.
3. **Smaller/quantized model** so the load itself is faster (ties back to
   Lesson 074's quantization content).

## Autoscaling: the trigger metric is different

A typical web API autoscales on request rate or CPU usage. A model server
is often better scaled on:
- **GPU utilization** (if GPU-bound) rather than CPU, which may sit idle
  while the GPU is saturated — a metric most CPU-oriented autoscalers
  don't expose out of the box, requiring GPU-aware tooling (e.g. KServe's
  autoscaling, or a custom metric fed from `nvidia-smi`).
- **In-flight request count / queue depth** rather than raw request rate,
  since a single inference request can be expensive (seconds, not
  milliseconds) — a small number of concurrent requests can already
  saturate a single instance, unlike a typical CRUD endpoint that
  comfortably handles hundreds of concurrent lightweight requests per
  instance.

## Secrets and config: this part is identical to what you already know

API keys, database URLs, and model registry credentials belong in the
platform's secret manager (AWS Secrets Manager, GCP Secret Manager, or
the target platform's env-var UI) — never baked into the image, exactly
the same discipline as any Node/Spring service you've deployed. Nothing
ML-specific here.

## Cost: the one number that changes the conversation

A `t3.small` running a Node API costs a few dollars a month. A single
GPU instance (e.g. an AWS `g4dn.xlarge` or a cloud A10/A100 instance) can
cost anywhere from ~$0.50/hr to several dollars/hr — meaning an
always-on GPU endpoint can run into hundreds or thousands of dollars a
month if not scaled to zero when idle. This is the single biggest
practical reason ML deployment conversations spend so much time on
autoscaling-to-zero and quantization/CPU-inference fallbacks: the
infrastructure cost per request is qualitatively higher than a typical
web service, so idle capacity is expensive in a way it usually isn't for
a stateless API.

## What's genuinely new vs. what transfers

**Transfers directly**: secrets management, the general PaaS vs. VM vs.
Kubernetes decision framework, container registries, CI/CD to trigger
deploys, load balancing, TLS termination — all identical to deploying any
containerized backend service.

**Genuinely new**: GPU availability as a deployment-target filter,
cold-start cost driven by model-loading time rather than process startup,
autoscaling on GPU/queue-depth metrics instead of CPU/request-rate, and a
materially higher cost-per-idle-instance that makes scale-to-zero a much
more central design concern than it typically is for CRUD services.
