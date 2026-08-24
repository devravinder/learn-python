# 01 — Concepts: Containerizing ML Apps

## If you've containerized a Node/Spring service, this is mostly familiar

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Same structure as a `Dockerfile` for any backend service: base image,
install dependencies, copy code, expose a port, define the start command.
`python:3.11-slim` is the Python equivalent of `node:20-alpine` — a
minimal base image to keep the final image smaller.

## Wrinkle 1: images get large, fast

ML dependencies (`torch`, `transformers`) are large — a naive image can
easily reach several GB, compared to a typical Node service's tens/
hundreds of MB. Mitigations:
- **Multi-stage builds** (same technique as a Node/Go multi-stage build
  that discards build tools from the final image): install dependencies
  in one stage, copy only the needed artifacts into a slim final stage.
- **CPU-only PyTorch builds** when you don't need GPU support in a given
  deployment target — the GPU-enabled build is substantially larger.
- **`.dockerignore`** — exactly like `.gitignore`, exclude datasets,
  notebooks, `__pycache__`, and anything not needed at runtime.

```dockerfile
# multi-stage example
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Wrinkle 2: where do model weights live?

Three common patterns, each with real tradeoffs:
1. **Baked into the image** (`COPY model.pt /app/model.pt`): simplest,
   fully self-contained, but bloats the image and requires a rebuild for
   every model update.
2. **Downloaded at container startup**: keeps the image small, but adds
   startup latency and a runtime dependency on wherever the weights are
   hosted being reachable.
3. **Mounted as a volume** (`docker run -v /host/models:/app/models`):
   image stays generic/reusable across model versions, weights live
   outside the container entirely — the standard choice for production
   deployments where models update independently of application code.

## Wrinkle 3: GPU access from inside a container

```bash
docker run --gpus all my-ml-app
```

Requires the NVIDIA Container Toolkit installed on the host and a
CUDA-compatible base image (e.g. `nvidia/cuda:12.1.0-runtime-ubuntu22.04`
instead of plain `python:3.11-slim`) — a genuinely ML-specific concern
with no direct equivalent in typical Node/Java containerization, since
those workloads essentially never need GPU passthrough.

## docker-compose for local multi-service development

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models/gpt_checkpoint.pt
```

Identical in spirit to any `docker-compose.yml` you've written for a
Node/Postgres/Redis stack — the model-serving container is just another
service in the compose file, no different in kind from an API server
depending on a database container.

## Health checks in the container spec itself

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1
```

Directly uses Lesson 078's `/health` endpoint — Docker (and orchestrators
built on top of it, like Kubernetes) can automatically restart a container
that stops reporting healthy, exactly the same operational pattern as any
other containerized service you've deployed.

## What's genuinely new vs. what transfers (again)

**Transfers directly**: Dockerfile structure, multi-stage builds,
docker-compose for local dev, health checks, the entire mental model of
"package once, run anywhere."

**Genuinely new**: image size management specifically driven by ML
library weight, the three-way tradeoff for where model weights live
(baked in / downloaded / mounted), and GPU passthrough when relevant — a
short, specific list, not a wholesale reinvention of containerization
knowledge you already have.
