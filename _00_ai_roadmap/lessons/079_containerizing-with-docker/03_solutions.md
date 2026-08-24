# 03 — Solutions: Containerizing ML Apps

*(No running Docker daemon was available in the authoring sandbox — the
`docker` CLI was present but couldn't connect to a daemon, so none of this
was independently executed. Every command/Dockerfile follows standard,
well-documented Docker usage precisely; run it yourself to confirm the
actual sizes/behavior on your machine.)*

## 1. Basic Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t my-llm-api .
docker run -p 8000:8000 my-llm-api
curl -X POST localhost:8000/predict -d '{"text": "hello"}' -H "Content-Type: application/json"
```

The `curl` should work identically to running the app natively — the
container is just an isolated environment running the same process,
exposed via the mapped port.

## 2. `.dockerignore` effect on build context

```text
__pycache__/
*.pyc
.venv/
venv/
data/raw_training_data/
*.pt
```

Before adding this, `docker build`'s reported "Sending build context to
Docker daemon" size includes everything in the directory (potentially
gigabytes of training data/checkpoints you don't want copied into every
build); after adding it, that number should drop substantially — Docker
never even sends the excluded files to the build process, the same
mechanism `.gitignore` uses for `git add`.

## 3. Multi-stage vs single-stage image size

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
WORKDIR /app
ENV PATH=/root/.local/bin:$PATH
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker images | grep my-llm-api
```

The multi-stage image should be noticeably smaller than the single-stage
version — build-time-only files (pip's cache, any compiler toolchains
needed to build certain Python packages from source) never make it into
the final image's layers.

## 4. HEALTHCHECK

```dockerfile
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

```bash
docker run -d -p 8000:8000 --name llm-api my-llm-api
docker ps   # STATUS column should show "(health: starting)" then "(healthy)"
```

Point the app at a nonexistent checkpoint path and rebuild/rerun — `docker
ps` should show `(unhealthy)` once the health check's retry budget is
exhausted, exactly mirroring what Lesson 078's `/health` endpoint reports
directly.

## 5. docker-compose with a mounted model volume

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

```bash
docker compose up
# replace ./models/gpt_checkpoint.pt with a new checkpoint file
docker compose restart api
```

Since the model file lives on the host filesystem (mounted, not copied
into the image), swapping the file and restarting the container (not
rebuilding the image) should pick up the new model — no `docker build`
step needed at all for a model update, exactly the deployment-flexibility
benefit `01_concepts.md` describes.

## 6. GPU access

```bash
docker run --gpus all -p 8000:8000 my-llm-api-cuda
```

With the plain `python:3.11-slim` base image, `torch.cuda.is_available()`
should return `False` inside the container even on a GPU-equipped host —
the base image needs CUDA libraries and drivers-compatible tooling
present (e.g. `nvidia/cuda:12.1.0-runtime-ubuntu22.04` as the base, plus
a CUDA-enabled PyTorch install) for GPU passthrough to actually work; the
host having a GPU is necessary but not sufficient on its own.
