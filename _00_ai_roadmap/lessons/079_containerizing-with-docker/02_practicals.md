# 02 — Practicals: Containerizing ML Apps

1. Write a `Dockerfile` for Lesson 078's FastAPI app (single-stage,
   simplest version). Build it (`docker build -t my-llm-api .`) and run it
   (`docker run -p 8000:8000 my-llm-api`). Confirm you can `curl` the
   `/predict` endpoint from your host machine, exactly as if it were
   running natively.

2. Write a `.dockerignore` excluding `__pycache__`, `*.pyc`, any local
   `venv`/`.venv`, and your raw training data folder. Rebuild and compare
   the build context size reported by Docker (`Sending build context to
   Docker daemon: X MB`) before and after adding `.dockerignore`.

3. Convert your Dockerfile to the multi-stage version from
   `01_concepts.md`. Compare final image sizes (`docker images`) between
   the single-stage and multi-stage versions.

4. Add a `HEALTHCHECK` instruction pointing at Lesson 078's `/health`
   endpoint. Run the container and check `docker ps` — confirm the
   `STATUS` column shows `(healthy)` after the check interval passes, and
   `(unhealthy)` if you deliberately break the health endpoint (e.g. point
   the app at a nonexistent model checkpoint).

5. Write a `docker-compose.yml` that runs your API container with a
   mounted volume for model weights (rather than baking them into the
   image) and an environment variable for the model path. Confirm changing
   the mounted file and restarting the container picks up the new model,
   without rebuilding the image at all — the key benefit of the
   mounted-volume approach from `01_concepts.md`.

6. If you have GPU access: add `--gpus all` to your `docker run` command
   with a CUDA-compatible base image, and confirm `torch.cuda.is_available()`
   returns `True` from inside the running container (it should return
   `False` with the plain `python:3.11-slim` base image, even on a
   GPU-equipped host — confirming the base image, not just the host
   hardware, determines GPU visibility inside the container).
