# Lesson 080 — Deploying an Inference Endpoint to the Cloud

**Module:** 15 — Productionization & Career Transition
**Prerequisites:** [079](../079_containerizing-with-docker/README.md)
**Estimated time:** 1.5–2 hours

## Objective

Taking the Docker image from Lesson 079 and getting it running behind a
public URL. The container is already built and portable — this lesson is
about the deployment target options, the tradeoffs between them, and the
handful of concerns (cold starts, autoscaling, secrets, cost) that are
genuinely different for a GPU-hungry model server versus a typical
stateless web API.

## Contents

1. [01_concepts.md](01_concepts.md)
2. [02_practicals.md](02_practicals.md)
3. [03_solutions.md](03_solutions.md)

## Resources

- [Fly.io docs](https://fly.io/docs/)
- [Render docs](https://render.com/docs)
- [AWS ECS documentation](https://docs.aws.amazon.com/ecs/)
- [Hugging Face Inference Endpoints](https://huggingface.co/docs/inference-endpoints)
