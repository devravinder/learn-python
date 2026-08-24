# Multi-stage: build the static React app, then serve it with a tiny
# static file server — same pattern as Lesson 079's multi-stage build,
# applied to a frontend instead of a Python backend.
FROM node:20-slim AS builder
WORKDIR /app
COPY package.json .
RUN npm install
COPY . .
# Vite bakes VITE_* env vars into the bundle at build time, not at
# container runtime — so this must be a build ARG, not a `docker run -e`
# / compose `environment:` value (which would silently have no effect).
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

FROM node:20-slim
WORKDIR /app
RUN npm install -g serve
COPY --from=builder /app/dist ./dist
EXPOSE 5173
CMD ["serve", "-s", "dist", "-l", "5173"]
