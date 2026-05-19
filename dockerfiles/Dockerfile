# syntax=docker/dockerfile:1.7
# =============================================================================
# Multi-stage build for toxic_comment_classifier
#
# Stage 1 (builder): installs all pinned dependencies into an isolated prefix
# so that the final runtime image contains only what is needed to run the
# package, not pip's build/cache artifacts.
#
# Stage 2 (runtime): copies the prepared site-packages prefix from the
# builder and installs the project itself in editable mode for development
# parity with `pip install -e .` on the host.
#
# Base image: python:3.11-slim-bookworm pins the Debian release (bookworm)
# so image contents do not drift as new Debian releases ship. Using the
# `slim` flavor keeps the image small while still providing apt + glibc.
# =============================================================================

# --- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

WORKDIR /app

# Install dependencies into /root/.local via pip's --user flag. This keeps the
# install isolated to a single prefix that can be copied wholesale into the
# runtime stage, avoiding any pip metadata or cache leaking forward.
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim-bookworm

WORKDIR /app

# Pull in the populated user-site from the builder stage.
COPY --from=builder /root/.local /root/.local

# PATH: make the user-site console scripts (dvc, mlflow, uvicorn, ...) visible.
# PYTHONUNBUFFERED: stream logs immediately instead of buffering, which is the
#   expected behavior inside a container where stdout is a pipe.
# PYTHONDONTWRITEBYTECODE: skip writing .pyc files; they are not useful when
#   the container filesystem is ephemeral and they slow first import slightly.
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy the project source. .dockerignore keeps virtualenvs, caches, DVC data,
# and other host-only state out of the build context.
COPY . .

# Editable install registers the `toxic_comment_classifier` package against the
# copied source tree. This matches `make install` on the host.
RUN pip install --no-cache-dir -e .

# FastAPI / uvicorn default port. The training entrypoint does not bind it,
# but exposing here documents the contract for serving containers later.
EXPOSE 8000

# Default to training. Override with `docker compose run <svc> <cmd>` or
# `docker run ... <image> <cmd>` to run prediction, the API, tests, etc.
ENTRYPOINT ["python", "-m", "toxic_comment_classifier.train_model"]
