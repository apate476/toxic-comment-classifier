# Dockerfiles Directory

Container build context for `toxic_comment_classifier`. This README is the
single source of truth for building and running the project inside Docker.

## Contents

- **`Dockerfile`** — Multi-stage build (`builder` → runtime). Base image is `python:3.11-slim-bookworm`, pinned to the Debian *bookworm* release so the OS layer does not drift between rebuilds.

Related files (kept in the project root):

- **`../docker-compose.yaml`** — Compose V2 service definition with bind mounts for `data/`, `models/`, `mlruns/`, `configs/`, and `reports/`.
- **`../.dockerignore`** — Keeps virtualenvs, caches, DVC-pulled datasets, model checkpoints, MLflow runs, and secrets (`.dvc/config.local`) out of the build context.
- **`../scripts/docker_smoke.py`** — One-shot smoke test that verifies imports, path resolution, bind mounts, heavy deps, and raw-data reads from inside the container.

## Prerequisites

1. **Docker Desktop** (or an equivalent Docker Engine) running locally.
   Requires Docker Compose V2 — invoke it as `docker compose` (with a space),
   not the legacy hyphenated `docker-compose`.
2. **DVC-pulled data on the host.** `data/raw/*.csv` is bind-mounted into the
   container rather than baked into the image, so the host must have run
   `dvc pull` first. See `data/README.md` for OAuth setup.
3. **Empty mount targets exist on the host.** The repo ships `.gitkeep`-ed
   `mlruns/`, `models/`, `reports/`, and `configs/` so Docker does not
   silently create them as root-owned.

## Image layout

1. **Stage 1 — `builder`**: installs everything in `requirements.txt` into `/root/.local` via `pip install --user --no-cache-dir`. Isolating the install into a single prefix lets us copy it forward without dragging in pip's metadata or cache.
2. **Stage 2 — runtime**: copies the prepared `/root/.local` from the builder, prepends it to `PATH`, then `pip install -e .` registers the project package against the copied source tree (parity with `make install` on the host).

Environment defaults:

- `PYTHONUNBUFFERED=1` — flush stdout/stderr immediately (correct behavior in a pipe).
- `PYTHONDONTWRITEBYTECODE=1` — skip `.pyc` files on an ephemeral filesystem.
- `EXPOSE 8000` — documents the FastAPI/uvicorn port for future serving containers.
- `ENTRYPOINT` runs `toxic_comment_classifier.train_model`; override at the CLI to run anything else.

## Bind mounts (host ↔ container)

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `./data` | `/app/data` | read-only | DVC-pulled Jigsaw CSVs |
| `./models` | `/app/models` | read-write | Trained model artifacts |
| `./mlruns` | `/app/mlruns` | read-write | MLflow run metadata + artifacts |
| `./configs` | `/app/configs` | read-only | Hydra / YAML configuration |
| `./reports` | `/app/reports` | read-write | Metrics, predictions, figures |

Edits to host configs take effect on the next container run without rebuilding.

## Usage

```bash
# --- Build ------------------------------------------------------------------
# From the repository root:
docker compose build
# or, manually:
docker build -f dockerfiles/Dockerfile -t toxic_comment_classifier:latest .

# --- Run --------------------------------------------------------------------
# Default entrypoint (training):
docker compose up

# One-shot command with auto-cleanup:
docker compose run --rm toxic_comment_classifier \
    python -m toxic_comment_classifier.predict_model --input data/raw/test.csv

# Run the test suite inside the container:
docker compose run --rm --entrypoint pytest toxic_comment_classifier -v

# Interactive shell inside the image:
docker compose run --rm --entrypoint bash toxic_comment_classifier
```

## Smoke test

After a fresh build, verify the image with `scripts/docker_smoke.py`. It is
not part of the image (lives outside the bind mounts), so mount it ad-hoc:

```bash
docker compose run --rm --entrypoint python \
    -v "$(pwd)/scripts:/app/scripts:ro" \
    toxic_comment_classifier scripts/docker_smoke.py
```

Expected last line: `SMOKE TEST: PASS`.

The script checks:

- Package import (`toxic_comment_classifier` → `/app/src/...`).
- `PROJECT_ROOT` resolves to `/app` (not the host path).
- All five bind mounts are populated.
- Heavy deps load (torch, transformers, mlflow, pandas, sklearn).
- DVC-pulled CSVs read cleanly through the bind mount.

## Troubleshooting

- **`FileNotFoundError: /app/data/raw/*.csv`** — host has not run `dvc pull` yet.
- **`mlruns` is owned by root** — happens if Docker creates the dir before the host does. Fix: `sudo rm -rf mlruns && mkdir mlruns && touch mlruns/.gitkeep`.
- **Build context is huge / build is slow** — confirm `.dockerignore` excludes `.venv/`, `data/`, `mlruns/`, and `.history/`.
- **CUDA shows `False` inside the container** — expected on macOS; Docker Desktop does not expose host GPUs. CUDA-tagged torch wheels still install but run in CPU mode.

## Phase

Phase 2 deliverable — Docker infrastructure setup.
