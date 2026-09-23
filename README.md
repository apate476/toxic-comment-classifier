# Toxic Comment Classifier

[![CI](https://github.com/apate476/toxic-comment-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/apate476/toxic-comment-classifier/actions/workflows/ci.yml)
[![Code quality](https://github.com/apate476/toxic-comment-classifier/actions/workflows/codecheck.yaml/badge.svg)](https://github.com/apate476/toxic-comment-classifier/actions/workflows/codecheck.yaml)
[![Docker publish](https://github.com/apate476/toxic-comment-classifier/actions/workflows/docker-publish.yaml/badge.svg)](https://github.com/apate476/toxic-comment-classifier/actions/workflows/docker-publish.yaml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Multi-label toxicity classification for online comments, built end to end: a reproducible scikit-learn baseline, a full MLOps pipeline around it (Hydra, MLflow, DVC, Docker), and a FastAPI inference service deployed on Google Cloud Run behind GitHub Actions CI/CD.

A single comment can carry several kinds of toxicity at once, so each comment is scored independently against six labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

**Live API:** <https://toxic-comment-api-491682843765.us-central1.run.app> &nbsp;|&nbsp; **Interactive OpenAPI docs:** [`/docs`](https://toxic-comment-api-491682843765.us-central1.run.app/docs)

## Try it

Against the deployed service, no setup required:

```bash
# Liveness probe
curl https://toxic-comment-api-491682843765.us-central1.run.app/health
# {"status":"ok","service":"toxic-comment-classifier-api","model_available":true, ...}

# Score a batch of comments (1-100 per request)
curl -X POST https://toxic-comment-api-491682843765.us-central1.run.app/predict \
  -H 'Content-Type: application/json' \
  -d '{"comments": ["you are a wonderful person", "i will hunt you down"]}'
```

```jsonc
{
  "predictions": [
    {
      "comment": "you are a wonderful person",
      "labels": ["non_toxic"],
      "probabilities": {
        "toxic": 0.4719, "severe_toxic": 0.0522, "obscene": 0.2216,
        "threat": 0.028,  "insult": 0.2205,       "identity_hate": 0.0557
      }
    }
  ]
}
```

Or run the whole pipeline locally:

```bash
git clone https://github.com/apate476/toxic-comment-classifier.git
cd toxic-comment-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

make train      # trains the baseline, writes models/ and reports/
make predict    # scores data/raw/test.csv into reports/predictions.csv
make test       # 59 tests, 90% coverage

uvicorn api.main:app --reload   # serve locally on http://127.0.0.1:8000/docs
```

Full setup notes, including Docker and the Hydra/MLflow workflows, are in [Setup Instructions](#setup-instructions).

## At a glance

| | |
| --- | --- |
| **Task** | Multi-label text classification across 6 toxicity labels |
| **Data** | Jigsaw Toxic Comment Classification Challenge, 159,571 Wikipedia talk-page comments |
| **Model** | TF-IDF (50k features, uni+bigrams) with One-vs-Rest Logistic Regression |
| **Validation** | Micro F1 **0.658**, micro precision **0.886**, Hamming loss **0.020** |
| **Serving** | FastAPI on Cloud Run, `GET /health` and `POST /predict` |
| **Reproducibility** | Hydra config groups, MLflow tracking, DVC-versioned data, multi-stage Docker build |
| **CI/CD** | pytest on Python 3.10/3.11/3.12 at 90% coverage, ruff, mypy, CML training reports on every PR, Docker Hub publish |

## Project status

| Phase | Scope | Status |
| --- | --- | --- |
| **Phase 1** | Reproducible TF-IDF + One-vs-Rest Logistic Regression baseline, data validation, metrics, model artifact | Delivered |
| **Phase 2** | Hydra configuration management, MLflow experiment tracking, DVC, Docker, profiling, structured logging | Delivered |
| **Phase 3** | Test suite and CI, CML training reports, Docker publishing, FastAPI service, Cloud Run deployment, Streamlit demo | Delivered |
| **Next** | Fine-tuned DistilBERT to lift recall on the rare labels (`threat`, `identity_hate`), load testing, monitoring dashboards | Open |

## Documentation

| Document | What it covers |
| --- | --- |
| [PHASE1.md](PHASE1.md) | Project design and baseline model development |
| [PHASE2.md](PHASE2.md) | Containerization, configuration, tracking, and monitoring |
| [PHASE3.md](PHASE3.md) | CI/CD, CML, and cloud deployment |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and component boundaries |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branching model, commit conventions, and review process |
| [docs/](docs/) | API reference, GCP deployment, Vertex AI training, Streamlit demo |
| [data/README.md](data/README.md) | Data folder layout and DVC handling |
| [TESTING.md](TESTING.md) | Running the test suite locally and in CI |

## Team Information

- **Project Lead:** team_toxic (apate424@depaul.edu)
- **Team Members:** Taha Patil, Arya Patel, Bilal Qader, Asad Khan

## Overview

toxic_comment_classifier is a machine learning project focused on detecting toxic language in online comments. The goal is to classify each comment into one or more toxicity categories, including toxic, severe_toxic, obscene, threat, insult, and identity_hate.

For Phase 1, the project establishes a reproducible baseline using TF-IDF feature extraction and a One-vs-Rest Logistic Regression classifier. This baseline provides an initial performance reference before more advanced models, such as transformer-based architectures, are explored in later phases.

The repository follows an MLOps-oriented structure with separate folders for source code, data, tests, reports, model artifacts, and documentation. Data handling is supported through DVC, while model training and prediction are implemented as command-line entrypoints.

**Key Objectives:**

- Build a reproducible baseline model for multi-label toxic comment classification.
- Establish clear project structure, data handling, and model training workflows.
- Save baseline metrics and predictions for evaluation and future comparison.

**Business Context:**
Online platforms receive millions of user comments daily, making manual moderation impossible at scale. Toxic content left unmoderated leads to user churn, reputational damage, and potential legal liability. An automated classifier that can detect multiple forms of toxicity simultaneously provides a scalable solution for real-time content moderation.

**Technical Approach:**
The task is framed as multi-label classification, meaning a single comment can simultaneously belong to multiple toxicity categories. The Phase 1 deliverable is a TF-IDF + One-vs-Rest Logistic Regression baseline trained on the Jigsaw Toxic Comment Classification dataset (~159,000 Wikipedia talk-page comments labeled across 6 toxicity categories). Phase 3 wrapped that baseline in a FastAPI service deployed to Cloud Run, with GitHub Actions running tests, linting, type checks, CML training reports, and Docker publishing on every pull request. The next planned upgrade swaps the classifier for a fine-tuned DistilBERT model — a lightweight transformer that retains ~97% of BERT's language understanding at a fraction of the size — using a 6-unit sigmoid output head for the multi-label task.

The MLOps infrastructure prioritizes reproducibility and collaboration. MLflow tracks all experiment parameters, metrics, and model artifacts. DVC with a Google Drive remote handles data versioning. GitHub Actions powers CI/CD (Phase 3), running linting via ruff, type checking via mypy, and tests on every pull request. All feature development occurs on the dev branch via short-lived feature branches, with main reserved for end of phase merges.

**Expected Outcomes by Phase:**

- **Phase 1 (delivered):** Reproducible TF-IDF + Logistic Regression baseline with logged metrics, a versioned model artifact, and documented data handling.
- **Phase 2 (delivered):** Containerized training environment, Hydra config management, MLflow experiment tracking, profiling reports, and structured logging.
- **Phase 3 (delivered):** FastAPI inference service deployed to Cloud Run, GitHub Actions CI/CD on every PR (tests, lint, type checks, CML training reports, Docker publishing), and a Streamlit demo client.
- **Next:** fine-tuned DistilBERT classifier to lift recall on the rare labels, plus load testing and monitoring dashboards.

## Scope & Objectives

**Problem Statement:**
Many online platforms struggle to moderate toxic user-generated content at scale. This project builds an automated multi-label toxic comment classifier that detects six categories of toxicity simultaneously, delivered across three phases: a reproducible TF-IDF + Logistic Regression baseline (Phase 1), the surrounding MLOps tooling — containerization, configuration management, experiment tracking, and structured logging (Phase 2), and a FastAPI inference service deployed to Cloud Run behind GitHub Actions CI/CD (Phase 3).

**Goals:**

- **Phase 1** — Train a reproducible TF-IDF + One-vs-Rest Logistic Regression baseline on the Jigsaw dataset for multi-label classification ✅
- **Phase 2** — Establish a reproducible MLOps pipeline with Hydra, MLflow, DVC, Docker, profiling, and structured logging ✅
- **Phase 3** — Ship a FastAPI inference endpoint on Cloud Run with GitHub Actions CI/CD, CML training reports, and automated Docker publishing ✅

**Success Metrics:**

- Macro F1 Score across all 6 labels
- ROC-AUC per label on the test set
- Experiment reproducibility via MLflow with fixed random seeds
- Model artifact versioning
- CI/CD pipeline status on pull requests into dev

## Dataset

The project uses a toxic comment classification dataset containing online comments labeled across six toxicity categories.

### Label Columns

| Label         | Description                           |
| ------------- | ------------------------------------- |
| toxic         | General toxic or harmful language     |
| severe_toxic  | Strongly toxic language               |
| obscene       | Obscene or inappropriate language     |
| threat        | Threatening language                  |
| insult        | Insulting language                    |
| identity_hate | Hate speech targeting identity groups |

### Dataset Files

| File                       | Purpose                                    |
| -------------------------- | ------------------------------------------ |
| `data/raw/train.csv`       | Training data with comment text and labels |
| `data/raw/test.csv`        | Test data with comment text                |
| `data/raw/test_labels.csv` | Label file for test data                   |
| `reports/predictions.csv`  | Generated model predictions                |

The training file contains 159,571 labeled comments. For Phase 1, the model uses an 80/20 train-validation split from `train.csv`.

### Selection Rationale

**Selected Dataset:** Jigsaw Toxic Comment Classification Challenge
**Source:** Kaggle mirror — `julian3833/jigsaw-toxic-comment-classification-challenge`

**Justification:**

- The 6-label multi-label structure directly matches the problem requirements, alternatives like Twitter Hate Speech and Civil Comments only provide binary toxic or non-toxic labels
- Large enough at approximately 159,000 comments to fine-tune a transformer model effectively
- Originates from real Wikipedia talk page edits providing naturally occurring toxic and non-toxic text
- Widely used in NLP research providing a reliable benchmark for comparing results

### Schema

**Size:** ~159,571 comments

**Features:**

- `id` — unique comment identifier
- `comment_text` — raw Wikipedia talk page comment
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — binary labels (0 or 1)

**Format:** CSV

**Source:** Wikipedia talk page edits, labeled by human raters via the Jigsaw/Conversation AI project

## Model Considerations

- **TF-IDF + One-vs-Rest Logistic Regression** *(Phase 1, delivered)* — lightweight baseline that establishes a performance floor quickly and trains end-to-end in seconds on CPU.
- **TF-IDF + LightGBM** *(optional Phase 2 comparison)* — stronger classical baseline that handles non-linear feature interactions better than logistic regression.
- **DistilBERT with a 6-unit sigmoid output head** *(next planned upgrade)* — transformer model whose pretrained language understanding is expected to lift recall on the rare labels (`threat`, `identity_hate`); requires GPU access and replaces the sklearn pipeline.

## Architecture Diagram

```text
Raw Data
   |
   v
Data Validation
   |
   v
TF-IDF Feature Extraction
   |
   v
One-vs-Rest Logistic Regression
   |
   v
Validation Metrics + Saved Model
   |
   v
Predictions on Test Data
```

## Phase Deliverables

### Phase 1: Project Design & Model Development

See [PHASE1.md](PHASE1.md) for the detailed Phase 1 checklist and model training summary.

### Phase 2: Containerization & Monitoring

See [PHASE2.md](PHASE2.md) for the Phase 2 checklist.

## Phase 2 Additions

This phase introduces configuration management, structured logging, containerization, profiling, and experiment tracking. Full documentation lives in [PHASE2.md](./PHASE2.md).

### Configuration Management with Hydra

All hyperparameters, paths, and model knobs are managed by [Hydra](https://hydra.cc/). The config tree lives in `configs/` with one subfolder per config group (`data/`, `features/`, `model/`, `training/`). Run training with defaults or override any value on the CLI without editing code:

```bash
# Default baseline run
python -m toxic_comment_classifier.train_model

# Override hyperparameters
python -m toxic_comment_classifier.train_model model.C=10 features.max_features=20000
```

Every run writes a full config snapshot and override list to `outputs/<date>/<time>/.hydra/`, making each experiment reproducible.

### Application Logging

Logging is centralized in `src/toxic_comment_classifier/logging_config.py`. Console output uses `rich.logging.RichHandler` for colored, leveled output during development. A `RotatingFileHandler` writes structured plain-text logs to `logs/training.log` (and `logs/prediction.log` for inference), capped at 5 MB per file with up to 5 backups. Uncaught exceptions are rendered with `rich.traceback.install()` for source context in errors.

### Phase 3: CI/CD & Deployment

See [PHASE3.md](PHASE3.md) for the Phase 3 checklist.

## Setup Instructions

### Prerequisites

- Python 3.11+
- Git
- pip
- Optional: Docker and Docker Compose

### Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/apate476/toxic-comment-classifier.git
cd toxic-comment-classifier
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

For development tools, install:

```bash
pip install -r requirements_dev.txt
```

If the project uses the `src/` layout, install the package in editable mode:

```bash
pip install -e .
```

If editable installation is not available, run commands with:

```bash
PYTHONPATH=src
```

## Development Setup

Set up pre-commit hooks:

```bash
pre-commit install
```

Run tests to verify the environment:

```bash
pytest tests/
```

## Running the Pipeline

### Train the Baseline Model

From the project root, run:

```bash
python -m toxic_comment_classifier.train_model --data-path data/raw
```

This trains the Phase 1 baseline model using `data/raw/train.csv`.

The trained model is saved to:

```text
models/baseline_tfidf_logreg.joblib
```

The validation metrics are saved to:

```text
reports/baseline_metrics.json
```

### Generate Predictions

After training, run:

```bash
python -m toxic_comment_classifier.predict_model --input data/raw/test.csv
```

Predictions are saved to:

```text
reports/predictions.csv
```

### Common Make Commands

```bash
# Prepare data
make data

# Train the model
make train

# Generate predictions
make predict

# Run tests
make test

# Run linting checks
make lint

# Auto-format code
make format

# See all available commands
make help
```

## Containerization

The project ships a reproducible Docker setup so training and inference run identically on any host. Build configuration lives in `dockerfiles/Dockerfile`; runtime orchestration lives in `docker-compose.yaml` at the repository root.

### Image overview

- Base image: `python:3.11-slim-bookworm` (pinned to the Debian *bookworm* release).
- Multi-stage build: dependencies are installed into an isolated user-site in a `builder` stage and copied into a clean runtime stage.
- `.dockerignore` keeps virtualenvs, DVC-pulled datasets, MLflow runs, model artifacts, secrets, and caches out of the build context.

### Bind mounts (host ↔ container)

| Host path   | Container path | Mode       | Purpose                         |
| ----------- | -------------- | ---------- | ------------------------------- |
| `./data`    | `/app/data`    | read-only  | DVC-pulled Jigsaw CSVs          |
| `./models`  | `/app/models`  | read-write | Trained model artifacts         |
| `./mlruns`  | `/app/mlruns`  | read-write | MLflow run metadata + artifacts |
| `./configs` | `/app/configs` | read-only  | Hydra / YAML configuration      |
| `./reports` | `/app/reports` | read-write | Metrics, predictions, figures   |

### Build and run

```bash
# Build the image
docker compose build

# Run the default entrypoint (training)
docker compose up

# Run a different command (predict, test, shell, etc.)
docker compose run --rm toxic_comment_classifier \
    python -m toxic_comment_classifier.predict_model --input data/raw/test.csv

# Interactive shell inside the image
docker compose run --rm --entrypoint bash toxic_comment_classifier
```

> Prerequisite: Docker Desktop (or an equivalent Docker Engine) must be running. The repo expects `data/raw/*.csv` to already be DVC-pulled on the host because `data/` is bind-mounted into the container rather than baked into the image.

## Phase 2 Tooling Guide

Phase 2 adds five operational tools on top of the Phase 1 baseline: configuration management (Hydra), experiment tracking (MLflow), structured logging (Rich), profiling (cProfile / memory-profiler), and containerization (Docker). This section documents how to set up and use each one. The full deliverable checklist lives in [PHASE2.md](PHASE2.md); system diagrams live in [ARCHITECTURE.md](ARCHITECTURE.md).

### Documentation Map

| Topic                            | Where to look                                                       |
| --------------------------------- | ------------------------------------------------------------------ |
| Project overview, setup, commands | This README                                                         |
| Phase 2 tool usage                | This section                                                        |
| System architecture + diagrams    | [ARCHITECTURE.md](ARCHITECTURE.md)                                  |
| Phase deliverable checklists       | [PHASE1.md](PHASE1.md) / [PHASE2.md](PHASE2.md) / [PHASE3.md](PHASE3.md) |
| Contribution workflow              | [CONTRIBUTING.md](CONTRIBUTING.md)                                  |

### Configuration Management (Hydra)

All hyperparameters, paths, and model knobs are managed by **Hydra** (`hydra-core==1.3.2`). No code edits are needed to run a new experiment — every value is defined in YAML under `configs/` and can be overridden from the command line.

**Config layout**

```text
configs/
├── config.yaml           # Root config: defaults list + global seed
├── data/jigsaw.yaml      # Dataset paths, split ratio, column names
├── features/tfidf.yaml   # TF-IDF vectorizer settings
├── model/logreg.yaml     # Logistic Regression hyperparameters
└── training/default.yaml # Output paths and filenames
```

The root `config.yaml` composes the groups through a `defaults` list and sets a global `seed: 42`.

**Running with different configurations**

```bash
# 1. Default baseline run (C=1.0, max_features=50000, ngram_range=[1,2])
python -m toxic_comment_classifier.train_model

# 2. Stronger regularization with a smaller vocabulary
python -m toxic_comment_classifier.train_model model.C=10 features.max_features=20000

# 3. Unigrams only
python -m toxic_comment_classifier.train_model features.ngram_range=[1,1]

# 4. Override multiple groups at once
python -m toxic_comment_classifier.train_model model.C=0.5 model.penalty=l1 data.val_split=0.1
```

Override syntax is `key=value` or `group.subkey=value` — no argparse flags, no code changes. Every run writes a full config snapshot to `outputs/<date>/<time>/.hydra/config.yaml` and the override list to `overrides.yaml`, so each experiment is reproducible from disk.

### Experiment Tracking (MLflow)

Experiment tracking uses **MLflow** (`mlflow==3.11.1`) with a local file-based tracking store. No server setup is required — runs are written to the `mlruns/` directory at the repo root.

**Setup**

MLflow is installed with the project dependencies (`pip install -r requirements.txt`). The tracking URI and experiment name are configured in code (`scripts/run_mlflow_experiments.py`):

```python
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("toxic-comment-phase2")
```

**Running tracked experiments**

```bash
# Trains and logs 3 experiment configurations to MLflow
python scripts/run_mlflow_experiments.py
```

This trains three pre-defined configurations — `baseline_tfidf_logreg`, `smaller_tfidf_logreg`, and `balanced_tfidf_logreg` — and for each run logs the hyperparameters (`max_features`, `ngram_range`, `C`, `class_weight`), the metrics (`micro_f1`, `macro_f1`, `micro_precision`, `micro_recall`, `hamming_loss`), and the trained model artifact. A combined comparison is also written to `reports/experiments/experiment_results.csv` and `.json`.

**Viewing and comparing runs**

```bash
# Launch the MLflow UI, then open http://localhost:5000
mlflow ui

# Generate comparison charts from the logged results
python scripts/plot_experiment_results.py
```

**Selecting the best model:** compare runs in the MLflow UI (or the `experiment_results.csv` table) and pick the configuration with the highest **macro F1** — macro F1 weights the rare labels (`threat`, `identity_hate`) equally with common ones, which matters for this imbalanced multi-label dataset.

### Logging

Logging is centralized in `src/toxic_comment_classifier/logging_config.py`. The `setup_logging()` function attaches two handlers to the root logger:

- **Console** — `rich.logging.RichHandler` with colored levels, timestamps, and pretty tracebacks (for interactive development).
- **File** — `RotatingFileHandler` writing plain text to `logs/`, capped at 5 MB per file with up to 5 backups (~25 MB total, so disk usage is bounded).

Both `train_model.py` and `predict_model.py` call `setup_logging()` once at the start of `main()`. Inference logs route to a separate file by passing `setup_logging(log_filename="prediction.log")`.

**Usage example**

```python
import logging
from toxic_comment_classifier.logging_config import setup_logging

setup_logging()                       # training -> logs/training.log
logger = logging.getLogger(__name__)

logger.info("Loading training data from %s", data_path)
logger.warning("Found %d rows with empty comment_text", n_empty)
logger.error("Validation failed: missing label column", exc_info=True)
```

**Log levels:** `INFO` for normal progress (paths, timing, completion), `WARNING` for recoverable issues, `ERROR` for caught exceptions with context, `DEBUG` for verbose tracing (disabled by default). The composed Hydra config is logged at the top of every run, so any teammate reading `logs/training.log` can see exactly which hyperparameters produced the saved model.

### Debugging & Profiling

**Interactive debugging.** Drop a breakpoint anywhere in the code and run the entrypoint normally:

```python
breakpoint()          # built-in pdb; or: import ipdb; ipdb.set_trace()
```

To debug inside the container, run with an interactive shell:

```bash
docker compose run --rm --entrypoint bash toxic_comment_classifier
```

**Container smoke test.** `scripts/docker_smoke.py` verifies package import, config path resolution, bind-mount visibility, dependency versions, and that DVC-pulled data is reachable from inside the container:

```bash
docker compose run --rm --entrypoint python toxic_comment_classifier scripts/docker_smoke.py
```

**CPU profiling.** `scripts/profile_training.py` wraps the training pipeline in `cProfile` and writes both a binary and a human-readable report:

```bash
python scripts/profile_training.py
# -> reports/profiling/training_cpu_profile.prof   (open with snakeviz/pstats)
# -> reports/profiling/training_cpu_profile.txt    (top 30 functions by cumulative time)
```

**Memory profiling.** `scripts/profile_memory.py` samples peak memory during training using `memory-profiler`:

```bash
python scripts/profile_memory.py
# -> reports/profiling/training_memory_profile.txt  (starting / peak / increase MiB)
```

**Framework profiling (Scalene).** `scripts/profile_scalene.py` runs training under [Scalene](https://github.com/plasma-umass/scalene), which attributes time spent inside C extensions (TF-IDF, BLAS) back to the originating Python line — something `cProfile` cannot do for this sklearn pipeline:

```bash
python scripts/profile_scalene.py
# -> reports/profiling/training_scalene_profile.html   (line-by-line CPU + memory)
```

### Performance Guide

Use this loop to profile and optimize:

1. **Measure first.** Run `scripts/profile_training.py`, `scripts/profile_memory.py`, and `scripts/profile_scalene.py` to capture a baseline before changing anything. Each tool reports something the others miss — see PHASE2.md §3 for the matrix.
2. **Find the bottleneck.** Open `training_cpu_profile.txt` for the cProfile cumulative view, or the Scalene HTML report for line-level CPU + memory attribution inside sklearn / NumPy. For this TF-IDF + Logistic Regression pipeline, vectorization and the per-label classifier `fit` dominate runtime.
3. **Optimize one thing.** Typical levers: lower `features.max_features`, narrow `features.ngram_range`, or adjust the solver. Change one config value at a time.
4. **Re-measure.** Re-run the profiler and compare. Training timings are also persisted to `reports/baseline_metrics.json` as `fit_seconds` / `predict_seconds`, and across MLflow runs you can chart them with `scripts/plot_experiment_results.py`.
5. **Document the result.** Record the before/after numbers so the optimization is justified, not assumed.

### How the Tools Work Together

A single training run touches every Phase 2 tool in sequence:

```text
configs/ (Hydra)
   │  composes cfg at runtime, snapshots to outputs/<date>/<time>/.hydra/
   ▼
train_model.py  ──▶  logging_config.py  ──▶  logs/training.log  (Rich + rotating file)
   │
   ├──▶  reads data from data/raw/  (DVC-pulled, bind-mounted in Docker)
   │
   ├──▶  MLflow logs params + metrics + model artifact ──▶  mlruns/
   │
   └──▶  writes models/*.joblib  +  reports/baseline_metrics.json

Docker  wraps the whole pipeline so it runs identically on any host.
cProfile / memory-profiler  attach to the same train_model.main() entrypoint.
```

In short: **Hydra** decides *what* runs, **logging** records *what happened*, **MLflow** records *what the results were*, **profiling** measures *how fast/heavy it was*, and **Docker** guarantees it all behaves the same everywhere.

### Examples — Common Workflows

```bash
# Full local setup
make dev

# Baseline training run (default config)
make train

# Experiment sweep with overrides
python -m toxic_comment_classifier.train_model model.C=10 features.max_features=20000

# Tracked multi-experiment comparison
python scripts/run_mlflow_experiments.py && mlflow ui

# Profile the pipeline
python scripts/profile_training.py
python scripts/profile_memory.py
python scripts/profile_scalene.py

# Containerized training
docker compose build && docker compose up
```

### Troubleshooting

| Symptom                                          | Likely cause                              | Fix                                                                                          |
| ------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: toxic_comment_classifier`   | Package not installed in editable mode    | Run `pip install -e .` (or `make install`); or prefix commands with `PYTHONPATH=src`         |
| `Cannot find primary config 'config'` (Hydra)     | Command run from outside the repo root    | `cd` to the repo root before running `train_model`                                          |
| Profiling scripts can't import the package        | Editable install missing                  | `pip install -e .`, then run scripts from the repo root                                      |
| `No module named memory_profiler`                 | Profiling dependency not installed        | `pip install memory-profiler` (included in `requirements.txt`)                               |
| `mlflow ui` fails — port 5000 in use              | Another process holds the port            | `mlflow ui --port 5001`                                                                       |
| MLflow runs don't appear in the UI                | UI started from a different directory     | Run `mlflow ui` from the repo root so it reads `./mlruns`                                    |
| `dvc pull` fails with an auth error               | Google Drive credentials not configured   | Set local `gdrive_client_id` / `gdrive_client_secret`, then re-run `dvc pull`                |
| Docker run can't find `data/raw/*.csv`            | Data not DVC-pulled on the host           | Run `dvc pull` on the host first — `data/` is bind-mounted, not baked into the image         |
| `docker compose` build fails — Docker not running | Docker Desktop is stopped                 | Start Docker Desktop and retry                                                                |
| `pre-commit` blocks a commit                      | ruff / mypy found issues                  | Run `make format` then `make lint`; fix remaining type errors before committing              |

### Version Compatibility

The project targets **Python 3.11+**. All runtime versions are pinned in `requirements.txt`; development tools are pinned in `requirements_dev.txt`. Key versions:

| Tool            | Version    | Purpose                                  |
| --------------- | ---------- | ---------------------------------------- |
| Python          | 3.11+      | Runtime                                  |
| numpy           | 2.4.x      | Numerical computing                      |
| pandas          | 2.3.x      | Data loading / manipulation              |
| scikit-learn    | 1.8.x      | TF-IDF, Logistic Regression, metrics     |
| joblib          | 1.5.x      | Model persistence                        |
| hydra-core      | 1.3.2      | Configuration management                 |
| omegaconf       | 2.3.x      | Config object / type system              |
| mlflow          | 3.11.x     | Experiment tracking                      |
| dvc             | 3.67.x     | Data versioning (with `dvc-gdrive`)      |
| rich            | 15.0.x     | Console logging / tracebacks             |
| memory-profiler | latest     | Memory profiling                         |
| matplotlib      | 3.10.x     | Experiment comparison plots              |
| pytest          | 9.0.x      | Test runner                              |
| ruff            | 0.15.x     | Lint + format                            |
| mypy            | 1.20.x     | Static type checking                     |
| pre-commit      | 4.6.x      | Git hook automation                      |
| Docker          | 24+ engine | Containerization                         |

> Reproduce the exact environment with `pip install -r requirements.txt`. If you upgrade a pinned package, re-run `make test` and the profiling scripts to confirm nothing regressed.

## Baseline Model Performance

The Phase 1 baseline model uses TF-IDF vectorization with a One-vs-Rest Logistic Regression classifier. The model was trained on `data/raw/train.csv`, which contains 159,571 labeled comments, using an 80/20 train-validation split.

### Model Configuration

| Component          | Value                           |
| ------------------ | ------------------------------- |
| Feature extraction | TF-IDF                          |
| Maximum features   | 50,000                          |
| N-gram range       | Unigrams and bigrams            |
| Stop words         | English                         |
| Classifier         | One-vs-Rest Logistic Regression |
| Solver             | liblinear                       |
| Max iterations     | 1000                            |
| Validation split   | 20%                             |
| Random seed        | 42                              |

### Validation Metrics

| Metric          |  Score |
| --------------- | -----: |
| Micro F1        | 0.6581 |
| Macro F1        | 0.4738 |
| Micro Precision | 0.8865 |
| Micro Recall    | 0.5233 |
| Hamming Loss    | 0.0201 |

### Per-Label Breakdown

| Label           | Precision | Recall | F1     | Support |
| --------------- | --------: | -----: | -----: | ------: |
| `toxic`         |    0.9208 | 0.5821 | 0.7133 |   3,056 |
| `severe_toxic`  |    0.5484 | 0.2118 | 0.3056 |     321 |
| `obscene`       |    0.9229 | 0.6006 | 0.7277 |   1,715 |
| `threat`        |    0.6471 | 0.1486 | 0.2418 |      74 |
| `insult`        |    0.8306 | 0.4771 | 0.6061 |   1,614 |
| `identity_hate` |    0.7333 | 0.1497 | 0.2486 |     294 |
| **micro avg**   |    0.8865 | 0.5233 | 0.6581 |   7,074 |
| **macro avg**   |    0.7672 | 0.3617 | 0.4738 |   7,074 |

Generated by `sklearn.metrics.classification_report`; the full report and confusion matrix grid live in `reports/`.

The baseline is precision-heavy: when it flags a comment, it is usually right, but it misses roughly half the toxic examples. The gap widens on the rare labels, where `threat` (74 positives) and `identity_hate` (294 positives) recall below 0.15. TF-IDF has too few occurrences of those patterns to separate them, which is the expected failure mode for a linear bag-of-ngrams model on a heavily imbalanced multi-label task.

That breakdown is what motivates the DistilBERT upgrade: pretrained contextual embeddings do not need the rare labels to be frequent in this dataset to recognize them, so the expected win is concentrated in exactly the labels where this baseline is weakest.

## Technology Stack

### Core Dependencies

- **numpy** - Numerical computing
- **pandas** - Data manipulation and CSV loading
- **scikit-learn** - TF-IDF vectorization, Logistic Regression, metrics, and train-validation splitting
- **joblib** - Model persistence
- **pyyaml** - Configuration file support

### Data Version Control

- **DVC** - Data versioning and remote data storage support

### Development Tools

- **pytest** - Testing framework
- **pytest-cov** - Test coverage
- **ruff** - Linting and formatting
- **mypy** - Static type checking
- **pre-commit** - Git hook automation

### MLOps & Experiment Tooling (Phase 2)

- **Docker / Docker Compose** - Reproducible containerized training and inference
- **MLflow** - Experiment tracking for parameters, metrics, and model artifacts
- **Hydra / OmegaConf** - Hierarchical configuration management with CLI overrides
- **Rich** - Colored, leveled console logging and pretty tracebacks
- **cProfile + memory-profiler** - CPU and memory profiling of the training pipeline

### Serving & CI/CD (Phase 3)

- **FastAPI / Uvicorn** - Real-time multi-label inference API with Pydantic request validation
- **Google Cloud Run** - Serverless container hosting for the inference service, with Artifact Registry for images
- **Streamlit** - Demo client that calls the deployed endpoint (`demo/streamlit_app.py`)
- **GitHub Actions** - pytest across Python 3.10/3.11/3.12, ruff, mypy, Continuous ML (CML) training reports, and Docker image publishing
- **Codecov** - Coverage reporting on every pull request

## Inference API

The trained pipeline is served by a FastAPI application in `api/`, containerized via `dockerfiles/Dockerfile.api` and deployed to Google Cloud Run.

**Base URL:** <https://toxic-comment-api-491682843765.us-central1.run.app>
**Interactive docs:** [`/docs`](https://toxic-comment-api-491682843765.us-central1.run.app/docs) (Swagger UI) and [`/redoc`](https://toxic-comment-api-491682843765.us-central1.run.app/redoc)

| Method | Path       | Purpose                                                      |
| ------ | ---------- | ------------------------------------------------------------ |
| `GET`  | `/health`  | Liveness and readiness probe; reports whether the model loaded |
| `POST` | `/predict` | Multi-label toxicity scores for 1-100 comments per request     |

**Request schema** (`api/schemas.py`):

```json
{ "comments": ["you are a wonderful person", "i will hunt you down"] }
```

Each comment must be 1 to 10,000 characters, with 1 to 100 comments per request. Requests outside those bounds are rejected with a 422 by Pydantic validation; if the model artifact is missing, `/predict` returns a 503.

**Response schema:** one entry per input comment, echoing the comment alongside a binary prediction and a probability for each of the six labels, plus the identifier of the model artifact that served the request.

> **Note:** the currently deployed Cloud Run revision predates the schema in `api/schemas.py`. It returns `labels` as a list of the labels that fired rather than a per-label boolean map, and omits the `version` field. Redeploying from `main` picks up the current contract.

**Run the API locally:**

```bash
uvicorn api.main:app --reload            # http://127.0.0.1:8000/docs
MODEL_PATH=models/baseline_tfidf_logreg.joblib uvicorn api.main:app
```

The model path is read from the `MODEL_PATH` environment variable and defaults to `models/baseline_tfidf_logreg.joblib`. The model is loaded once at startup by a lifespan handler rather than per request.

Deployment steps for Artifact Registry and Cloud Run are documented in [docs/gcp_deployment.md](docs/gcp_deployment.md) and [docs/section3_part2_deployment.md](docs/section3_part2_deployment.md).

## Continuous Integration & Continuous Delivery

The repository ships two GitHub Actions workflows that automate model training reporting and container publishing on every push to `main` and every pull request targeting `main`.

| Workflow                      | File                                  | Trigger                  | What it does                                                                 |
| ----------------------------- | ------------------------------------- | ------------------------ | ---------------------------------------------------------------------------- |
| **CML (Continuous ML)**       | `.github/workflows/cml.yaml`          | `push` + `pull_request` to `main` | Trains the baseline on a committed sample dataset and posts the classification report + confusion matrix back to the PR / commit. |
| **Docker Publish**            | `.github/workflows/docker-publish.yaml` | `push` + `pull_request` to `main` | Builds the multi-stage image with GHA layer caching. On `main`, publishes `latest` + long-SHA tags to Docker Hub and runs a smoke test against the published image. |

The pre-existing `ci.yml` (lint, type-check, pytest) and `codecheck.yaml` (lint + type-check on dev/main) workflows remain in place and continue to gate every PR.

### Continuous ML (CML) workflow

The CML workflow uses [iterative.ai's CML](https://cml.dev/) to attach human-readable training reports to PRs and commits.

**Trigger and outputs:**

- **Pull request to `main`**: CML posts a PR review comment containing the classification report, the per-label confusion matrix, and the full metrics JSON.
- **Push to `main`**: CML posts the same report as a commit comment.

**Pipeline steps** (full source in `.github/workflows/cml.yaml`):

1. `actions/checkout@v5` — clone the repo.
2. `astral-sh/setup-uv@v8.2.0` — install [uv](https://github.com/astral-sh/uv) with Python 3.11. (Pinned to an immutable patch tag — `setup-uv` stopped publishing floating `@v8` major tags as of v8.0.0 for supply-chain security.)
3. `iterative/setup-cml@v2` (`vega: false`) — install the CML CLI.
4. `uv venv --python 3.11` + `uv pip install -e .` — set up an isolated environment and install the project in editable mode.
5. `python -m toxic_comment_classifier.train_model data.raw_path=data/sample data.train_file=train_sample.csv mlflow.enabled=false` — train on the committed sample (no DVC pull required).
6. `actions/upload-artifact@v4` — archive `reports/` and `models/` for download.
7. Build `report.md` (classification report + confusion-matrix image + metrics JSON) and call `cml comment create --publish report.md`.

**Required secrets:** *none* — CML uses the default `GITHUB_TOKEN` provided to every workflow run.

**Expected artifacts** (after a successful run):

| Artifact                                  | Source                                                 |
| ----------------------------------------- | ------------------------------------------------------ |
| `reports/classification_report.txt`       | `sklearn.metrics.classification_report` per label      |
| `reports/confusion_matrix.png`            | `multilabel_confusion_matrix` + `ConfusionMatrixDisplay` grid |
| `reports/baseline_metrics.json`           | micro/macro F1, precision, recall, Hamming loss, timing |
| `models/baseline_tfidf_logreg.joblib`     | trained pipeline                                        |

**Sample dataset:** the workflow trains on `data/sample/train_sample.csv` — a 600-row stratified subset of the full Jigsaw training set, committed to the repo so the workflow runs end-to-end without DVC credentials. The full training set in `data/raw/train.csv` remains DVC-tracked and is the source of truth for local and Docker training.

**Triggering CML locally:** open a PR against `main`. Within ~1–2 minutes the **CML / train-and-report** check appears in the PR; when it goes green the bot comment is posted.

### Docker Publish workflow

The Docker workflow builds the multi-stage image defined in `dockerfiles/Dockerfile` using modern Docker GitHub Actions.

**Behavior matrix:**

| Event                       | Build | Push to Docker Hub | Smoke test       |
| --------------------------- | :---: | :----------------: | ---------------- |
| `pull_request` to `main`    | ✅    | ❌                  | Against the locally loaded image |
| `push` to `main`            | ✅    | ✅                  | Against the published `latest` tag |

**Tagging strategy** (via `docker/metadata-action@v5`):

- `latest` — only on pushes to the default branch (`main`).
- `sha-<long-sha>` — every build, both PR and push, so a specific commit's image is always reproducible.

**Caching:** GitHub Actions cache (`cache-from: type=gha`, `cache-to: type=gha,mode=max`) is enabled on both PR and push builds, so unchanged layers are reused across runs.

**Smoke test:** after a `main` push, the workflow pulls `<DOCKER_HUB_REPOSITORY>:latest` and runs `docker run --rm <image>:latest python -c "import toxic_comment_classifier ..."`. The container's exit code propagates back to the workflow — any ImportError or runtime crash fails the job.

### Required GitHub secrets

Configure these once under **Settings → Secrets and variables → Actions** in the GitHub repo:

| Secret                  | Used by                  | Description                                                                 |
| ----------------------- | ------------------------ | --------------------------------------------------------------------------- |
| `DOCKER_HUB_USERNAME`   | `docker-publish.yaml`    | Docker Hub login. Used by `docker/login-action@v3` to authenticate the push. |
| `DOCKER_HUB_TOKEN`      | `docker-publish.yaml`    | Docker Hub **access token** (not the account password). Create one at [hub.docker.com/settings/security](https://hub.docker.com/settings/security). Scope: Read, Write, Delete. |
| `DOCKER_HUB_REPOSITORY` | `docker-publish.yaml`    | Fully-qualified target repo, e.g. `apate476/toxic-comment-classifier`. Used as the `images:` input for `docker/metadata-action`. |

CML does **not** require any custom secrets — it authenticates against GitHub using the workflow's built-in `GITHUB_TOKEN`.

### Example workflow runs

Live run history for every workflow is on the [Actions tab](https://github.com/apate476/toxic-comment-classifier/actions). Each pull request into `main` picks up a CML bot comment carrying the classification report and confusion matrix for that revision.

![CI passing](docs/images/ci-passing.png)

_Test, lint, and type-check workflows green on `main`._

## Project Structure

This project uses the modern `src/` layout. The importable package lives in `src/toxic_comment_classifier/`, which keeps source code separate from project configuration, data, reports, and tests.

```text
toxic-comment-classifier/
├── src/
│   └── toxic_comment_classifier/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│       ├── train_model.py
│       ├── predict_model.py
│       ├── data/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── utils/
│       └── visualization/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_data.py
│   └── test_model.py
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── test_labels.csv
│   └── processed/
├── models/
│   └── baseline_tfidf_logreg.joblib
├── reports/
│   ├── baseline_metrics.json
│   ├── predictions.csv
│   └── figures/
├── notebooks/
├── docs/
├── configs/
├── dockerfiles/
├── api/
├── .github/
├── PHASE1.md
├── PHASE2.md
├── PHASE3.md
├── requirements.txt
├── requirements_dev.txt
├── pyproject.toml
├── Makefile
├── docker-compose.yaml
├── LICENSE
└── README.md
```

## Code Organization

The main training and prediction entrypoints are:

```text
src/toxic_comment_classifier/train_model.py
src/toxic_comment_classifier/predict_model.py
```

The training script loads the raw training data, validates the required columns, trains the baseline model, evaluates it on a validation split, saves the model artifact, and writes metrics to the reports folder.

The prediction script loads the saved model, scores the test comments, and writes predicted labels to `reports/predictions.csv`.

## Data Handling

Raw data is stored under:

```text
data/raw/
```

Processed or transformed data should be stored under:

```text
data/processed/
```

Large data files are managed with DVC instead of being committed directly to Git. This keeps the repository lightweight while preserving reproducibility.

The raw data validation tests check that:

- Training data has the expected columns.
- Training data is not empty.
- Comment text values are not missing.
- Label columns contain binary values.
- Test data has the expected structure.
- Missing files raise the expected error.

## Version Control Workflow

The project uses a feature-branch workflow during development. Team members work on separate branches for data handling, model training, documentation, and project proposal updates. Changes are reviewed through pull requests before final submission.

Commits should be descriptive and focused. Example commit messages include:

```text
feat(model): add baseline toxic comment classifier
docs: update phase 1 model documentation
test(data): add raw data validation tests
chore(data): configure DVC remote
```

Before final submission, the repository should contain the completed Phase 1 implementation, generated reports, and updated documentation.

## Contribution Summary

- [x] Development environment has been set up
- [x] Repository structure follows an MLOps-oriented `src/` layout
- [x] Data versioning support has been configured with DVC
- [x] Raw data validation tests have been added
- [x] Baseline model has been implemented and trained
- [x] Evaluation metrics have been generated and saved
- [x] Test set predictions have been generated
- [x] Documentation has been updated for Phase 1
- [x] Configuration management migrated to Hydra config groups
- [x] Experiment tracking wired up with MLflow
- [x] Training and inference containerized with a multi-stage Docker build
- [x] CPU and memory profiling reports generated
- [x] Test suite expanded to 59 tests at 90% coverage
- [x] GitHub Actions CI added for tests, linting, and type checking
- [x] CML training reports posted automatically on pull requests
- [x] Docker images published to Docker Hub from `main`
- [x] FastAPI inference service implemented and deployed to Cloud Run
- [x] Streamlit demo client built against the deployed endpoint

## References

- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
