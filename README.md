## Scope & Objectives

**Problem Statement:**
Many online platforms struggle to moderate toxic user-generated content at scale. This project builds an automated multi-label toxic comment classifier using DistilBERT to detect six categories of toxicity simultaneously.

**Goals:**

- Fine-tune DistilBERT on the Jigsaw dataset for multi-label classification
- Deploy a FastAPI inference endpoint for real-time predictions
- Establish a reproducible MLOps pipeline with MLflow, DVC, and GitHub Actions CI/CD

**Success Metrics:**

- Macro F1 Score across all 6 labels
- ROC-AUC per label on the test set
- Experiment reproducibility via MLflow with fixed random seeds
- Model artifact versioning
- CI/CD pipeline status on pull requests into dev

## Detailed Description

**Business Context:**
Online platforms receive millions of user comments daily, making manual moderation impossible at scale. Toxic content left unmoderated leads to user churn, reputational damage, and potential legal liability. An automated classifier that can detect multiple forms of toxicity simultaneously provides a scalable solution for real-time content moderation.

**Technical Approach:**
This project fine-tunes DistilBERT, a lightweight transformer model that retains approximately 97% of BERT's language understanding while being significantly faster and smaller. The task is framed as multi-label classification, meaning a single comment can simultaneously belong to multiple toxicity categories. A 6-unit sigmoid output head replaces the default classification head to support this. The model is trained on the Jigsaw Toxic Comment Classification dataset consisting of approximately 159,000 Wikipedia talk page comments labeled across 6 toxicity categories.

The MLOps infrastructure prioritizes reproducibility and collaboration. MLflow tracks all experiment parameters, metrics, and model artifacts. DVC with a Google Drive remote handles data versioning. GitHub Actions powers CI/CD, running linting via ruff, type checking via mypy, and tests on every pull request. All feature development occurs on the dev branch via short-lived feature branches, with main reserved for end of phase merges.

**Expected Outcomes:**
By the end of Phase 1 the project will deliver a trained DistilBERT multi-label classifier with logged metrics and versioned model artifacts, a FastAPI endpoint for real-time toxicity predictions, a fully reproducible MLOps pipeline, and comprehensive documentation covering data handling, model architecture, and API usage.

## Dataset Selection

**Selected Dataset:** Jigsaw Toxic Comment Classification Challenge
**Source:** Kaggle mirror — `julian3833/jigsaw-toxic-comment-classification-challenge`

**Justification:**

- The 6-label multi-label structure directly matches the problem requirements, alternatives like Twitter Hate Speech and Civil Comments only provide binary toxic or non-toxic labels
- Large enough at approximately 159,000 comments to fine-tune a transformer model effectively
- Originates from real Wikipedia talk page edits providing naturally occurring toxic and non-toxic text
- Widely used in NLP research providing a reliable benchmark for comparing results

## Dataset Description

**Size:** ~159,571 comments

**Features:**

- `id` — unique comment identifier
- `comment_text` — raw Wikipedia talk page comment
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — binary labels (0 or 1)

**Format:** CSV

**Source:** Wikipedia talk page edits, labeled by human raters via the Jigsaw/Conversation AI project

## Model Considerations

- DistilBERT with a 6-unit sigmoid output head — best suited for multi-label text classification given its strong natural language understanding from pretraining
- TF-IDF + Logistic Regression — lightweight baseline suitable for establishing a performance floor quickly
- TF-IDF + LightGBM — stronger classical baseline that handles non-linear feature interactions better than logistic regression

# toxic_comment_classifier

A multi-label toxic comment classification project built with a reproducible MLOps structure. The Phase 1 baseline model uses TF-IDF text features with a One-vs-Rest Logistic Regression classifier to predict six toxicity labels: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

The project includes structured source code, data versioning support with DVC, baseline model training, prediction generation, evaluation metrics, tests, and documentation for future MLOps phases.

## Team Information

- **Project Lead:** team_toxic (apate424@depaul.edu)
- **Team Members:** Taha Patil, Arya Patel, Bilal Qader, Asad Khan

## Project Overview

toxic_comment_classifier is a machine learning project focused on detecting toxic language in online comments. The goal is to classify each comment into one or more toxicity categories, including toxic, severe_toxic, obscene, threat, insult, and identity_hate.

For Phase 1, the project establishes a reproducible baseline using TF-IDF feature extraction and a One-vs-Rest Logistic Regression classifier. This baseline provides an initial performance reference before more advanced models, such as transformer-based architectures, are explored in later phases.

The repository follows an MLOps-oriented structure with separate folders for source code, data, tests, reports, model artifacts, and documentation. Data handling is supported through DVC, while model training and prediction are implemented as command-line entrypoints.

**Key Objectives:**

- Build a reproducible baseline model for multi-label toxic comment classification.
- Establish clear project structure, data handling, and model training workflows.
- Save baseline metrics and predictions for evaluation and future comparison.

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
git clone <repository-url>
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

### Performance Guide

Use this loop to profile and optimize:

1. **Measure first.** Run `scripts/profile_training.py` and `scripts/profile_memory.py` to capture a baseline before changing anything.
2. **Find the bottleneck.** Open `training_cpu_profile.txt` (sorted by cumulative time) — for this TF-IDF + Logistic Regression pipeline, vectorization and the per-label classifier `fit` dominate runtime.
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

The baseline model shows strong precision, meaning that predicted toxicity labels are usually reliable. The lower recall indicates that the model misses some toxic examples, which is expected for a simple baseline on an imbalanced multi-label text classification dataset.

Future work may include experimenting with transformer-based models such as DistilBERT to improve recall and overall classification performance.

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

### Planned (Phase 3)

- **FastAPI / Uvicorn** - Real-time model serving API
- **GitHub Actions** - CI/CD automation (lint, type-check, test on every PR)

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

## Documentation

Important documentation files:

| File             | Description                                                            |
| ---------------- | ---------------------------------------------------------------------- |
| `README.md`      | Main project overview, setup instructions, model summary, and commands |
| `PHASE1.md`      | Phase 1 checklist and deliverables                                     |
| `PHASE2.md`      | Phase 2 checklist                                                      |
| `PHASE3.md`      | Phase 3 checklist                                                      |
| `data/README.md` | Data folder documentation                                              |
| `docs/`          | Additional project documentation                                       |

## Contribution Summary

- [x] Development environment has been set up
- [x] Repository structure follows an MLOps-oriented `src/` layout
- [x] Data versioning support has been configured with DVC
- [x] Raw data validation tests have been added
- [x] Baseline model has been implemented and trained
- [x] Evaluation metrics have been generated and saved
- [x] Test set predictions have been generated
- [x] Documentation has been updated for Phase 1
- [x] Phase 1 documentation prepared for submission

## References

- [Phase 1 — Project Design & Model Development](PHASE1.md)
- [Phase 2 — Containerization & Monitoring](PHASE2.md)
- [Phase 3 — CI/CD & Deployment](PHASE3.md)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
