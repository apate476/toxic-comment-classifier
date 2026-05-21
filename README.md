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

- Base image: `python:3.11-slim-bookworm` (pinned to the Debian _bookworm_ release).
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

### Future MLOps Tools

- **Docker** - Containerized execution
- **FastAPI** - Model serving API
- **MLflow** - Experiment tracking
- **GitHub Actions** - CI/CD automation

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
