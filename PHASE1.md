# PHASE 1: Project Design & Model Development

## Overview

Phase 1 establishes the foundation for the `toxic_comment_classifier` MLOps project. This phase covers project planning, repository organization, team collaboration setup, data handling, baseline model development, and documentation.

By the end of Phase 1, the project includes a structured repository, validated raw data, a trained baseline model, saved evaluation metrics, generated predictions, and clear documentation for future development.

---

## 1. Project Proposal

### Scope & Objectives

The goal of this project is to build a multi-label machine learning classifier that detects toxic language in online comments. Each comment can belong to one or more toxicity categories, including `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

The main objective for Phase 1 is to establish a working baseline model and a reproducible MLOps project structure. This baseline will be used as a reference point for future improvements in later phases.

### Problem Statement

Online platforms often receive large volumes of user-generated comments. Manually reviewing every comment is time-consuming and difficult to scale. A toxic comment classifier can help identify harmful or inappropriate content more efficiently by automatically predicting toxicity labels for each comment.

This project focuses on building a reliable baseline system that can classify comments into multiple toxicity categories. The classifier is intended to support moderation workflows by flagging comments that may require review.

### Success Metrics

The baseline model is evaluated using metrics suitable for multi-label classification:

| Metric | Purpose |
|---|---|
| Micro F1 | Measures overall balance between precision and recall across all labels |
| Macro F1 | Measures average F1 across labels, treating each label equally |
| Micro Precision | Measures how often predicted toxic labels are correct |
| Micro Recall | Measures how many true toxic labels are detected |
| Hamming Loss | Measures the fraction of incorrect labels across all predictions |

The Phase 1 goal is not to produce the final best model, but to establish a working and reproducible baseline.

### Detailed Project Description

`toxic_comment_classifier` is a machine learning project designed to classify online comments into multiple toxicity categories. The project addresses the challenge of detecting harmful language in user-generated content, which is important for online communities, social platforms, forums, and content moderation systems.

The business context for this project is content safety. Online platforms need scalable tools that can help identify comments containing toxic, insulting, obscene, threatening, or identity-based hateful language. A machine learning classifier can assist moderation teams by flagging potentially harmful comments for additional review. This can reduce manual workload and improve response time when handling harmful content.

The technical approach for Phase 1 uses a traditional machine learning baseline. Text comments are converted into numerical features using TF-IDF vectorization. A One-vs-Rest Logistic Regression classifier is then trained to predict six toxicity labels. This approach is simple, interpretable, fast to train, and appropriate for establishing a first performance benchmark.

The repository follows an MLOps-oriented structure with separate folders for source code, data, reports, tests, documentation, and model artifacts. Data handling is supported with DVC, which allows large data files to be tracked without committing them directly to Git. Training and prediction are implemented as command-line entrypoints so future team members can reproduce the model workflow.

The expected outcome of Phase 1 is a working baseline model, saved validation metrics, generated test predictions, and documentation explaining how the system is organized and how to run it. Future phases can improve the model by adding experiment tracking, containerization, monitoring, CI/CD, deployment, and more advanced model architectures such as transformer-based models.

### Dataset Selection

The project uses a toxic comment classification dataset containing online comments and multiple binary toxicity labels. This dataset is appropriate because it directly matches the project goal: predicting whether a comment belongs to one or more toxic content categories.

The dataset was selected because:

- It contains real text comments.
- It supports multi-label classification.
- It includes the target toxicity categories required for the project.
- It is commonly used for toxic comment classification experiments.
- It is suitable for both baseline machine learning models and future deep learning models.

### Dataset Description

The raw dataset is stored under:

```text
data/raw/
```

The main files are:

| File | Description |
|---|---|
| `train.csv` | Training data containing comment text and toxicity labels |
| `test.csv` | Test comments used for prediction generation |
| `test_labels.csv` | Label file for the test dataset |
| `raw.dvc` | DVC tracking file for the raw dataset |

The training dataset contains 159,571 labeled comments.

The expected columns in `train.csv` are:

| Column | Description |
|---|---|
| `id` | Unique identifier for each comment |
| `comment_text` | Raw comment text |
| `toxic` | Binary label for toxic language |
| `severe_toxic` | Binary label for severe toxicity |
| `obscene` | Binary label for obscene language |
| `threat` | Binary label for threatening language |
| `insult` | Binary label for insulting language |
| `identity_hate` | Binary label for identity-based hate speech |

### Model Considerations

Several model approaches were considered for this project:

| Model | Reason for Consideration |
|---|---|
| TF-IDF + Logistic Regression | Fast, reliable, interpretable baseline for text classification |
| TF-IDF + Linear SVM | Strong traditional text classification approach |
| Naive Bayes | Simple baseline for text data |
| DistilBERT | Strong future transformer-based model for improved language understanding |
| BERT-based models | More advanced future option for improved contextual text classification |

For Phase 1, the selected baseline is:

```text
TF-IDF + One-vs-Rest Logistic Regression
```

This model was selected because it is easy to train, reproducible, and appropriate for quickly establishing baseline performance.

### Open-Source Tools

The project uses the following open-source tools:

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| pandas | Data loading and manipulation |
| scikit-learn | TF-IDF vectorization, model training, splitting, and metrics |
| joblib | Saving and loading trained model artifacts |
| pytest | Automated testing |
| ruff | Linting and formatting |
| mypy | Static type checking |
| DVC | Data versioning and remote data tracking |
| GitHub | Version control and collaboration |
| pre-commit | Automated code quality checks |

---

## 2. Code Organization & Setup

### Repository Structure

The project follows a modern MLOps-style `src/` layout.

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
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── notebooks/
├── docs/
├── configs/
├── dockerfiles/
├── api/
├── PHASE1.md
├── PHASE2.md
├── PHASE3.md
├── requirements.txt
├── requirements_dev.txt
├── pyproject.toml
├── Makefile
└── README.md
```

### Environment Setup

The project uses a Python virtual environment.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

Because the project uses the `src/` layout, the package can be installed in editable mode:

```bash
pip install -e .
```

If editable installation is not available, commands can be run with:

```bash
PYTHONPATH=src
```

### Dependency Management

Runtime dependencies are maintained in:

```text
requirements.txt
```

Development dependencies are maintained in:

```text
requirements_dev.txt
```

Important runtime dependencies include:

- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`
- `pyyaml`

Important development dependencies include:

- `pytest`
- `pytest-cov`
- `ruff`
- `mypy`
- `pre-commit`

### Code Organization

The project separates responsibilities across folders:

| Folder | Purpose |
|---|---|
| `src/` | Main Python package |
| `tests/` | Unit and integration tests |
| `data/raw/` | Original immutable data |
| `data/processed/` | Processed or transformed data |
| `models/` | Saved model artifacts |
| `reports/` | Metrics, predictions, and generated reports |
| `notebooks/` | Exploratory notebooks |
| `docs/` | Additional documentation |
| `configs/` | Configuration files |
| `api/` | Future API service |
| `dockerfiles/` | Future Docker configuration |

---

## 3. Version Control & Collaboration

### Branching Strategy

The project uses a feature-branch workflow. Development work is completed on separate branches and merged into `main` through pull requests.

Current feature branches include:

```text
feature/data-handling
feature/documentation
feature/model-training
feature/project-proposal
```

### Pull Request Process

The team uses pull requests to review work before it becomes part of the final project submission. Each pull request should include a clear description of the changes made, the files updated, and any commands used to verify the work.

Before submission, the final repository should include the completed Phase 1 code, data handling setup, baseline model training workflow, generated reports, and updated documentation.


### Commit Discipline

Commits should be descriptive and focused.

Examples:

```text
feat(model): add baseline toxic comment classifier
docs: update phase 1 model documentation
test(data): add raw data validation tests
chore(data): configure DVC remote
```

### Team Roles

| Role | Responsibility |
|---|---|
| Project Lead | Coordinates project direction and final review |
| Data Handling | Sets up raw data, DVC, validation, and data documentation |
| Model Training | Implements baseline model, metrics, predictions, and model persistence |
| Documentation | Updates README, phase documentation, and project reports |
| Reviewers | Review pull requests before merge |

### Code Review Guidelines

Before merging a pull request, reviewers should check that:

- The code runs without errors.
- The project structure is not broken.
- Required files are updated.
- Tests pass.
- Documentation matches the actual implementation.
- Large artifacts are not accidentally committed unless agreed on by the team.
- Commit messages are clear.

---

## 4. Data Handling

### Raw Data

Raw data is stored in:

```text
data/raw/
```

The raw dataset includes:

```text
train.csv
test.csv
test_labels.csv
```

Raw data should be treated as immutable and should not be manually edited.

### Processed Data

Processed data should be stored in:

```text
data/processed/
```

This folder is reserved for cleaned, transformed, or feature-engineered data created by reproducible scripts.

### Data Cleaning and Preprocessing

For Phase 1, the baseline model performs minimal preprocessing directly in the training pipeline:

- Loads `train.csv`
- Validates required columns
- Fills missing comment text values with empty strings
- Converts text into TF-IDF features
- Splits data into training and validation sets

Since the baseline uses TF-IDF, traditional numeric normalization is not required. Text features are generated through TF-IDF vectorization.

### Data Splits

The baseline model uses an 80/20 train-validation split from `data/raw/train.csv`.

| Split | Purpose |
|---|---|
| 80% training | Model fitting |
| 20% validation | Baseline performance evaluation |

The split uses a fixed random seed for reproducibility.

### Data Validation

Data validation tests are included in:

```text
tests/test_data.py
```

The tests check that:

- Training data has the expected columns.
- Training data is not empty.
- `comment_text` values are not missing.
- Label columns are binary.
- Test data has the expected columns.
- Missing files raise the expected error.

### DVC Setup

DVC has been configured for data tracking. The raw dataset is tracked through:

```text
data/raw.dvc
```

DVC configuration is stored in:

```text
.dvc/config
```

The actual large raw data files are not intended to be directly committed to Git. DVC allows the team to track dataset versions while storing the underlying data remotely.

---

## 5. Model Training

### Baseline Model

A Phase 1 baseline model was implemented using:

```text
TF-IDF + One-vs-Rest Logistic Regression
```

The model predicts six labels:

```text
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

The training script is located at:

```text
src/toxic_comment_classifier/train_model.py
```

The prediction script is located at:

```text
src/toxic_comment_classifier/predict_model.py
```

### Training Command

From the project root:

```bash
python -m toxic_comment_classifier.train_model --data-path data/raw
```

If the package is not installed in editable mode:

```bash
PYTHONPATH=src python -m toxic_comment_classifier.train_model --data-path data/raw
```

### Prediction Command

After training:

```bash
python -m toxic_comment_classifier.predict_model --input data/raw/test.csv
```

If the package is not installed in editable mode:

```bash
PYTHONPATH=src python -m toxic_comment_classifier.predict_model --input data/raw/test.csv
```

### Model Configuration

| Component | Value |
|---|---|
| Feature extraction | TF-IDF |
| Maximum features | 50,000 |
| N-gram range | Unigrams and bigrams |
| Stop words | English |
| Classifier | One-vs-Rest Logistic Regression |
| Solver | liblinear |
| Max iterations | 1000 |
| Validation split | 20% |
| Random seed | 42 |

### Evaluation Metrics

The baseline model was trained on `data/raw/train.csv`, which contains 159,571 labeled comments. The model was evaluated using an 80/20 train-validation split.

| Metric | Score |
|---|---:|
| Micro F1 | 0.6581 |
| Macro F1 | 0.4738 |
| Micro Precision | 0.8865 |
| Micro Recall | 0.5233 |
| Hamming Loss | 0.0201 |

### Model Persistence

The trained model is saved to:

```text
models/baseline_tfidf_logreg.joblib
```

The validation metrics are saved to:

```text
reports/baseline_metrics.json
```

Predictions for the test set are saved to:

```text
reports/predictions.csv
```

### Reproducibility

Training reproducibility is supported through:

- Fixed random seed
- Logging during training and prediction
- Saved model artifact
- Saved metrics file
- Documented training command
- Consistent train-validation split

### Baseline Interpretation

The baseline model has strong micro precision, meaning that when it predicts a toxicity label, it is usually correct. The lower micro recall indicates that the model misses some toxic examples. This is expected for a simple baseline on an imbalanced multi-label text dataset.

Future improvements may include:

- Class imbalance handling
- Threshold tuning
- Additional text preprocessing
- More advanced feature engineering
- Transformer-based models such as DistilBERT
- Experiment tracking with MLflow

---

## 6. Documentation & Reporting

### README

The project README includes:

- Project overview and objectives
- Dataset description
- Setup instructions
- Training and prediction commands
- Baseline model performance
- Technology stack
- Project structure
- Version control workflow
- Documentation references

### Code Docstrings

Main training and prediction functions include docstrings describing their purpose and expected behavior.

### Code Style

The project includes tooling for code quality:

| Tool | Purpose |
|---|---|
| ruff | Linting and formatting |
| mypy | Static type checking |
| pytest | Testing |
| pre-commit | Automated checks before commits |

### Makefile

The Makefile provides common project commands, including:

```bash
make setup
make train
make test
make lint
make format
make clean
```

### Reports

The main Phase 1 output files are:

| File | Description |
|---|---|
| `reports/baseline_metrics.json` | Baseline validation metrics |
| `reports/predictions.csv` | Predictions generated from the test dataset |

---

## Phase 1 Completion Checklist

### Project Proposal

- [x] Scope and objectives defined
- [x] Detailed project description added
- [x] Dataset selected and justified
- [x] Dataset characteristics documented
- [x] Model considerations documented
- [x] Open-source tools documented

### Code Organization & Setup

- [x] GitHub repository created
- [x] MLOps-style project structure created
- [x] Python virtual environment supported
- [x] Dependency files included
- [x] `src/` layout used
- [x] Installation documented

### Version Control & Collaboration

- [x] Feature branches used
- [x] Pull request workflow established
- [x] Team responsibilities documented
- [x] Commit message examples documented
- [x] Code review expectations documented

### Data Handling

- [x] Raw data files available
- [x] Data folder documented
- [x] Data validation tests added
- [x] Train-validation split implemented
- [x] DVC configured for raw data tracking
- [x] Data loading workflow validated

### Model Training

- [x] Baseline model implemented
- [x] Baseline model trained
- [x] Hyperparameters documented
- [x] Evaluation metrics calculated
- [x] Model artifact saved
- [x] Metrics saved
- [x] Predictions generated
- [x] Reproducibility supported through seed and logging

### Documentation & Reporting

- [x] README updated
- [x] Phase 1 documentation updated
- [x] Training and prediction commands documented
- [x] Baseline model performance documented
- [x] Reports generated
- [x] Pull request review and final merge

---

## Phase 1 Summary

Phase 1 successfully establishes a working foundation for the `toxic_comment_classifier` project. The repository now includes an organized MLOps structure, DVC-supported data handling, raw data validation tests, a trained baseline model, evaluation metrics, generated predictions, and updated documentation.

The Phase 1 baseline uses TF-IDF feature extraction with One-vs-Rest Logistic Regression. It provides a clear performance reference that future phases can improve through better preprocessing, hyperparameter tuning, class imbalance strategies, experiment tracking, and more advanced models.