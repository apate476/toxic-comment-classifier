# Testing Guide

## Running Tests Locally

### Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```

### Run all tests
```bash
pytest tests/ -v
```

### Run with coverage
```bash
pytest tests/ --cov=toxic_comment_classifier --cov-report=term-missing
```

### Run a specific test file
```bash
pytest tests/test_training.py -v
```

### Run a specific test
```bash
pytest tests/test_training.py::TestValidateTrainingData::test_passes_with_valid_data -v
```

## Running Tests in CI

Tests run automatically via GitHub Actions on every push and PR to `main` or `dev`.

Workflows:
- `.github/workflows/ci.yml` - runs ruff, mypy, and pytest across Python 3.10, 3.11, 3.12
- `.github/workflows/codecheck.yaml` - runs ruff and mypy only

Coverage reports are uploaded to Codecov after every CI run.

## Pre-commit Hooks

Install hooks:
```bash
pre-commit install
```

Run manually on all files:
```bash
pre-commit run --all-files
```

Skip hooks in an emergency:
```bash
git commit -m "message" --no-verify
```
