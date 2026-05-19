"""Training entrypoint for the toxic comment classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxic_comment_classifier.config import DEFAULT_CONFIG, MODELS_DIR, PROCESSED_DATA_DIR
from toxic_comment_classifier.logging_config import get_logger, setup_logging
from toxic_comment_classifier.utils.seed import set_seed

logger = get_logger(__name__)

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _resolve_train_file(data_path: Path) -> Path:
    """Return the path to the training dataset."""
    candidates = [
        data_path / "train.csv" if data_path.is_dir() else data_path,
        Path("data/processed/train.csv"),
        Path("data/raw/train.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("train.csv was not found in the provided path, data/processed, or data/raw.")


def _validate_training_data(df: pd.DataFrame) -> None:
    """Validate that the training dataset contains the required columns."""
    required_columns = ["comment_text", *LABEL_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Training data is empty.")


def train(data_path: Path, model_dir: Path, epochs: int, batch_size: int, lr: float) -> None:
    """Train and save a baseline multi-label text classification model."""
    train_file = _resolve_train_file(data_path)

    logger.info("Loading training data from %s", train_file)
    df = pd.read_csv(train_file)
    _validate_training_data(df)

    texts = df["comment_text"].fillna("")
    labels = df[LABEL_COLUMNS]

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=DEFAULT_CONFIG.training.seed,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=DEFAULT_CONFIG.training.seed,
                    )
                ),
            ),
        ]
    )

    logger.info(
        "Training baseline model with %d training rows and %d validation rows",
        len(x_train),
        len(x_val),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_val)

    metrics: dict[str, Any] = {
        "model_type": "TF-IDF + OneVsRest Logistic Regression",
        "training_file": str(train_file),
        "rows": int(len(df)),
        "validation_split": 0.2,
        "random_state": DEFAULT_CONFIG.training.seed,
        "labels": LABEL_COLUMNS,
        "micro_f1": float(f1_score(y_val, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_val, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_val, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_val, predictions)),
        "hyperparameters": {
            "tfidf_max_features": 50_000,
            "tfidf_ngram_range": [1, 2],
            "tfidf_stop_words": "english",
            "classifier": "LogisticRegression",
            "C": 1.0,
            "penalty": "l2",
            "solver": "liblinear",
            "max_iter": 1000,
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "baseline_tfidf_logreg.joblib"
    metrics_path = reports_dir / "baseline_metrics.json"

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)


def main() -> None:
    """Run model training from the command line."""
    cfg = DEFAULT_CONFIG.training

    parser = argparse.ArgumentParser(description="Train the toxic comment classifier.")
    parser.add_argument("--data-path", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    train(
        data_path=args.data_path,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
    )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
