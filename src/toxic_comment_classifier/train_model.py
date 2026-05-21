"""Training entrypoint for the toxic comment classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import hydra
import joblib
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from rich.traceback import install as install_rich_traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxic_comment_classifier.logging_config import get_logger, setup_logging
from toxic_comment_classifier.utils.seed import set_seed

logger = get_logger(__name__)


def _resolve_train_file(raw_path: Path, train_filename: str) -> Path:
    """Return the path to the training dataset, searching common locations."""
    candidates = [
        raw_path / train_filename,
        Path("data/processed") / train_filename,
        Path("data/raw") / train_filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"{train_filename} was not found in {raw_path}, data/processed, or data/raw.")


def _validate_training_data(df: pd.DataFrame, text_column: str, label_columns: list[str]) -> None:
    """Validate that the training dataset contains the required columns."""
    required_columns = [text_column, *label_columns]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    if df.empty:
        raise ValueError("Training data is empty.")


def train(cfg: DictConfig) -> None:
    """Train and save a baseline multi-label text classification model."""
    # Resolve all paths to absolute. Hydra runs may execute from any cwd,
    # so we anchor data and output paths to the project root.
    raw_path = Path(to_absolute_path(cfg.data.raw_path))
    model_dir = Path(to_absolute_path(cfg.training.model_dir))
    reports_dir = Path(to_absolute_path(cfg.training.reports_dir))

    train_file = _resolve_train_file(raw_path, cfg.data.train_file)

    logger.info("Loading training data from %s", train_file)
    df = pd.read_csv(train_file)

    label_columns = list(cfg.data.label_columns)
    _validate_training_data(df, cfg.data.text_column, label_columns)

    texts = df[cfg.data.text_column].fillna("")
    labels = df[label_columns]

    x_train, x_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=cfg.data.val_split,
        random_state=cfg.seed,
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=cfg.features.max_features,
                    ngram_range=tuple(cfg.features.ngram_range),
                    stop_words=cfg.features.stop_words,
                    min_df=cfg.features.min_df,
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        C=cfg.model.C,
                        penalty=cfg.model.penalty,
                        solver=cfg.model.solver,
                        max_iter=cfg.model.max_iter,
                        random_state=cfg.seed,
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
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_elapsed = time.perf_counter() - fit_start
    logger.info("Model fit completed in %.2fs", fit_elapsed)

    predict_start = time.perf_counter()
    predictions = model.predict(x_val)
    predict_elapsed = time.perf_counter() - predict_start
    logger.info("Validation prediction completed in %.2fs", predict_elapsed)

    metrics: dict[str, Any] = {
        "model_type": "TF-IDF + OneVsRest Logistic Regression",
        "training_file": str(train_file),
        "rows": int(len(df)),
        "fit_seconds": round(fit_elapsed, 2),
        "predict_seconds": round(predict_elapsed, 2),
        "validation_split": cfg.data.val_split,
        "random_state": cfg.seed,
        "labels": label_columns,
        "micro_f1": float(f1_score(y_val, predictions, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_val, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_val, predictions, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_val, predictions)),
        "hyperparameters": {
            "tfidf_max_features": cfg.features.max_features,
            "tfidf_ngram_range": list(cfg.features.ngram_range),
            "tfidf_stop_words": cfg.features.stop_words,
            "tfidf_min_df": cfg.features.min_df,
            "classifier": "LogisticRegression",
            "C": cfg.model.C,
            "penalty": cfg.model.penalty,
            "solver": cfg.model.solver,
            "max_iter": cfg.model.max_iter,
        },
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / cfg.training.model_filename
    metrics_path = reports_dir / cfg.training.metrics_filename

    joblib.dump(model, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info("Saved model to %s", model_path)
    logger.info("Saved metrics to %s", metrics_path)

    # Hydra writes the composed config and overrides to a run-scoped directory
    # automatically; log the path so users know where to look.
    run_dir = HydraConfig.get().runtime.output_dir
    logger.info("Hydra run artifacts written to %s", run_dir)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run model training from the command line via Hydra."""
    install_rich_traceback(show_locals=False)
    setup_logging()
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    train(cfg)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
