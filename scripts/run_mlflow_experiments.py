"""Run MLflow experiments for toxic comment classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data/raw/train.csv")
REPORT_DIR = Path("reports/experiments")
MODEL_DIR = Path("models")
MLRUNS_DIR = Path("mlruns")

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


EXPERIMENTS: list[dict[str, Any]] = [
    {
        "name": "baseline_tfidf_logreg",
        "max_features": 50_000,
        "ngram_range": (1, 2),
        "C": 1.0,
        "class_weight": None,
    },
    {
        "name": "smaller_tfidf_logreg",
        "max_features": 25_000,
        "ngram_range": (1, 1),
        "C": 1.0,
        "class_weight": None,
    },
    {
        "name": "balanced_tfidf_logreg",
        "max_features": 50_000,
        "ngram_range": (1, 2),
        "C": 1.0,
        "class_weight": "balanced",
    },
]


def load_data() -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load data and create train/validation split."""
    df = pd.read_csv(DATA_PATH)

    df["comment_text"] = df["comment_text"].fillna("")

    x = df["comment_text"]
    y = df[LABEL_COLUMNS]

    return cast(
        tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame],
        train_test_split(x, y, test_size=0.2, random_state=42),
    )


def calculate_metrics(y_true: pd.DataFrame, y_pred: Any) -> dict[str, float]:
    """Calculate multi-label classification metrics."""
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build TF-IDF + One-vs-Rest Logistic Regression pipeline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config["max_features"],
                    ngram_range=config["ngram_range"],
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        C=config["C"],
                        class_weight=config["class_weight"],
                    )
                ),
            ),
        ]
    )


def run_experiments() -> None:
    """Run and track multiple experiments with MLflow."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment("toxic-comment-phase2")

    x_train, x_val, y_train, y_val = load_data()

    results = []

    for config in EXPERIMENTS:
        with mlflow.start_run(run_name=config["name"]):
            model = build_pipeline(config)

            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)

            metrics = calculate_metrics(y_val, y_pred)

            model_path = MODEL_DIR / f"{config['name']}.joblib"
            joblib.dump(model, model_path)

            mlflow.log_param("max_features", config["max_features"])
            mlflow.log_param("ngram_range", str(config["ngram_range"]))
            mlflow.log_param("C", config["C"])
            mlflow.log_param("class_weight", config["class_weight"])

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.log_artifact(str(model_path))

            row = {
                "experiment_name": config["name"],
                **config,
                **metrics,
                "model_path": str(model_path),
            }
            results.append(row)

            print(f"Finished experiment: {config['name']}")
            print(metrics)

    results_df = pd.DataFrame(results)
    results_csv = REPORT_DIR / "experiment_results.csv"
    results_json = REPORT_DIR / "experiment_results.json"

    results_df.to_csv(results_csv, index=False)
    results_json.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"Experiment results saved to {results_csv}")
    print(f"Experiment results saved to {results_json}")


if __name__ == "__main__":
    run_experiments()
