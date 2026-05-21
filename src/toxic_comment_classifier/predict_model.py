"""Prediction entrypoint for the toxic comment classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from toxic_comment_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR
from toxic_comment_classifier.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _resolve_input_file(input_path: Path) -> Path:
    """Return the path to the input dataset."""
    candidates = [
        input_path,
        Path("data/processed/test.csv"),
        Path("data/raw/test.csv"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Input file was not found in the provided path, data/processed, or data/raw.")


def predict(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Generate predictions from a saved model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path}")

    input_file = _resolve_input_file(input_path)

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)

    logger.info("Loading input data from %s", input_file)
    df = pd.read_csv(input_file)

    if "comment_text" not in df.columns:
        raise ValueError("Input data must contain a comment_text column.")

    texts = df["comment_text"].fillna("")
    predictions = model.predict(texts)

    output_df = pd.DataFrame(predictions, columns=LABEL_COLUMNS)

    if "id" in df.columns:
        output_df.insert(0, "id", df["id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    logger.info("Saved predictions to %s", output_path)


def main() -> None:
    """Run prediction from the command line."""
    parser = argparse.ArgumentParser(description="Generate toxic comment predictions.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODELS_DIR / "baseline_tfidf_logreg.joblib",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "test.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/predictions.csv"),
    )
    args = parser.parse_args()

    setup_logging(log_filename="prediction.log")
    predict(args.model_path, args.input, args.output)

    logger.info("Prediction complete")


if __name__ == "__main__":
    main()
