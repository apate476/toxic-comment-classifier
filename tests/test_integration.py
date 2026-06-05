"""Integration tests for the full training pipeline."""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from toxic_comment_classifier.train_model import train

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _make_cfg(tmp_path: Path, train_file: Path) -> DictConfig:
    """Build a minimal Hydra-style config for testing."""
    return OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "raw_path": str(tmp_path),
                "train_file": train_file.name,
                "text_column": "comment_text",
                "label_columns": LABEL_COLUMNS,
                "val_split": 0.2,
            },
            "features": {
                "max_features": 100,
                "ngram_range": [1, 1],
                "stop_words": "english",
                "min_df": 1,
            },
            "model": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 100,
            },
            "training": {
                "model_dir": str(tmp_path / "models"),
                "model_filename": "model.joblib",
                "reports_dir": str(tmp_path / "reports"),
                "metrics_filename": "metrics.json",
            },
            "mlflow": {
                "enabled": False,
                "tracking_uri": "file:mlruns",
                "experiment_name": "test",
                "run_name": "test-run",
            },
        }
    )


def _make_train_csv(path: Path, n: int = 50) -> None:
    """Write a minimal training CSV."""
    data = {
        "id": list(range(n)),
        "comment_text": [f"comment number {i}" for i in range(n)],
        **{col: [i % 2 for i in range(n)] for col in LABEL_COLUMNS},
    }
    pd.DataFrame(data).to_csv(path, index=False)


@pytest.fixture(autouse=True)
def mock_hydra_config(tmp_path: Path) -> Generator[None, None, None]:
    """Patch HydraConfig so train() can run outside of Hydra."""
    mock = MagicMock()
    mock.runtime.output_dir = str(tmp_path)
    with patch("toxic_comment_classifier.train_model.HydraConfig") as hydra_mock:
        hydra_mock.get.return_value = mock
        yield


class TestFullTrainingPipeline:
    def test_train_produces_model_and_metrics(self, tmp_path: Path) -> None:
        """Full train() call should produce a saved model and metrics file."""
        train_file = tmp_path / "train.csv"
        _make_train_csv(train_file)

        cfg = _make_cfg(tmp_path, train_file)
        train(cfg)

        model_path = tmp_path / "models" / "model.joblib"
        metrics_path = tmp_path / "reports" / "metrics.json"

        assert model_path.exists(), "Model file was not created"
        assert metrics_path.exists(), "Metrics file was not created"

    def test_metrics_contain_expected_keys(self, tmp_path: Path) -> None:
        """Metrics JSON should contain all expected keys after training."""
        train_file = tmp_path / "train.csv"
        _make_train_csv(train_file)

        cfg = _make_cfg(tmp_path, train_file)
        train(cfg)

        metrics_path = tmp_path / "reports" / "metrics.json"
        with metrics_path.open() as f:
            metrics = json.load(f)

        for key in ["micro_f1", "macro_f1", "hamming_loss", "fit_seconds", "predict_seconds"]:
            assert key in metrics, f"Missing key: {key}"

    def test_saved_model_can_predict(self, tmp_path: Path) -> None:
        """Model saved by train() should load and produce predictions."""
        train_file = tmp_path / "train.csv"
        _make_train_csv(train_file)

        cfg = _make_cfg(tmp_path, train_file)
        train(cfg)

        model = joblib.load(tmp_path / "models" / "model.joblib")
        predictions = model.predict(["this is a test comment"])
        assert predictions.shape[1] == len(LABEL_COLUMNS)
