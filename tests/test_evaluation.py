"""Tests for evaluation metric helpers."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from toxic_comment_classifier.evaluation.metrics import classification_report, regression_report
from toxic_comment_classifier.visualization.visualize import plot_confusion_matrix, plot_training_history


class TestClassificationReport:
    def test_returns_expected_keys(self) -> None:
        """Report should contain accuracy, precision, recall, and f1."""
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 1, 0]
        result = classification_report(y_true, y_pred)
        assert set(result.keys()) == {"accuracy", "precision", "recall", "f1"}

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should return 1.0 for all metrics."""
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 1, 0]
        result = classification_report(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        """All wrong predictions should return 0.0 accuracy."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        result = classification_report(y_true, y_pred)
        assert result["accuracy"] == 0.0


class TestRegressionReport:
    def test_returns_expected_keys(self) -> None:
        """Report should contain mae, mse, rmse, and r2."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        result = regression_report(y_true, y_pred)
        assert set(result.keys()) == {"mae", "mse", "rmse", "r2"}

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should return 0 error and r2 of 1.0."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        result = regression_report(y_true, y_pred)
        assert result["mae"] == 0.0
        assert result["mse"] == 0.0
        assert result["rmse"] == 0.0
        assert result["r2"] == 1.0

    def test_rmse_is_sqrt_of_mse(self) -> None:
        """RMSE should equal the square root of MSE."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.5, 2.5, 3.5]
        result = regression_report(y_true, y_pred)
        assert abs(result["rmse"] - np.sqrt(result["mse"])) < 1e-10


class TestPlotTrainingHistory:
    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_training_history should write a file when output_path is given."""
        path = str(tmp_path / "history.png")
        plot_training_history({"loss": [0.5, 0.4, 0.3]}, output_path=path)
        assert os.path.exists(path)

    def test_runs_without_output_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """plot_training_history should not raise when output_path is None."""
        monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
        plot_training_history({"loss": [0.5, 0.4]})


class TestPlotConfusionMatrix:
    def test_saves_to_file(self, tmp_path: Path) -> None:
        """plot_confusion_matrix should write a file when output_path is given."""
        cm = np.array([[10, 2], [3, 15]])
        path = str(tmp_path / "cm.png")
        plot_confusion_matrix(cm, labels=["neg", "pos"], output_path=path)
        assert os.path.exists(path)
