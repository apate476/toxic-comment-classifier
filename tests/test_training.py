"""Unit tests for training pipeline components."""

from __future__ import annotations

import pandas as pd
import pytest

from toxic_comment_classifier.train_model import _flatten_params, _resolve_tracking_uri, _validate_training_data

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class TestValidateTrainingData:
    def test_passes_with_valid_data(self) -> None:
        """Validation should not raise on a well-formed DataFrame."""
        df = pd.DataFrame({"comment_text": ["hello"], **{col: [0] for col in LABEL_COLUMNS}})
        _validate_training_data(df, "comment_text", LABEL_COLUMNS)

    def test_raises_on_missing_column(self) -> None:
        """Validation should raise ValueError when a required column is absent."""
        df = pd.DataFrame({"comment_text": ["hello"]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_training_data(df, "comment_text", LABEL_COLUMNS)

    def test_raises_on_empty_dataframe(self) -> None:
        """Validation should raise ValueError when the DataFrame is empty."""
        df = pd.DataFrame({"comment_text": [], **{col: [] for col in LABEL_COLUMNS}})
        with pytest.raises(ValueError, match="empty"):
            _validate_training_data(df, "comment_text", LABEL_COLUMNS)


class TestResolveTrackingUri:
    def test_passthrough_http(self) -> None:
        """HTTP URIs should be returned unchanged."""
        uri = "http://localhost:5000"
        assert _resolve_tracking_uri(uri) == uri

    def test_passthrough_https(self) -> None:
        """HTTPS URIs should be returned unchanged."""
        uri = "https://mlflow.example.com"
        assert _resolve_tracking_uri(uri) == uri

    def test_file_uri_is_resolved(self) -> None:
        """file: URIs should be converted to absolute paths."""
        result = _resolve_tracking_uri("file:mlruns")
        assert not result.startswith("file:")
        assert "mlruns" in result


class TestFlattenParams:
    def test_flat_dict(self) -> None:
        """Flat DictConfig should produce dot-notation keys."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"seed": 42, "model": {"C": 1.0}})
        result = _flatten_params(cfg)
        assert result["seed"] == 42
        assert result["model.C"] == 1.0

    def test_mlflow_keys_excluded(self) -> None:
        """mlflow.* keys should be stripped from the output."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"seed": 42, "mlflow": {"enabled": True, "experiment_name": "test"}})
        result = _flatten_params(cfg)
        assert not any(k.startswith("mlflow.") for k in result)
