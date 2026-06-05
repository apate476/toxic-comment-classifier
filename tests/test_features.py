"""Tests for feature engineering."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from toxic_comment_classifier.data.loaders import load_processed, save_processed
from toxic_comment_classifier.data.make_dataset import process_data
from toxic_comment_classifier.features.build_features import build_features
from toxic_comment_classifier.logging_config import get_logger, setup_logging
from toxic_comment_classifier.utils.io import load_json, save_json


class TestBuildFeatures:
    def test_returns_dataframe(self) -> None:
        """build_features should return a DataFrame."""
        df = pd.DataFrame({"comment_text": ["hello", "world"]})
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_mutate_input(self) -> None:
        """build_features should not modify the original DataFrame."""
        df = pd.DataFrame({"comment_text": ["hello"]})
        original_len = len(df)
        build_features(df)
        assert len(df) == original_len

    def test_preserves_columns(self) -> None:
        """build_features should preserve all input columns."""
        df = pd.DataFrame({"comment_text": ["hello"], "toxic": [0]})
        result = build_features(df)
        assert list(result.columns) == list(df.columns)

    def test_preserves_row_count(self) -> None:
        """build_features should preserve the number of rows."""
        df = pd.DataFrame({"comment_text": ["a", "b", "c"]})
        result = build_features(df)
        assert len(result) == 3


class TestProcessData:
    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """process_data should create the output directory if it does not exist."""
        output_dir = tmp_path / "processed"
        process_data(tmp_path, output_dir)
        assert output_dir.exists()

    def test_runs_without_error(self, tmp_path: Path) -> None:
        """process_data should complete without raising."""
        process_data(tmp_path, tmp_path / "out")


class TestJsonIO:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """save_json then load_json should return the original object."""
        data = {"key": "value", "num": 42}
        path = tmp_path / "test.json"
        save_json(data, path)
        result = load_json(path)
        assert result == data

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_json should create missing parent directories."""
        path = tmp_path / "nested" / "dir" / "file.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_load_raises_on_missing_file(self, tmp_path: Path) -> None:
        """load_json should raise when the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")


class TestSetupLogging:
    def test_runs_without_error(self, tmp_path: Path) -> None:
        """setup_logging should complete without raising."""
        setup_logging(log_dir=tmp_path)

    def test_creates_log_file(self, tmp_path: Path) -> None:
        """setup_logging should create the log file."""
        setup_logging(log_dir=tmp_path, log_filename="test.log")
        assert (tmp_path / "test.log").exists()

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling setup_logging twice should not raise."""
        setup_logging(log_dir=tmp_path)
        setup_logging(log_dir=tmp_path)

    def test_get_logger_returns_logger(self) -> None:
        """get_logger should return a logging.Logger instance."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)


class TestLoaders:
    def test_save_and_load_processed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """save_processed then load_processed should return equivalent data."""
        from toxic_comment_classifier import config
        from toxic_comment_classifier.data import loaders

        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path)
        monkeypatch.setattr(loaders, "PROCESSED_DATA_DIR", tmp_path)
        df = pd.DataFrame({"comment_text": ["hello", "world"], "toxic": [0, 1]})
        save_processed(df, "test.csv")
        result = load_processed("test.csv")
        assert list(result.columns) == list(df.columns)
        assert len(result) == 2


class TestMakeDatasetMain:
    def test_main_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """main() should run without error using tmp dirs."""
        import sys

        from toxic_comment_classifier import config
        from toxic_comment_classifier.data.make_dataset import main

        monkeypatch.setattr(sys, "argv", ["make_dataset"])
        monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path)
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "processed")
        main()
