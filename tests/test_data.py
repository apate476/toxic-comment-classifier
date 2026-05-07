"""Smoke tests for the raw data loaders.

These tests assume the DVC-tracked dataset has been pulled to ``data/raw``.
They are skipped automatically if the files are missing so that CI runs
without the dataset still pass.
"""

from __future__ import annotations

import pytest

from toxic_comment_classifier.config import RAW_DATA_DIR
from toxic_comment_classifier.data import load_raw

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
TEST_LABELS_FILE = "test_labels.csv"

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _require(filename: str) -> None:
    if not (RAW_DATA_DIR / filename).exists():
        pytest.skip(f"{filename} not found in {RAW_DATA_DIR}; run `dvc pull` first")


class TestLoadRaw:
    def test_train_has_expected_columns(self) -> None:
        _require(TRAIN_FILE)
        df = load_raw(TRAIN_FILE)
        assert list(df.columns) == ["id", "comment_text", *LABEL_COLUMNS]

    def test_train_is_non_empty(self) -> None:
        _require(TRAIN_FILE)
        df = load_raw(TRAIN_FILE)
        assert len(df) > 0
        assert df["comment_text"].notna().all()

    def test_train_labels_are_binary(self) -> None:
        _require(TRAIN_FILE)
        df = load_raw(TRAIN_FILE)
        for col in LABEL_COLUMNS:
            assert set(df[col].unique()).issubset({0, 1})

    def test_test_has_expected_columns(self) -> None:
        _require(TEST_FILE)
        df = load_raw(TEST_FILE)
        assert list(df.columns) == ["id", "comment_text"]

    def test_test_labels_has_expected_columns(self) -> None:
        _require(TEST_LABELS_FILE)
        df = load_raw(TEST_LABELS_FILE)
        assert list(df.columns) == ["id", *LABEL_COLUMNS]

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_raw("definitely_not_a_real_file.csv")
