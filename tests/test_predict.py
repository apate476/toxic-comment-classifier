"""Tests for prediction pipeline components."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from toxic_comment_classifier.predict_model import _resolve_input_file, predict


class TestResolveInputFile:
    def test_raises_when_no_file_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise FileNotFoundError when no candidate exists."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _resolve_input_file(tmp_path / "nonexistent.csv")

    def test_returns_existing_file(self, tmp_path: Path) -> None:
        """Should return the path when the file exists."""
        f = tmp_path / "test.csv"
        f.write_text("id,comment_text\n1,hello\n")
        result = _resolve_input_file(f)
        assert result == f


class TestPredict:
    def test_raises_when_model_missing(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when model file does not exist."""
        with pytest.raises(FileNotFoundError, match="Model file was not found"):
            predict(
                tmp_path / "no_model.joblib",
                tmp_path / "input.csv",
                tmp_path / "output.csv",
            )

    def test_raises_when_no_comment_text_column(self, tmp_path: Path) -> None:
        """Should raise ValueError when input data lacks comment_text column."""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", OneVsRestClassifier(LogisticRegression())),
            ]
        )
        texts = ["hello world", "toxic comment", "nice day"]
        y = np.array([[1, 0], [0, 1], [0, 0]])
        model.fit(texts, y)

        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)

        input_path = tmp_path / "input.csv"
        input_path.write_text("wrong_col\nhello\n")

        with pytest.raises(ValueError, match="comment_text"):
            predict(model_path, input_path, tmp_path / "output.csv")

    def test_predict_writes_output(self, tmp_path: Path) -> None:
        """Predict should write a CSV with label columns."""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("clf", OneVsRestClassifier(LogisticRegression())),
            ]
        )
        texts = ["hello world", "toxic comment", "nice day", "bad word", "good post", "hate this"]
        y = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
        )
        model.fit(texts, y)

        model_path = tmp_path / "model.joblib"
        joblib.dump(model, model_path)

        input_path = tmp_path / "input.csv"
        input_path.write_text("id,comment_text\n1,hello world\n2,toxic comment\n")

        output_path = tmp_path / "output.csv"
        predict(model_path, input_path, output_path)

        assert output_path.exists()
        result = pd.read_csv(output_path)
        assert "toxic" in result.columns
        assert len(result) == 2
