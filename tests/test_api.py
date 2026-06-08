"""Tests for the FastAPI inference service (api/main.py)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from api.main import app
from api.schemas import LABEL_COLUMNS

TRAIN_TEXTS = [
    "you are a wonderful kind person",
    "have a great day my friend",
    "you are stupid and i hate you",
    "i will hurt you idiot scum",
]
# Each of the six labels has at least one positive and one negative example.
TRAIN_LABELS = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1, 1],
]


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    """Train a tiny pipeline matching the production artifact and save it."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=200))),
        ]
    )
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture()
def client(model_file: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient with the toy model loaded through the lifespan handler."""
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_all_labels(client: TestClient) -> None:
    response = client.post("/predict", json={"comments": ["you are lovely", "i hate you idiot"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for prediction in body["predictions"]:
        assert set(prediction["labels"]) == set(LABEL_COLUMNS)
        assert set(prediction["probabilities"]) == set(LABEL_COLUMNS)
        for probability in prediction["probabilities"].values():
            assert 0.0 <= probability <= 1.0
    assert body["version"] == "model.joblib"


def test_predict_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/predict", json={"comments": []})
    assert response.status_code == 422


def test_predict_rejects_empty_string(client: TestClient) -> None:
    response = client.post("/predict", json={"comments": [""]})
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json={"comments": ["hello"]})
    assert response.status_code == 503
