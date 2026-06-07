"""FastAPI inference service for the toxic comment classifier.

Serves the TF-IDF + One-vs-Rest logistic regression pipeline trained by
``toxic_comment_classifier.train_model`` (a joblib-serialized sklearn
``Pipeline``).

Run locally from the repository root:

    uvicorn api.main:app --reload

Configuration (environment variables):

- ``MODEL_PATH`` — path to the joblib model artifact
  (default: ``models/baseline_tfidf_logreg.joblib``).

Endpoints:

- ``GET /health`` — liveness probe; reports whether the model is loaded.
- ``POST /predict`` — multi-label toxicity predictions for 1-100 comments.

Interactive OpenAPI docs are served at ``/docs`` (Swagger UI) and ``/redoc``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import (
    LABEL_COLUMNS,
    CommentPrediction,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/baseline_tfidf_logreg.joblib")

#: Mutable application state populated by the lifespan handler.
_state: dict[str, Any] = {"model": None, "version": "unloaded"}


def _model_path() -> Path:
    """Resolve the model artifact path from the environment."""
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup and release it at shutdown."""
    path = _model_path()
    if path.exists():
        logger.info("Loading model from %s", path)
        _state["model"] = joblib.load(path)
        _state["version"] = path.name
        logger.info("Model loaded: %s", path.name)
    else:
        logger.warning("Model artifact not found at %s; /predict will return 503", path)
    yield
    _state["model"] = None
    _state["version"] = "unloaded"


app = FastAPI(
    title="Toxic Comment Classifier API",
    description="Multi-label toxicity classification for user comments (Jigsaw labels).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Liveness/readiness probe for Cloud Run and load balancers."""
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classify comments against the six Jigsaw toxicity labels."""
    model = _state["model"]
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is not loaded (expected artifact at {_model_path()}).",
        )

    try:
        raw_preds = np.asarray(model.predict(request.comments))
        if hasattr(model, "predict_proba"):
            raw_probs = np.asarray(model.predict_proba(request.comments))
        else:  # pragma: no cover - all sklearn OvR-LogReg pipelines expose predict_proba
            raw_probs = raw_preds.astype(float)
    except Exception as exc:  # noqa: BLE001 - surface model errors as a clean 500
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    predictions = [
        CommentPrediction(
            comment=comment,
            labels={label: bool(raw_preds[i][j]) for j, label in enumerate(LABEL_COLUMNS)},
            probabilities={label: round(float(raw_probs[i][j]), 4) for j, label in enumerate(LABEL_COLUMNS)},
        )
        for i, comment in enumerate(request.comments)
    ]
    return PredictResponse(predictions=predictions, version=str(_state["version"]))
