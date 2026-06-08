import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/baseline_tfidf_logreg.joblib"))

app = FastAPI(
    title="Toxic Comment Classifier API",
    description="FastAPI inference service for toxic comment classification.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    comments: list[str] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    predictions: list[dict]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def format_probabilities(raw_probabilities, index: int) -> dict | None:
    if raw_probabilities is None:
        return None

    try:
        if isinstance(raw_probabilities, list):
            return {
                label: round(float(class_probs[index][1]), 4)
                for label, class_probs in zip(LABEL_COLUMNS, raw_probabilities, strict=False)
            }

        probability_row = np.asarray(raw_probabilities[index]).ravel()
        return {
            label: round(float(probability), 4)
            for label, probability in zip(LABEL_COLUMNS, probability_row, strict=False)
        }
    except Exception:
        return None


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "toxic-comment-classifier-api",
        "model_available": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    comments = request.comments

    try:
        predicted_labels = model.predict(comments)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error

    raw_probabilities = None
    if hasattr(model, "predict_proba"):
        try:
            raw_probabilities = model.predict_proba(comments)
        except Exception:
            raw_probabilities = None

    predictions = []

    for index, comment in enumerate(comments):
        prediction_row = np.asarray(predicted_labels[index]).astype(int).ravel()

        detected_labels = [label for label, value in zip(LABEL_COLUMNS, prediction_row, strict=False) if value == 1]

        if not detected_labels:
            detected_labels = ["non_toxic"]

        predictions.append(
            {
                "comment": comment,
                "labels": detected_labels,
                "probabilities": format_probabilities(raw_probabilities, index),
            }
        )

    return PredictionResponse(predictions=predictions)
