"""Pydantic request/response schemas for the inference API.

Kept free of any ``toxic_comment_classifier`` imports so the API can be
packaged into a slim serving container without the full training stack.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

#: Label order must match the training pipeline
#: (see ``toxic_comment_classifier.predict_model.LABEL_COLUMNS``).
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

CommentText = Annotated[str, StringConstraints(min_length=1, max_length=10_000)]


class PredictRequest(BaseModel):
    """Request body for ``POST /predict``."""

    comments: list[CommentText] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="One or more comment texts to classify (1-100 items, each 1-10,000 characters).",
        examples=[["you are a wonderful person", "i will hunt you down"]],
    )


class CommentPrediction(BaseModel):
    """Per-comment prediction with binary labels and per-label probabilities."""

    comment: str = Field(description="The input comment text, echoed back.")
    labels: dict[str, bool] = Field(description="Binary prediction for each of the six toxicity labels.")
    probabilities: dict[str, float] = Field(description="Predicted probability for each label, rounded to 4 decimals.")


class PredictResponse(BaseModel):
    """Response body for ``POST /predict``."""

    predictions: list[CommentPrediction]
    version: str = Field(description="Identifier of the model artifact serving the request.")


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    model_loaded: bool
