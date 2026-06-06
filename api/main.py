from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(
    title="Toxic Comment Classifier API",
    description="FastAPI inference service for toxic comment classification.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    comments: list[str]


class PredictionResponse(BaseModel):
    predictions: list[dict]


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "toxic-comment-classifier-api"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    predictions = []

    for comment in request.comments:
        predictions.append(
            {
                "comment": comment,
                "label": "placeholder",
                "toxic_probability": None,
            }
        )

    return PredictionResponse(predictions=predictions)