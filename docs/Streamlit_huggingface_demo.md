# Streamlit / Hugging Face Demo Deployment

## Overview

This document describes the Streamlit frontend deployed on Hugging Face Spaces for the toxic comment classifier project.

The frontend provides a user-friendly interface for testing the deployed toxic comment classifier API. It sends user-entered comments to the Cloud Run FastAPI backend and displays real model prediction results.

## Deployed Hugging Face Space

The Streamlit demo is deployed at:

```text
https://huggingface.co/spaces/taha7908/patil
```

## Backend API

The Hugging Face Space calls the deployed Cloud Run API:

```text
https://toxic-comment-api-491682843765.us-central1.run.app
```

## Demo Features

The Streamlit app includes:

- API health check
- Model availability check
- Single-comment prediction
- Multi-comment prediction
- Predicted toxicity labels
- Toxic probability display
- Class probability table
- Raw API response display

## Verified Functionality

The deployed Hugging Face demo was tested successfully.

The `/health` endpoint returned a healthy response and confirmed that the trained model was available:

```json
{
  "status": "ok",
  "service": "toxic-comment-classifier-api",
  "model_available": true,
  "model_path": "models/baseline_tfidf_logreg.joblib"
}
```

The `/predict` endpoint returned real model predictions with labels and class probabilities instead of placeholder responses.

Example prediction output:

```json
{
  "predictions": [
    {
      "comment": "You are awful and disgusting.",
      "labels": ["toxic"],
      "probabilities": {
        "toxic": 0.5128,
        "severe_toxic": 0.0527,
        "obscene": 0.2286,
        "threat": 0.0281,
        "insult": 0.2755,
        "identity_hate": 0.0563
      }
    }
  ]
}
```

## Deployment Notes

The Hugging Face Space was deployed manually through the Hugging Face web interface.

The source Streamlit app is included in this repository under:

```text
demo/streamlit_app.py
```

The dependency file is included under:

```text
demo/requirements.txt
```

The frontend does not store or load the machine learning model directly. The trained model runs inside the Cloud Run FastAPI service, and the Hugging Face Space only calls the deployed API.

## Section 3 Integration

This deployment completes the Streamlit / Hugging Face portion of the GCP deployment section.

The final serving flow is:

```text
Hugging Face Streamlit frontend
        ↓
Google Cloud Run FastAPI backend
        ↓
Trained toxic comment classifier model
        ↓
Predicted labels and class probabilities
```

## Evidence to Include

Recommended screenshots for documentation or presentation:

- Hugging Face Space running successfully
- Streamlit health check showing `model_available=true`
- Streamlit prediction result showing labels and class probabilities
- Cloud Run `/health` curl response
- Cloud Run `/predict` curl response
- Artifact Registry image
- GitHub Actions GCP authentication check
