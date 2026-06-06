# Section 3: GCP Deployment

## Overview

This section documents the GCP setup, Artifact Registry image push, FastAPI inference API, and Cloud Run deployment for the toxic comment classifier project.

## GCP Project

Project ID:

```text
mlops-toxic-pahse3-498520
```

Region:

```text
us-central1
```

## Enabled GCP Services

The following GCP services were enabled:

- Artifact Registry
- Cloud Build
- Cloud Run
- Vertex AI

## Artifact Registry

Docker repository:

```text
toxic-comment-images
```

Full Artifact Registry path:

```text
us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images
```

## FastAPI Inference Service

The FastAPI service was added under:

```text
api/main.py
```

The API exposes two endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Confirms the API service is running |
| `/predict` | POST | Accepts comment text and returns predictions |

## API Dockerfile

A separate API-serving Dockerfile was created:

```text
dockerfiles/Dockerfile.api
```

This file is separate from the existing training Dockerfile. The training Dockerfile is used for model training, while this API Dockerfile starts a long-running FastAPI server required by Cloud Run.

A lightweight API dependency file was also added:

```text
requirements-api.txt
```

This keeps the Cloud Run image smaller by installing only the dependencies needed for the current API service.

## Local API Testing

The API was tested locally using Uvicorn:

```bash
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Successful response:

```json
{"status":"ok","service":"toxic-comment-classifier-api"}
```

Prediction test:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments":["you are nice","this is a bad comment"]}'
```

Successful response:

```json
{"predictions":[{"comment":"you are nice","label":"placeholder","toxic_probability":null},{"comment":"this is a bad comment","label":"placeholder","toxic_probability":null}]}
```

## Docker Build and Push

The API image was built for `linux/amd64` and pushed to Artifact Registry using Docker Buildx:

```bash
docker buildx build \
  --no-cache \
  --platform linux/amd64 \
  --provenance=false \
  -f dockerfiles/Dockerfile.api \
  -t us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-comment-api:cloudrun \
  --push \
  .
```

## Cloud Run Deployment

The API image was deployed to Cloud Run:

```bash
gcloud run deploy toxic-comment-api \
  --image us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-comment-api:cloudrun \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080
```

Cloud Run service URL:

```text
https://toxic-comment-api-491682843765.us-central1.run.app
```

## Cloud Run Endpoint Testing

Health endpoint:

```bash
curl https://toxic-comment-api-491682843765.us-central1.run.app/health
```

Successful response:

```json
{"status":"ok","service":"toxic-comment-classifier-api"}
```

Prediction endpoint:

```bash
curl -X POST https://toxic-comment-api-491682843765.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"comments":["you are nice","this is a bad comment"]}'
```

Successful response:

```json
{"predictions":[{"comment":"you are nice","label":"placeholder","toxic_probability":null},{"comment":"this is a bad comment","label":"placeholder","toxic_probability":null}]}
```

## Notes

The current `/predict` endpoint returns placeholder predictions. This confirms that the FastAPI service, Docker image, Artifact Registry push, and Cloud Run deployment are working correctly.

