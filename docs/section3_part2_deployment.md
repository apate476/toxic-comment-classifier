# Section 3 — Part 2: FastAPI, Cloud Run & Streamlit/HuggingFace Deployment

This document covers the serving half of the GCP deployment: the FastAPI
inference service, the Cloud Run deployment of that service, and the public
Streamlit demo on HuggingFace Spaces.

Part 1 (project setup, Artifact Registry, training job) is documented
separately; this part assumes the training Docker image workflow from
Section 2 already pushes images to the registry.

## Project context

| Item | Value |
|---|---|
| GCP project ID | `mlops-toxic-pahse3-498520` |
| Region | `us-central1` |
| Artifact Registry repo | `toxic-comment-images` |
| Registry path | `us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images` |

---

## 1. FastAPI inference service

### Files

| File | Purpose |
|---|---|
| `api/main.py` | FastAPI app: lifespan model loading, `/health`, `/predict` |
| `api/schemas.py` | Pydantic request/response validation |
| `tests/test_api.py` | Unit tests (TestClient against a toy trained pipeline) |
| `requirements-api.txt` | Slim serving dependencies (no torch/dvc/mlflow) |

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe; reports whether the model artifact is loaded |
| `/predict` | POST | Multi-label predictions for 1–100 comments |
| `/docs` | GET | Auto-generated Swagger UI (OpenAPI spec at `/openapi.json`) |

### Request/response contract

Request — `comments` must contain 1–100 strings, each 1–10,000 characters
(enforced by Pydantic; violations return `422`):

```json
POST /predict
{ "comments": ["you are a wonderful person", "i will hunt you down"] }
```

Response:

```json
{
  "predictions": [
    {
      "comment": "i will hunt you down",
      "labels": { "toxic": true, "severe_toxic": false, "obscene": false,
                  "threat": true, "insult": false, "identity_hate": false },
      "probabilities": { "toxic": 0.91, "severe_toxic": 0.12, "obscene": 0.08,
                         "threat": 0.77, "insult": 0.31, "identity_hate": 0.05 }
    }
  ],
  "version": "baseline_tfidf_logreg.joblib"
}
```

Error behavior: `503` if the model artifact is missing, `422` on invalid
input, `500` (with detail) if inference itself fails.

### Run locally

```bash
pip install -r requirements-api.txt
# train first if models/baseline_tfidf_logreg.joblib does not exist:
#   python -m toxic_comment_classifier.train_model
uvicorn api.main:app --reload
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["you are great"]}'
```

The model path is configurable with `MODEL_PATH` (default
`models/baseline_tfidf_logreg.joblib`).

---

## 2. Cloud Run deployment

### Container

`dockerfiles/Dockerfile.api` builds a slim serving image
(`python:3.11-slim-bookworm`, only `requirements-api.txt`, the `api/`
package, and the model artifact). It honors Cloud Run's injected `PORT`.

```bash
# Build (from repo root; requires the trained model in models/)
docker build -f dockerfiles/Dockerfile.api -t toxic-api .

# Test locally the same way Cloud Run runs it
docker run -p 8080:8080 -e PORT=8080 toxic-api
curl localhost:8080/health
```

### Push to Artifact Registry

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag toxic-api \
  us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-api:v1

docker push \
  us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-api:v1
```

### Deploy

```bash
gcloud run deploy toxic-comment-api \
  --project mlops-toxic-pahse3-498520 \
  --region us-central1 \
  --image us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-api:v1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --concurrency 80
```

Auto-scaling: `min-instances 0` scales to zero when idle (no cost),
`max-instances 3` caps spend; Cloud Run scales between them on request
load. The first request after idle pays a cold start (~2–4 s for this
image).

### Invoke the deployed service

```bash
SERVICE_URL=$(gcloud run services describe toxic-comment-api \
  --project mlops-toxic-pahse3-498520 --region us-central1 \
  --format 'value(status.url)')

curl "$SERVICE_URL/health"
curl -X POST "$SERVICE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"comments": ["you are great", "i will hurt you"]}'
```

---

## 3. Streamlit demo on HuggingFace Spaces

### Files

| File | Purpose |
|---|---|
| `streamlit_demo/app.py` | Streamlit UI (text input, threshold slider, per-label probability bars) |
| `streamlit_demo/requirements.txt` | Space dependencies |
| `streamlit_demo/README.md` | Space configuration (YAML front matter) + deploy guide |

### Backend resolution

The app uses the bundled joblib model when present, otherwise it calls the
Cloud Run API (Space secret `API_URL`). This means the Space keeps working
even if the Cloud Run service is torn down for cost reasons — just bundle
the artifact.

### Deploy

1. Create a Space at huggingface.co (SDK: **Streamlit**, hardware: free CPU).
2. Push the demo files:

```bash
git clone https://huggingface.co/spaces/<username>/toxic-comment-classifier hf-space
cp streamlit_demo/app.py streamlit_demo/requirements.txt streamlit_demo/README.md hf-space/
mkdir -p hf-space/models && cp models/baseline_tfidf_logreg.joblib hf-space/models/
cd hf-space && git add . && git commit -m "deploy streamlit demo" && git push
```

3. (Remote-API mode) In the Space settings, add secret `API_URL` =
   the Cloud Run service URL.

### Feature walkthrough

- Paste a comment, pick a decision threshold, hit **Classify**.
- Flagged labels are summarized in a banner; all six label probabilities
  render as progress bars with percentages.
- The sidebar shows which backend (bundled model vs. remote API) served
  the prediction.

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `503 Model is not loaded` | Artifact missing from image — ensure `models/baseline_tfidf_logreg.joblib` exists at build time (train locally or `dvc pull` first) |
| `422 Unprocessable Entity` | Empty list, empty string, >100 comments, or >10,000 chars per comment |
| Cloud Run "container failed to start" | App must listen on `$PORT`; keep the shell-form `CMD` in `Dockerfile.api` |
| Slow first request | Cold start from `min-instances 0`; set `--min-instances 1` (costs more) |
| HF Space build fails | `requirements.txt` must be at the Space root; check the Space build logs |

## 5. Cost notes

- Cloud Run at `min-instances 0` bills only per-request CPU/memory; demo
  traffic stays within the free tier.
- Artifact Registry charges for storage (~$0.10/GB·month) — delete old
  image tags.
- HuggingFace Spaces free CPU tier is sufficient for the demo.
- Cleanup: `gcloud run services delete toxic-comment-api` and delete unused
  registry tags when the course phase is graded.
