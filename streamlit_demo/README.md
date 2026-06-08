---
title: Toxic Comment Classifier
emoji: 🛡️
colorFrom: red
colorTo: indigo
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
license: mit
---

# Toxic Comment Classifier — Streamlit Demo

Interactive demo for the Phase 3 toxic comment classifier (TF-IDF +
One-vs-Rest logistic regression, trained on the Jigsaw dataset).

This directory is the source for the HuggingFace Space. The YAML front
matter above is the Space configuration — keep it as the first thing in
this file.

## Backends

The app picks a backend in this order:

1. **Bundled model** — if `models/baseline_tfidf_logreg.joblib` is present
   (or `MODEL_PATH` points to an artifact), inference runs in-process.
2. **Remote API** — otherwise it calls the Cloud Run FastAPI service. Set
   the Space secret `API_URL` to the service URL
   (Settings → Variables and secrets in the Space).

## Run locally

```bash
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/app.py
```

## Deploy to HuggingFace Spaces

```bash
# One-time: create the Space (SDK: Streamlit), then:
git clone https://huggingface.co/spaces/<username>/toxic-comment-classifier hf-space
cp streamlit_demo/app.py streamlit_demo/requirements.txt streamlit_demo/README.md hf-space/
mkdir -p hf-space/models && cp models/baseline_tfidf_logreg.joblib hf-space/models/  # optional: bundle model
cd hf-space && git add . && git commit -m "deploy streamlit demo" && git push
```

The full walkthrough lives in `docs/section3_part2_deployment.md`.
