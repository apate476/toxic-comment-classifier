"""Streamlit demo for the toxic comment classifier (HuggingFace Spaces).

The app classifies a comment against the six Jigsaw toxicity labels.
It resolves a backend in this order:

1. A local joblib model artifact (``MODEL_PATH``, default
   ``models/baseline_tfidf_logreg.joblib``) — used when the Space bundles
   the model file.
2. A remote FastAPI endpoint (``API_URL`` environment variable or Space
   secret pointing at the Cloud Run service) — used when the model is
   served remotely.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

LABEL_DISPLAY = {
    "toxic": "Toxic",
    "severe_toxic": "Severely toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}

DEFAULT_MODEL_PATH = "models/baseline_tfidf_logreg.joblib"


@st.cache_resource
def load_local_model() -> Any | None:
    """Load the joblib pipeline if it is bundled with the app."""
    path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def get_api_url() -> str | None:
    """Resolve the remote API URL from secrets or the environment."""
    url = os.environ.get("API_URL", "")
    try:
        url = st.secrets.get("API_URL", url)
    except FileNotFoundError:
        pass
    return url.rstrip("/") or None


def predict_local(model: Any, text: str) -> dict[str, float]:
    """Run inference with the bundled sklearn pipeline."""
    probabilities = model.predict_proba([text])[0]
    return {label: float(probabilities[i]) for i, label in enumerate(LABEL_COLUMNS)}


def predict_remote(api_url: str, text: str) -> dict[str, float]:
    """Run inference through the deployed FastAPI service."""
    response = requests.post(f"{api_url}/predict", json={"comments": [text]}, timeout=30)
    response.raise_for_status()
    return response.json()["predictions"][0]["probabilities"]


def main() -> None:
    st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🛡️", layout="centered")
    st.title("🛡️ Toxic Comment Classifier")
    st.caption(
        "Multi-label toxicity detection (TF-IDF + One-vs-Rest logistic regression) "
        "trained on the Jigsaw toxic comment dataset. MLOps course project, Phase 3."
    )

    model = load_local_model()
    api_url = get_api_url()

    if model is not None:
        st.sidebar.success("Backend: bundled model artifact")
    elif api_url is not None:
        st.sidebar.info(f"Backend: remote API ({api_url})")
    else:
        st.error(
            "No backend available. Bundle the model artifact with the Space "
            "or set the API_URL secret to the Cloud Run service URL."
        )
        st.stop()

    with st.form("classify"):
        text = st.text_area(
            "Comment text",
            placeholder="Type or paste a comment to analyze…",
            height=140,
            max_chars=10_000,
        )
        threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
        submitted = st.form_submit_button("Classify", type="primary", use_container_width=True)

    if not submitted:
        return
    if not text.strip():
        st.warning("Please enter a comment first.")
        return

    with st.spinner("Classifying…"):
        try:
            scores = predict_local(model, text) if model is not None else predict_remote(str(api_url), text)
        except Exception as exc:  # noqa: BLE001 - show backend errors in the UI
            st.error(f"Prediction failed: {exc}")
            return

    flagged = [label for label, score in scores.items() if score >= threshold]
    if flagged:
        st.error("Flagged: " + ", ".join(LABEL_DISPLAY[label] for label in flagged))
    else:
        st.success("No toxicity detected at the current threshold.")

    st.subheader("Label probabilities")
    for label in LABEL_COLUMNS:
        score = scores[label]
        col_name, col_bar, col_value = st.columns([2, 5, 1])
        col_name.write(LABEL_DISPLAY[label])
        col_bar.progress(min(max(score, 0.0), 1.0))
        col_value.write(f"{score:.2%}")


if __name__ == "__main__":
    main()
