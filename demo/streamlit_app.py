import requests
import streamlit as st

API_URL = "https://toxic-comment-api-491682843765.us-central1.run.app"

st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Toxic Comment Classifier Demo")
st.write(
    "This Streamlit app calls the deployed Cloud Run FastAPI service "
    "and displays real model predictions from the toxic comment classifier."
)

left_col, right_col = st.columns([2, 1])

with right_col:
    st.subheader("Deployment Info")
    st.markdown(
        """
        **Frontend:** Streamlit  
        **Backend:** FastAPI  
        **Serving:** Google Cloud Run  
        **Registry:** Artifact Registry  
        **Training:** Vertex AI / Dockerized training  
        """
    )

    st.subheader("Cloud Run API")
    st.code(API_URL, language="text")

    st.info(
        "Enter one or more comments. The app sends them to the deployed API "
        "and displays predicted toxicity labels and probabilities."
    )

with left_col:
    st.subheader("1. API Health Check")

    if st.button("Run Health Check", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            response.raise_for_status()
            health = response.json()

            if health.get("model_available"):
                st.success("API is healthy and the model is available.")
            else:
                st.warning("API is running, but the model is not available.")

            st.json(health)
        except requests.RequestException as error:
            st.error(f"Health check failed: {error}")

    st.divider()

    st.subheader("2. Prediction Demo")

    sample_option = st.selectbox(
        "Choose a sample or enter your own:",
        [
            "Custom input",
            "Positive example",
            "Toxic example",
            "Multiple comments",
        ],
    )

    default_text = ""

    if sample_option == "Positive example":
        default_text = "Thank you for your help. I really appreciate it."
    elif sample_option == "Toxic example":
        default_text = "You are awful and disgusting."
    elif sample_option == "Multiple comments":
        default_text = (
            "Thank you for your help. I really appreciate it.\n"
            "You are awful and disgusting.\n"
            "Have a great day."
        )

    comments_text = st.text_area(
        "Comments",
        value=default_text,
        height=160,
        placeholder="Type one comment per line...",
    )

    comments = [line.strip() for line in comments_text.splitlines() if line.strip()]
    st.caption(f"Comments ready to submit: {len(comments)}")

    if st.button("Run Prediction", type="primary", use_container_width=True):
        if not comments:
            st.warning("Please enter at least one comment.")
        else:
            payload = {"comments": comments}

            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                    timeout=20,
                )
                response.raise_for_status()
                result = response.json()

                st.success("Prediction request completed.")

                predictions = result.get("predictions", [])

                st.subheader("Prediction Results")

                for index, item in enumerate(predictions, start=1):
                    comment = item.get("comment", "")
                    labels = item.get("labels", [])
                    probabilities = item.get("probabilities", {})

                    toxic_probability = probabilities.get("toxic")
                    predicted_label = ", ".join(labels) if labels else "unknown"

                    with st.container(border=True):
                        st.markdown(f"### Comment {index}")
                        st.write(comment)

                        metric_col1, metric_col2 = st.columns(2)
                        metric_col1.metric("Predicted Label", predicted_label)
                        metric_col2.metric(
                            "Toxic Probability",
                            "N/A"
                            if toxic_probability is None
                            else f"{toxic_probability:.4f}",
                        )

                        if probabilities:
                            st.markdown("#### Class Probabilities")

                            probability_rows = [
                                {
                                    "Label": label,
                                    "Probability": probability,
                                }
                                for label, probability in probabilities.items()
                            ]

                            st.dataframe(probability_rows, use_container_width=True)

                with st.expander("Raw API Response"):
                    st.json(result)

            except requests.RequestException as error:
                st.error("Prediction request failed.")
                st.exception(error)
