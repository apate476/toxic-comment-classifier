#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="mlops-toxic-pahse3-498520"
REGION="us-central1"
REPOSITORY="toxic-comment-images"
IMAGE_NAME="toxic-comment-training"
IMAGE_TAG="vertex"
JOB_NAME="toxic-comment-training-job"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Training image: ${IMAGE_URI}"

gcloud config set project "${PROJECT_ID}"

echo "Submitting Vertex AI custom training job..."

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri="${IMAGE_URI}"

echo "Vertex AI custom training job submitted."
