# Vertex AI Custom Training Job Setup

## Overview

This document describes the Vertex AI custom training job setup for the toxic comment classifier project.

Vertex AI custom training allows the project to run model training using a custom container image. For this project, the training job is intended to use the finalized training Docker image produced from the Docker/CML workflow.

## GCP Project

Project ID:

```text
mlops-toxic-pahse3-498520
```

Region:

```text
us-central1
```

## Required APIs

The following GCP services were enabled for Phase 3:

- Vertex AI
- Artifact Registry
- Cloud Build
- Cloud Run

## Artifact Registry

Docker repository:

```text
toxic-comment-images
```

Artifact Registry path:

```text
us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images
```

## Expected Training Image

The Vertex AI custom training job is configured to use the following expected training image:

```text
us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-comment-training:vertex
```

This image should be built from the finalized training Dockerfile/workflow after the Docker/CML section is finalized.

## Submit Training Job Script

A helper script was added for submitting the Vertex AI custom training job:

```text
scripts/submit_vertex_training_job.sh
```

The script sets the GCP project, builds the expected training image URI, and submits a Vertex AI custom job using `gcloud ai custom-jobs create`.

## Training Job Command

The script submits the job using the following command structure:

```bash
gcloud ai custom-jobs create \
  --region="us-central1" \
  --display-name="toxic-comment-training-job" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri="us-central1-docker.pkg.dev/mlops-toxic-pahse3-498520/toxic-comment-images/toxic-comment-training:vertex"
```

## Worker Pool Configuration

The current setup uses:

```text
machine-type=n1-standard-4
replica-count=1
```

This provides a basic CPU-based training configuration suitable for a lightweight custom training job. The machine type can be changed later if the training workload requires more resources.

## Notes

The Vertex AI API is enabled and the custom training job submission script is prepared.

The actual execution of the Vertex AI job depends on the finalized training container image and data/model artifact workflow from the Docker/CML section. This keeps the training workflow separate from the Cloud Run serving workflow and avoids modifying the existing training Dockerfile.

## Status

- Vertex AI API enabled
- Artifact Registry available for training images
- Vertex AI custom training job submit script added
- Training job setup documented
- Actual training job execution pending finalized training image
