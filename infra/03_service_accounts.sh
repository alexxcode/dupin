#!/usr/bin/env bash
# SA de serving: SOLO lectura de objetos sobre el bucket de artifacts (mínimo privilegio).
set -euo pipefail
: "${PROJECT_ID:?export PROJECT_ID=dupin-dupin}"
SA="dupin-serving"
SA_EMAIL="${SA}@${PROJECT_ID}.iam.gserviceaccount.com"
ART_BUCKET="gs://${PROJECT_ID}-artifacts"

if ! gcloud iam service-accounts describe "$SA_EMAIL" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SA" --display-name="Dupin serving (Cloud Run)"
fi

# objectViewer acotado al bucket de artifacts, no a todo el proyecto.
gsutil iam ch "serviceAccount:${SA_EMAIL}:roles/storage.objectViewer" "$ART_BUCKET"
echo "SA listo: $SA_EMAIL (objectViewer sobre $ART_BUCKET)"
