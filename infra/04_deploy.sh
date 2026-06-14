#!/usr/bin/env bash
# Build (Cloud Build) + deploy (Cloud Run), con verificación del bundle.
set -euo pipefail
: "${PROJECT_ID:?export PROJECT_ID=dupin-dupin}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-dupin}"
SERVICE="${SERVICE:-dupin}"
MODEL_VERSION="${MODEL_VERSION:-m-v1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/dupin:latest"
BUNDLE_URI="gs://${PROJECT_ID}-artifacts/${MODEL_VERSION}"
SA_EMAIL="dupin-serving@${PROJECT_ID}.iam.gserviceaccount.com"

# 1) Verificar que el bundle existe ANTES de desplegar (rechaza si falta).
echo "Verificando bundle en ${BUNDLE_URI}..."
gsutil -q stat "${BUNDLE_URI}/manifest.json" || { echo "FALTA manifest.json en ${BUNDLE_URI}"; exit 1; }
gsutil -q stat "${BUNDLE_URI}/model.joblib"  || { echo "FALTA model.joblib en ${BUNDLE_URI}"; exit 1; }
echo "Bundle OK."

# 2) Build + push de la imagen.
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_IMAGE="${IMAGE}" .

# 3) Deploy a Cloud Run (escala a cero; descarga el bundle al arrancar).
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --service-account="${SA_EMAIL}" \
  --set-env-vars="DUPIN_BUNDLE_URI=${BUNDLE_URI}" \
  --memory=1Gi --cpu=1 --min-instances=0 \
  --allow-unauthenticated

echo "Desplegado. Prueba: curl \$(gcloud run services describe ${SERVICE} --region ${REGION} --format='value(status.url)')/health"
