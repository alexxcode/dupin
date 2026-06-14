#!/usr/bin/env bash
# Crea el repositorio Docker en Artifact Registry (idempotente).
set -euo pipefail
: "${PROJECT_ID:?export PROJECT_ID=dupin-dupin}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-dupin}"

if gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  echo "Ya existe el repo AR: $REPO"
else
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker --location="$REGION" \
    --description="Imágenes de Dupin"
  echo "Repo AR creado: $REPO"
fi
