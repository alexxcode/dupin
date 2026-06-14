#!/usr/bin/env bash
# Habilita las APIs necesarias. Requiere: export PROJECT_ID=dupin-dupin
set -euo pipefail
: "${PROJECT_ID:?export PROJECT_ID=dupin-dupin}"

gcloud config set project "$PROJECT_ID"
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
echo "APIs habilitadas en $PROJECT_ID."
