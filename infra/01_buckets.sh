#!/usr/bin/env bash
# Crea los buckets raw / features / artifacts (idempotente).
set -euo pipefail
: "${PROJECT_ID:?export PROJECT_ID=dupin-dupin}"
REGION="${REGION:-us-central1}"

for name in raw features artifacts; do
  bucket="gs://${PROJECT_ID}-${name}"
  if gsutil ls -b "$bucket" >/dev/null 2>&1; then
    echo "Ya existe: $bucket"
  else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "$bucket"
    echo "Creado: $bucket"
  fi
done
