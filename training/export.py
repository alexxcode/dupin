"""ModelBundle — el artefacto versionado: modelo + esquema + umbrales + metadata.

Un modelo de Dupin NO es un archivo: es el conjunto completo, publicado y cargado
junto (Invariante 5). `load` rechaza si falta cualquier componente. La función
`decide` vive aquí y la comparten entrenamiento y serving (paridad de la decisión).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import joblib

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"

# Decisiones posibles. Score alto = más riesgo.
APPROVE, REVIEW, BLOCK = "approve", "review", "block"


def decide(score: float, t_review: float, t_block: float) -> str:
    """Mapea un score continuo a la decisión. t_block >= t_review."""
    if score >= t_block:
        return BLOCK
    if score >= t_review:
        return REVIEW
    return APPROVE


@dataclass
class ModelBundle:
    model: Any                       # estimador con predict_proba
    feature_names: list[str]
    feature_version: str
    model_version: str
    thresholds: dict[str, float]     # {"review": ..., "block": ..., "review_budget":..., "block_budget":...}
    metadata: dict = field(default_factory=dict)

    # ── Inferencia ──
    def score(self, X) -> Any:
        """Probabilidad de fraude. X debe tener las columnas feature_names en orden."""
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def decide_score(self, score: float) -> str:
        return decide(score, self.thresholds["review"], self.thresholds["block"])

    # ── Persistencia ──
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, os.path.join(directory, MODEL_FILE))
        manifest = {
            "feature_version": self.feature_version,
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "thresholds": self.thresholds,
            "metadata": self.metadata,
        }
        with open(os.path.join(directory, MANIFEST_FILE), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> "ModelBundle":
        """Carga el bundle. Rechaza (ValueError) si falta cualquier componente."""
        model_path = os.path.join(directory, MODEL_FILE)
        manifest_path = os.path.join(directory, MANIFEST_FILE)
        if not os.path.exists(model_path):
            raise ValueError(f"Bundle inválido: falta {MODEL_FILE} en {directory}")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Bundle inválido: falta {MANIFEST_FILE} en {directory}")

        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        for key in ("feature_version", "model_version", "feature_names", "thresholds"):
            if key not in man or man[key] in (None, [], {}):
                raise ValueError(f"Bundle inválido: manifest sin '{key}'")
        for thr in ("review", "block"):
            if thr not in man["thresholds"]:
                raise ValueError(f"Bundle inválido: thresholds sin '{thr}'")

        model = joblib.load(model_path)
        return cls(
            model=model,
            feature_names=man["feature_names"],
            feature_version=man["feature_version"],
            model_version=man["model_version"],
            thresholds=man["thresholds"],
            metadata=man.get("metadata", {}),
        )
