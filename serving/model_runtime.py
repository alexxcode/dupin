"""ModelRuntime — singleton de serving: carga el bundle y puntúa en vivo.

Al arrancar descarga el bundle desde GCS (o ruta local), valida que su
`feature_version` coincide con el código de features cargado, y mantiene un
`FeatureState` en memoria (cold-start) que se actualiza con cada transacción.

Usa el MISMO `features_from_state` del entrenamiento: paridad por construcción.
Rechaza arrancar si falta cualquier componente del bundle (Invariante 5).
"""
from __future__ import annotations

import os
import tempfile
import time
from threading import Lock

import pandas as pd

from features.build_features import features_from_state
from features.config import DEFAULT_CONFIG, FEATURE_NAMES, FeatureConfig
from serving.explain import explain
from training.export import APPROVE, ModelBundle


def _download_bundle_from_gcs(gcs_prefix: str, local_dir: str) -> str:
    """Descarga model.joblib + manifest.json desde gs://bucket/prefix a local_dir."""
    from google.cloud import storage

    assert gcs_prefix.startswith("gs://")
    _, _, rest = gcs_prefix.partition("gs://")
    bucket_name, _, prefix = rest.partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    os.makedirs(local_dir, exist_ok=True)
    for fname in ("model.joblib", "manifest.json"):
        blob = bucket.blob(f"{prefix.rstrip('/')}/{fname}")
        blob.download_to_filename(os.path.join(local_dir, fname))
    return local_dir


def _read_bytes(uri: str) -> bytes:
    """Lee bytes desde gs:// o ruta local."""
    if uri.startswith("gs://"):
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, obj = rest.partition("/")
        return storage.Client().bucket(bucket_name).blob(obj).download_as_bytes()
    with open(uri, "rb") as f:
        return f.read()


class ModelRuntime:
    def __init__(self, bundle: ModelBundle, config: FeatureConfig = DEFAULT_CONFIG):
        if bundle.feature_version != config.feature_version:
            raise ValueError(
                f"feature_version del bundle ({bundle.feature_version}) != código "
                f"({config.feature_version}); el modelo y las features no coinciden."
            )
        self.bundle = bundle
        self.config = config
        self._scope = set(config.scope_types)
        self._lock = Lock()
        self.scored_count = 0
        self.warm_entities = 0
        # Estado de entidades en vivo. Arranca vacío; un warm-state opcional lo
        # precarga con la historia previa al demo (ver _load_warm_state).
        from features.entity_state import FeatureState
        self.state = FeatureState(config)

    def _load_warm_state(self, uri: str) -> None:
        """Precarga el FeatureState desde un snapshot (gs:// o local, .json/.gz).

        Hace que el scoring en vivo vea la misma historia que la evaluación
        offline en lugar de arrancar cold-start. Valida feature_version."""
        import gzip
        import json

        from features.entity_state import FeatureState

        data = _read_bytes(uri)
        if uri.endswith(".gz"):
            data = gzip.decompress(data)
        snap = json.loads(data)
        snap_fv = snap.get("feature_version")
        if snap_fv != self.config.feature_version:
            raise ValueError(
                f"warm-state feature_version ({snap_fv}) != código "
                f"({self.config.feature_version})"
            )
        self.state = FeatureState.from_snapshot(snap, self.config)
        self.warm_entities = len(snap.get("dest", {}))

    @classmethod
    def from_uri(cls, uri: str, config: FeatureConfig = DEFAULT_CONFIG) -> "ModelRuntime":
        """Carga desde una ruta local o gs://. Lanza si el bundle está incompleto."""
        if uri.startswith("gs://"):
            local = _download_bundle_from_gcs(uri, tempfile.mkdtemp(prefix="dupin_bundle_"))
        else:
            local = uri
        bundle = ModelBundle.load(local)   # rechaza si falta componente
        runtime = cls(bundle, config)
        warm_uri = os.environ.get("DUPIN_WARM_STATE_URI", "")
        if warm_uri:
            runtime._load_warm_state(warm_uri)
        return runtime

    def score(self, step: int, tx_type: str, amount: float, name_orig: str, name_dest: str) -> dict:
        """Puntúa una transacción cruda y actualiza el estado. Thread-safe."""
        t0 = time.perf_counter()
        cfg = self.config
        with self._lock:
            # Fuera de superficie de fraude → aprobar sin modelo (decisión explícita).
            if tx_type not in self._scope:
                self.state.update(name_orig, name_dest, step, amount)
                self.scored_count += 1
                return {
                    "score": 0.0,
                    "decision": APPROVE,
                    "scorable": False,
                    "reasons": [{"feature": "type", "message": f"Tipo '{tx_type}' fuera de la superficie de fraude (solo TRANSFER/CASH_OUT)", "contribution": None}],
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "model_version": self.bundle.model_version,
                    "feature_version": self.bundle.feature_version,
                }

            dest_state = self.state.dest_state(name_dest)        # estado PREVIO
            orig_pc = self.state.orig_prior_count(name_orig)
            feats = features_from_state(step, tx_type, amount, dest_state, orig_pc, cfg)

            X = pd.DataFrame([feats])[FEATURE_NAMES]
            score = float(self.bundle.model.predict_proba(X)[:, 1][0])
            decision = self.bundle.decide_score(score)
            reasons = explain(self.bundle.model, feats)

            # Incorporar al estado DESPUÉS de puntuar (causalidad).
            self.state.update(name_orig, name_dest, step, amount)
            self.scored_count += 1

        return {
            "score": score,
            "decision": decision,
            "scorable": True,
            "reasons": reasons,
            "latency_ms": (time.perf_counter() - t0) * 1000,
            "model_version": self.bundle.model_version,
            "feature_version": self.bundle.feature_version,
        }
