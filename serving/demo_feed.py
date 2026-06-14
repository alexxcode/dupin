"""Feed de transacciones para el dashboard en vivo.

Primario: una muestra real del periodo de test publicada en GCS (CC BY-SA → vive
en GCS, no en git). Fallback: un feed sintético para desarrollo local sin GCS.
Se cachea en memoria tras la primera carga.
"""
from __future__ import annotations

import json
import os
import random

_cache: list[dict] | None = None


def _synthetic(n: int = 800, seed: int = 42) -> list[dict]:
    """Feed sintético plausible: receptores que repiten + ráfagas sospechosas."""
    rng = random.Random(seed)
    receivers = [f"C{rng.randint(10**8, 10**9)}" for _ in range(60)]
    hot = receivers[:6]  # receptores "calientes" que concentran ráfagas
    rows: list[dict] = []
    step = 480
    for i in range(n):
        if i % 40 == 0:
            step += 1
        if rng.random() < 0.06:  # ráfaga sospechosa a un receptor caliente
            dest = rng.choice(hot)
            amount = rng.uniform(2e5, 8e6)
            is_fraud = int(rng.random() < 0.45)
        else:
            dest = rng.choice(receivers)
            amount = rng.uniform(10, 4e5)
            is_fraud = 0
        rows.append({
            "step": step,
            "type": "TRANSFER" if rng.random() < 0.3 else "CASH_OUT",
            "amount": round(amount, 2),
            "nameOrig": f"C{rng.randint(10**8, 10**9)}",
            "nameDest": dest,
            "isFraud": is_fraud,
        })
    return rows


def _load_from_gcs(uri: str) -> list[dict] | None:
    try:
        from google.cloud import storage

        _, _, rest = uri.partition("gs://")
        bucket_name, _, obj = rest.partition("/")
        text = storage.Client().bucket(bucket_name).blob(obj).download_as_text()
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except Exception:  # noqa: BLE001 - cae al fallback sintético
        return None


def load_demo_feed(limit: int = 3000) -> list[dict]:
    global _cache
    if _cache is None:
        uri = os.environ.get("DUPIN_DEMO_FEED_URI", "")
        rows = _load_from_gcs(uri) if uri.startswith("gs://") else None
        _cache = rows if rows else _synthetic()
    return _cache[:limit]
