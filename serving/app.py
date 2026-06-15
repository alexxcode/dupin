"""FastAPI de Dupin — scoring de transacciones en tiempo real.

Carga el bundle al arrancar (env DUPIN_BUNDLE_URI; gs:// o ruta local). Expone:
  POST /v1/score   transacción cruda → score + decisión + razón + latency_ms
  GET  /health     estado del servicio (reporta si el modelo está cargado)
  GET  /version    versiones de modelo y features (503 si no hay modelo)
El dashboard en vivo (Fase 6) se sirve same-origin desde aquí.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from serving.demo_feed import load_demo_feed
from serving.model_runtime import ModelRuntime
from serving.schemas import Reason, ScoreRequest, ScoreResponse

_runtime: ModelRuntime | None = None
_load_error: str | None = None


def load_runtime() -> None:
    """Intenta cargar el bundle desde DUPIN_BUNDLE_URI. No lanza: guarda el error."""
    global _runtime, _load_error
    _runtime, _load_error = None, None    # reset antes de (re)cargar
    uri = os.environ.get("DUPIN_BUNDLE_URI")
    if not uri:
        _load_error = "DUPIN_BUNDLE_URI no configurado"
        return
    try:
        _runtime = ModelRuntime.from_uri(uri)
        _load_error = None
    except Exception as exc:  # noqa: BLE001 - se reporta vía /health
        _runtime, _load_error = None, str(exc)


def set_runtime(runtime: ModelRuntime | None) -> None:
    """Inyección directa (tests)."""
    global _runtime, _load_error
    _runtime = runtime
    _load_error = None if runtime else "runtime no cargado"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime()
    yield


app = FastAPI(title="Dupin", description="Real-time transaction fraud scoring", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _runtime else "degraded",
        "model_loaded": _runtime is not None,
        "model_version": _runtime.bundle.model_version if _runtime else None,
        "scored_count": _runtime.scored_count if _runtime else 0,
        "warm_entities": _runtime.warm_entities if _runtime else 0,
        "warm_error": _runtime.warm_error if _runtime else None,
        "error": _load_error,
    }


@app.get("/version")
def version() -> dict:
    if _runtime is None:
        raise HTTPException(status_code=503, detail=f"Modelo no cargado: {_load_error}")
    return {
        "model_version": _runtime.bundle.model_version,
        "feature_version": _runtime.bundle.feature_version,
        "n_features": len(_runtime.bundle.feature_names),
        "thresholds": _runtime.bundle.thresholds,
    }


@app.post("/v1/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if _runtime is None:
        raise HTTPException(status_code=503, detail=f"Modelo no cargado: {_load_error}")
    result = _runtime.score(req.step, req.type, req.amount, req.nameOrig, req.nameDest)
    return ScoreResponse(
        score=result["score"],
        decision=result["decision"],
        scorable=result["scorable"],
        reasons=[Reason(**r) for r in result["reasons"]],
        latency_ms=result["latency_ms"],
        model_version=result["model_version"],
        feature_version=result["feature_version"],
    )


@app.get("/v1/demo-feed")
def demo_feed(limit: int = 3000) -> list[dict]:
    """Feed de transacciones para el dashboard (muestra real de GCS o sintético)."""
    return load_demo_feed(limit)


# Dashboard en vivo, servido same-origin. Se monta AL FINAL para no eclipsar las
# rutas de API; cualquier path no-API cae al estático (index.html en "/").
_DASHBOARD_DIR = Path(__file__).parent / "dashboard"
app.mount("/", StaticFiles(directory=str(_DASHBOARD_DIR), html=True), name="dashboard")
