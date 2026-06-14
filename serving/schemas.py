"""Contrato de la API de scoring (pydantic)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Transacción cruda a puntuar. Sin columnas de balance (prohibidas por fuga)."""

    step: int = Field(..., ge=0, description="Hora de simulación (1 step = 1 hora).")
    type: str = Field(..., description="Tipo de operación PaySim.")
    amount: float = Field(..., ge=0, description="Monto de la transacción.")
    nameOrig: str = Field(..., min_length=1, description="Cuenta originadora.")
    nameDest: str = Field(..., min_length=1, description="Cuenta receptora.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 500, "type": "TRANSFER", "amount": 181000.0,
                "nameOrig": "C840083671", "nameDest": "C38997010",
            }
        }
    }


class Reason(BaseModel):
    feature: str
    message: str
    contribution: float | None = None   # atribución del modelo (None si no disponible)


class ScoreResponse(BaseModel):
    score: float                # probabilidad de fraude [0,1]
    decision: str               # approve / review / block
    scorable: bool              # False si fuera de superficie de fraude
    reasons: list[Reason]
    latency_ms: float
    model_version: str
    feature_version: str
