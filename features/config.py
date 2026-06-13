"""FeatureConfig — configuración versionada de la capa de features (feat-v1).

Cambiar cualquier parámetro aquí obliga a incrementar `feature_version` y
regenerar la matriz de features (Invariante 9). La matriz está acoplada a esta
config.

Decisiones de diseño ancladas en la Fase 1 (ver docs/phase1_findings.md):
- El comportamiento se acumula sobre la cuenta RECEPTORA (`nameDest`): los
  originadores son de un solo uso (99.85% únicos), no tienen historia.
- La superficie de scoring es {TRANSFER, CASH_OUT}: el 100% del fraude vive ahí.
- Ventanas temporales: 24h y 168h (step = 1 hora) + agregado histórico + recencia.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str = "feat-v1"

    # Entidad sobre la que se acumula comportamiento (cuenta receptora).
    dest_col: str = "nameDest"
    orig_col: str = "nameOrig"
    time_col: str = "step"
    amount_col: str = "amount"
    type_col: str = "type"
    label_col: str = "isFraud"

    # Superficie de scoring: solo estos tipos reciben score/label. La HISTORIA por
    # entidad se acumula desde TODAS las transacciones, no solo estas.
    scope_types: tuple[str, ...] = ("TRANSFER", "CASH_OUT")

    # Ventanas de agregación en steps (horas). short / medium.
    windows: dict[str, int] = field(default_factory=lambda: {"24h": 24, "168h": 168})

    # Defaults para entidad sin historia (nueva) o estadística indefinida.
    # IDÉNTICOS en batch y online — es lo que garantiza la paridad de los faltantes.
    amount_ratio_default: float = 1.0   # sin desviación: monto == "promedio"
    amount_z_default: float = 0.0       # sin desviación estandarizada
    recency_default: float = 99999.0    # "hace mucho / nunca" para entidad nueva

    @property
    def max_window(self) -> int:
        return max(self.windows.values())


# Nombres de las features que consume el modelo, EN ORDEN. El bundle (Fase 4)
# congela este orden; el serving valida contra él. No reordenar sin re-versionar.
FEATURE_NAMES: list[str] = [
    # Point features (sin historia; columnas de balance EXCLUIDAS por fuga).
    "log_amount",         # log1p(amount)
    "type_transfer",      # 1.0 si TRANSFER, 0.0 si CASH_OUT
    "hour",               # step % 24
    # Comportamiento del receptor (causal, solo tx anteriores).
    "dest_prior_count",   # nº de tx recibidas antes por el dest (histórico)
    "dest_is_new",        # 1.0 si el dest no tenía historia
    "dest_amount_ratio",  # amount / promedio histórico recibido por el dest
    "dest_amount_z",      # z-score del monto vs historia del dest
    "dest_recency",       # steps desde la última tx recibida por el dest
    "dest_cnt_24h",       # nº de tx al dest en las últimas 24h
    "dest_cnt_168h",      # nº de tx al dest en las últimas 168h
    "dest_amt_sum_24h",   # suma de montos al dest en las últimas 24h
    "dest_amt_sum_168h",  # suma de montos al dest en las últimas 168h
    # Comportamiento del originador (casi siempre 0: single-use; señal débil).
    "orig_prior_count",   # nº de tx enviadas antes por el orig
]

# Columnas de metadatos que acompañan a la matriz (no son input del modelo).
META_COLUMNS: list[str] = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]

DEFAULT_CONFIG = FeatureConfig()
