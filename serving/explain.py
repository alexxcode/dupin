"""explain — factores de la decisión por predicción.

Fiel a la atribución nativa del boosting (LightGBM `pred_contrib`): se eligen los
features que MÁS pesaron en ESTA predicción (por |contribución|) y se describen
con su valor. El signo de la contribución da la dirección (sube/baja el riesgo),
así que la explicación sirve igual para un bloqueo que para una aprobación —no
solo para "flags". Si el modelo no soporta pred_contrib (p. ej. en tests), cae a
un ranking heurístico por valor.
"""
from __future__ import annotations

import math

import pandas as pd

from features.config import FEATURE_NAMES


def _label(name: str, feats: dict[str, float]) -> str:
    """Descripción legible del valor del feature (neutral; la dirección la da el signo)."""
    v = feats[name]
    if name == "log_amount":
        return f"Monto ${math.expm1(v):,.0f}"
    if name == "type_transfer":
        return "Operación TRANSFER" if v >= 1 else "Operación CASH_OUT"
    if name == "hour":
        return f"Hora del día: {int(v)}h"
    if name == "dest_prior_count":
        return "Receptor sin historial" if v == 0 else f"Receptor con {int(v)} tx previas"
    if name == "dest_is_new":
        return "Receptor sin historial (cuenta nueva)" if v >= 1 else "Receptor con historial"
    if name == "dest_amount_ratio":
        return f"Monto {v:.1f}× el promedio del receptor"
    if name == "dest_amount_z":
        return f"Monto {v:+.1f}σ vs el patrón del receptor"
    if name == "dest_recency":
        return "Receptor sin tx previa" if v >= 9999 else f"Última tx al receptor hace {int(v)}h"
    if name == "dest_cnt_24h":
        return f"{int(v)} tx al receptor en 24h"
    if name == "dest_cnt_168h":
        return f"{int(v)} tx al receptor en 7 días"
    if name == "dest_amt_sum_24h":
        return f"${v:,.0f} recibidos en 24h"
    if name == "dest_amt_sum_168h":
        return f"${v:,.0f} recibidos en 7 días"
    if name == "orig_prior_count":
        return f"{int(v)} tx previas del originador"
    return name


def _contributions(model, X: pd.DataFrame) -> dict[str, float] | None:
    """Contribución por feature (LightGBM, en log-odds). None si no se soporta.

    Signo: > 0 empuja hacia FRAUDE, < 0 hacia legítimo.
    """
    try:
        contrib = model.predict(X[FEATURE_NAMES], pred_contrib=True)
    except (TypeError, AttributeError, Exception):  # noqa: BLE001
        return None
    row = contrib[0]
    return {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}


# Reglas de respaldo (sin atribución): features de valor sospechoso, dirección +.
_FALLBACK = [
    ("dest_is_new", lambda v: v >= 1.0),
    ("dest_amount_ratio", lambda v: v >= 2.0),
    ("dest_amount_z", lambda v: abs(v) >= 3.0),
    ("dest_cnt_24h", lambda v: v >= 3.0),
    ("log_amount", lambda v: v >= 12.0),
    ("type_transfer", lambda v: v >= 1.0),
]


def explain(model, feats: dict[str, float], top_k: int = 4) -> list[dict]:
    """Top-k factores de la decisión, rankeados por |contribución| del modelo.

    Cada item: {feature, message, contribution}. `contribution` > 0 sube el riesgo,
    < 0 lo baja (el dashboard lo muestra con dirección y color).
    """
    X = pd.DataFrame([feats])[FEATURE_NAMES]
    contrib = _contributions(model, X)

    if contrib is not None:
        order = sorted(FEATURE_NAMES, key=lambda n: abs(contrib[n]), reverse=True)
        out = []
        for name in order:
            if abs(contrib[name]) < 1e-9:
                continue
            out.append({"feature": name, "message": _label(name, feats),
                        "contribution": contrib[name]})
            if len(out) >= top_k:
                break
        return out

    # Fallback heurístico (sin pred_contrib): factores de riesgo por valor.
    out = []
    for name, fires in _FALLBACK:
        if fires(feats[name]):
            out.append({"feature": name, "message": _label(name, feats), "contribution": None})
        if len(out) >= top_k:
            break
    return out
