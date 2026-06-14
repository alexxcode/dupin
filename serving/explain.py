"""explain — razón legible del flag por predicción.

Combina dos cosas: la atribución nativa del boosting (LightGBM `pred_contrib`,
SHAP-like) para RANKEAR qué empujó la decisión, y plantillas legibles que
traducen los valores de las features a lenguaje de risk lead ("monto 12× el
promedio del receptor"). Si el modelo no soporta pred_contrib (p. ej. en tests),
cae a un ranking heurístico — las plantillas no dependen del modelo.
"""
from __future__ import annotations

import pandas as pd

from features.config import FEATURE_NAMES

# (feature, ¿dispara?, render) — orden = prioridad por defecto si no hay atribución.
_RULES: list[tuple[str, "callable", "callable"]] = [
    ("dest_is_new", lambda v: v >= 1.0, lambda f: "Receptor sin historial previo (cuenta nueva)"),
    ("dest_amount_ratio", lambda v: v >= 2.0, lambda f: f"Monto {f['dest_amount_ratio']:.1f}× el promedio histórico del receptor"),
    ("dest_amount_z", lambda v: abs(v) >= 3.0, lambda f: f"Monto {f['dest_amount_z']:.1f}σ respecto al patrón del receptor"),
    ("dest_cnt_24h", lambda v: v >= 3.0, lambda f: f"{int(f['dest_cnt_24h'])} transacciones al receptor en 24h (velocidad alta)"),
    ("dest_cnt_168h", lambda v: v >= 8.0, lambda f: f"{int(f['dest_cnt_168h'])} transacciones al receptor en 7 días"),
    ("dest_recency", lambda v: v <= 1.0, lambda f: "Transacción casi inmediata tras la anterior al receptor"),
    ("log_amount", lambda v: v >= 12.0, lambda f: "Monto elevado"),
    ("type_transfer", lambda v: v >= 1.0, lambda f: "Operación tipo TRANSFER"),
]


def _contributions(model, X: pd.DataFrame) -> dict[str, float] | None:
    """Atribución por feature (LightGBM). None si el modelo no la soporta."""
    try:
        contrib = model.predict(X[FEATURE_NAMES], pred_contrib=True)
    except (TypeError, AttributeError, Exception):  # noqa: BLE001 - cualquier modelo no-LGBM
        return None
    row = contrib[0]
    # pred_contrib devuelve n_features + 1 (último = base/expected). Descartar base.
    return {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}


def explain(model, feats: dict[str, float], top_k: int = 3) -> list[dict]:
    """Top-k razones legibles que empujaron la decisión, rankeadas por atribución.

    `feats` es el vector de features de la transacción (dict por nombre).
    """
    X = pd.DataFrame([feats])[FEATURE_NAMES]
    contrib = _contributions(model, X)

    triggered: list[dict] = []
    for prio, (name, fires, render) in enumerate(_RULES):
        if fires(feats[name]):
            c = contrib[name] if contrib is not None else None
            # salience: atribución del modelo si existe; si no, prioridad de la regla.
            salience = c if c is not None else (len(_RULES) - prio)
            triggered.append({
                "feature": name,
                "message": render(feats),
                "contribution": c,
                "_salience": salience,
            })

    triggered.sort(key=lambda r: r["_salience"], reverse=True)
    for r in triggered:
        r.pop("_salience")
    return triggered[:top_k]
