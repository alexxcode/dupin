"""Auditoría de fuga — detecta features que filtran el futuro o la etiqueta.

Comprobación automática: si una sola feature, usada como score por sí misma,
separa el fraude de forma sospechosamente perfecta (AUC ≈ 1), es señal de fuga
(de etiqueta o de futuro). Las features honestas de Dupin no deberían superar el
umbral; las columnas de balance prohibidas SÍ (control positivo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def single_feature_aucs(X: pd.DataFrame, y) -> dict[str, float]:
    """AUC de cada feature usada como score (agnóstico a la dirección)."""
    y = np.asarray(y).astype(int)
    res: dict[str, float] = {}
    for col in X.columns:
        v = X[col].to_numpy(dtype="float64")
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if np.all(v == v[0]):          # constante: no separa nada
            res[col] = 0.5
            continue
        auc = roc_auc_score(y, v)
        res[col] = float(max(auc, 1.0 - auc))   # dirección-agnóstico
    return res


def audit(X: pd.DataFrame, y, suspicious: float = 0.98) -> dict:
    """Audita una matriz de features. `passed` es True si ninguna feature aislada
    alcanza un AUC sospechoso."""
    aucs = single_feature_aucs(X, y)
    flagged = {k: a for k, a in aucs.items() if a >= suspicious}
    ranked = dict(sorted(aucs.items(), key=lambda kv: kv[1], reverse=True))
    return {
        "single_feature_auc": ranked,
        "flagged": flagged,
        "suspicious_threshold": suspicious,
        "passed": len(flagged) == 0,
    }
