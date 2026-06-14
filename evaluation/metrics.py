"""Métricas de fraude con desbalance — precision-recall y punto de operación.

La cabecera NO es accuracy ni ROC-AUC a secas (engañan con desbalance extremo):
es precision-recall y el punto de operación en lenguaje de negocio —cuánto fraude
se atrapa a cambio de revisar cuántas operaciones legítimas (Invariante 4).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def average_precision(y_true, scores) -> float:
    """Área bajo la curva precision-recall (PR-AUC). Métrica de cabecera."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true, scores) -> float:
    """ROC-AUC. Reportada solo como referencia, no como cabecera."""
    return float(roc_auc_score(y_true, scores))


def pr_curve(y_true, scores):
    """precision, recall, thresholds (para graficar la curva completa)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return precision, recall, thresholds


def threshold_for_review_budget(scores, budget: float) -> float:
    """Umbral tal que se marca como máximo `budget` fracción de operaciones.

    Es el cuantil (1 - budget) de los scores. Se fija sobre VALIDACIÓN y se aplica
    tal cual al test (el review_rate del test puede diferir: eso es realista).
    """
    return float(np.quantile(scores, 1.0 - budget))


def operating_point(y_true, scores, threshold: float) -> dict:
    """Confusión + recall/precision/review_rate a un umbral dado."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n = len(y)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "review_rate": (tp + fp) / n if n else 0.0,
        "n_flagged": tp + fp,
        "n": n,
    }


def select_threshold(y_val, scores_val, budget: float = 0.01) -> float:
    """Regla de cabecera: umbral por presupuesto de revisión sobre validación."""
    return threshold_for_review_budget(scores_val, budget)


def recall_at_review_rate(y_true, scores, review_rate: float) -> dict:
    """Punto de operación al fijar el umbral PARA lograr `review_rate` EN ESTE set.

    Es la ENVOLVENTE de capacidad del modelo (qué recall es alcanzable a ese
    presupuesto), no el umbral desplegado (que se fija sobre validación). Útil
    para mostrar el techo honesto sin el sesgo de transferencia val→test.
    """
    thr = threshold_for_review_budget(scores, review_rate)
    return operating_point(y_true, scores, thr)


def recall_review_curve(y_true, scores, rates=None) -> list[dict]:
    """Curva recall vs presupuesto de revisión sobre un mismo set (envolvente)."""
    if rates is None:
        rates = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    return [recall_at_review_rate(y_true, scores, r) for r in rates]


def cost_curve(
    y_true, scores, fn_cost: float = 100.0, fp_cost: float = 1.0, n_points: int = 200
) -> dict:
    """Curva de costo de negocio: costo esperado vs umbral (FN caro, FP barato).

    Secundaria al recall@budget; traduce el trade-off a unidades de costo. Default:
    un fraude perdido cuesta 100× una revisión innecesaria.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores)
    qs = np.linspace(0.0, 1.0, n_points)
    thresholds = np.quantile(s, qs)
    costs, recalls, review_rates = [], [], []
    for thr in thresholds:
        op = operating_point(y, s, thr)
        costs.append(op["fn"] * fn_cost + op["fp"] * fp_cost)
        recalls.append(op["recall"])
        review_rates.append(op["review_rate"])
    costs = np.asarray(costs)
    best = int(np.argmin(costs))
    return {
        "thresholds": thresholds.tolist(),
        "costs": costs.tolist(),
        "recalls": recalls,
        "review_rates": review_rates,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "best_threshold": float(thresholds[best]),
        "best_cost": float(costs[best]),
    }
