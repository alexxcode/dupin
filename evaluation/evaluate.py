"""evaluate — protocolo de evaluación honesto, produce la tabla temporal-vs-aleatorio.

Entrena un clasificador sobre el split TEMPORAL y reporta su punto de operación
sobre el test futuro (el número desplegable). Repite con un split ALEATORIO del
mismo tamaño como diagnóstico de optimismo. El gap entre ambos es el resultado
central por el eje de fuga TEMPORAL. (El eje de fuga de ETIQUETA —columnas de
balance— se cuantifica aparte; ver notebook 03 y leakage_audit.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from data.splits import (
    DEFAULT_SPLIT,
    TemporalSplit,
    assign_temporal,
    random_assign_like,
    split_report,
)
from evaluation import metrics
from features.config import FEATURE_NAMES


def default_baseline(seed: int = 42) -> HistGradientBoostingClassifier:
    """Gradient boosting tabular (sklearn, sin deps extra; robusto a outliers de
    z-score). La Fase 4 puede cambiarlo por XGBoost/LightGBM para el bundle."""
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        max_iter=300,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _fit(make_model, X_tr, y_tr, seed):
    """Fit con ponderación de clase (scale_pos_weight)."""
    clf = make_model(seed)
    pos = int(y_tr.sum())
    neg = len(y_tr) - pos
    spw = neg / pos if pos else 1.0
    sample_weight = np.where(y_tr == 1, spw, 1.0)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clf


def _regime(make_model, X, y, assign, budget, seed):
    """Evalúa un régimen (temporal o aleatorio) dado el vector de asignación.

    Ajusta UNA vez sobre train; el umbral se elige sobre val y se aplica al test.
    """
    tr = (assign == "train").to_numpy()
    va = (assign == "val").to_numpy()
    te = (assign == "test").to_numpy()

    clf = _fit(make_model, X[tr], y[tr], seed)
    sc_val = clf.predict_proba(X[va])[:, 1]
    sc_test = clf.predict_proba(X[te])[:, 1]

    thr = metrics.select_threshold(y[va], sc_val, budget)
    op = metrics.operating_point(y[te], sc_test, thr)
    summary = {
        "pr_auc": metrics.average_precision(y[te], sc_test),
        "roc_auc": metrics.roc_auc(y[te], sc_test),
        # Prevalencia del test: clave para interpretar el PR-AUC. El PR-AUC de un
        # clasificador aleatorio == prevalencia, así que NO es comparable entre
        # regímenes con distinta prevalencia (temporal-cola vs aleatorio-global).
        "test_prevalence": float(y[te].mean()),
        "operating_point": op,
    }
    return summary, y[te], sc_test, clf


def evaluate(
    matrix: pd.DataFrame,
    feature_names: list[str] = FEATURE_NAMES,
    label_col: str = "isFraud",
    split: TemporalSplit = DEFAULT_SPLIT,
    budget: float = 0.01,
    seed: int = 42,
    make_model=default_baseline,
    return_scores: bool = False,
) -> dict:
    """Produce el reporte temporal-vs-aleatorio al presupuesto de revisión dado.

    Si `return_scores`, adjunta bajo la clave privada `_scores` los (y_test,
    scores) de cada régimen para graficar curvas. Esa clave NO es serializable a
    JSON: el llamador debe quitarla antes de guardar (report.save_json la ignora).
    """
    X = matrix[feature_names].to_numpy(dtype="float32")
    y = matrix[label_col].to_numpy().astype(int)

    temporal_assign = assign_temporal(matrix, split)
    random_assign = random_assign_like(temporal_assign, seed)

    temporal, y_te_t, sc_te_t, clf_t = _regime(
        make_model, X, y, temporal_assign, budget, seed
    )
    random, y_te_r, sc_te_r, _ = _regime(
        make_model, X, y, random_assign, budget, seed
    )

    report = {
        "budget": budget,
        "split_config": {
            "train_max_step": split.train_max_step,
            "val_max_step": split.val_max_step,
        },
        "split_report": split_report(matrix, split, label_col),
        "temporal": temporal,
        "random": random,
        "gap": {
            # Cabecera del eje TEMPORAL: el recall@budget al MISMO punto de
            # operación de negocio. Cuánto sobreestima el split aleatorio el
            # fraude atrapado en producción.
            "recall": (
                random["operating_point"]["recall"]
                - temporal["operating_point"]["recall"]
            ),
            # PR-AUC: contextual, NO comparable entre regímenes (distinta
            # prevalencia de test). Ver test_prevalence en cada régimen.
            "pr_auc_confounded_by_prevalence": random["pr_auc"] - temporal["pr_auc"],
        },
    }
    if return_scores:
        report["_scores"] = {
            "temporal": (y_te_t, sc_te_t),
            "random": (y_te_r, sc_te_r),
            "temporal_model": clf_t,
        }
    return report
