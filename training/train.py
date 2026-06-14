"""train — split temporal → fit → selección de umbrales → bundle.

Entrena sobre el split TEMPORAL de entrenamiento, fija los umbrales review/block
por presupuesto sobre VALIDACIÓN, evalúa sobre el test futuro (punto desplegado
val→test Y envolvente honesta recall-vs-presupuesto) y ensambla el bundle
versionado. El modelo es agnóstico: `make_model` decide la familia (LightGBM en
Colab, HistGB en tests).
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from data.splits import DEFAULT_SPLIT, TemporalSplit, assign_temporal, split_report
from evaluation import metrics
from features.config import FEATURE_NAMES
from training.export import ModelBundle


def _scale_pos_weight(y) -> float:
    pos = int(np.sum(y))
    neg = len(y) - pos
    return neg / pos if pos else 1.0


def train_and_build(
    matrix: pd.DataFrame,
    make_model,
    feature_names: list[str] = FEATURE_NAMES,
    label_col: str = "isFraud",
    split: TemporalSplit = DEFAULT_SPLIT,
    review_budget: float = 0.01,
    block_budget: float = 0.001,
    feature_version: str = "feat-v1",
    model_version: str = "m-v1",
    seed: int = 42,
) -> ModelBundle:
    """Entrena y devuelve el bundle listo para publicar."""
    X = matrix[feature_names]
    y = matrix[label_col].to_numpy().astype(int)

    a = assign_temporal(matrix, split)
    tr = (a == "train").to_numpy()
    va = (a == "val").to_numpy()
    te = (a == "test").to_numpy()

    spw = _scale_pos_weight(y[tr])
    clf = make_model()
    sample_weight = np.where(y[tr] == 1, spw, 1.0)
    clf.fit(X[tr], y[tr], sample_weight=sample_weight)

    sc_val = clf.predict_proba(X[va])[:, 1]
    sc_test = clf.predict_proba(X[te])[:, 1]

    # Umbrales por presupuesto, fijados sobre VALIDACIÓN.
    t_review = metrics.threshold_for_review_budget(sc_val, review_budget)
    t_block = metrics.threshold_for_review_budget(sc_val, block_budget)

    # Punto de operación DESPLEGADO (umbral val aplicado al test futuro).
    deployed_review = metrics.operating_point(y[te], sc_test, t_review)
    deployed_block = metrics.operating_point(y[te], sc_test, t_block)

    # ENVOLVENTE honesta: recall alcanzable si se gasta el presupuesto exacto en test.
    envelope_review = metrics.recall_at_review_rate(y[te], sc_test, review_budget)
    curve = metrics.recall_review_curve(y[te], sc_test)

    thresholds = {
        "review": float(t_review),
        "block": float(t_block),
        "review_budget": review_budget,
        "block_budget": block_budget,
    }
    metadata = {
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seed": seed,
        "scale_pos_weight": spw,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "split": split_report(matrix, split, label_col),
        "test_metrics": {
            "pr_auc": metrics.average_precision(y[te], sc_test),
            "roc_auc": metrics.roc_auc(y[te], sc_test),
            "test_prevalence": float(y[te].mean()),
            "deployed_review": deployed_review,
            "deployed_block": deployed_block,
            "envelope_review": envelope_review,
            "recall_review_curve": curve,
        },
    }
    return ModelBundle(
        model=clf,
        feature_names=list(feature_names),
        feature_version=feature_version,
        model_version=model_version,
        thresholds=thresholds,
        metadata=metadata,
    )
