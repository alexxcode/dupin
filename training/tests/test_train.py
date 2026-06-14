"""Test de integración: train_and_build produce un bundle válido y desplegable."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from data.splits import TemporalSplit
from features.config import FEATURE_NAMES
from training.export import APPROVE, BLOCK, REVIEW
from training.train import train_and_build


def _synthetic_matrix(n_per_seg=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lo, hi in [(0, 432), (432, 480), (480, 744)]:
        for _ in range(n_per_seg):
            y = int(rng.random() < 0.2)
            row = {f: rng.normal(0, 1) for f in FEATURE_NAMES}
            row["dest_amount_ratio"] = rng.normal(3.0 if y else 0.5, 0.5)
            row["step"] = int(rng.integers(lo, hi))
            row["isFraud"] = y
            rows.append(row)
    return pd.DataFrame(rows)


def _make_model():
    return HistGradientBoostingClassifier(max_iter=80, random_state=0)


def test_train_and_build_produces_valid_bundle(tmp_path):
    matrix = _synthetic_matrix()
    bundle = train_and_build(
        matrix, make_model=_make_model,
        split=TemporalSplit(432, 480), review_budget=0.1, block_budget=0.02,
    )

    # Umbrales coherentes: block más estricto que review.
    assert bundle.thresholds["block"] >= bundle.thresholds["review"]
    assert bundle.feature_version == "feat-v1"
    assert bundle.model_version == "m-v1"

    # Metadata con métricas de test y curva envolvente.
    tm = bundle.metadata["test_metrics"]
    assert 0.0 <= tm["pr_auc"] <= 1.0
    assert len(tm["recall_review_curve"]) > 0

    # Round-trip y decisión funcionan.
    bundle.save(str(tmp_path))
    from training.export import ModelBundle
    loaded = ModelBundle.load(str(tmp_path))
    s = loaded.score(matrix.head(5))
    assert len(s) == 5
    assert loaded.decide_score(1.0) in (APPROVE, REVIEW, BLOCK)


def test_thresholds_separate_decisions():
    matrix = _synthetic_matrix()
    bundle = train_and_build(
        matrix, make_model=_make_model,
        split=TemporalSplit(432, 480), review_budget=0.1, block_budget=0.02,
    )
    # Un score por encima de block debe bloquear; por debajo de review, aprobar.
    assert bundle.decide_score(bundle.thresholds["block"] + 1e-9) == BLOCK
    assert bundle.decide_score(bundle.thresholds["review"] - 1e-9) == APPROVE
