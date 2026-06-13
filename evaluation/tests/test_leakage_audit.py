"""Tests de la auditoría de fuga: caza features que filtran, deja pasar las sanas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.leakage_audit import audit, single_feature_aucs


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({
        "leaky": y.astype(float) + rng.normal(0, 0.01, n),  # casi == etiqueta
        "noise": rng.normal(0, 1, n),                        # sin relación
        "weak": y * 0.3 + rng.normal(0, 1, n),               # señal débil
        "const": np.ones(n),                                 # constante
    })
    return X, y


def test_leaky_feature_is_flagged():
    X, y = _data()
    rep = audit(X, y, suspicious=0.98)
    assert "leaky" in rep["flagged"]
    assert not rep["passed"]


def test_clean_features_pass():
    X, y = _data()
    rep = audit(X[["noise", "weak", "const"]], y, suspicious=0.98)
    assert rep["passed"]
    assert rep["flagged"] == {}


def test_constant_feature_auc_is_half():
    X, y = _data()
    aucs = single_feature_aucs(X[["const"]], y)
    assert aucs["const"] == 0.5


def test_auc_is_direction_agnostic():
    # Una feature perfectamente ANTI-correlacionada también debe dar AUC alto.
    n = 1000
    y = np.array([0, 1] * (n // 2))
    X = pd.DataFrame({"inverted": 1.0 - y + np.random.default_rng(0).normal(0, 0.01, n)})
    aucs = single_feature_aucs(X, y)
    assert aucs["inverted"] > 0.98
