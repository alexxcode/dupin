"""Test de integración: evaluate() corre de extremo a extremo y reporta el gap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.splits import TemporalSplit
from evaluation.evaluate import evaluate
from evaluation import report as report_mod
from features.config import FEATURE_NAMES


def _synthetic_matrix(n_per_seg=400, seed=0) -> pd.DataFrame:
    """Matriz con las columnas reales y señal correlacionada con la etiqueta,
    con fraude en los tres segmentos temporales."""
    rng = np.random.default_rng(seed)
    rows = []
    # Tres bloques temporales: train (<432), val ([432,480)), test (>=480).
    for lo, hi in [(0, 432), (432, 480), (480, 744)]:
        for _ in range(n_per_seg):
            y = int(rng.random() < 0.2)
            row = {f: rng.normal(0, 1) for f in FEATURE_NAMES}
            # Señal: dest_amount_ratio más alto en fraude.
            row["dest_amount_ratio"] = rng.normal(3.0 if y else 0.5, 0.5)
            row["step"] = int(rng.integers(lo, hi))
            row["isFraud"] = y
            rows.append(row)
    return pd.DataFrame(rows)


def test_evaluate_end_to_end_structure():
    matrix = _synthetic_matrix()
    rep = evaluate(matrix, budget=0.1, seed=0, split=TemporalSplit(432, 480))

    for key in ("budget", "split_report", "temporal", "random", "gap"):
        assert key in rep
    for regime in ("temporal", "random"):
        op = rep[regime]["operating_point"]
        assert 0.0 <= rep[regime]["pr_auc"] <= 1.0
        assert 0.0 <= op["recall"] <= 1.0
        assert op["review_rate"] <= 0.2  # alrededor del budget 0.1

    # El reporte se formatea sin error.
    assert "Régimen" in report_mod.format_table(rep)
    assert "Segmento" in report_mod.format_split(rep)


def test_split_report_has_fraud_on_all_segments():
    matrix = _synthetic_matrix()
    rep = evaluate(matrix, budget=0.1, seed=0, split=TemporalSplit(432, 480))
    sr = rep["split_report"]
    for seg in ("train", "val", "test"):
        assert sr[seg]["rows"] > 0
        assert sr[seg]["fraud"] > 0
