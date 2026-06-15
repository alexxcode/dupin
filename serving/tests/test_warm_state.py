"""Test: ModelRuntime precarga el warm-state y el scoring deja de ser cold-start."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features.build_features import build_state
from features.config import FEATURE_NAMES, DEFAULT_CONFIG
from training.train import train_and_build


def _matrix(n_per_seg=300, seed=0) -> pd.DataFrame:
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


def _history() -> pd.DataFrame:
    rows = [
        (100, "TRANSFER", "Oa", "Dwarm", 100.0),
        (200, "CASH_OUT", "Ob", "Dwarm", 300.0),
        (450, "TRANSFER", "Oc", "Dwarm", 200.0),
    ]
    return pd.DataFrame(rows, columns=["step", "type", "nameOrig", "nameDest", "amount"])


def test_runtime_loads_warm_state_and_scores_with_history(tmp_path, monkeypatch):
    # Bundle de juguete.
    bundle = train_and_build(
        _matrix(), make_model=lambda: HistGradientBoostingClassifier(max_iter=40, random_state=0),
        review_budget=0.1, block_budget=0.02,
    )
    bdir = tmp_path / "bundle"
    bundle.save(str(bdir))

    # Snapshot de warm-state con historia para 'Dwarm'.
    state = build_state(_history(), DEFAULT_CONFIG, max_step=480)
    snap_path = tmp_path / "warm.json"
    snap_path.write_text(json.dumps(state.snapshot()))

    monkeypatch.setenv("DUPIN_WARM_STATE_URI", str(snap_path))
    from serving.model_runtime import ModelRuntime
    rt = ModelRuntime.from_uri(str(bdir))

    assert rt.warm_entities >= 1
    # 'Dwarm' tiene historia → no es cuenta nueva al puntuar.
    res = rt.score(500, "TRANSFER", 250.0, "Onew", "Dwarm")
    assert res["scorable"] is True
    assert rt.state.dest_state("Dwarm").count == 4  # 3 previos + el recién puntuado
