"""Tests del bundle: decisión, round-trip de persistencia, validación estricta."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from training.export import APPROVE, BLOCK, REVIEW, ModelBundle, decide
from features.config import FEATURE_NAMES


def test_decide_thresholds():
    assert decide(0.05, t_review=0.3, t_block=0.8) == APPROVE
    assert decide(0.50, t_review=0.3, t_block=0.8) == REVIEW
    assert decide(0.90, t_review=0.3, t_block=0.8) == BLOCK
    # Bordes: >= es inclusivo.
    assert decide(0.30, 0.3, 0.8) == REVIEW
    assert decide(0.80, 0.3, 0.8) == BLOCK


def _toy_bundle() -> ModelBundle:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = (X["dest_amount_ratio"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    return ModelBundle(
        model=model,
        feature_names=list(FEATURE_NAMES),
        feature_version="feat-v1",
        model_version="m-v1",
        thresholds={"review": 0.3, "block": 0.8, "review_budget": 0.01, "block_budget": 0.001},
        metadata={"note": "toy"},
    )


def test_save_load_roundtrip(tmp_path):
    b = _toy_bundle()
    b.save(str(tmp_path))
    loaded = ModelBundle.load(str(tmp_path))

    assert loaded.feature_names == b.feature_names
    assert loaded.feature_version == "feat-v1"
    assert loaded.thresholds["review"] == 0.3
    # Mismas predicciones tras el round-trip.
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(20, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    np.testing.assert_allclose(b.score(X), loaded.score(X))


def test_load_rejects_missing_model(tmp_path):
    b = _toy_bundle()
    b.save(str(tmp_path))
    (tmp_path / "model.joblib").unlink()
    with pytest.raises(ValueError, match="model.joblib"):
        ModelBundle.load(str(tmp_path))


def test_load_rejects_missing_manifest(tmp_path):
    b = _toy_bundle()
    b.save(str(tmp_path))
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest.json"):
        ModelBundle.load(str(tmp_path))


def test_load_rejects_incomplete_thresholds(tmp_path):
    import json
    b = _toy_bundle()
    b.save(str(tmp_path))
    man_path = tmp_path / "manifest.json"
    man = json.loads(man_path.read_text())
    del man["thresholds"]["block"]
    man_path.write_text(json.dumps(man))
    with pytest.raises(ValueError, match="block"):
        ModelBundle.load(str(tmp_path))
