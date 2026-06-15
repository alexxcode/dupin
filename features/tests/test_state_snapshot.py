"""Tests del snapshot de estado (warm-start): round-trip, poda, corte por step."""
from __future__ import annotations

import pandas as pd
import pytest

from features.build_features import build_state, features_from_state
from features.config import FEATURE_NAMES, DEFAULT_CONFIG
from features.entity_state import FeatureState

CFG = DEFAULT_CONFIG


def _stream() -> pd.DataFrame:
    rows = [
        (1, "TRANSFER", "O1", "D1", 100.0),
        (1, "CASH_OUT", "O2", "D1", 200.0),
        (2, "PAYMENT", "O3", "D1", 50.0),
        (5, "TRANSFER", "O4", "D1", 300.0),
        (10, "TRANSFER", "O1", "D2", 10.0),
        (480, "CASH_OUT", "O5", "D1", 400.0),
    ]
    return pd.DataFrame(rows, columns=["step", "type", "nameOrig", "nameDest", "amount"])


def _feats(state: FeatureState, step, ttype, amount, orig, dest):
    return features_from_state(
        step, ttype, amount, state.dest_state(dest), state.orig_prior_count(orig), CFG
    )


def test_build_state_respects_max_step():
    state = build_state(_stream(), CFG, max_step=480)
    # Procesa steps 1,1,2,5,10 pero NO el 480.
    assert state.dest_state("D1").count == 4
    assert state.orig_prior_count("O1") == 2  # step1 y step10
    assert state.dest_state("D2").count == 1


def test_snapshot_roundtrip_preserves_features():
    state = build_state(_stream(), CFG, max_step=480)
    restored = FeatureState.from_snapshot(state.snapshot(), CFG)
    # Para una tx de prueba, el vector debe ser idéntico antes y después.
    a = _feats(state, 480, "CASH_OUT", 500.0, "O5", "D1")
    b = _feats(restored, 480, "CASH_OUT", 500.0, "O5", "D1")
    for f in FEATURE_NAMES:
        assert a[f] == pytest.approx(b[f]), f


def test_snapshot_prune_to_subset():
    state = build_state(_stream(), CFG)
    snap = state.snapshot(dest_ids={"D1"}, orig_ids={"O1"})
    assert set(snap["dest"]) == {"D1"}
    assert set(snap["orig"]) == {"O1"}
    restored = FeatureState.from_snapshot(snap, CFG)
    assert restored.dest_state("D2") is None          # podado
    assert restored.dest_state("D1") is not None
    assert restored.orig_prior_count("O1") == 2


def test_warm_state_makes_dest_not_new():
    """El punto del warm-start: un receptor con historia NO sale como nuevo."""
    restored = FeatureState.from_snapshot(build_state(_stream(), CFG, 480).snapshot(), CFG)
    feats = _feats(restored, 480, "CASH_OUT", 400.0, "O5", "D1")
    assert feats["dest_is_new"] == 0.0
    assert feats["dest_prior_count"] == 4.0
    # Sin warm-start (estado vacío) saldría nuevo:
    cold = _feats(FeatureState(CFG), 480, "CASH_OUT", 400.0, "O5", "D1")
    assert cold["dest_is_new"] == 1.0
