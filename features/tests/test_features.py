"""Tests de la capa de features: fuga temporal, paridad, defaults, dtype/shape.

Romper cualquiera de estos introduce un bug silencioso que solo aparece en
producción (ver Invariantes 1, 2, 9).
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from features.build_features import build_features, features_from_state
from features.config import FEATURE_NAMES, META_COLUMNS, DEFAULT_CONFIG
from features.entity_state import FeatureState

CFG = DEFAULT_CONFIG


def _stream() -> pd.DataFrame:
    """Stream sintético, ya en orden temporal (step asc, estable).

    Incluye un PAYMENT (fuera de superficie) que NO se emite pero SÍ alimenta la
    historia del receptor D1. Montos distintos para identificar filas sin ambig.
    """
    rows = [
        # step, type,       nameOrig, nameDest, amount, isFraud
        (1,   "TRANSFER",  "O1", "D1", 100.0, 0),
        (1,   "CASH_OUT",  "O2", "D1", 200.0, 0),
        (2,   "PAYMENT",   "O3", "D1",  50.0, 0),   # no-scope; alimenta historia
        (3,   "TRANSFER",  "O1", "D2",  10.0, 0),   # O1 reaparece como orig
        (5,   "TRANSFER",  "O4", "D1", 300.0, 1),
        (200, "CASH_OUT",  "O5", "D1", 400.0, 0),   # fuera de ventana 168h
    ]
    return pd.DataFrame(
        rows, columns=["step", "type", "nameOrig", "nameDest", "amount", "isFraud"]
    )


def _row(out: pd.DataFrame, amount: float) -> pd.Series:
    return out.loc[out["amount"] == amount].iloc[0]


# ───────────────────────────── shape / dtype ─────────────────────────────
def test_columns_and_dtype():
    out = build_features(_stream(), CFG)
    assert list(out.columns) == META_COLUMNS + FEATURE_NAMES
    for f in FEATURE_NAMES:
        assert out[f].dtype == "float32", f"{f} no es float32"


def test_scope_filter_emits_only_transfer_and_cashout():
    df = _stream()
    out = build_features(df, CFG)
    # 6 filas, 1 es PAYMENT -> 5 emitidas.
    assert len(out) == 5
    assert set(out["type"]) == {"TRANSFER", "CASH_OUT"}
    # La historia SÍ incluyó el PAYMENT: ver test_history_includes_nonscope.


# ───────────────────────────── defaults ─────────────────────────────
def test_new_entity_defaults():
    df = pd.DataFrame(
        [(1, "TRANSFER", "OX", "DX", 123.0, 0)],
        columns=["step", "type", "nameOrig", "nameDest", "amount", "isFraud"],
    )
    r = build_features(df, CFG).iloc[0]
    assert r["dest_is_new"] == 1.0
    assert r["dest_prior_count"] == 0.0
    assert r["dest_amount_ratio"] == pytest.approx(CFG.amount_ratio_default)
    assert r["dest_amount_z"] == pytest.approx(CFG.amount_z_default)
    assert r["dest_recency"] == pytest.approx(CFG.recency_default)
    assert r["dest_cnt_24h"] == 0.0
    assert r["dest_cnt_168h"] == 0.0
    assert r["dest_amt_sum_24h"] == 0.0
    assert r["orig_prior_count"] == 0.0


# ───────────────────────────── valores conocidos ─────────────────────────────
def test_known_values_row_step5():
    out = build_features(_stream(), CFG)
    r = _row(out, 300.0)  # step5, D1 con priors (100,200,50)
    # count=3, mean=116.6667, sumsq=52500
    mean = 350.0 / 3.0
    var = 52500.0 / 3.0 - mean * mean
    std = math.sqrt(var)
    assert r["dest_prior_count"] == 3.0
    assert r["dest_is_new"] == 0.0
    assert r["dest_amount_ratio"] == pytest.approx(300.0 / mean, rel=1e-5)
    assert r["dest_amount_z"] == pytest.approx((300.0 - mean) / std, rel=1e-5)
    assert r["dest_recency"] == 3.0  # step5 - last_step2
    assert r["dest_cnt_24h"] == 3.0
    assert r["dest_cnt_168h"] == 3.0
    assert r["dest_amt_sum_24h"] == pytest.approx(350.0)
    assert r["hour"] == 5.0
    assert r["type_transfer"] == 1.0
    assert r["log_amount"] == pytest.approx(math.log1p(300.0))


def test_window_eviction_row_step200():
    out = build_features(_stream(), CFG)
    r = _row(out, 400.0)  # step200: todos los priors fuera de 168h
    assert r["dest_prior_count"] == 4.0          # histórico ve los 4
    assert r["dest_cnt_24h"] == 0.0              # ninguno en ventana
    assert r["dest_cnt_168h"] == 0.0
    assert r["dest_amt_sum_168h"] == 0.0
    assert r["dest_recency"] == 195.0            # 200 - 5


def test_orig_prior_count_reappearance():
    out = build_features(_stream(), CFG)
    r = _row(out, 10.0)  # D2, orig O1 que ya fue orig en la fila de 100.0
    assert r["orig_prior_count"] == 1.0


def test_history_includes_nonscope():
    """El PAYMENT (50.0) no se emite, pero cuenta en la historia de D1."""
    out = build_features(_stream(), CFG)
    r = _row(out, 300.0)
    # Si el PAYMENT NO contara, prior_count sería 2 y sum_24h 300, no 3 y 350.
    assert r["dest_prior_count"] == 3.0
    assert r["dest_amt_sum_24h"] == pytest.approx(350.0)


# ───────────────────────────── determinismo ─────────────────────────────
def test_determinism():
    """Misma entrada -> misma salida, bit a bit. (El desempate dentro de un mismo
    step es el orden de emisión, así que barajar la entrada SÍ puede cambiar el
    resultado en empates; eso es correcto, no un fallo de determinismo.)"""
    df = _stream()
    a = build_features(df, CFG)
    b = build_features(df.copy(), CFG)
    pd.testing.assert_frame_equal(a, b)


# ───────────────────────────── SIN FUGA TEMPORAL ─────────────────────────────
def test_no_future_leakage_prefix_invariance():
    """Invariante clave: el vector de una tx no depende de transacciones futuras.

    Construir sobre el stream completo debe dar, para cada fila emitida, el mismo
    vector que construir solo sobre el prefijo hasta esa fila (inclusive). Si
    alguna feature filtrara futuro, ambos diferirían.
    """
    df = _stream().reset_index(drop=True)
    full = build_features(df, CFG)

    # Para cada posición del stream que sea de superficie, recomputa por prefijo.
    for pos in range(len(df)):
        if df.loc[pos, "type"] not in CFG.scope_types:
            continue
        prefix = build_features(df.iloc[: pos + 1], CFG)
        emitted = prefix.iloc[-1]  # la última emitida es la fila `pos`
        amount = df.loc[pos, "amount"]
        ref = _row(full, amount)
        for f in FEATURE_NAMES:
            assert emitted[f] == pytest.approx(ref[f], rel=1e-6, abs=1e-6), (
                f"Fuga en '{f}' (pos={pos}): prefijo={emitted[f]} full={ref[f]}"
            )


# ───────────────────────────── PARIDAD batch ≡ online ─────────────────────────────
def test_parity_online_replay_equals_batch():
    """El serving online (FeatureState + features_from_state) debe reproducir
    exactamente la salida del batch. Misma definición, mismo resultado."""
    df = _stream().sort_values("step", kind="stable").reset_index(drop=True)
    batch = build_features(df, CFG)

    # Replay manual estilo serving.
    state = FeatureState(CFG)
    online_rows = []
    for _, tx in df.iterrows():
        ds = state.dest_state(tx["nameDest"])
        opc = state.orig_prior_count(tx["nameOrig"])
        if tx["type"] in CFG.scope_types:
            feats = features_from_state(
                int(tx["step"]), tx["type"], float(tx["amount"]), ds, opc, CFG
            )
            online_rows.append(feats)
        state.update(tx["nameOrig"], tx["nameDest"], int(tx["step"]), float(tx["amount"]))

    online = pd.DataFrame(online_rows, columns=FEATURE_NAMES).astype("float32")
    pd.testing.assert_frame_equal(
        batch[FEATURE_NAMES].reset_index(drop=True), online.reset_index(drop=True)
    )
