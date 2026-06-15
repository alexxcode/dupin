"""build_features — única fuente de verdad transacción → vector.

`features_from_state` es LA definición de cómo una transacción se convierte en un
vector de features, dado el estado previo de sus entidades. La usan por igual:
  - el batch offline (`build_features`), que reproduce el stream en orden temporal;
  - el serving online (Fase 5), que mantiene el mismo `FeatureState`.

Por eso es físicamente imposible que entrenamiento y producción difieran
(paridad por construcción · Invariante 1). Ninguna feature usa información
posterior al instante evaluado (sin fuga temporal · Invariante 2).
"""
from __future__ import annotations

import math

import pandas as pd

from features.config import FEATURE_NAMES, META_COLUMNS, FeatureConfig
from features.entity_state import DestState, FeatureState


def features_from_state(
    step: int,
    tx_type: str,
    amount: float,
    dest_state: DestState | None,
    orig_prior_count: int,
    config: FeatureConfig,
) -> dict[str, float]:
    """Vector de features para una transacción dado el estado PREVIO.

    `dest_state` refleja solo transacciones anteriores (None si el receptor es
    nuevo). No muta nada. Devuelve un dict con las claves de FEATURE_NAMES.
    """
    # ── Point features ──
    log_amount = math.log1p(amount)
    type_transfer = 1.0 if tx_type == "TRANSFER" else 0.0
    hour = float(step % 24)

    # ── Comportamiento del receptor ──
    if dest_state is None or dest_state.count == 0:
        dest_prior_count = 0.0
        dest_is_new = 1.0
        dest_amount_ratio = config.amount_ratio_default
        dest_amount_z = config.amount_z_default
        dest_recency = config.recency_default
        dest_cnt_24h = dest_cnt_168h = 0.0
        dest_amt_sum_24h = dest_amt_sum_168h = 0.0
    else:
        mean = dest_state.mean()
        std = dest_state.std()
        dest_prior_count = float(dest_state.count)
        dest_is_new = 0.0
        dest_amount_ratio = amount / mean if mean > 0.0 else config.amount_ratio_default
        dest_amount_z = (amount - mean) / std if std > 0.0 else config.amount_z_default
        dest_recency = float(dest_state.recency(step))
        w24 = config.windows["24h"]
        w168 = config.windows["168h"]
        dest_cnt_24h = float(dest_state.window_count(step, w24))
        dest_cnt_168h = float(dest_state.window_count(step, w168))
        dest_amt_sum_24h = dest_state.window_sum(step, w24)
        dest_amt_sum_168h = dest_state.window_sum(step, w168)

    return {
        "log_amount": log_amount,
        "type_transfer": type_transfer,
        "hour": hour,
        "dest_prior_count": dest_prior_count,
        "dest_is_new": dest_is_new,
        "dest_amount_ratio": dest_amount_ratio,
        "dest_amount_z": dest_amount_z,
        "dest_recency": dest_recency,
        "dest_cnt_24h": dest_cnt_24h,
        "dest_cnt_168h": dest_cnt_168h,
        "dest_amt_sum_24h": dest_amt_sum_24h,
        "dest_amt_sum_168h": dest_amt_sum_168h,
        "orig_prior_count": float(orig_prior_count),
    }


def build_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    with_meta: bool = True,
) -> pd.DataFrame:
    """Construye la matriz de features reproduciendo el stream en orden temporal.

    La historia por entidad se acumula desde TODAS las filas de `df`; solo se
    EMITEN las filas cuyo tipo está en `config.scope_types`. Para cada transacción
    se computan las features con el estado previo y LUEGO se incorpora al estado
    (regla causal estricta).

    `df` debe contener: step, type, amount, nameOrig, nameDest [, isFraud].
    Devuelve un DataFrame con FEATURE_NAMES (float32) + META_COLUMNS (si with_meta).
    """
    # Orden temporal estable: dentro del mismo step se conserva el orden original
    # del CSV (orden de emisión del simulador ≈ orden de llegada).
    ordered = df.sort_values(config.time_col, kind="stable")

    has_label = config.label_col in ordered.columns
    state = FeatureState(config)
    scope = set(config.scope_types)

    # Acumulamos tuplas (no dicts) por memoria: ~millones de filas en el batch.
    feat_rows: list[tuple] = []
    meta_rows: list[tuple] = []

    # itertuples es el camino rápido; accedemos por posición fija de columna.
    cols = [
        config.time_col, config.type_col, config.amount_col,
        config.orig_col, config.dest_col,
    ]
    if has_label:
        cols.append(config.label_col)
    sub = ordered[cols]

    for row in sub.itertuples(index=False, name=None):
        if has_label:
            step, tx_type, amount, orig_id, dest_id, label = row
        else:
            step, tx_type, amount, orig_id, dest_id = row
            label = 0
        step = int(step)
        amount = float(amount)

        dest_state = state.dest_state(dest_id)          # estado PREVIO
        orig_pc = state.orig_prior_count(orig_id)

        if tx_type in scope:
            feats = features_from_state(
                step, tx_type, amount, dest_state, orig_pc, config
            )
            feat_rows.append(tuple(feats[name] for name in FEATURE_NAMES))
            if with_meta:
                meta_rows.append((step, tx_type, amount, orig_id, dest_id, int(label)))

        # Incorporar la tx al estado DESPUÉS de emitir (causalidad).
        state.update(orig_id, dest_id, step, amount)

    out = pd.DataFrame(feat_rows, columns=FEATURE_NAMES).astype("float32")
    if with_meta:
        meta = pd.DataFrame(meta_rows, columns=META_COLUMNS)
        out = pd.concat([meta.reset_index(drop=True), out], axis=1)
    return out


def build_state(
    df: pd.DataFrame, config: FeatureConfig, max_step: int | None = None
) -> FeatureState:
    """Reproduce el stream y devuelve el FeatureState tras procesar las tx con
    `step < max_step` (o todas si None).

    Se usa para construir el snapshot de warm-start del serving: el estado de las
    entidades tal como estaba justo ANTES del periodo de demo, para que el scoring
    en vivo vea la misma historia que la evaluación offline (no cold-start).
    """
    ordered = df.sort_values(config.time_col, kind="stable")
    state = FeatureState(config)
    cols = [config.time_col, config.type_col, config.amount_col,
            config.orig_col, config.dest_col]
    for row in ordered[cols].itertuples(index=False, name=None):
        step, _tx_type, amount, orig_id, dest_id = row
        step = int(step)
        if max_step is not None and step >= max_step:
            break  # ordenado por step asc → podemos cortar
        state.update(orig_id, dest_id, step, float(amount))
    return state
