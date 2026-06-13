"""Particiones temporales — la vara de medir honesta (Fase 3).

El resultado de cabecera SIEMPRE se obtiene con split temporal: train sobre el
pasado, val/test sobre el futuro, sin solapamiento (Invariante 3). El split
aleatorio existe solo como diagnóstico de optimismo, con los MISMOS tamaños de
segmento para que la comparación sea apples-to-apples.

Corte (Fase 1): step = 1 hora; 744 steps = 30 días. Volumen front-loaded, así que
el corte día-20 (step 480) deja un test de cola más denso en fraude — eso es el
"futuro distinto del pasado" y es parte honesta de la evaluación, no un defecto.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    # train = step < train_max_step
    # val   = train_max_step <= step < val_max_step
    # test  = step >= val_max_step
    train_max_step: int = 432   # ~18 días
    val_max_step: int = 480     # corte día 20; val = últimos ~2 días pre-corte
    time_col: str = "step"


DEFAULT_SPLIT = TemporalSplit()


def assign_temporal(df: pd.DataFrame, split: TemporalSplit = DEFAULT_SPLIT) -> pd.Series:
    """Etiqueta cada fila como 'train' / 'val' / 'test' según su step."""
    s = df[split.time_col].to_numpy()
    out = np.where(
        s < split.train_max_step, "train",
        np.where(s < split.val_max_step, "val", "test"),
    )
    return pd.Series(out, index=df.index, name="split")


def random_assign_like(
    temporal_assign: pd.Series, seed: int = 42
) -> pd.Series:
    """Asignación ALEATORIA con los mismos tamaños de segmento que la temporal.

    Mantener los tamaños constantes aísla el efecto del orden (aleatorio vs
    temporal): el gap resultante es optimismo puro por fuga temporal, no por
    tamaños distintos.
    """
    counts = temporal_assign.value_counts()
    labels = np.array(
        ["train"] * int(counts.get("train", 0))
        + ["val"] * int(counts.get("val", 0))
        + ["test"] * int(counts.get("test", 0))
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return pd.Series(labels, index=temporal_assign.index, name="split_random")


def split_report(
    df: pd.DataFrame,
    split: TemporalSplit = DEFAULT_SPLIT,
    label_col: str = "isFraud",
) -> dict:
    """Conteos por segmento: filas, fraude, tasa, rango de steps. Para verificar
    que hay fraude suficiente a ambos lados antes de confiar en el número."""
    a = assign_temporal(df, split)
    rep: dict[str, dict] = {}
    for seg in ("train", "val", "test"):
        part = df[a == seg]
        rep[seg] = {
            "rows": int(len(part)),
            "fraud": int(part[label_col].sum()) if len(part) else 0,
            "fraud_rate": float(part[label_col].mean()) if len(part) else 0.0,
            "step_min": int(part[split.time_col].min()) if len(part) else None,
            "step_max": int(part[split.time_col].max()) if len(part) else None,
        }
    return rep
