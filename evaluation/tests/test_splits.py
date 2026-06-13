"""Tests del split temporal: sin solapamiento, cobertura total, tamaños espejo."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.splits import (
    DEFAULT_SPLIT,
    TemporalSplit,
    assign_temporal,
    random_assign_like,
    split_report,
)


def _df(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": rng.integers(0, 744, size=n),
        "isFraud": rng.integers(0, 2, size=n),
    })


def test_temporal_boundaries_no_overlap():
    df = _df()
    split = TemporalSplit(train_max_step=432, val_max_step=480)
    a = assign_temporal(df, split)
    assert (df.loc[a == "train", "step"] < 432).all()
    assert (df.loc[a == "val", "step"] >= 432).all()
    assert (df.loc[a == "val", "step"] < 480).all()
    assert (df.loc[a == "test", "step"] >= 480).all()


def test_every_row_assigned_once():
    df = _df()
    a = assign_temporal(df, DEFAULT_SPLIT)
    assert set(a.unique()) <= {"train", "val", "test"}
    assert len(a) == len(df)
    assert a.isna().sum() == 0


def test_random_split_mirrors_temporal_sizes():
    df = _df()
    a = assign_temporal(df, DEFAULT_SPLIT)
    r = random_assign_like(a, seed=1)
    assert a.value_counts().to_dict() == r.value_counts().to_dict()
    # Pero las asignaciones difieren (no es la misma partición).
    assert (a.to_numpy() != r.to_numpy()).any()


def test_split_report_counts():
    df = _df()
    rep = split_report(df, DEFAULT_SPLIT)
    total = sum(rep[s]["rows"] for s in ("train", "val", "test"))
    assert total == len(df)
    assert sum(rep[s]["fraud"] for s in ("train", "val", "test")) == int(df["isFraud"].sum())
