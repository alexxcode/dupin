"""Tests de métricas: presupuesto de revisión, punto de operación, PR-AUC, costo."""
from __future__ import annotations

import numpy as np
import pytest

from evaluation import metrics


def test_threshold_for_review_budget_limits_flagged():
    rng = np.random.default_rng(0)
    scores = rng.random(10000)
    thr = metrics.threshold_for_review_budget(scores, budget=0.01)
    review_rate = (scores >= thr).mean()
    assert review_rate == pytest.approx(0.01, abs=0.005)


def test_operating_point_confusion_counts():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    op = metrics.operating_point(y, scores, threshold=0.5)
    assert op["tp"] == 2 and op["fp"] == 0
    assert op["fn"] == 0 and op["tn"] == 2
    assert op["recall"] == 1.0
    assert op["precision"] == 1.0
    assert op["review_rate"] == 0.5


def test_average_precision_perfect_ranking():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert metrics.average_precision(y, scores) == pytest.approx(1.0)


def test_recall_drops_when_threshold_too_high():
    y = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.65]
    op = metrics.operating_point(y, scores, threshold=0.9)
    assert op["tp"] == 0 and op["recall"] == 0.0


def test_recall_at_review_rate_hits_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(10000) < 0.05).astype(int)
    scores = rng.random(10000) * 0.5 + y * 0.5  # señal
    op = metrics.recall_at_review_rate(y, scores, review_rate=0.02)
    assert op["review_rate"] == pytest.approx(0.02, abs=0.01)


def test_recall_review_curve_monotonic_recall():
    rng = np.random.default_rng(1)
    y = (rng.random(5000) < 0.1).astype(int)
    scores = rng.random(5000) * 0.5 + y * 0.5
    curve = metrics.recall_review_curve(y, scores, rates=[0.01, 0.05, 0.1, 0.2])
    recalls = [op["recall"] for op in curve]
    # Más presupuesto de revisión nunca reduce el recall.
    assert all(b >= a - 1e-9 for a, b in zip(recalls, recalls[1:]))


def test_cost_curve_prefers_catching_fraud_when_fn_expensive():
    # Señal perfecta: el umbral óptimo debe atrapar todo el fraude (recall alto).
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.6, 1.0, 10)])
    cc = metrics.cost_curve(y, scores, fn_cost=100.0, fp_cost=1.0)
    op = metrics.operating_point(y, scores, cc["best_threshold"])
    assert op["recall"] >= 0.9
