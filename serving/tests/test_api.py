"""Tests del serving: endpoints, scoring, fuera de superficie, health sin modelo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier

import serving.app as app_module
from features.config import FEATURE_NAMES
from training.train import train_and_build


def _synthetic_matrix(n_per_seg=300, seed=0) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("bundle")
    bundle = train_and_build(
        _synthetic_matrix(),
        make_model=lambda: HistGradientBoostingClassifier(max_iter=60, random_state=0),
        review_budget=0.1, block_budget=0.02,
    )
    bundle.save(str(d))
    return str(d)


@pytest.fixture
def client(bundle_dir, monkeypatch):
    monkeypatch.setenv("DUPIN_BUNDLE_URI", bundle_dir)
    with TestClient(app_module.app) as c:
        yield c


def test_health_ok_when_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["status"] == "ok"


def test_version(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "m-v1"
    assert body["feature_version"] == "feat-v1"
    assert body["n_features"] == len(FEATURE_NAMES)


def test_score_transfer_returns_full_contract(client):
    r = client.post("/v1/score", json={
        "step": 500, "type": "TRANSFER", "amount": 500000.0,
        "nameOrig": "C1", "nameDest": "Cdest",
    })
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in ("approve", "review", "block")
    assert body["scorable"] is True
    assert body["latency_ms"] >= 0.0
    assert isinstance(body["reasons"], list)


def test_out_of_surface_is_approved(client):
    r = client.post("/v1/score", json={
        "step": 500, "type": "PAYMENT", "amount": 100.0,
        "nameOrig": "C1", "nameDest": "M2",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["scorable"] is False
    assert body["decision"] == "approve"


def test_missing_field_is_422(client):
    r = client.post("/v1/score", json={"step": 1, "type": "TRANSFER", "amount": 10.0})
    assert r.status_code == 422


def test_state_accumulates_across_calls(client):
    before = client.get("/health").json()["scored_count"]
    client.post("/v1/score", json={"step": 1, "type": "TRANSFER", "amount": 10.0, "nameOrig": "A", "nameDest": "B"})
    client.post("/v1/score", json={"step": 2, "type": "TRANSFER", "amount": 20.0, "nameOrig": "C", "nameDest": "B"})
    after = client.get("/health").json()["scored_count"]
    assert after == before + 2


def test_demo_feed_returns_transactions(client):
    r = client.get("/v1/demo-feed?limit=50")
    assert r.status_code == 200
    feed = r.json()
    assert isinstance(feed, list) and len(feed) == 50
    for k in ("step", "type", "amount", "nameOrig", "nameDest"):
        assert k in feed[0]


def test_dashboard_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Dupin" in r.text


def test_health_degraded_without_model(monkeypatch):
    monkeypatch.delenv("DUPIN_BUNDLE_URI", raising=False)
    with TestClient(app_module.app) as c:
        h = c.get("/health")
        assert h.status_code == 200
        assert h.json()["model_loaded"] is False
        assert c.get("/version").status_code == 503
        assert c.post("/v1/score", json={
            "step": 1, "type": "TRANSFER", "amount": 10.0, "nameOrig": "A", "nameDest": "B",
        }).status_code == 503
