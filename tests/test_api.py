from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ARTIFACTS_OK = (
    Path("artifacts/modelo_lstm.keras").exists() or Path("artifacts/modelo_lstm.h5").exists()
)


@pytest.fixture(scope="module")
def client():
    """TestClient como context manager - dispara lifespan (carrega modelo)."""
    # Força URI inválida pra MLflow falhar rápido e cair no fallback local
    import os

    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:1"

    from src.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "online"
    assert "model_version" in payload


def test_generate_api_key(client):
    r = client.post("/generate_api_key")
    assert r.status_code == 200
    assert "api_key" in r.json()
    assert len(r.json()["api_key"]) == 32  # 16 bytes em hex


def test_predict_requires_api_key_header(client):
    r = client.post("/predict", json={"prices": [100.0] * 60})
    assert r.status_code == 422  # header obrigatorio ausente


def test_predict_rejects_invalid_api_key(client):
    r = client.post(
        "/predict",
        json={"prices": [100.0] * 60},
        headers={"x-api-key": "fake-key-nao-existe"},
    )
    assert r.status_code == 401


def test_metrics_endpoint_exposed(client):
    """Gap 01: /metrics deve estar exposto para Prometheus scrapear."""
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    # Verifica que ha pelo menos algumas metricas conhecidas
    assert "http_request" in body or "lstm_predictions_total" in body


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="Sem artefato treinado em artifacts/")
def test_predict_happy_path(client):
    """Fluxo feliz: gera key + manda 60 precos + recebe 10 previsoes."""
    api_key = client.post("/generate_api_key").json()["api_key"]
    r = client.post(
        "/predict",
        json={"prices": [100.0 + i * 0.1 for i in range(60)]},
        headers={"x-api-key": api_key},
    )
    assert r.status_code == 200
    payload = r.json()
    assert len(payload["previsoes_10_dias"]) == 10
    assert all(isinstance(p, (int, float)) for p in payload["previsoes_10_dias"])
    assert "model_version" in payload
    assert "tempo_ms" in payload


@pytest.mark.integration
@pytest.mark.skipif(not ARTIFACTS_OK, reason="Sem artefato treinado em artifacts/")
def test_predict_rejects_insufficient_history(client):
    """Pydantic deve barrar listas menores que TIME_STEPS."""
    api_key = client.post("/generate_api_key").json()["api_key"]
    r = client.post(
        "/predict",
        json={"prices": [100.0] * 5},
        headers={"x-api-key": api_key},
    )
    assert r.status_code in (400, 422)


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("artifacts/reference_prices.parquet").exists(),
    reason="Sem referencia de drift",
)
def test_drift_endpoint_detects_distribution_shift(client):
    """Endpoint /monitoring/drift retorna 'retrain' para precos drasticamente diferentes."""
    api_key = client.post("/generate_api_key").json()["api_key"]
    # Precos artificialmente altos forcam drift
    drifted = [500.0 + i * 0.5 for i in range(150)]
    r = client.post(
        "/monitoring/drift",
        json={"prices": drifted},
        headers={"x-api-key": api_key},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] in ("warning", "retrain")
    assert payload["psi"] > 0.10


@pytest.mark.integration
@pytest.mark.skipif(
    not Path("artifacts/reference_prices.parquet").exists(),
    reason="Sem referencia de drift",
)
def test_drift_csv_endpoint_with_drifted_data(client, tmp_path):
    """Endpoint /monitoring/drift_csv retorna 'retrain' para CSV drifted."""
    import pandas as pd

    csv_path = tmp_path / "drifted.csv"
    pd.DataFrame({"Close": [500.0 + i * 0.5 for i in range(150)]}).to_csv(csv_path, index=False)

    api_key = client.post("/generate_api_key").json()["api_key"]
    with open(csv_path, "rb") as f:
        r = client.post(
            "/monitoring/drift_csv",
            files={"file": ("drifted.csv", f, "text/csv")},
            headers={"x-api-key": api_key},
        )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] in ("warning", "retrain")
    assert payload["psi"] > 0.10


@pytest.mark.skipif(
    not Path("artifacts/reference_prices.parquet").exists(),
    reason="Sem referencia de drift",
)
def test_drift_csv_rejects_non_csv(client):
    api_key = client.post("/generate_api_key").json()["api_key"]
    r = client.post(
        "/monitoring/drift_csv",
        files={"file": ("test.txt", b"foo,bar\n1,2", "text/plain")},
        headers={"x-api-key": api_key},
    )
    assert r.status_code == 400


@pytest.mark.skipif(
    not Path("artifacts/reference_prices.parquet").exists(),
    reason="Sem referencia de drift",
)
def test_drift_csv_rejects_insufficient_rows(client, tmp_path):
    """CSV com menos de 100 valores deve ser rejeitado."""
    import pandas as pd
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame({"Close": [100.0, 101.0, 102.0]}).to_csv(csv_path, index=False)

    api_key = client.post("/generate_api_key").json()["api_key"]
    with open(csv_path, "rb") as f:
        r = client.post(
            "/monitoring/drift_csv",
            files={"file": ("tiny.csv", f, "text/csv")},
            headers={"x-api-key": api_key},
        )
    assert r.status_code == 400
    assert "100" in r.json()["detail"]


def test_drift_csv_returns_503_when_no_reference(client, monkeypatch, tmp_path):
    """Quando referencia nao existe, endpoint deve retornar 503."""
    from src.serving import app as app_module

    # Aponta para path que nao existe
    monkeypatch.setattr(
        app_module,
        "FALLBACK_REFERENCE_PATH",
        tmp_path / "nao_existe.parquet",
    )

    api_key = client.post("/generate_api_key").json()["api_key"]
    fake_csv = b"Close\n" + b"\n".join(str(100 + i).encode() for i in range(150))
    r = client.post(
        "/monitoring/drift_csv",
        files={"file": ("test.csv", fake_csv, "text/csv")},
        headers={"x-api-key": api_key},
    )
    assert r.status_code == 503
    assert "nao encontrada" in r.json()["detail"].lower()
