from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.monitoring.drift import detect_drift

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
TIME_STEPS = int(os.getenv("TIME_STEPS", "50"))
FUTURE_DAYS = int(os.getenv("FUTURE_DAYS", "10"))
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "lstm-price-forecaster")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

BASE_DIR = Path(__file__).resolve().parent
FALLBACK_MODEL_PATH = BASE_DIR.parent.parent / "artifacts" / "modelo_lstm.keras"
FALLBACK_SCALER_PATH = BASE_DIR.parent.parent / "artifacts" / "scaler.pkl"
FALLBACK_REFERENCE_PATH = BASE_DIR.parent.parent / "artifacts" / "reference_prices.parquet"


# ---------------------------------------------------------------------------
# Metricas Prometheus customizadas
# ---------------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "lstm_predictions_total",
    "Total de predicoes realizadas",
    ["endpoint", "status"],
)
PREDICTION_LATENCY = Histogram(
    "lstm_prediction_latency_seconds",
    "Latencia de inferencia do modelo LSTM",
    ["endpoint"],
)
PREDICTION_VALUE = Histogram(
    "lstm_prediction_value",
    "Distribuicao dos valores preditos - usado para drift",
    buckets=[0, 25, 50, 75, 100, 150, 200, 300, 500, 1000],
)


# ---------------------------------------------------------------------------
# Carga do modelo
# ---------------------------------------------------------------------------
model: Any = None
scaler: Any = None
model_version: str = "unknown"


def load_model_from_mlflow() -> tuple[Any, Any, str]:
    """Carrega modelo do MLflow Model Registry.

    Tenta primeiro por alias (recomendado no MLflow 3.x), depois cai
    para stage (compatibilidade com MLflow 2.x). Se o scaler nao estiver
    disponivel como artefato do run, usa o scaler local como fallback.
    """
    import mlflow
    import mlflow.tensorflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    alias = os.getenv("MLFLOW_MODEL_ALIAS", "production").lower()
    mv = None
    label = "unknown"
    model_uri = None

    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
        label = f"v{mv.version} (@{alias})"
        model_uri = f"models:/{MODEL_NAME}@{alias}"
    except Exception as exc:
        logger.info("Alias '%s' nao encontrado (%s), tentando stage", alias, exc)
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise RuntimeError(
                f"Nenhuma versao com alias='{alias}' nem stage='{MODEL_STAGE}' para {MODEL_NAME}"
            ) from exc
        mv = versions[0]
        label = f"v{mv.version} (stage={MODEL_STAGE})"
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

    loaded_model = mlflow.tensorflow.load_model(model_uri)

    loaded_scaler = None
    try:
        scaler_local_path = client.download_artifacts(str(mv.run_id), "scaler.pkl")
        loaded_scaler = joblib.load(scaler_local_path)
        logger.info("Scaler carregado do MLflow run %s", mv.run_id)
    except Exception as exc:
        logger.warning("Scaler nao disponivel no MLflow run (%s), usando local", exc)
        if FALLBACK_SCALER_PATH.exists():
            loaded_scaler = joblib.load(FALLBACK_SCALER_PATH)
            logger.info("Scaler carregado do disco: %s", FALLBACK_SCALER_PATH)
        else:
            raise RuntimeError(
                f"Scaler nao encontrado nem no MLflow nem em {FALLBACK_SCALER_PATH}"
            ) from exc

    return loaded_model, loaded_scaler, label


def load_model_from_disk() -> tuple[Any, Any, str]:
    """Fallback: carrega do disco local."""
    if not FALLBACK_MODEL_PATH.exists() or not FALLBACK_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Artefatos nao encontrados em {FALLBACK_MODEL_PATH.parent}. "
            "Rode `python -m src.models.train` primeiro."
        )
    loaded_model = keras.models.load_model(FALLBACK_MODEL_PATH, compile=False)
    loaded_scaler = joblib.load(FALLBACK_SCALER_PATH)
    return loaded_model, loaded_scaler, "local-fallback"


def _load_resources() -> None:
    """Tenta carregar do MLflow; se falhar, usa fallback local."""
    global model, scaler, model_version
    try:
        model, scaler, model_version = load_model_from_mlflow()
        logger.info("Modelo carregado do MLflow: %s", model_version)
    except Exception as exc:
        logger.warning("Falha ao carregar do MLflow (%s). Tentando fallback local.", exc)
        try:
            model, scaler, model_version = load_model_from_disk()
            logger.info("Modelo carregado do disco: %s", model_version)
        except Exception as exc2:
            logger.error("Falha critica ao carregar modelo: %s", exc2)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle do app: carrega modelo no startup, libera no shutdown."""
    logger.info("Iniciando aplicacao - carregando modelo...")
    _load_resources()
    yield
    logger.info("Encerrando aplicacao")


app = FastAPI(
    title="LSTM Price Forecaster - Datathon Fase 05",
    version="0.1.0",
    description="API de previsao de precos com instrumentacao Prometheus e integracao MLflow.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# ---------------------------------------------------------------------------
# Autenticacao via API Key
# ---------------------------------------------------------------------------
API_KEYS: set[str] = set()
_initial_key = os.getenv("INITIAL_API_KEY")
if _initial_key:
    API_KEYS.add(_initial_key)


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Valida a API key vinda no header x-api-key."""
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="API Key invalida ou nao registrada")


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------
class PriceRequest(BaseModel):
    """Requisição com preços históricos."""

    prices: list[float] = Field(..., min_length=TIME_STEPS, description="Precos historicos.")


class PredictionResponse(BaseModel):
    """Resposta de previsão."""

    previsoes_10_dias: list[float]
    model_version: str
    tempo_ms: float


class DriftRequest(BaseModel):
    """Janela atual de precos para checar drift contra a referencia."""

    prices: list[float] = Field(..., min_length=100, description="Precos recentes.")


class DriftResponse(BaseModel):
    """Resultado da analise de drift."""

    psi: float
    status: str
    n_reference: int
    n_current: int
    message: str
    model_version: str


# ---------------------------------------------------------------------------
# Logica de inferencia
# ---------------------------------------------------------------------------
def predict_next_days(prices: np.ndarray, n_days: int) -> list[float]:
    """Prediz os próximos n_days usando forecasting recursivo.

    Args:
        prices: Array 1D com preços históricos (>= TIME_STEPS).
        n_days: Quantos dias à frente prever.

    Returns:
        Lista com n_days previsões.

    Raises:
        HTTPException: Se modelo não carregado ou input insuficiente.
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo nao carregado. Verifique logs do servidor.",
        )

    if len(prices) < TIME_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"Necessarios pelo menos {TIME_STEPS} precos historicos.",
        )

    prices_2d = prices.reshape(-1, 1)
    scaled = scaler.transform(prices_2d)

    input_seq = scaled[-TIME_STEPS:]
    predictions: list[float] = []

    for _ in range(n_days):
        x = input_seq.reshape(1, TIME_STEPS, 1)
        pred_scaled = model.predict(x, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(round(float(pred), 2))
        PREDICTION_VALUE.observe(float(pred))
        input_seq = np.append(input_seq[1:], pred_scaled, axis=0)

    return predictions


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def health() -> dict[str, Any]:
    """Healthcheck."""
    return {
        "status": "online",
        "modelo": "ativo" if model is not None else "INATIVO",
        "model_version": model_version,
        "keras_version": keras.__version__,
    }


@app.post("/generate_api_key")
def generate_api_key() -> dict[str, str]:
    """Gera nova API key. Em producao, mover para Redis persistente."""
    api_key = secrets.token_hex(16)
    API_KEYS.add(api_key)
    logger.info("Nova API key gerada")
    return {"api_key": api_key, "aviso": "Guarde esta chave."}


@app.post("/predict", dependencies=[Depends(verify_api_key)], response_model=PredictionResponse)
def predict_json(data: PriceRequest) -> PredictionResponse:
    """Predição a partir de JSON."""
    start = time.time()
    endpoint = "predict_json"
    try:
        prices = np.array(data.prices, dtype=float)
        with PREDICTION_LATENCY.labels(endpoint=endpoint).time():
            preds = predict_next_days(prices, FUTURE_DAYS)
        PREDICTION_COUNTER.labels(endpoint=endpoint, status="success").inc()
        return PredictionResponse(
            previsoes_10_dias=preds,
            model_version=model_version,
            tempo_ms=round((time.time() - start) * 1000, 2),
        )
    except HTTPException:
        PREDICTION_COUNTER.labels(endpoint=endpoint, status="error").inc()
        raise
    except Exception as exc:
        PREDICTION_COUNTER.labels(endpoint=endpoint, status="error").inc()
        logger.exception("Erro em /predict")
        raise HTTPException(500, str(exc)) from exc


@app.post(
    "/predict_csv",
    dependencies=[Depends(verify_api_key)],
    response_model=PredictionResponse,
)
def predict_csv(
    file: UploadFile = File(...),
    column_name: str | None = None,
) -> PredictionResponse:
    """Predição a partir de arquivo CSV."""
    start = time.time()
    endpoint = "predict_csv"

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo CSV valido")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(400, "Erro ao ler CSV.") from exc

    if df.empty:
        raise HTTPException(400, "CSV vazio")

    col = column_name or df.columns[0]
    if col not in df.columns:
        raise HTTPException(400, f"Coluna '{col}' nao encontrada. Disponiveis: {list(df.columns)}")

    try:
        prices = df[col].dropna().values.astype(float)
    except ValueError as exc:
        raise HTTPException(400, "Coluna contem dados nao numericos.") from exc

    try:
        with PREDICTION_LATENCY.labels(endpoint=endpoint).time():
            preds = predict_next_days(prices, FUTURE_DAYS)
        PREDICTION_COUNTER.labels(endpoint=endpoint, status="success").inc()
        return PredictionResponse(
            previsoes_10_dias=preds,
            model_version=model_version,
            tempo_ms=round((time.time() - start) * 1000, 2),
        )
    except HTTPException:
        PREDICTION_COUNTER.labels(endpoint=endpoint, status="error").inc()
        raise


@app.post(
    "/monitoring/drift",
    dependencies=[Depends(verify_api_key)],
    response_model=DriftResponse,
)
def check_drift(data: DriftRequest) -> DriftResponse:
    """Calcula PSI entre uma janela atual de precos e a referencia de treino.

    Resolve GAP 06: endpoint operacional para detectar drift sob demanda.
    Thresholds: PSI > 0.10 = warning, PSI > 0.20 = trigger de retreino.
    """
    if not FALLBACK_REFERENCE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Referencia nao encontrada em {FALLBACK_REFERENCE_PATH}. "
                "Rode `python -m src.models.train` primeiro."
            ),
        )

    reference_df = pd.read_parquet(FALLBACK_REFERENCE_PATH)
    reference = reference_df.iloc[:, 0].values.astype(float)
    current = np.array(data.prices, dtype=float)

    report = detect_drift(reference, current)
    return DriftResponse(
        psi=report.psi,
        status=report.status,
        n_reference=report.n_reference,
        n_current=report.n_current,
        message=report.message,
        model_version=model_version,
    )


@app.post(
    "/monitoring/drift_csv",
    dependencies=[Depends(verify_api_key)],
    response_model=DriftResponse,
)
def check_drift_csv(
    file: UploadFile = File(...),
    column_name: str | None = None,
) -> DriftResponse:
    """Calcula PSI a partir de um arquivo CSV de precos atuais.

    Versao do /monitoring/drift que aceita CSV em vez de JSON. Util para
    pipelines em batch que ja exportam dados em arquivo.

    Args:
        file: CSV com pelo menos uma coluna numerica de precos.
        column_name: Nome da coluna a usar. Se None, usa a primeira coluna.

    Returns:
        DriftResponse com PSI, status e mensagem.
    """
    if not FALLBACK_REFERENCE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Referencia nao encontrada em {FALLBACK_REFERENCE_PATH}. "
                "Rode `python -m src.models.train` primeiro."
            ),
        )

    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Envie um arquivo CSV valido")

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(400, "Erro ao ler CSV.") from exc

    if df.empty:
        raise HTTPException(400, "CSV vazio")

    col = column_name or df.columns[0]
    if col not in df.columns:
        raise HTTPException(
            400,
            f"Coluna '{col}' nao encontrada. Disponiveis: {list(df.columns)}",
        )

    try:
        current = df[col].dropna().values.astype(float)
    except ValueError as exc:
        raise HTTPException(400, "Coluna contem dados nao numericos.") from exc

    if len(current) < 100:
        raise HTTPException(
            400,
            f"CSV precisa de pelo menos 100 valores. Recebido: {len(current)}",
        )

    reference_df = pd.read_parquet(FALLBACK_REFERENCE_PATH)
    reference = reference_df.iloc[:, 0].values.astype(float)

    report = detect_drift(reference, current)
    return DriftResponse(
        psi=report.psi,
        status=report.status,
        n_reference=report.n_reference,
        n_current=report.n_current,
        message=report.message,
        model_version=model_version,
    )
