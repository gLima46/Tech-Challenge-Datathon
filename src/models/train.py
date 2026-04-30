from __future__ import annotations

import hashlib
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.features.feature_engineering import prepare_training_data, validate_prices
from src.features.ingestion import download_prices

logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Fixa seeds para reprodutibilidade."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(
    input_shape: tuple[int, int],
    lstm_units: list[int],
    dropout: float,
    optimizer: str,
    loss: str,
) -> Sequential:
    """Constrói o modelo LSTM.

    Args:
        input_shape: (time_steps, n_features).
        lstm_units: Lista de units por camada LSTM.
        dropout: Taxa de dropout.
        optimizer: Nome do optimizer.
        loss: Nome da função de perda.

    Returns:
        Modelo Keras compilado.
    """
    layers: list[Any] = []
    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        if i == 0:
            layers.append(LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            layers.append(LSTM(units, return_sequences=return_sequences))
        layers.append(Dropout(dropout))
    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def evaluate_model(
    model: Sequential,
    x: np.ndarray,
    y: np.ndarray,
    scaler: Any,
) -> dict[str, float]:
    """Avalia modelo retornando métricas no espaco original (nao-normalizado).

    Args:
        model: Modelo treinado.
        x: Features de validação.
        y: Target normalizado de validação.
        scaler: Scaler ajustado no treino.

    Returns:
        Dicionário com MAE, RMSE, MAPE.
    """
    y_pred = model.predict(x, verbose=0)
    y_real = scaler.inverse_transform(y)
    y_pred_real = scaler.inverse_transform(y_pred)

    mae = float(mean_absolute_error(y_real, y_pred_real))
    rmse = float(np.sqrt(mean_squared_error(y_real, y_pred_real)))
    mape = float(np.mean(np.abs((y_real - y_pred_real) / y_real)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def get_git_sha() -> str:
    """Retorna o SHA do commit atual (GAP 05: rastreabilidade de codigo)."""
    try:
        output = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        decoded: str = output.decode("utf-8")
        stripped: str = decoded.strip()
        return stripped
    except Exception:
        return "unknown"


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash determinístico de um DataFrame (versão de dados, GAP 05)."""
    return hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12]


def train(config_path: str | Path = "configs/model_config.yaml") -> str:
    """Executa o pipeline completo de treino com MLflow tracking.

    Args:
        config_path: Caminho para o YAML de configuração.

    Returns:
        run_id do experimento MLflow criado.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seeds(cfg["training"]["seed"])

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = download_prices(
        symbol=cfg["data"]["symbol"],
        start_date=cfg["data"]["start_date"],
        end_date=cfg["data"]["end_date"],
    )
    df = validate_prices(df)
    data_hash = hash_dataframe(df)

    x_train, x_val, y_train, y_val, scaler = prepare_training_data(
        df=df,
        target_column=cfg["data"]["target_column"],
        time_steps=cfg["features"]["time_steps"],
        train_split=cfg["features"]["train_split"],
    )

    with mlflow.start_run(run_name=f"lstm-{cfg['data']['symbol']}") as run:
        mlflow.log_params(
            {
                "symbol": cfg["data"]["symbol"],
                "start_date": cfg["data"]["start_date"],
                "end_date": cfg["data"]["end_date"],
                "time_steps": cfg["features"]["time_steps"],
                "train_split": cfg["features"]["train_split"],
                "lstm_units": str(cfg["model"]["lstm_units"]),
                "dropout": cfg["model"]["dropout"],
                "optimizer": cfg["model"]["optimizer"],
                "epochs": cfg["training"]["epochs"],
                "batch_size": cfg["training"]["batch_size"],
                "seed": cfg["training"]["seed"],
            }
        )

        mlflow.set_tags(
            {
                "model_name": cfg["mlflow"]["registered_model_name"],
                "model_type": "regression-timeseries-lstm",
                "framework": "tensorflow-keras",
                "owner": os.getenv("MLFLOW_OWNER", "grupo-XX"),
                "risk_level": "medium",
                "training_data_version": data_hash,
                "git_sha": get_git_sha(),
                "phase": "datathon-fase05",
            }
        )

        model = build_model(
            input_shape=(x_train.shape[1], 1),
            lstm_units=cfg["model"]["lstm_units"],
            dropout=cfg["model"]["dropout"],
            optimizer=cfg["model"]["optimizer"],
            loss=cfg["model"]["loss"],
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=cfg["training"]["early_stopping_patience"],
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg["training"]["reduce_lr_factor"],
                patience=cfg["training"]["reduce_lr_patience"],
                min_lr=cfg["training"]["reduce_lr_min"],
            ),
        ]

        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=cfg["training"]["epochs"],
            batch_size=cfg["training"]["batch_size"],
            callbacks=callbacks,
            verbose=2,
        )

        metrics = evaluate_model(model, x_val, y_val, scaler)
        mlflow.log_metrics(metrics)
        logger.info("Métricas finais: %s", metrics)

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)

        model_path = artifacts_dir / "modelo_lstm.keras"
        scaler_path = artifacts_dir / "scaler.pkl"
        model.save(model_path)

        import joblib

        joblib.dump(scaler, scaler_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))
        mlflow.tensorflow.log_model(model, artifact_path="keras_model")

        reference_window_size = 252
        reference_df = df[[cfg["data"]["target_column"]]].tail(reference_window_size)
        reference_path = artifacts_dir / "reference_prices.parquet"
        reference_df.to_parquet(reference_path)
        mlflow.log_artifact(str(reference_path))
        logger.info(
            "Referencia para drift salva em %s (%d linhas)",
            reference_path,
            len(reference_df),
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/keras_model"
        try:
            mlflow.register_model(
                model_uri=model_uri,
                name=cfg["mlflow"]["registered_model_name"],
            )
            logger.info("Modelo registrado no Model Registry")
        except Exception as exc:
            logger.warning("Nao foi possivel registrar no Model Registry: %s", exc)

        logger.info("Run concluido: %s", run_id)
        run_id_str: str = str(run_id)
        return run_id_str


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    train()
