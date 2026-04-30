from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


PRICE_SCHEMA = DataFrameSchema(
    {
        "Close": Column(float, Check.gt(0), nullable=False),
        "Open": Column(float, Check.gt(0), nullable=False, required=False),
        "High": Column(float, Check.gt(0), nullable=False, required=False),
        "Low": Column(float, Check.gt(0), nullable=False, required=False),
        "Volume": Column(
            "int64",
            Check.ge(0),
            nullable=False,
            required=False,
            coerce=True,
        ),
    },
    strict=False,
)


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame de preços contra o schema de contrato.

    Args:
        df: DataFrame bruto vindo da ingestão.

    Returns:
        Mesmo DataFrame após validação (lança erro se invalido).
    """
    logger.info("Validando schema de %d linhas", len(df))
    return PRICE_SCHEMA.validate(df)


def create_sequences(
    data: np.ndarray,
    time_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cria janelas deslizantes X, y para treino de LSTM.

    Args:
        data: Array 2D (n_samples, 1) com preços normalizados.
        time_steps: Tamanho da janela de lookback.

    Returns:
        Tupla (X, y) onde X tem shape (n, time_steps, 1) e y tem shape (n, 1).

    Raises:
        ValueError: Se `data` tem menos linhas que `time_steps`.
    """
    if len(data) <= time_steps:
        raise ValueError(f"Dados insuficientes: {len(data)} linhas para time_steps={time_steps}")

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in range(time_steps, len(data)):
        x_list.append(data[i - time_steps : i])
        y_list.append(data[i])

    X = np.array(x_list)
    y = np.array(y_list)
    logger.info("Sequências criadas: X=%s, y=%s", X.shape, y.shape)
    return X, y


def fit_scaler(prices: np.ndarray) -> MinMaxScaler:
    """Ajusta um MinMaxScaler nos preços de treino.

    Args:
        prices: Array 2D (n, 1) com preços brutos.

    Returns:
        Scaler ajustado pronto para transform / inverse_transform.
    """
    scaler = MinMaxScaler()
    scaler.fit(prices)
    return scaler


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    time_steps: int,
    train_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepara X_train, X_val, y_train, y_val e scaler.

    Args:
        df: DataFrame de preços validado.
        target_column: Coluna a prever (ex: 'Close').
        time_steps: Janela de lookback.
        train_split: Fração de treino (ex: 0.5).

    Returns:
        (X_train, X_val, y_train, y_val, scaler).
    """
    prices = df[[target_column]].values.astype(float)
    scaler = fit_scaler(prices)
    prices_scaled = scaler.transform(prices)

    x_all, y_all = create_sequences(prices_scaled, time_steps)

    split_index = int(len(x_all) * train_split)
    x_train, x_val = x_all[:split_index], x_all[split_index:]
    y_train, y_val = y_all[:split_index], y_all[split_index:]

    logger.info(
        "Split: train=%d amostras, val=%d amostras",
        len(x_train),
        len(x_val),
    )
    return x_train, x_val, y_train, y_val, scaler
