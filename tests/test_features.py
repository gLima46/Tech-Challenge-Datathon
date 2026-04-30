from __future__ import annotations

import numpy as np
import pandera as pa
import pytest
from src.features.feature_engineering import (
    create_sequences,
    fit_scaler,
    prepare_training_data,
    validate_prices,
)


def test_validate_prices_accepts_valid_df(sample_prices_df):
    result = validate_prices(sample_prices_df)
    assert len(result) == len(sample_prices_df)


def test_validate_prices_rejects_negative_close(sample_prices_df):
    sample_prices_df.loc[0, "Close"] = -1.0
    with pytest.raises(pa.errors.SchemaError):
        validate_prices(sample_prices_df)


def test_create_sequences_shapes(small_series):
    time_steps = 10
    X, y = create_sequences(small_series, time_steps)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90, 1)


def test_create_sequences_values(small_series):
    X, y = create_sequences(small_series, time_steps=3)
    # primeira janela = [0, 1, 2], target = 3
    np.testing.assert_array_equal(X[0], np.array([[0], [1], [2]]))
    np.testing.assert_array_equal(y[0], np.array([3]))


def test_create_sequences_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="Dados insuficientes"):
        create_sequences(np.array([[1.0], [2.0]]), time_steps=10)


def test_fit_scaler_range(sample_prices_df):
    prices = sample_prices_df[["Close"]].values
    scaler = fit_scaler(prices)
    scaled = scaler.transform(prices)
    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0


def test_prepare_training_data_split(sample_prices_df):
    x_train, x_val, y_train, y_val, scaler = prepare_training_data(
        df=sample_prices_df,
        target_column="Close",
        time_steps=20,
        train_split=0.7,
    )
    total = len(x_train) + len(x_val)
    assert total == len(sample_prices_df) - 20
    assert x_train.shape[1] == 20
    assert x_train.shape[2] == 1
    # split deve respeitar a fração
    assert abs(len(x_train) / total - 0.7) < 0.05
