from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """DataFrame sintético simulando saída do yfinance.

    Passeio aleatório geométrico com 500 observações.
    """
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.normal(loc=0.0005, scale=0.02, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close * (1 + rng.normal(0, 0.005, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.01, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.01, n))),
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }
    )


@pytest.fixture
def small_series() -> np.ndarray:
    """Série numérica simples para teste de sequências."""
    return np.array([[x] for x in range(100)], dtype=float)
