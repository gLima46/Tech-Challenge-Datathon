from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from src.features.ingestion import download_prices


@pytest.fixture
def fake_yf_response() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.5, 101.5, 102.5],
            "Volume": [1000000, 1100000, 1200000],
        }
    )


@patch("src.features.ingestion.yf.download")
def test_download_prices_returns_dataframe(mock_download, fake_yf_response):
    mock_download.return_value = fake_yf_response
    result = download_prices("DIS", "2020-01-01", "2020-01-04")
    assert len(result) == 3
    assert "Close" in result.columns


@patch("src.features.ingestion.yf.download")
def test_download_prices_raises_on_empty(mock_download):
    mock_download.return_value = pd.DataFrame()
    with pytest.raises(ValueError, match="Nenhum dado"):
        download_prices("INVALID", "2020-01-01", "2020-01-04")


@patch("src.features.ingestion.yf.download")
def test_download_prices_writes_parquet(mock_download, fake_yf_response, tmp_path):
    mock_download.return_value = fake_yf_response
    output = tmp_path / "out.parquet"
    download_prices("DIS", "2020-01-01", "2020-01-04", output_path=output)
    assert output.exists()
    loaded = pd.read_parquet(output)
    assert len(loaded) == 3


@patch("src.features.ingestion.yf.download")
def test_download_prices_handles_multiindex_columns(mock_download):
    multi = pd.DataFrame(
        {
            ("Close", "DIS"): [100.0, 101.0],
            ("Volume", "DIS"): [1000.0, 1100.0],
        }
    )
    multi.columns = pd.MultiIndex.from_tuples(multi.columns)
    mock_download.return_value = multi
    result = download_prices("DIS", "2020-01-01", "2020-01-03")
    assert "Close" in result.columns
