from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Baixa preços históricos de um ativo via yfinance.

    Args:
        symbol: Ticker do ativo (ex: 'DIS', 'PETR4.SA').
        start_date: Data inicial no formato 'YYYY-MM-DD'.
        end_date: Data final no formato 'YYYY-MM-DD'.
        output_path: Se fornecido, salva o resultado em parquet.

    Returns:
        DataFrame com colunas Open, High, Low, Close, Volume.

    Raises:
        ValueError: Se o download retornar DataFrame vazio.
    """
    logger.info("Baixando %s de %s a %s", symbol, start_date, end_date)
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    if df.empty:
        raise ValueError(f"Nenhum dado retornado para {symbol} no intervalo informado.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    logger.info("Baixadas %d linhas", len(df))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        logger.info("Dados salvos em %s", output_path)

    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Download de preços para o pipeline.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    download_prices(args.symbol, args.start_date, args.end_date, args.output)
