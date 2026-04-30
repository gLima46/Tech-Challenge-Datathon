from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Resultado de uma análise de drift."""

    psi: float
    status: str
    n_reference: int
    n_current: int
    message: str

    def to_dict(self) -> dict:
        """Serializa o report como dicionário JSON-friendly."""
        return {
            "psi": round(self.psi, 4),
            "status": self.status,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "message": self.message,
        }


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calcula o Population Stability Index entre duas distribuições.

    PSI = sum( (current% - ref%) * ln(current% / ref%) )

    Args:
        reference: Distribuição de referência (ex: treino).
        current: Distribuição observada em produção.
        n_bins: Número de bins para discretização.

    Returns:
        Valor de PSI (>= 0, quanto maior mais drift).
    """
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    eps = 1e-6
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    warning_threshold: float = 0.10,
    retrain_threshold: float = 0.20,
    min_samples: int = 100,
) -> DriftReport:
    """Executa detecção de drift e retorna um report estruturado.

    Args:
        reference: Distribuição de referência.
        current: Distribuição atual.
        warning_threshold: PSI que dispara warning.
        retrain_threshold: PSI que dispara retreino.
        min_samples: Mínimo de amostras para confiar no cálculo.

    Returns:
        DriftReport com PSI, status e mensagem.
    """
    if len(current) < min_samples:
        return DriftReport(
            psi=0.0,
            status="stable",
            n_reference=len(reference),
            n_current=len(current),
            message=f"Amostras insuficientes (n={len(current)} < {min_samples})",
        )

    psi = calculate_psi(reference, current)

    if psi >= retrain_threshold:
        status = "retrain"
        message = f"PSI={psi:.3f} >= {retrain_threshold} - TRIGGER DE RETREINO"
    elif psi >= warning_threshold:
        status = "warning"
        message = f"PSI={psi:.3f} >= {warning_threshold} - atencao ao drift"
    else:
        status = "stable"
        message = f"PSI={psi:.3f} - distribuicao estavel"

    logger.info(message)
    return DriftReport(
        psi=psi,
        status=status,
        n_reference=len(reference),
        n_current=len(current),
        message=message,
    )


def run_drift_check(
    reference_path: Path | str,
    current_path: Path | str,
    column: str = "Close",
    **kwargs,
) -> DriftReport:
    """Helper para rodar drift a partir de arquivos parquet/csv."""
    ref_df = _read_any(reference_path)
    cur_df = _read_any(current_path)
    return detect_drift(
        reference=ref_df[column].values.astype(float),
        current=cur_df[column].values.astype(float),
        **kwargs,
    )


def _read_any(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
