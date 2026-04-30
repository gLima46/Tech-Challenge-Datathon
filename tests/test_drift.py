from __future__ import annotations

import numpy as np
import pytest
from src.monitoring.drift import calculate_psi, detect_drift


@pytest.fixture
def stable_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 1000)
    cur = rng.normal(100, 10, 1000)
    return ref, cur


@pytest.fixture
def drifted_distributions():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 1000)
    cur = rng.normal(150, 15, 1000)  # média e variância deslocadas
    return ref, cur


def test_psi_is_low_for_similar_distributions(stable_distributions):
    ref, cur = stable_distributions
    psi = calculate_psi(ref, cur)
    assert psi < 0.10


def test_psi_is_high_for_drifted_distributions(drifted_distributions):
    ref, cur = drifted_distributions
    psi = calculate_psi(ref, cur)
    assert psi > 0.20


def test_detect_drift_status_stable(stable_distributions):
    ref, cur = stable_distributions
    report = detect_drift(ref, cur)
    assert report.status == "stable"


def test_detect_drift_status_retrain(drifted_distributions):
    ref, cur = drifted_distributions
    report = detect_drift(ref, cur)
    assert report.status == "retrain"


def test_detect_drift_skips_when_few_samples():
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 10, 1000)
    cur = rng.normal(150, 15, 50)  # abaixo do min_samples
    report = detect_drift(ref, cur, min_samples=100)
    assert report.status == "stable"
    assert "insuficientes" in report.message.lower()


def test_drift_report_to_dict_has_required_keys(stable_distributions):
    ref, cur = stable_distributions
    report = detect_drift(ref, cur)
    d = report.to_dict()
    assert {"psi", "status", "n_reference", "n_current", "message"} <= set(d.keys())
