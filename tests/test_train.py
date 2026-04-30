from __future__ import annotations

from src.models.train import build_model, get_git_sha, hash_dataframe, set_seeds


def test_set_seeds_is_deterministic():
    """Mesma seed deve gerar mesmas sequencias."""
    import numpy as np

    set_seeds(42)
    a = np.random.rand(10)
    set_seeds(42)
    b = np.random.rand(10)
    assert (a == b).all()


def test_build_model_has_correct_input_shape():
    model = build_model(
        input_shape=(60, 1),
        lstm_units=[32, 32],
        dropout=0.2,
        optimizer="adam",
        loss="mse",
    )
    # Layer 0 eh InputLayer ou primeira LSTM, dependendo da versao
    assert model.input_shape == (None, 60, 1)
    assert model.output_shape == (None, 1)


def test_build_model_compiles():
    model = build_model(
        input_shape=(30, 1),
        lstm_units=[16],
        dropout=0.1,
        optimizer="adam",
        loss="mse",
    )
    assert model.optimizer is not None


def test_hash_dataframe_is_deterministic(sample_prices_df):
    h1 = hash_dataframe(sample_prices_df)
    h2 = hash_dataframe(sample_prices_df)
    assert h1 == h2
    assert len(h1) == 12


def test_hash_dataframe_changes_with_data(sample_prices_df):
    h1 = hash_dataframe(sample_prices_df)
    modified = sample_prices_df.copy()
    modified.loc[0, "Close"] = 999.99
    h2 = hash_dataframe(modified)
    assert h1 != h2


def test_get_git_sha_returns_string():
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0
