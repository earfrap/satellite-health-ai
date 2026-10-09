"""Tests for the M4 Autoencoder detector."""

import numpy as np
import pandas as pd
import pytest

from satellite_health.models.autoencoder import AutoencoderDetector


def make_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(128, 6)),
        columns=[f"channel_{i}" for i in range(41, 47)],
    )


def test_autoencoder_training_and_scores():
    X = make_data()
    detector = AutoencoderDetector(epochs=3, batch_size=32).fit(X)
    scores = detector.score_samples(X)

    assert isinstance(scores, pd.Series)
    assert scores.index.equals(X.index)
    assert len(scores) == len(X)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    assert len(detector.loss_history_) == 3


def test_autoencoder_reproducibility():
    X = make_data()

    first = AutoencoderDetector(epochs=3, batch_size=32).fit(X)
    second = AutoencoderDetector(epochs=3, batch_size=32).fit(X)

    np.testing.assert_allclose(
        first.score_samples(X),
        second.score_samples(X),
        rtol=1e-6,
        atol=1e-6,
    )


def test_autoencoder_requires_training():
    with pytest.raises(RuntimeError):
        AutoencoderDetector().score_samples(make_data())
