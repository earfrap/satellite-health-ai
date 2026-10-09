import numpy as np
import pandas as pd
import pytest

from satellite_health.models.isolation_forest import IsolationForestDetector


def test_isolation_forest():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(300, 6)),
        columns=[f"channel_{i}" for i in range(41, 47)],
    )

    detector = IsolationForestDetector().fit(X)
    scores = detector.score_samples(X)

    assert isinstance(scores, pd.Series)
    assert len(scores) == len(X)
    assert scores.index.equals(X.index)
    assert np.isfinite(scores).all()

    repeated = IsolationForestDetector().fit(X)
    np.testing.assert_allclose(
        scores.to_numpy(),
        repeated.score_samples(X).to_numpy(),
    )


def test_isolation_forest_requires_training():
    X = pd.DataFrame(
        np.ones((3, 6)),
        columns=[f"channel_{i}" for i in range(41, 47)],
    )

    with pytest.raises(RuntimeError):
        IsolationForestDetector().score_samples(X)
