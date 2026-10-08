
import numpy as np
import pandas as pd
import pytest

from satellite_health.models.robust_threshold import RobustThresholdDetector


def test_robust_score():
    X_train = pd.DataFrame({
        "channel_a": [1., 2., 3., 4., 5.],
        "channel_b": [10., 20., 30., 40., 50.],
    })

    detector = RobustThresholdDetector().fit(X_train)

    X_test = pd.DataFrame({
        "channel_a": [3., 5.],
        "channel_b": [30., 50.],
    })

    scores = detector.score_samples(X_test)

    assert scores.shape == (2, 2)
    assert np.allclose(scores.iloc[0], 0.0)
    assert np.allclose(scores.iloc[1], 2.0 / 1.4826)


def test_zero_mad_rejected():
    X = pd.DataFrame({
        "channel_a": [1., 1., 1., 1.],
    })

    with pytest.raises(ValueError, match="zero MAD"):
        RobustThresholdDetector().fit(X)


def test_score_before_fit_rejected():
    detector = RobustThresholdDetector()

    with pytest.raises(RuntimeError, match="fitted first"):
        detector.score_samples(pd.DataFrame({"a": [1.]}))


def test_channel_mismatch_rejected():
    X = pd.DataFrame({"a": [1., 2., 3.]})
    detector = RobustThresholdDetector().fit(X)

    with pytest.raises(ValueError, match="Input channels differ"):
        detector.score_samples(pd.DataFrame({"b": [1., 2., 3.]}))

