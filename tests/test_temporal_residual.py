"""Tests for causal temporal reference functions."""

import pandas as pd

from satellite_health.models.temporal_residual import (
    causal_rolling_median,
    TemporalResidualDetector,
)


def test_causal_median_excludes_current_observation():
    timestamps = pd.to_datetime([
        "2000-01-01 10:00:00",
        "2000-01-01 10:05:00",
        "2000-01-01 10:10:00",
        "2000-01-01 10:15:00",
    ])

    X = pd.DataFrame(
        {"channel_41": [10.0, 12.0, 14.0, 1000.0]},
        index=timestamps,
    )

    reference = causal_rolling_median(
        X,
        window="30min",
        min_periods=3,
    )

    assert pd.isna(reference.iloc[0, 0])
    assert pd.isna(reference.iloc[1, 0])
    assert pd.isna(reference.iloc[2, 0])
    assert reference.iloc[3, 0] == 12.0


def test_causal_median_uses_physical_time_window():
    timestamps = pd.to_datetime([
        "2000-01-01 10:00:00",
        "2000-01-01 10:15:00",
        "2000-01-01 10:16:00",
        "2000-01-01 10:39:00",
        "2000-01-01 10:40:00",
    ])

    X = pd.DataFrame(
        {"channel_41": [100.0, 10.0, 20.0, 30.0, 999.0]},
        index=timestamps,
    )

    reference = causal_rolling_median(
        X,
        window="30min",
        min_periods=3,
    )

    assert reference.iloc[-1, 0] == 20.0



def test_causal_median_rejects_duplicate_timestamps():
    import pytest

    timestamps = pd.to_datetime([
        "2000-01-01 10:00:00",
        "2000-01-01 10:00:00",
        "2000-01-01 10:05:00",
    ])

    X = pd.DataFrame(
        {"channel_41": [10.0, 12.0, 14.0]},
        index=timestamps,
    )

    with pytest.raises(ValueError, match="Duplicate timestamps"):
        causal_rolling_median(X)


def test_temporal_detector_scores_large_residual():
    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })

    detector = TemporalResidualDetector().fit(training)

    evaluation = pd.DataFrame({
        "channel_41": [0.0, 1.0, 100.0, float("nan")],
    })

    scores = detector.score_samples(evaluation)

    assert scores.iloc[0, 0] == 0.0
    assert scores.iloc[2, 0] > scores.iloc[1, 0]
    assert scores.iloc[2, 0] > 10.0
    assert pd.isna(scores.iloc[3, 0])


def test_temporal_detector_rejects_infinite_training_values():
    import pytest

    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, float("inf")],
    })

    with pytest.raises(ValueError, match="infinite"):
        TemporalResidualDetector().fit(training)



def test_temporal_detector_rejects_infinite_evaluation_values():
    import pytest

    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })

    detector = TemporalResidualDetector().fit(training)

    evaluation = pd.DataFrame({
        "channel_41": [0.0, float("inf")],
    })

    with pytest.raises(ValueError, match="infinite"):
        detector.score_samples(evaluation)




def test_temporal_detector_rejects_different_channels():
    import pytest

    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, 2.0],
    })

    detector = TemporalResidualDetector().fit(training)

    evaluation = pd.DataFrame({
        "channel_42": [0.0, 1.0],
    })

    with pytest.raises(ValueError, match="Input channels differ"):
        detector.score_samples(evaluation)


def test_temporal_detector_uses_valid_residuals_per_channel():
    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "channel_42": [float("nan"), -2.0, 0.0, 2.0, 4.0],
    })

    detector = TemporalResidualDetector().fit(training)

    assert detector.median_["channel_41"] == 0.0
    assert detector.median_["channel_42"] == 1.0



def test_temporal_detector_rejects_channel_without_valid_residuals():
    import pytest

    training = pd.DataFrame({
        "channel_41": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "channel_42": [float("nan")] * 5,
    })

    with pytest.raises(ValueError, match="no valid residuals"):
        TemporalResidualDetector().fit(training)



