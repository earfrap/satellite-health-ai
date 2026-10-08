"""Tests for causal temporal reference functions."""

import pandas as pd

from satellite_health.models.temporal_residual import (
    causal_rolling_median,
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


