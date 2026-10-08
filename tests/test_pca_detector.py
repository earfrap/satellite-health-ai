
"""Tests for E2 robust scaling."""

import numpy as np
import pandas as pd
import pytest

from satellite_health.models.pca_detector import (
    RobustChannelScaler,
    PCAAnomalyDetector,
)


def make_training_data():
    return pd.DataFrame({
        "channel_41": [1.0, 2.0, 3.0, 4.0, 5.0],
        "channel_42": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


def test_training_medians_transform_to_zero():
    X = make_training_data()

    scaler = RobustChannelScaler().fit(X)
    Z = scaler.transform(X)

    assert np.allclose(Z.median().to_numpy(), 0.0)


def test_transform_reuses_training_parameters():
    X = make_training_data()
    scaler = RobustChannelScaler().fit(X)

    original_median = scaler.median_.copy()
    original_scale = scaler.scale_.copy()

    future = pd.DataFrame({
        "channel_41": [100.0],
        "channel_42": [1000.0],
    })

    Z = scaler.transform(future)

    assert (Z.to_numpy() > 0).all()
    pd.testing.assert_series_equal(scaler.median_, original_median)
    pd.testing.assert_series_equal(scaler.scale_, original_scale)


def test_missing_values_are_rejected():
    X = make_training_data()
    X.loc[0, "channel_41"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        RobustChannelScaler().fit(X)


def test_changed_channels_are_rejected():
    X = make_training_data()
    scaler = RobustChannelScaler().fit(X)

    changed = X.rename(columns={"channel_41": "channel_99"})

    with pytest.raises(ValueError, match="channels"):
        scaler.transform(changed)




def make_correlated_data():
    """Two channels with an exact linear relationship."""
    return pd.DataFrame({
        "channel_41": [1.0, 2.0, 3.0, 4.0, 5.0],
        "channel_44": [2.0, 4.0, 6.0, 8.0, 10.0],
    })


def test_pca_reconstructs_linear_relationship():
    X = make_correlated_data()

    detector = PCAAnomalyDetector(n_components=1).fit(X)
    scores = detector.score_samples(X)

    assert np.allclose(scores.to_numpy(), 0.0, atol=1e-10)


def test_pca_detects_broken_relationship():
    X = make_correlated_data()
    detector = PCAAnomalyDetector(n_components=1).fit(X)

    normal = pd.DataFrame({
        "channel_41": [3.0],
        "channel_44": [6.0],
    })

    unusual = pd.DataFrame({
        "channel_41": [3.0],
        "channel_44": [20.0],
    })

    normal_score = detector.score_samples(normal).iloc[0]
    unusual_score = detector.score_samples(unusual).iloc[0]

    assert unusual_score > normal_score
    assert unusual_score > 0


def test_pca_requires_fitting():
    detector = PCAAnomalyDetector(n_components=1)

    with pytest.raises(RuntimeError, match="fitted"):
        detector.score_samples(make_correlated_data())




def test_channel_contributions_sum_to_spe():
    X = make_correlated_data()
    detector = PCAAnomalyDetector(n_components=1).fit(X)

    contributions = detector.channel_contributions(X)
    scores = detector.score_samples(X)

    assert np.allclose(
        contributions.sum(axis=1).to_numpy(),
        scores.to_numpy(),
    )

    assert contributions.index.equals(X.index)
    assert contributions.columns.equals(X.columns)


def test_channel_contributions_detect_broken_relationship():
    X = make_correlated_data()
    detector = PCAAnomalyDetector(n_components=1).fit(X)

    unusual = pd.DataFrame({
        "channel_41": [3.0],
        "channel_44": [20.0],
    })

    contributions = detector.channel_contributions(unusual)

    assert (contributions.to_numpy() >= 0).all()
    assert contributions.sum(axis=1).iloc[0] > 0
