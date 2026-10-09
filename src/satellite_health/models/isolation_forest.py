"""Isolation Forest anomaly detector for spacecraft telemetry."""

import pandas as pd
from sklearn.ensemble import IsolationForest

from satellite_health.models.pca_detector import RobustChannelScaler


class IsolationForestDetector:
    """Isolation Forest with training-derived robust scaling."""

    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler_ = None
        self.model_ = None

    def fit(self, X: pd.DataFrame):
        """Fit the detector using nominal training telemetry."""
        scaler = RobustChannelScaler().fit(X)
        Z = scaler.transform(X)

        model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=256,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(Z)

        self.scaler_ = scaler
        self.model_ = model
        return self

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        """Return anomaly scores; higher means more anomalous."""
        if self.model_ is None:
            raise RuntimeError("Detector must be fitted first.")

        Z = self.scaler_.transform(X)
        scores = -self.model_.score_samples(Z)

        return pd.Series(
            scores,
            index=X.index,
            name="isolation_forest_score",
        )
