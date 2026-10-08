
import numpy as np
import pandas as pd


class RobustThresholdDetector:
    """Independent per-channel robust anomaly scoring."""

    def __init__(self):
        self.median_ = None
        self.scale_ = None
        self.columns_ = None

    def fit(self, X: pd.DataFrame):
        if X.empty:
            raise ValueError("Training data cannot be empty.")

        if X.isna().any().any():
            raise ValueError("Training data contain missing values.")

        median = X.median()
        mad = (X - median).abs().median()

        if (mad <= 0).any():
            raise ValueError("At least one channel has zero MAD.")

        self.median_ = median
        self.scale_ = 1.4826 * mad
        self.columns_ = list(X.columns)
        return self

    def score_samples(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise RuntimeError("Detector must be fitted first.")

        if list(X.columns) != self.columns_:
            raise ValueError("Input channels differ from training.")

        if X.isna().any().any():
            raise ValueError("Input data contain missing values.")

        return (X - self.median_).abs().div(self.scale_)

