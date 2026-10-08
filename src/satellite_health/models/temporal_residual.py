"""Causal temporal reference functions for E1."""

import numpy as np
import pandas as pd


def causal_rolling_median(
    X: pd.DataFrame,
    window: str = "30min",
    min_periods: int = 10,
) -> pd.DataFrame:
    """Compute a time-based median using strictly past observations."""

    if X.empty:
        raise ValueError("Input data cannot be empty.")

    if not isinstance(X.index, pd.DatetimeIndex):
        raise TypeError("Input index must be a DatetimeIndex.")

    if not X.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be chronologically sorted.")

    if X.index.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed.")

    if X.isna().any().any():
        raise ValueError("Input data contain missing values.")

    if min_periods < 1:
        raise ValueError("min_periods must be positive.")

    return X.rolling(
        window=window,
        min_periods=min_periods,
        closed="left",
    ).median()


class TemporalResidualDetector:
    """Robust per-channel scoring of causal temporal residuals."""

    def __init__(self):
        self.median_ = None
        self.scale_ = None
        self.columns_ = None

    def fit(self, residuals: pd.DataFrame):
        if np.isinf(residuals.to_numpy()).any():
            raise ValueError("Training residuals contain infinite values.")
        valid = residuals

        if valid.empty:
            raise ValueError("No valid training residuals.")

        if valid.count().eq(0).any():
            raise ValueError("At least one channel has no valid residuals.")            
            
        median = valid.median()
        mad = (valid - median).abs().median()

        if (mad <= 0).any():
            raise ValueError("At least one channel has zero residual MAD.")

        self.median_ = median
        self.scale_ = 1.4826 * mad
        self.columns_ = list(valid.columns)

        return self

    def score_samples(self, residuals: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise RuntimeError("Detector must be fitted first.")

        if list(residuals.columns) != self.columns_:
            raise ValueError("Input channels differ from training.")

        if np.isinf(residuals.to_numpy()).any():
            raise ValueError("Evaluation residuals contain infinite values.")

        scores = (
            (residuals - self.median_)
            .abs()
            .div(self.scale_)
        )

        return scores



