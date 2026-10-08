"""Causal temporal reference functions for E1."""

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
