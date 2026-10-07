from pathlib import Path

import pandas as pd


def audit_channel(path: str | Path) -> dict:
    """Compute basic integrity and temporal statistics for one telemetry channel."""

    path = Path(path)
    df = pd.read_pickle(path)

    if df.shape[1] != 1:
        raise ValueError(
            f"Expected one telemetry column, found {df.shape[1]}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Telemetry index must be a pandas DatetimeIndex")

    channel = df.columns[0]
    intervals = df.index.to_series().diff().dt.total_seconds().dropna()

    summary = {
        "channel": channel,
        "n_samples": len(df),
        "start_time": df.index.min(),
        "end_time": df.index.max(),
        "dtype": str(df[channel].dtype),
        "missing_values": int(df[channel].isna().sum()),
        "duplicate_timestamps": int(df.index.duplicated().sum()),
        "monotonic": bool(df.index.is_monotonic_increasing),
        "median_interval_s": float(intervals.median()),
        "mean_interval_s": float(intervals.mean()),
        "min_interval_s": float(intervals.min()),
        "max_interval_s": float(intervals.max()),
        "memory_mib": float(df.memory_usage(deep=True).sum() / 1024**2),
    }

    return summary
