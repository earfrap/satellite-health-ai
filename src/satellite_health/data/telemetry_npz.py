"""Load ESA telemetry from portable NumPy NPZ archives."""

import numpy as np
import pandas as pd


def load_telemetry_npz(path):
    """Load six-channel telemetry without pickle dependencies."""
    with np.load(path, allow_pickle=False) as archive:
        values = archive["values"]
        timestamps = archive["timestamps_ns"]
        columns = archive["columns"].tolist()

    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("Expected six telemetry channels.")

    if timestamps.ndim != 1 or len(timestamps) != len(values):
        raise ValueError("Invalid telemetry timestamps.")

    if len(columns) != 6:
        raise ValueError("Invalid telemetry column names.")

    index = pd.to_datetime(timestamps, utc=True)

    if index.hasnans or not index.is_monotonic_increasing:
        raise ValueError("Invalid telemetry time index.")

    return pd.DataFrame(values, index=index, columns=columns)
