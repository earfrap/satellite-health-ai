"""Export ESA telemetry pickle files to NumPy-compatible NPZ archives."""

import argparse

import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["interim"] / "mission1" / "group8"

    source = base / f"telemetry_{args.year}.pkl"
    destination = base / f"telemetry_{args.year}.npz"

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing archive: {destination}"
        )

    telemetry = pd.read_pickle(source)

    if not isinstance(telemetry, pd.DataFrame):
        raise TypeError("Expected telemetry DataFrame.")

    if not isinstance(telemetry.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex.")

    if telemetry.index.tz is not None:
        raise ValueError(
            "Expected timezone-naive timestamps, matching original telemetry."
        )

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in telemetry.dtypes):
        raise TypeError("Telemetry contains non-numeric columns.")

    columns = list(telemetry.columns)

    if not all(isinstance(column, str) for column in columns):
        raise TypeError("Expected string column names.")

    values = telemetry.to_numpy(dtype=np.float64, copy=True)
    timestamps = telemetry.index.asi8.copy()

    if values.shape[1] != 6:
        raise ValueError(f"Expected six channels, found {values.shape[1]}.")

    np.savez_compressed(
        destination,
        values=values,
        timestamps_ns=timestamps,
        columns=np.asarray(columns, dtype=np.str_),
    )

    print("Source:", source)
    print("Destination:", destination)
    print("Samples:", len(telemetry))
    print("Channels:", columns)
    print("NPZ EXPORT PASSED")


if __name__ == "__main__":
    main()
