
"""Prepare synchronized Group 8 telemetry for a selected year."""

import argparse

import pandas as pd

from satellite_health.data.paths import load_data_paths


CHANNELS = [f"channel_{i}" for i in range(41, 47)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year", type=int, required=True, choices=range(2001, 2014)
    )
    args = parser.parse_args()

    year = args.year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")

    paths = load_data_paths("configs/data/paths.local.yaml")
    source_root = (
        paths["raw"] / "ESA-Mission1" / "extracted_channels"
    )
    output_dir = paths["interim"] / "mission1" / "group8"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    reference_index = None

    for channel in CHANNELS:
        print(f"Loading {channel}...")

        df = pd.read_pickle(source_root / channel)
        subset = df.loc[
            (df.index >= start) & (df.index < end)
        ].copy()
        del df

        if subset.empty:
            raise ValueError(f"{channel}: no data for {year}")

        if not subset.index.is_monotonic_increasing:
            raise ValueError(f"{channel}: unsorted timestamps")

        if subset.index.has_duplicates:
            raise ValueError(f"{channel}: duplicate timestamps")

        if reference_index is None:
            reference_index = subset.index
        elif not subset.index.equals(reference_index):
            raise ValueError(
                f"{channel}: timestamp alignment mismatch"
            )

        print(f"  {len(subset):,} samples — aligned")
        frames.append(subset)

    telemetry = pd.concat(frames, axis=1)

    if telemetry.isna().any().any():
        raise ValueError("Missing values in synchronized telemetry")

    if telemetry.columns.duplicated().any():
        raise ValueError("Duplicate channel columns")

    output_path = output_dir / f"telemetry_{year}.pkl"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    telemetry.to_pickle(output_path)

    print(f"\nYear: {year}")
    print(f"Shape: {telemetry.shape}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
