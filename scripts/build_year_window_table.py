"""Build annual Group 8 window tables for E4."""

import argparse
import pandas as pd
from satellite_health.data.paths import load_data_paths
from satellite_health.preprocessing.windowing import (
    assign_training_label,
)
from satellite_health.preprocessing.windowing import (
    assign_training_label,
    event_overlap_by_category,
)


def assign_e4_label(
    anomaly_overlap: float,
    rare_event_overlap: float,
    communication_gap_overlap: float,
) -> str:
    """Assign E4 labels while protecting communication gaps."""

    # Preserve the existing E0-E3 labeling convention.
    original_label = assign_training_label(
        anomaly_overlap=anomaly_overlap,
        rare_event_overlap=rare_event_overlap,
        threshold=0.5,
    )

    # Communication gaps take precedence over nominal/boundary labels.
    if communication_gap_overlap > 0:
        if original_label in ("NOMINAL", "BOUNDARY"):
            return "COMMUNICATION_GAP"

    return original_label

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=range(2001, 2014),
    )
    args = parser.parse_args()

    year = args.year
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year + 1}-01-01")

    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry_path = (
        paths["interim"]
        / "mission1"
        / "group8"
        / f"telemetry_{year}.pkl"
    )
    telemetry = pd.read_pickle(telemetry_path)

    mission_root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )
    anomaly_types = pd.read_csv(
        mission_root / "anomaly_types.csv"
    )

    labels = labels[
        labels["Channel"].isin(telemetry.columns)
    ].copy()

    events = (
        labels.groupby("ID")
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
        )
        .reset_index()
    )

    events = events.merge(
        anomaly_types[["ID", "Category"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    events["event_start"] = (
        events["event_start"].dt.tz_localize(None)
    )
    events["event_end"] = (
        events["event_end"].dt.tz_localize(None)
    )

    # Include all events intersecting this year.
    events = events[
        (events["event_start"] < year_end)
        & (events["event_end"] > year_start)
    ].copy()

    print(f"Year: {year}")
    print(f"Telemetry shape: {telemetry.shape}")
    print(f"ESA events intersecting year: {len(events)}")
    print(events["Category"].value_counts())


    window = pd.Timedelta(minutes=30)

    sample_counts = (
        telemetry["channel_41"]
        .resample("30min")
        .count()
    )

    rows = []

    for window_start, sample_count in sample_counts.items():
        window_end = window_start + window

        relevant_events = events[
            (events["event_start"] < window_end)
            & (events["event_end"] > window_start)
        ]

        overlaps = event_overlap_by_category(
            window_start,
            window_end,
            relevant_events,
        )

        anomaly_overlap = overlaps.get("Anomaly", 0.0)
        rare_event_overlap = overlaps.get("Rare Event", 0.0)
        gap_overlap = overlaps.get("Communication Gap", 0.0)

        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "sample_count": int(sample_count),
                "anomaly_overlap": anomaly_overlap,
                "rare_event_overlap": rare_event_overlap,
                "communication_gap_overlap": gap_overlap,
                "training_label": assign_e4_label(
                    anomaly_overlap,
                    rare_event_overlap,
                    gap_overlap,
                ),
            }
        )

    windows = pd.DataFrame(rows)

    output_dir = (
        paths["interim"] / "mission1" / "group8"
    )
    output_path = output_dir / f"windows_{year}.csv"

    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    windows.to_csv(output_path, index=False)

    print(f"\nWindows: {len(windows):,}")
    print("\nTraining labels:")
    print(windows["training_label"].value_counts())
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
