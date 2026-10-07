from pathlib import Path

import pandas as pd

from satellite_health.data.paths import load_data_paths


GROUP_8_CHANNELS = {
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
}


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    mission_root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )

    anomaly_types = pd.read_csv(
        mission_root / "anomaly_types.csv"
    )

    # Keep only label rows involving Group 8.
    group8_labels = labels[
        labels["Channel"].isin(GROUP_8_CHANNELS)
    ].copy()

    # Collapse channel-level labels into one row per event.
    events = (
        group8_labels
        .groupby("ID")
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
            affected_channels=("Channel", "nunique"),
        )
        .reset_index()
    )

    # Add ESA event metadata.
    events = events.merge(
        anomaly_types,
        on="ID",
        how="left",
        validate="one_to_one",
    )

    events["duration_hours"] = (
        events["event_end"] - events["event_start"]
    ).dt.total_seconds() / 3600

    events["year"] = events["event_start"].dt.year

    print("=== GROUP 8 EVENT AUDIT ===")
    print(f"Events: {len(events)}")

    print("\nEvents by year and category:")
    yearly = pd.crosstab(
        events["year"],
        events["Category"],
        margins=True,
    )
    print(yearly)

    print("\nEvent duration [hours]:")
    print(
        events["duration_hours"].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )

    print("\nAffected Group 8 channels per event:")
    print(events["affected_channels"].value_counts().sort_index())

    print("\nChronological event list:")
    columns = [
        "ID",
        "event_start",
        "event_end",
        "duration_hours",
        "affected_channels",
        "Category",
        "Dimensionality",
        "Locality",
        "Length",
    ]

    print(
        events[columns]
        .sort_values("event_start")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
