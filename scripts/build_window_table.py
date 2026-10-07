import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.preprocessing.windowing import (
    assign_training_label,
    event_overlap_by_category,
)


WINDOW = pd.Timedelta(minutes=30)


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry = pd.read_pickle(
        paths["interim"]
        / "mission1"
        / "group8"
        / "telemetry_2000.pkl"
    )

    mission_root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )

    anomaly_types = pd.read_csv(
        mission_root / "anomaly_types.csv"
    )

    # Keep only events involving Group 8 channels.
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

    events = events[
        events["event_start"].dt.year == 2000
    ].copy()

    # Fixed 30-minute calendar windows.
    sample_counts = (
        telemetry["channel_41"]
        .resample("30min")
        .count()
    )

    rows = []

    for window_start, sample_count in sample_counts.items():
        window_end = window_start + WINDOW

        # Only inspect events that can intersect this window.
        relevant_events = events[
            (events["event_start"] < window_end)
            & (events["event_end"] > window_start)
        ]

        overlaps = event_overlap_by_category(
            window_start,
            window_end,
            relevant_events,
        )

        rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "sample_count": int(sample_count),
                "anomaly_overlap": overlaps.get(
                    "Anomaly", 0.0
                ),
                "rare_event_overlap": overlaps.get(
                    "Rare Event", 0.0
                ),
            }
        )

    windows = pd.DataFrame(rows)
    

    windows["training_label"] = windows.apply(
        lambda row: assign_training_label(
            anomaly_overlap=row["anomaly_overlap"],
            rare_event_overlap=row["rare_event_overlap"],
            threshold=0.5,
        ),
        axis=1,
    )


    output_dir = (
        paths["interim"]
        / "mission1"
        / "group8"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "windows_2000.csv"
    windows.to_csv(output_path, index=False)

    print("=== 2000 WINDOW TABLE ===")
    print(f"Windows: {len(windows):,}")

    print("\nSample count:")
    print(windows["sample_count"].describe())

    print("\nWindows touching an Anomaly:")
    print((windows["anomaly_overlap"] > 0).sum())

    print("\nWindows touching a Rare Event:")
    print((windows["rare_event_overlap"] > 0).sum())

    print("\nAnomaly overlap:")
    print(
        windows.loc[
            windows["anomaly_overlap"] > 0,
            "anomaly_overlap",
        ].describe()
    )

    print("\nRare Event overlap:")
    print(
        windows.loc[
            windows["rare_event_overlap"] > 0,
            "rare_event_overlap",
        ].describe()
    )



    print("\nTraining labels:")
    print(windows["training_label"].value_counts())

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
