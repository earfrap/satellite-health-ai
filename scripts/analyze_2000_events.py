import pandas as pd

from satellite_health.data.paths import load_data_paths


CONTEXT_HOURS = 6


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

    # Only annotations for the six channels in our Development Slice.
    labels = labels[
        labels["Channel"].isin(telemetry.columns)
    ].copy()

    events = (
        labels.groupby("ID")
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
            affected_channels=("Channel", "nunique"),
        )
        .reset_index()
    )

    events = events.merge(
        anomaly_types[["ID", "Category"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    # Convert timezone-aware ESA labels to the naive timestamps
    # used by the telemetry DatetimeIndex.
    events["event_start"] = events["event_start"].dt.tz_localize(None)
    events["event_end"] = events["event_end"].dt.tz_localize(None)

    # Development Slice = events starting in 2000.
    events = events[
        events["event_start"].dt.year == 2000
    ].copy()

    results = []

    for _, row in events.iterrows():
        event_id = row["ID"]
        start = row["event_start"]
        end = row["event_end"]

        before = telemetry.loc[
            start - pd.Timedelta(hours=CONTEXT_HOURS):start
        ]

        event = telemetry.loc[start:end]

        after = telemetry.loc[
            end:end + pd.Timedelta(hours=CONTEXT_HOURS)
        ]

        for channel in telemetry.columns:
            context_std = (
                before[channel].std()
                + after[channel].std()
            ) / 2

            context_range = (
                (before[channel].max() - before[channel].min())
                + (after[channel].max() - after[channel].min())
            ) / 2

            event_std = event[channel].std()
            event_range = event[channel].max() - event[channel].min()

            results.append({
                "event_id": event_id,
                "category": row["Category"],
                "channel": channel,
                "event_samples": len(event),
                "duration_minutes": (
                    end - start
                ).total_seconds() / 60,
                "std_ratio": event_std / context_std,
                "range_ratio": event_range / context_range,
            })

    results = pd.DataFrame(results)

    print("=== DEVELOPMENT SLICE EVENT ANALYSIS ===")
    print(f"Events: {results['event_id'].nunique()}")
    print(f"Event-channel observations: {len(results)}")

    print("\nEvents by category:")
    print(
        results[
            ["event_id", "category"]
        ]
        .drop_duplicates()
        ["category"]
        .value_counts()
    )

    print("\nSTD ratio by category:")
    print(
        results.groupby("category")["std_ratio"]
        .describe()
        .round(3)
    )

    print("\nRange ratio by category:")
    print(
        results.groupby("category")["range_ratio"]
        .describe()
        .round(3)
    )

    print("\nMean ratios per event:")
    event_summary = (
        results.groupby(
            ["event_id", "category"],
            as_index=False,
        )
        .agg(
            event_samples=("event_samples", "first"),
            duration_minutes=("duration_minutes", "first"),
            mean_std_ratio=("std_ratio", "mean"),
            mean_range_ratio=("range_ratio", "mean"),
        )
        .sort_values(["category", "mean_std_ratio"])
    )

    print(
        event_summary.round(3).to_string(index=False)
    )

    output_dir = paths["outputs"] / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "event_statistics_2000.csv"
    results.to_csv(output_path, index=False)

    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()
