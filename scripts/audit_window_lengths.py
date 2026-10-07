import pandas as pd

from satellite_health.data.paths import load_data_paths


WINDOW_MINUTES = [15, 30, 60]


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry = pd.read_pickle(
        paths["interim"]
        / "mission1"
        / "group8"
        / "telemetry_2000.pkl"
    )

    labels = pd.read_csv(
        paths["raw"] / "ESA-Mission1" / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
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

    events["event_start"] = events["event_start"].dt.tz_localize(None)
    events["event_end"] = events["event_end"].dt.tz_localize(None)

    events = events[
        events["event_start"].dt.year == 2000
    ].copy()

    events["duration_min"] = (
        events["event_end"] - events["event_start"]
    ).dt.total_seconds() / 60

    print("=== WINDOW LENGTH AUDIT ===")

    print("\nEvent durations in 2000 [minutes]:")
    print(
        events["duration_min"]
        .describe(
            percentiles=[0.25, 0.5, 0.75]
        )
        .round(2)
    )

    print("\nCandidate windows:")

    for minutes in WINDOW_MINUTES:
        expected_samples = minutes * 2

        shorter = (
            events["duration_min"] < minutes
        ).sum()

        print(
            f"\n{minutes:>2} min window"
            f"\n  expected samples at 30 s cadence: {expected_samples}"
            f"\n  events shorter than window: {shorter}/{len(events)}"
            f"\n  approximate windows/year: "
            f"{365 * 24 * 60 // minutes:,}"
        )

    print("\nEvent durations:")
    print(
        events[
            ["ID", "duration_min"]
        ]
        .sort_values("duration_min")
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
