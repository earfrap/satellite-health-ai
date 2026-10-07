import pandas as pd

from satellite_health.data.paths import load_data_paths


EVENT_IDS = ["id_193", "id_35"]
CONTEXT_HOURS = 6


def compute_statistics(data: pd.DataFrame) -> pd.DataFrame:
    stats = pd.DataFrame({
        "mean": data.mean(),
        "std": data.std(),
        "min": data.min(),
        "max": data.max(),
    })

    stats["range"] = stats["max"] - stats["min"]

    return stats


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry_path = (
        paths["interim"]
        / "mission1"
        / "group8"
        / "telemetry_2000.pkl"
    )

    mission_root = paths["raw"] / "ESA-Mission1"

    telemetry = pd.read_pickle(telemetry_path)

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )

    for event_id in EVENT_IDS:

        event_rows = labels[
            (labels["ID"] == event_id)
            & (labels["Channel"].isin(telemetry.columns))
        ]

        event_start = event_rows["StartTime"].min().tz_localize(None)
        event_end = event_rows["EndTime"].max().tz_localize(None)

        before = telemetry.loc[
            event_start - pd.Timedelta(hours=CONTEXT_HOURS):
            event_start
        ]

        event = telemetry.loc[event_start:event_end]

        after = telemetry.loc[
            event_end:
            event_end + pd.Timedelta(hours=CONTEXT_HOURS)
        ]

        before_stats = compute_statistics(before)
        event_stats = compute_statistics(event)
        after_stats = compute_statistics(after)

        # Average before and after to obtain a local nominal context.
        context_std = (
            before_stats["std"] + after_stats["std"]
        ) / 2

        context_range = (
            before_stats["range"] + after_stats["range"]
        ) / 2

        comparison = pd.DataFrame({
            "context_std": context_std,
            "event_std": event_stats["std"],
            "std_ratio": event_stats["std"] / context_std,
            "context_range": context_range,
            "event_range": event_stats["range"],
            "range_ratio": event_stats["range"] / context_range,
        })

        print("\n" + "=" * 70)
        print(f"EVENT: {event_id}")
        print(f"Start: {event_start}")
        print(f"End:   {event_end}")
        print(f"Event samples: {len(event):,}")
        print("=" * 70)

        print(
            comparison.round(3).to_string()
        )


if __name__ == "__main__":
    main()
