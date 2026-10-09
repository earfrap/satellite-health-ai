"""Summarize ESA event detection across E4 evaluation years."""

import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.evaluation.event_metrics import (
    extract_alert_episodes,
    compute_event_detection,
)


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    mission_root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )
    types = pd.read_csv(mission_root / "anomaly_types.csv")

    channels = {f"channel_{i}" for i in range(41, 47)}
    labels = labels[labels["Channel"].isin(channels)]

    events = (
        labels.groupby("ID")
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
        )
        .reset_index()
        .merge(
            types[["ID", "Category"]],
            on="ID",
            how="left",
            validate="one_to_one",
        )
    )

    for col in ("event_start", "event_end"):
        events[col] = pd.to_datetime(events[col], utc=True)

    if events["Category"].isna().any():
        raise ValueError("Missing ESA event categories")

    base = paths["outputs"] / "experiments" / "e4"
    results = []

    for year in range(2001, 2014):
        windows = pd.read_csv(
            base / "comparisons" / f"comparison_{year}.csv",
            parse_dates=["window_start", "window_end"],
        )

        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

        yearly_events = events[
            (events["event_start"] < end)
            & (events["event_end"] > start)
        ].copy()

        for name in ("e0", "e1", "e2"):
            score = windows[f"{name}_score"]

            # Restore missing alerts for unscorable windows.
            alert = (
                windows[f"{name}_alert"]
                .astype("boolean")
                .mask(score.isna(), pd.NA)
            )

            detector_windows = windows[
                ["window_start", "window_end"]
            ].copy()
            detector_windows["alert"] = alert

            episodes = extract_alert_episodes(detector_windows)
            detection = compute_event_detection(
                yearly_events, episodes
            )

            for event in detection.itertuples(index=False):
                new_episodes = episodes[
                    (episodes["episode_start"] >= event.event_start)
                    & (episodes["episode_start"] < event.event_end)
                ]

                new_episode = not new_episodes.empty

                if new_episode:
                    first_start = new_episodes["episode_start"].min()

                    latency_minutes = (
                        first_start
                        + pd.Timedelta(minutes=30)
                        - event.event_start
                    ).total_seconds() / 60.0
                else:
                    latency_minutes = float("nan")


                results.append({
                    "year": year,
                    "detector": name.upper(),
                    "event_id": event.ID,
                    "category": event.Category,
                    "detected": bool(event.detected),
                    "preexisting_alert": bool(
                        event.preexisting_alert
                    ),
                    "new_episode_during_event": bool(new_episode),
                    "new_episode_latency_minutes": latency_minutes,
                })

    result = pd.DataFrame(results)

    output = base / "e4_event_detection_with_latency_2001_2013.csv"
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output}"
        )

    result.to_csv(output, index=False)

    print("Saved:", output)
    print("Rows:", len(result))
    print("Unique ESA event IDs:", result["event_id"].nunique())
    print("Years:", result["year"].nunique())
    print("Detectors:", result["detector"].unique().tolist())


if __name__ == "__main__":
    main()
