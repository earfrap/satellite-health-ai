"""Evaluate frozen E3 alerts against ESA events."""

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

    results = []

    for year in range(2001, 2014):
        path = (
            paths["outputs"]
            / "experiments"
            / "m3"
            / "comparisons"
            / f"e3_{year}.csv"
        )

        windows = pd.read_csv(
            path,
            parse_dates=["window_start", "window_end"],
        )

        score = windows["e3_score"]
        alert = (
            windows["e3_alert"]
            .astype("boolean")
            .mask(score.isna(), pd.NA)
        )

        detector_windows = windows[
            ["window_start", "window_end"]
        ].copy()
        detector_windows["alert"] = alert

        episodes = extract_alert_episodes(detector_windows)

        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

        yearly_events = events[
            (events["event_start"] < end)
            & (events["event_end"] > start)
        ].copy()

        detection = compute_event_detection(
            yearly_events, episodes
        )

        for event in detection.itertuples(index=False):
            new_episodes = episodes[
                (episodes["episode_start"] >= event.event_start)
                & (episodes["episode_start"] < event.event_end)
            ]

            results.append({
                "ID": event.ID,
                "Category": event.Category,
                "overlap": bool(event.detected),
                "new_episode": not new_episodes.empty,
            })

    result = pd.DataFrame(results)

    # Events crossing calendar years should be counted once.
    result = (
        result.groupby(["ID", "Category"], as_index=False)
        [["overlap", "new_episode"]]
        .max()
    )

    output_path = (
        paths["outputs"] / "experiments" / "m3"
        / "e3_event_detection.csv"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing overwrite: {output_path}")

    result.to_csv(output_path, index=False)
    print("Saved:", output_path)

    print("E3 — Isolation Forest")
    print()

    for category, subset in [
        ("All events", result),
        ("ESA Anomaly", result[result["Category"] == "Anomaly"]),
        ("ESA Rare Event", result[result["Category"] == "Rare Event"]),
    ]:
        n = len(subset)
        print(category)
        print(f"  Events: {n}")
        print(f"  Overlap: {subset['overlap'].sum()}/{n}")
        print(f"  New episodes: {subset['new_episode'].sum()}/{n}")
        print()


if __name__ == "__main__":
    main()
