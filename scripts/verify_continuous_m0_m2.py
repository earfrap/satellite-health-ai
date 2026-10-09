"""Verify continuous ESA event detection for E0, E1 and E2."""

import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.evaluation.event_metrics import (
    extract_alert_episodes,
    compute_event_detection,
)


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["outputs"] / "experiments" / "eval"
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
        raise ValueError("Missing ESA event categories.")

    original = pd.read_csv(
        base / "e4_event_detection_with_latency_2001_2013.csv"
    )

    evaluated_ids = set(original["event_id"].astype(str))
    events = events[
        events["ID"].astype(str).isin(evaluated_ids)
    ].copy()

    if len(events) != len(evaluated_ids):
        raise ValueError("ESA event IDs do not match.")

    frames = []

    for year in range(2001, 2014):
        frame = pd.read_csv(
            base / "comparisons" / f"comparison_{year}.csv",
            parse_dates=["window_start", "window_end"],
        )
        frames.append(frame)

    windows = (
        pd.concat(frames, ignore_index=True)
        .sort_values("window_start")
        .reset_index(drop=True)
    )

    if windows["window_start"].duplicated().any():
        raise ValueError("Duplicate telemetry windows.")

    print("Continuous windows:", len(windows))
    print("ESA events:", len(events))

    for model in ("e0", "e1", "e2"):
        score = windows[f"{model}_score"]

        detector_windows = windows[
            ["window_start", "window_end"]
        ].copy()

        detector_windows["alert"] = (
            windows[f"{model}_alert"]
            .astype("boolean")
            .mask(score.isna(), pd.NA)
        )

        episodes = extract_alert_episodes(detector_windows)
        detection = compute_event_detection(events, episodes)

        rows = []

        for event in detection.itertuples(index=False):
            new_episodes = episodes[
                (episodes["episode_start"] >= event.event_start)
                & (episodes["episode_start"] < event.event_end)
            ]

            rows.append({
                "ID": event.ID,
                "Category": event.Category,
                "overlap": bool(event.detected),
                "new_episode": not new_episodes.empty,
            })

        result = pd.DataFrame(rows)

        previous = original[
            original["detector"].str.upper() == model.upper()
        ]

        previous = (
            previous.groupby("event_id", as_index=False)
            .agg(
                overlap=("detected", "max"),
                new_episode=("new_episode_during_event", "max"),
            )
        )

        comparison = result.merge(
            previous,
            left_on="ID",
            right_on="event_id",
            suffixes=("_continuous", "_previous"),
            validate="one_to_one",
        )

        if len(comparison) != len(events):
            raise ValueError(f"Event mismatch for {model}.")

        output_path = (
            base / f"{model}_event_detection_continuous.csv"
        )

        if output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output_path}"
            )

        result.to_csv(output_path, index=False)

        print(f"\n{model.upper()}")
        print("Episodes:", len(episodes))

        for metric in ("overlap", "new_episode"):
            old = int(comparison[f"{metric}_previous"].sum())
            new = int(comparison[f"{metric}_continuous"].sum())
            changed = int(
                (
                    comparison[f"{metric}_previous"]
                    != comparison[f"{metric}_continuous"]
                ).sum()
            )

            print(
                f"{metric}: previous={old}, "
                f"continuous={new}, changed_events={changed}"
            )

        print("Saved:", output_path)


if __name__ == "__main__":
    main()
