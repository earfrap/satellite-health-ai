"""Verify ESA event metrics using continuous 2001–2013 alerts."""

import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.evaluation.event_metrics import (
    extract_alert_episodes,
    compute_event_detection,
)


def load_events(paths):
    root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )
    types = pd.read_csv(root / "anomaly_types.csv")

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

    return events


def verify_model(paths, events, model):
    frames = []

    for year in range(2001, 2014):
        path = (
            paths["outputs"]
            / "experiments"
            / model
            / "comparisons"
            / f"{model}_{year}.csv"
        )

        frame = pd.read_csv(
            path,
            parse_dates=["window_start", "window_end"],
        )

        frame["alert"] = (
            frame[f"{model}_alert"]
            .astype("boolean")
            .mask(frame[f"{model}_score"].isna(), pd.NA)
        )

        frames.append(
            frame[["window_start", "window_end", "alert"]]
        )

    windows = (
        pd.concat(frames, ignore_index=True)
        .sort_values("window_start")
        .reset_index(drop=True)
    )

    if windows["window_start"].duplicated().any():
        raise ValueError(f"Duplicate windows for {model}.")

    episodes = extract_alert_episodes(windows)

    previous_path = (
        paths["outputs"] / "experiments" / model
        / f"{model}_event_detection.csv"
    )
    previous = pd.read_csv(previous_path)

    # Use exactly the ESA events evaluated in the original protocol.
    evaluated_ids = set(previous["ID"].astype(str))
    selected_events = events[
        events["ID"].astype(str).isin(evaluated_ids)
    ].copy()

    if len(selected_events) != len(previous):
        raise ValueError("Event selection does not match previous evaluation.")

    detection = compute_event_detection(selected_events, episodes)

    results = []

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

    output_dir = (
        paths["outputs"] / "experiments" / model
    )
    output_path = (
        output_dir / f"{model}_event_detection_continuous.csv"
    )

    if output_path.exists():
        existing = pd.read_csv(output_path)
        if not existing.equals(result):
            raise FileExistsError(
                f"Existing verification differs: {output_path}"
            )
    else:
        result.to_csv(output_path, index=False)

    previous_path = (
        output_dir / f"{model}_event_detection.csv"
    )
    previous = pd.read_csv(previous_path)

    comparison = result.merge(
        previous,
        on=["ID", "Category"],
        suffixes=("_continuous", "_previous"),
        validate="one_to_one",
    )

    if len(comparison) != len(result) or len(result) != len(previous):
        raise ValueError("Event IDs differ between evaluations.")

    print(f"\n{model.upper()} — Continuous evaluation")
    print("Windows:", len(windows))
    print("Episodes:", len(episodes))
    print("Events:", len(result))

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


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    events = load_events(paths)

    for model in ("e3", "m4"):
        verify_model(paths, events, model)


if __name__ == "__main__":
    main()
