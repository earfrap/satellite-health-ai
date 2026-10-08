
"""Run the E1 causal temporal-residual baseline on the 2000 development slice."""
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.data.event_catalog import load_group_events
from satellite_health.models.temporal_residual import (
    causal_rolling_median,
    TemporalResidualDetector,
)
from satellite_health.evaluation.event_metrics import (
    extract_alert_episodes,
    classify_alert_episodes,
    compute_event_detection,
    compute_detection_latency,
)


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    base = paths["interim"] / "mission1" / "group8"
    raw = paths["raw"] / "ESA-Mission1"

    telemetry = pd.read_pickle(base / "telemetry_2000.pkl")
    windows = pd.read_csv(
        base / "windows_2000.csv",
        parse_dates=["window_start", "window_end"],
    )

    events = load_group_events(
        raw / "labels.csv",
        raw / "anomaly_types.csv",
        [f"channel_{i}" for i in range(41, 47)],
    )

    # ESA timestamps are UTC. The existing window timestamps
    # are timezone-naive but represent UTC clock times.
    windows["window_start"] = windows["window_start"].dt.tz_localize("UTC")
    windows["window_end"] = windows["window_end"].dt.tz_localize("UTC")

    print("=== E1 BASELINE: DATA LOADING ===")
    print("Telemetry shape:", telemetry.shape)
    print("Windows:", len(windows))
    print("Group 8 events:", len(events))
    print("Window timezone:", windows["window_start"].dt.tz)




    # Chronological development split (year 2000).
    train_end = pd.Timestamp("2000-06-01", tz="UTC")
    val_end = pd.Timestamp("2000-09-01", tz="UTC")

    # Training uses only NOMINAL windows.
    train_windows = windows[
        (windows["window_start"] < train_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    # Telemetry timestamps are naive UTC; localize without shifting.
    telemetry.index = telemetry.index.tz_localize("UTC")

    train_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(train_windows["window_start"])
    )




    # Compute the temporal reference on the full chronological series.
    # closed="left" excludes the current observation.
    reference = causal_rolling_median(
        telemetry,
        window="30min",
        min_periods=10,
    )

    residuals = telemetry - reference

    # Fit robust scales using nominal training residuals only.
    train_data = residuals.loc[train_mask]

    detector = TemporalResidualDetector().fit(train_data)

    # Maximum normalized residual across channels and time.
    window_scores = (
        detector.score_samples(residuals)
        .max(axis=1)
        .resample("30min")
        .max()
    )

    if window_scores.empty:
        raise ValueError("E1 produced no window scores.")

    if not window_scores.notna().all():
        raise ValueError("E1 produced windows without valid scores.")


    windows["e1_score"] = windows["window_start"].map(window_scores)

    if windows["e1_score"].isna().any():
        raise ValueError("Some windows have no E1 score.")

    # Calibrate using nominal validation windows only.
    validation_nominal = windows[
        (windows["window_start"] >= train_end)
        & (windows["window_start"] < val_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    threshold = validation_nominal["e1_score"].quantile(0.99)

    print("\n=== E1 TRAINING AND CALIBRATION ===")
    print("Training nominal windows:", len(train_windows))
    print("Training samples:", len(train_data))
    print("Validation nominal windows:", len(validation_nominal))
    print(f"Threshold (99th percentile): {threshold:.6f}")


    # Evaluate on the chronological test period.
    test_start = val_end
    test_end = pd.Timestamp("2001-01-01", tz="UTC")

    test = windows[
        (windows["window_start"] >= test_start)
        & (windows["window_start"] < test_end)
    ].copy()



    # Apply the calibrated threshold to validation and test windows.
    # Validation provides context for episodes crossing the test boundary.
    evaluation_context = windows[
        (windows["window_start"] >= train_end)
        & (windows["window_start"] < test_end)
    ].copy()

    evaluation_context["alert"] = (
        evaluation_context["e1_score"] > threshold
    )

    # Preserve the original test-period alert calculation.
    test["alert"] = test["e1_score"] > threshold

    # Build episodes with preceding temporal context.
    all_episodes = extract_alert_episodes(evaluation_context)

    # Retain episodes overlapping the test period.
    episodes = all_episodes[
        (all_episodes["episode_start"] < test_end)
        & (all_episodes["episode_end"] > test_start)
    ].copy()


    # Select original ESA events overlapping the test period.
    test_events = events[
        (events["event_start"] < test_end)
        & (events["event_end"] > test_start)
    ].copy()

    # Associate alert episodes with ESA events.
    classified = classify_alert_episodes(episodes, test_events)



    # Event-level detection using complete alert episodes.
    event_detection = compute_event_detection(
        test_events,
        episodes,
    )

    # Operational latency using alert-window closure times.
    event_latency = compute_detection_latency(
        test_events,
        evaluation_context,
    )

    # Verify that both evaluations refer to the same events.
    if not event_detection["ID"].equals(event_latency["ID"]):
        raise ValueError("Event order mismatch between evaluation functions.")

    # Combine event-level metrics.
    event_metrics = event_detection.copy()

    event_metrics["first_alert_time"] = (
        event_latency["first_alert_time"]
    )

    event_metrics["latency_minutes"] = (
        event_latency["latency_minutes"]
    )

    if not event_detection["detected"].equals(event_latency["detected"]):
        raise ValueError("Inconsistent event detection results.")



    test_days = (
        test["window_end"].max() - test["window_start"].min()
    ).total_seconds() / 86400

    print("\n=== E1 TEST EVALUATION ===")
    print("Test windows:", len(test))
    print("ESA test events:", len(test_events))
    print("Alerted windows:", int(test["alert"].sum()))
    print("Total alert episodes:", len(classified))

    print("\nEpisodes by association:")
    print(classified["association"].value_counts().to_string())

    unassociated = (
        classified["association"] == "UNASSOCIATED"
    ).sum()

    print(
        "Unassociated episodes/day:",
        round(unassociated / test_days, 3),
    )



    # Save reproducible E1 experiment outputs.
    output_dir = paths["outputs"] / "experiments" / "e1"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Window-level scores and alerts.
    windows.to_csv(
        output_dir / "window_scores_2000.csv",
        index=False,
    )

    # Test alert episodes with ESA associations.
    classified.to_csv(
        output_dir / "test_episodes_2000.csv",
        index=False,
    )

    # Event-level detection and operational latency.
    event_metrics.to_csv(
        output_dir / "test_event_metrics_2000.csv",
        index=False,
    )

    # Summary metrics and experiment configuration.
    summary = pd.DataFrame([{
        "experiment": "E1",
        "year": 2000,
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": val_end.isoformat(),
        "test_end_utc": test_end.isoformat(),
        "threshold_quantile": 0.99,
        "threshold": threshold,
        "test_windows": len(test),
        "alerted_windows": int(test["alert"].sum()),
        "total_episodes": len(classified),
        "anomaly_episodes": int(
            (classified["association"] == "ANOMALY").sum()
        ),
        "rare_event_episodes": int(
            (classified["association"] == "RARE_EVENT").sum()
        ),
        "unassociated_episodes": int(unassociated),
        "unassociated_episodes_per_day": unassociated / test_days,
    }])

    summary.to_csv(
        output_dir / "summary_2000.csv",
        index=False,
    )

    print("\n=== E1 OUTPUT FILES ===")
    print("Output directory:", output_dir)
    print("Saved window scores, test episodes and summary.")


if __name__ == "__main__":
    main()

