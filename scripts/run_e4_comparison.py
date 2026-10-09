"""Compare frozen E0, E1 and E2 detectors on unseen years."""
import pandas as pd
import argparse
import joblib
from satellite_health.models.temporal_residual import causal_rolling_median
from satellite_health.data.paths import load_data_paths
from satellite_health.evaluation.event_metrics import (
    extract_alert_episodes,
    compute_event_detection,
    compute_detection_latency,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2001)
    args = parser.parse_args()

    paths = load_data_paths("configs/data/paths.local.yaml")
    model_dir = paths["outputs"] / "experiments" / "e4" / "models"

    models = {}

    for name in ("e0", "e1", "e2"):
        artifact = joblib.load(
            model_dir / f"{name}_frozen_2000.joblib"
        )
        models[name] = artifact

        print(
            f"{name.upper()}: "
            f"threshold={artifact['threshold']:.9f}, "
            f"detector={type(artifact['detector']).__name__}"
        )

    print("Evaluation year:", args.year)

    # Load the evaluation year without modifying model parameters.
    base = paths["interim"] / "mission1" / "group8"

    telemetry = pd.read_pickle(
        base / f"telemetry_{args.year}.pkl"
    )

    windows = pd.read_csv(
        base / f"windows_{args.year}.csv",
        parse_dates=["window_start", "window_end"],
    )

    # Timestamps are stored as naive UTC.
    telemetry.index = telemetry.index.tz_localize("UTC")
    windows["window_start"] = windows["window_start"].dt.tz_localize("UTC")
    windows["window_end"] = windows["window_end"].dt.tz_localize("UTC")

    # All detectors must receive the same channels in the same order.
    for name, artifact in models.items():
        detector = artifact["detector"]

        if name == "e2":
            expected = detector.scaler_.columns_
        else:
            expected = detector.columns_

        if list(telemetry.columns) != expected:
            raise ValueError(
                f"{name.upper()}: channel order mismatch"
            )

    print("Telemetry samples:", len(telemetry))
    print("Evaluation windows:", len(windows))
    print("Channel alignment: OK")


    # E0: maximum robust score across channels and within each window.
    e0 = models["e0"]["detector"]

    e0_scores = (
        e0.score_samples(telemetry)
        .max(axis=1)
        .resample("30min")
        .max()
    )

    # E1: causal temporal reference, using only past observations.
    e1 = models["e1"]["detector"]

    reference = causal_rolling_median(
        telemetry,
        window="30min",
        min_periods=10,
    )

    residuals = telemetry - reference

    e1_scores = (
        e1.score_samples(residuals)
        .max(axis=1)
        .resample("30min")
        .max()
    )

    # E2: PCA reconstruction error.
    e2 = models["e2"]["detector"]

    e2_scores = (
        e2.score_samples(telemetry)
        .resample("30min")
        .max()
    )

    # Align scores with the same evaluation windows.
    for name, scores in [
        ("e0", e0_scores),
        ("e1", e1_scores),
        ("e2", e2_scores),
    ]:
        windows[f"{name}_score"] = windows["window_start"].map(scores)

        missing = windows[f"{name}_score"].isna().sum()

        print(f"{name.upper()} missing window scores: {missing}")

    # Apply the original frozen thresholds.
    for name in ("e0", "e1", "e2"):
        threshold = models[name]["threshold"]
        score = windows[f"{name}_score"]

        windows[f"{name}_alert"] = (
            score.gt(threshold)
            .astype("boolean")
            .mask(score.isna(), pd.NA)
        )

    # Compare alert rates by ESA window label.
    results = []

    for name in ("e0", "e1", "e2"):
        for label, group in windows.groupby("training_label"):
            results.append({
                "detector": name.upper(),
                "label": label,
                "windows": len(group),
                "alerts": int(group[f"{name}_alert"].sum()),
                "alert_fraction": group[f"{name}_alert"].mean(),
            })

    comparison = pd.DataFrame(results)

    print("\nE4 2001 comparison by ESA label:")
    print(comparison.to_string(index=False, float_format="%.3f"))

    # Monthly alert fractions outside ESA-annotated events.
    nominal = windows[
        windows["training_label"] == "NOMINAL"
    ].copy()

    nominal["month"] = (
        nominal["window_start"].dt.strftime("%Y-%m")
    )

    monthly = nominal.groupby("month")[
        ["e0_alert", "e1_alert", "e2_alert"]
    ].mean()

    print("\nMonthly NOMINAL alert fractions:")
    print(monthly.round(3).to_string())

    # Load ESA events using the same rules as annual window preparation.
    mission_root = paths["raw"] / "ESA-Mission1"

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )

    anomaly_types = pd.read_csv(
        mission_root / "anomaly_types.csv"
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

    events = events.merge(
        anomaly_types[["ID", "Category"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    # Match the UTC-aware timestamps used by the evaluation windows.
    for column in ("event_start", "event_end"):
        events[column] = pd.to_datetime(
            events[column], utc=True
        )

    year_start = pd.Timestamp(
        f"{args.year}-01-01", tz="UTC"
    )
    year_end = pd.Timestamp(
        f"{args.year + 1}-01-01", tz="UTC"
    )

    events = events[
        (events["event_start"] < year_end)
        & (events["event_end"] > year_start)
    ].copy()

    if events["Category"].isna().any():
        raise ValueError("ESA events with missing categories.")

    print("\nESA events by category:")
    print(events["Category"].value_counts().to_string())

    print("\nE4 2001 event-level detection:")

    for name in ("e0", "e1", "e2"):
        # Reuse the original E3 episode extraction.
        detector_windows = windows[
            ["window_start", "window_end", f"{name}_alert"]
        ].rename(columns={f"{name}_alert": "alert"})

        episodes = extract_alert_episodes(detector_windows)

        detection = compute_event_detection(
            events,
            episodes,
        )

        print(f"\n{name.upper()}")
        print("Alert episodes:", len(episodes))

        summary = (
            detection.groupby("Category")
            .agg(
                events=("detected", "size"),
                detected=("detected", "sum"),
                preexisting_alert=("preexisting_alert", "sum"),
            )
        )

        print(summary.to_string())

        anomaly_events = detection[
            detection["Category"] == "Anomaly"
        ]

        print("\nAnomaly event details:")
        print(
            anomaly_events[
                ["ID", "detected", "preexisting_alert"]
            ].to_string(index=False)
        )

        rare_events = detection[
            detection["Category"] == "Rare Event"
        ]

        print("\nRare Event details:")
        print(
            rare_events[
                [
                    "ID",
                    "event_start",
                    "detected",
                    "preexisting_alert",
                ]
            ].to_string(index=False)
        )

        latency = compute_detection_latency(
            events,
            detector_windows,
        )

        rare_latency = latency[
            latency["Category"] == "Rare Event"
        ]

        print("\nRare Event detection latency:")
        print(
            rare_latency[
                ["ID", "detected", "latency_minutes"]
            ].to_string(index=False)
        )

        anomaly_latency = latency[
            latency["Category"] == "Anomaly"
        ]

        print("\nAnomaly detection latency:")
        print(
            anomaly_latency[
                ["ID", "detected", "latency_minutes"]
            ].to_string(index=False)
        )

    # Save reproducible window-level evaluation results.
    output_dir = (
        paths["outputs"]
        / "experiments"
        / "e4"
        / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"comparison_{args.year}.csv"

    if output_file.exists():
        raise FileExistsError(
            f"Comparison already exists: {output_file}"
        )

    columns = [
        "window_start",
        "window_end",
        "training_label",
        "e0_score",
        "e0_alert",
        "e1_score",
        "e1_alert",
        "e2_score",
        "e2_alert",
    ]

    windows[columns].to_csv(output_file, index=False)

    print("\nSaved comparison:", output_file.name)
    print("Saved windows:", len(windows))

if __name__ == "__main__":
    main()
