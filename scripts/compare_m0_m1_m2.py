
"""Compare M0, M1 and M2 experiment summaries."""

import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    experiments_dir = paths["outputs"] / "experiments"

    summaries = []

    for experiment in ("m0", "m1", "m2"):
        summary_path = (
            experiments_dir / experiment / "summary_2000.csv"
        )

        summary = pd.read_csv(summary_path)
        if len(summary) != 1:
            raise ValueError(
                f"Expected one summary row for {experiment}."
            )

        summaries.append(summary)

    comparison = pd.concat(summaries, ignore_index=True)

    protocol_columns = [
        "year",
        "train_end_utc",
        "validation_end_utc",
        "test_end_utc",
        "threshold_quantile",
        "test_windows",
    ]

    for column in protocol_columns:
        if comparison[column].nunique() != 1:
            raise ValueError(
                f"Inconsistent evaluation protocol: {column}"
            )

    print("=== EVAL: PROTOCOL VALIDATION ===")
    print("Experiments:", comparison["experiment"].tolist())
    print("Common test windows:", comparison["test_windows"].iloc[0])
    print("Common threshold quantile:",
          comparison["threshold_quantile"].iloc[0])
    print("Protocol consistency: OK")
    # Operational comparison on the common test period.
    operational_columns = [
        "experiment",
        "test_windows",
        "alerted_windows",
        "total_episodes",
        "anomaly_episodes",
        "rare_event_episodes",
        "unassociated_episodes",
        "unassociated_episodes_per_day",
    ]

    operational = comparison[operational_columns].copy()

    operational["alerted_window_pct"] = (
        100
        * operational["alerted_windows"]
        / operational["test_windows"]
    )

    # Validate consistency of operational metrics.
    classified_total = (
        operational["anomaly_episodes"]
        + operational["rare_event_episodes"]
        + operational["unassociated_episodes"]
    )

    if not classified_total.equals(operational["total_episodes"]):
        raise ValueError("Episode categories do not sum to total episodes.")

    if (
        operational["alerted_windows"] > operational["test_windows"]
    ).any():
        raise ValueError("Alerted windows exceed test windows.")

    if (
        operational["total_episodes"] < 0
    ).any():
        raise ValueError("Negative episode count.")

    print("Operational metric consistency: OK")


    print("\n=== EVAL: OPERATIONAL COMPARISON ===")
    print(operational.round(3).to_string(index=False))



    # Compare detection and latency for the same ESA events.
    reference_events = pd.read_csv(
        experiments_dir / "m0" / "test_event_metrics_2000.csv"
    ).set_index("ID")

    if reference_events.index.has_duplicates:
        raise ValueError("Duplicate ESA event IDs in M0.")
    event_comparison = reference_events[
        ["Category", "event_start", "event_end"]
    ].copy()

    for experiment in ("m0", "m1", "m2"):
        events = pd.read_csv(
            experiments_dir
            / experiment
            / "test_event_metrics_2000.csv"
        ).set_index("ID")

        if set(events.index) != set(reference_events.index):
            raise ValueError(
                f"Different ESA events in {experiment}."
            )

        for column in ("Category", "event_start", "event_end"):
            if not events[column].equals(
                reference_events[column].reindex(events.index)
            ):
                raise ValueError(
                    f"Inconsistent event metadata in {experiment}: "
                    f"{column}"
                )

        label = experiment.upper()

        event_comparison[f"{label}_detected"] = events["detected"]
        event_comparison[f"{label}_latency_min"] = (
            events["latency_minutes"]
        )

    print("\n=== EVAL: EVENT-LEVEL COMPARISON ===")
    print(event_comparison.round(2).to_string())


    # Event-level coverage and latency summary.
    event_summary_rows = []

    for experiment in ("M0", "M1", "M2"):
        detected = event_comparison[f"{experiment}_detected"]
        latency = event_comparison[f"{experiment}_latency_min"]

        anomaly_mask = event_comparison["Category"] == "Anomaly"
        rare_mask = event_comparison["Category"] == "Rare Event"

        if latency[~detected].notna().any():
            raise ValueError(
                f"{experiment}: undetected events have latency values."
            )

        if latency[detected].isna().any():
            raise ValueError(
                f"{experiment}: detected events have missing latency."
            )

        if (latency[detected] < 0).any():
            raise ValueError(
                f"{experiment}: negative detection latency."
            )

        event_summary_rows.append({
            "experiment": experiment,
            "total_events": len(detected),
            "detected_events": int(detected.sum()),
            "anomaly_events": int(anomaly_mask.sum()),
            "detected_anomalies": int(detected[anomaly_mask].sum()),
            "rare_events": int(rare_mask.sum()),
            "detected_rare_events": int(detected[rare_mask].sum()),
            "median_latency_min": latency[detected].median(),
        })

    event_summary = pd.DataFrame(event_summary_rows)

    print("\n=== EVAL: EVENT-LEVEL SUMMARY ===")
    print(event_summary.round(3).to_string(index=False))


    # Save reproducible E3 comparison tables.
    output_dir = experiments_dir / "eval" / "development_2000"
    output_dir.mkdir(parents=True, exist_ok=True)

    operational.to_csv(
        output_dir / "operational_comparison_2000.csv",
        index=False,
    )

    event_comparison.to_csv(
        output_dir / "event_comparison_2000.csv",
        index=True,
        index_label="ID",
    )

    event_summary.to_csv(
        output_dir / "event_summary_2000.csv",
        index=False,
    )

    print("\n=== EVAL: OUTPUT FILES ===")
    print("Saved:", output_dir)
    print("Operational comparison: OK")
    print("Event comparison: OK")
    print("Event summary: OK")

if __name__ == "__main__":
    main()
