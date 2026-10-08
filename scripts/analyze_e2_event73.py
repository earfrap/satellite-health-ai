"""Reproducible explainability analysis for ESA event id_73."""

import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.models.pca_detector import PCAAnomalyDetector


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    # Load the exact thresholds from the original experiments.
    experiments_dir = paths["outputs"] / "experiments"

    e0_summary = pd.read_csv(
        experiments_dir / "e0/summary_2000.csv"
    )

    e2_summary = pd.read_csv(
        experiments_dir / "e2/summary_2000.csv"
    )

    e0_threshold = float(e0_summary.loc[0, "threshold"])
    e2_threshold = float(e2_summary.loc[0, "threshold"])

    X = pd.read_pickle(
        paths["interim"] / "mission1/group8/telemetry_2000.pkl"
    )

    windows = pd.read_csv(
        paths["interim"] / "mission1/group8/windows_2000.csv",
        parse_dates=["window_start"],
    )

    print("Telemetry samples:", len(X))
    print("Windows:", len(windows))

    # Select nominal training windows (January–May 2000).
    train_windows = windows[
        (windows["window_start"] < "2000-06-01")
        & (windows["training_label"] == "NOMINAL")
    ]

    train_mask = X.index.floor("30min").isin(
        pd.DatetimeIndex(train_windows["window_start"])
    )

    X_train = X.loc[train_mask]

    # Fit the same PCA configuration used in E2.
    detector = PCAAnomalyDetector(n_components=2)
    detector.fit(X_train)

    print("Nominal training samples:", len(X_train))
    print(
        "Explained variance:",
        round(detector.pca_.explained_variance_ratio_.sum(), 4),
    )


    # ESA event id_73 reference interval.
    event_start = pd.Timestamp("2000-12-22 16:58:58.314", tz="UTC")
    event_end = pd.Timestamp("2000-12-24 21:52:58.314", tz="UTC")

    # Load the original E2 window-level scores.
    e2_windows = pd.read_csv(
        experiments_dir / "e2/window_scores_2000.csv",
        parse_dates=["window_start"],
    )

    # Find alerted windows overlapping the ESA event.
    window_duration = pd.Timedelta("30min")

    event_alerts = e2_windows.loc[
        (e2_windows["window_start"] < event_end)
        & (
            e2_windows["window_start"] + window_duration
            > event_start
        )
        & (e2_windows["e2_score"] > e2_threshold)
    ].sort_values("window_start")

    if event_alerts.empty:
        raise ValueError("No E2 alert overlaps ESA event id_73.")

    # E2 window scores are UTC-aware; telemetry index is UTC-naive.
    first_alert_start_utc = event_alerts.iloc[0]["window_start"]

    first_alert_start = first_alert_start_utc.tz_localize(None)
    first_alert_end = first_alert_start + window_duration


    assert first_alert_start == pd.Timestamp(
        "2000-12-22 17:00:00"
    ), "Unexpected first E2 alert window for event id_73."    


    X_alert = X.loc[
        (X.index >= first_alert_start)
        & (X.index < first_alert_end)
    ]

    # Compute sample-level PCA scores inside the alert window.
    sample_scores = detector.score_samples(X_alert)

    peak_time = sample_scores.idxmax()
    peak_score = sample_scores.max()

    print("First E2 alert window:", first_alert_start)
    print("Peak timestamp:", peak_time)
    print("Peak SPE:", round(peak_score, 6))
    print("E2 threshold:", e2_threshold)
    print("Above threshold:", bool(peak_score > e2_threshold))


    # PCA reconstruction error contribution by channel.
    contributions = detector.channel_contributions(
        X.loc[[peak_time]]
    ).iloc[0]

    contribution_percent = (
        100 * contributions / contributions.sum()
    )

    explanation = pd.DataFrame({
        "SPE_contribution": contributions,
        "Contribution_percent": contribution_percent,
    })

    explanation = explanation.sort_values(
        "SPE_contribution",
        ascending=False,
    )

    print("\nPCA channel contributions:")
    print(explanation.round(3).to_string())

    print(
        "\nSum of contributions:",
        round(contributions.sum(), 6),
    )




    # Reconstruct E0 robust deviations from nominal training.
    median = X_train.median()
    mad = (X_train - median).abs().median()
    robust_scale = 1.4826 * mad

    observation = X.loc[peak_time]

    e0_deviation = (
        (observation - median).abs() / robust_scale
    )

    # Add E0 information to the PCA explanation table.
    explanation["E0_robust_deviation"] = e0_deviation



    print("\nE0 vs E2 channel comparison:")
    print(explanation.round(3).to_string())

    print("\nE0 maximum deviation:", round(e0_deviation.max(), 3))
    print("E0 threshold:", e0_threshold)
    print("E2 SPE:", round(peak_score, 3))
    print("E2 threshold:", e2_threshold)

    # Consistency checks for the event id_73 case study.
    import numpy as np

    np.testing.assert_allclose(
        contributions.sum(),
        peak_score,
        rtol=1e-10,
    )

    assert peak_score > e2_threshold, (
        "Expected E2 to exceed its threshold."
    )

    assert e0_deviation.max() < e0_threshold, (
        "Expected E0 to remain below its threshold."
    )

    print("\nAll explainability checks passed.")

    # Save the reproducible channel-level explanation.
    output_dir = paths["outputs"] / "experiments/e2"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "event73_channel_explanation.csv"

    explanation.to_csv(
        output_file,
        index=True,
        index_label="channel",
    )

    print("\nExplanation saved to:", output_file)

if __name__ == "__main__":
    main()