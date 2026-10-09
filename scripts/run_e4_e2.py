
"""Apply the frozen E2 PCA detector to Group 8 telemetry."""

import argparse

import joblib
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        choices=range(2001, 2014),
    )
    args = parser.parse_args()

    paths = load_data_paths("configs/data/paths.local.yaml")

    model_path = (
        paths["outputs"]
        / "experiments"
        / "e4"
        / "models"
        / "e2_frozen_2000.joblib"
    )

    artifact = joblib.load(model_path)
    detector = artifact["detector"]
    threshold = artifact["threshold"]

    telemetry_path = (
        paths["interim"]
        / "mission1"
        / "group8"
        / f"telemetry_{args.year}.pkl"
    )

    telemetry = pd.read_pickle(telemetry_path)

    # ESA timestamps are UTC; telemetry uses naive UTC timestamps.
    telemetry.index = telemetry.index.tz_localize("UTC")

    print("Year:", args.year)
    print("Telemetry shape:", telemetry.shape)
    print("Frozen threshold:", threshold)

    # Apply the previously fitted PCA and scaler.
    # No fit() or threshold recalibration is performed.
    scores = detector.score_samples(telemetry)

    window_scores = scores.resample("30min").max()

    alerts = window_scores > threshold

    print("Scored windows:", len(window_scores))
    print("Alerted windows:", int(alerts.sum()))
    print("Alert fraction:", float(alerts.mean()))


    monthly = pd.DataFrame({
        "score": window_scores,
        "alert": alerts,
    })

    monthly["month"] = monthly.index.to_period("M")

    monthly_summary = monthly.groupby("month").agg(
        windows=("alert", "size"),
        alerts=("alert", "sum"),
        alert_fraction=("alert", "mean"),
    )

    print("\nMonthly alert summary:")
    print(monthly_summary.to_string())


    windows_path = (
        paths["interim"]
        / "mission1"
        / "group8"
        / f"windows_{args.year}.csv"
    )

    windows = pd.read_csv(
        windows_path,
        parse_dates=["window_start"],
    )

    windows["window_start"] = pd.to_datetime(
        windows["window_start"],
        utc=True,
    )

    windows["alert"] = windows["window_start"].map(alerts)

    if windows["alert"].isna().any():
        raise ValueError("Missing alert decisions for ESA windows")

    nominal = windows.loc[
        windows["training_label"] == "NOMINAL"
    ].copy()

    nominal["month"] = nominal["window_start"].dt.to_period("M")

    nominal_summary = nominal.groupby("month").agg(
        nominal_windows=("alert", "size"),
        nominal_alerts=("alert", "sum"),
        nominal_alert_fraction=("alert", "mean"),
    )

    print("\nMonthly NOMINAL alert summary:")
    print(nominal_summary.to_string())

    contributions = detector.channel_contributions(telemetry)

    contributions["month"] = contributions.index.to_period("M")

    monthly_contributions = (
        contributions.groupby("month")
        .median()
    )

    print("\nMonthly median PCA channel contributions:")
    print(monthly_contributions.to_string())

    monthly_samples = (
        telemetry.groupby(telemetry.index.to_period("M"))
        .size()
    )

    print("\nMonthly telemetry sample counts:")
    print(monthly_samples.to_string())

    early = telemetry.loc[
        "2001-01-01":"2001-02-28"
    ]

    late = telemetry.loc[
        "2001-04-01":"2001-12-31"
    ]

    comparison = pd.DataFrame({
        "Jan-Feb median": early.median(),
        "Apr-Dec median": late.median(),
    })

    comparison["Difference"] = (
        comparison["Apr-Dec median"]
        - comparison["Jan-Feb median"]
    )

    print("\nChannel median comparison:")
    print(comparison.to_string())

    early_mad = (
        early - early.median()
    ).abs().median()

    late_mad = (
        late - late.median()
    ).abs().median()

    variability = pd.DataFrame({
        "Jan-Feb MAD": early_mad,
        "Apr-Dec MAD": late_mad,
    })

    variability["MAD ratio"] = (
        variability["Apr-Dec MAD"]
        / variability["Jan-Feb MAD"]
    )

    print("\nChannel variability comparison:")
    print(variability.to_string())


    early_corr = early.corr()
    late_corr = late.corr()

    corr_difference = late_corr - early_corr

    print("\nJan-Feb correlation matrix:")
    print(early_corr.round(3).to_string())

    print("\nApr-Dec correlation matrix:")
    print(late_corr.round(3).to_string())

    print("\nCorrelation changes (Apr-Dec minus Jan-Feb):")
    print(corr_difference.round(3).to_string())

    print("\nMonthly channel_41/channel_42 correlation:")

    for month, group in telemetry.groupby(
        telemetry.index.to_period("M")
    ):
        correlation = group[
            ["channel_41", "channel_42"]
        ].corr().iloc[0, 1]

        print(month, round(correlation, 3))

    scaled = detector.scaler_.transform(telemetry)

    monthly_scaled_medians = scaled.groupby(
        scaled.index.to_period("M")
    ).median()

    print("\nMonthly medians in frozen E2 scaled units:")
    print(monthly_scaled_medians.round(3).to_string())

    daily_alerts = alerts.resample("1D").mean()

    transition = daily_alerts.loc[
        "2001-03-20":"2001-04-15"
    ]

    print("\nDaily alert fraction during transition:")
    print(transition.round(3).to_string())

    transition_windows = windows.loc[
        (windows["window_start"] >= "2001-03-20")
        & (windows["window_start"] < "2001-04-16")
    ].copy()

    transition_windows["day"] = (
        transition_windows["window_start"].dt.date
    )

    daily_labels = pd.crosstab(
        transition_windows["day"],
        transition_windows["training_label"],
    )

    print("\nDaily ESA labels during transition:")
    print(daily_labels.to_string())


    nominal_transition = windows.loc[
        (windows["training_label"] == "NOMINAL")
        & (windows["window_start"] >= "2001-04-01")
        & (windows["window_start"] < "2001-04-21")
    ].copy()

    nominal_transition["day"] = (
        nominal_transition["window_start"].dt.date
    )

    nominal_daily = nominal_transition.groupby("day").agg(
        nominal_windows=("alert", "size"),
        nominal_alerts=("alert", "sum"),
        nominal_alert_fraction=("alert", "mean"),
    )

    print("\nDaily NOMINAL alerts after March event:")
    print(nominal_daily.to_string())

if __name__ == "__main__":
    main()
