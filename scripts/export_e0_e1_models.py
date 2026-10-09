"""Export frozen E0 and E1 detectors trained on 2000 data."""

import joblib
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.models.robust_threshold import RobustThresholdDetector
from satellite_health.models.temporal_residual import (
    causal_rolling_median,
    TemporalResidualDetector,
)


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    base = paths["interim"] / "mission1" / "group8"

    telemetry = pd.read_pickle(base / "telemetry_2000.pkl")
    windows = pd.read_csv(
        base / "windows_2000.csv",
        parse_dates=["window_start", "window_end"],
    )

    telemetry.index = telemetry.index.tz_localize("UTC")
    windows["window_start"] = windows["window_start"].dt.tz_localize("UTC")
    windows["window_end"] = windows["window_end"].dt.tz_localize("UTC")

    print("Telemetry samples:", len(telemetry))
    print("Windows:", len(windows))



    # Original chronological training split.
    train_end = pd.Timestamp("2000-06-01", tz="UTC")

    train_windows = windows[
        (windows["window_start"] < train_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    train_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(train_windows["window_start"])
    )

    # E0: robust statistics on raw telemetry.
    e0 = RobustThresholdDetector().fit(
        telemetry.loc[train_mask]
    )

    # E1: robust statistics on causal temporal residuals.
    reference = causal_rolling_median(
        telemetry,
        window="30min",
        min_periods=10,
    )

    residuals = telemetry - reference

    e1 = TemporalResidualDetector().fit(
        residuals.loc[train_mask]
    )

    print("Nominal training windows:", len(train_windows))
    print("E0 channels:", e0.columns_)
    print("E1 channels:", e1.columns_)
    print("E0 fitted:", e0.median_ is not None)
    print("E1 fitted:", e1.median_ is not None)

    # Reproduce original 30-minute window scores.
    e0_scores = (
        e0.score_samples(telemetry)
        .max(axis=1)
        .resample("30min")
        .max()
    )

    e1_scores = (
        e1.score_samples(residuals)
        .max(axis=1)
        .resample("30min")
        .max()
    )

    windows["e0_score"] = windows["window_start"].map(e0_scores)
    windows["e1_score"] = windows["window_start"].map(e1_scores)

    # Original nominal validation period.
    val_end = pd.Timestamp("2000-09-01", tz="UTC")

    validation_nominal = windows[
        (windows["window_start"] >= train_end)
        & (windows["window_start"] < val_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    e0_threshold = validation_nominal["e0_score"].quantile(0.99)
    e1_threshold = validation_nominal["e1_score"].quantile(0.99)

    print(f"E0 threshold: {e0_threshold:.9f}")
    print(f"E1 threshold: {e1_threshold:.9f}")

    # Check completeness and reproduce original alert decisions.
    assert windows["e0_score"].notna().all()
    assert windows["e1_score"].notna().all()

    windows["e0_alert"] = windows["e0_score"] > e0_threshold
    windows["e1_alert"] = windows["e1_score"] > e1_threshold

    # Original chronological test period: September–December 2000.
    test = windows[
        windows["window_start"] >= val_end
    ]

    print("Test windows:", len(test))
    print("E0 test alert windows:", int(test["e0_alert"].sum()))
    print("E1 test alert windows:", int(test["e1_alert"].sum()))

    # Verify expected original test alert counts.
    assert int(test["e0_alert"].sum()) == 137
    assert int(test["e1_alert"].sum()) == 59

    # Export frozen models for temporal generalization.
    model_dir = paths["outputs"] / "experiments" / "e4" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, detector, threshold in [
        ("e0", e0, e0_threshold),
        ("e1", e1, e1_threshold),
    ]:
        artifact = {
            "detector": detector,
            "threshold": float(threshold),
            "training_period": "2000-01-01/2000-06-01",
            "validation_period": "2000-06-01/2000-09-01",
            "score_aggregation": "30min_max",
            "threshold_quantile": 0.99,
        }

        if name == "e1":
            artifact["temporal_reference"] = {
                "method": "causal_rolling_median",
                "window": "30min",
                "min_periods": 10,
                "closed": "left",
            }

        output_file = model_dir / f"{name}_frozen_2000.joblib"

        if output_file.exists():
            raise FileExistsError(
                f"Frozen model already exists: {output_file}"
            )

        joblib.dump(artifact, output_file)

        # Verify the saved artifact can be loaded.
        restored = joblib.load(output_file)

        assert restored["threshold"] == artifact["threshold"]
        assert restored["detector"].columns_ == detector.columns_

        print(f"{name.upper()} frozen model saved and verified")

if __name__ == "__main__":
    main()
