"""Train and freeze E3 Isolation Forest using 2000 telemetry."""

import joblib
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.models.isolation_forest import IsolationForestDetector


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

    train_end = pd.Timestamp("2000-06-01", tz="UTC")
    validation_end = pd.Timestamp("2000-09-01", tz="UTC")

    train_windows = windows[
        (windows["window_start"] < train_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    train_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(train_windows["window_start"])
    )
    train_data = telemetry.loc[train_mask]

    detector = IsolationForestDetector().fit(train_data)

    validation_windows = windows[
        (windows["window_start"] >= train_end)
        & (windows["window_start"] < validation_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    validation_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(validation_windows["window_start"])
    )
    validation_data = telemetry.loc[validation_mask]

    validation_scores = (
        detector.score_samples(validation_data)
        .resample("30min")
        .max()
    )

    threshold = float(validation_scores.quantile(0.99))

    if not pd.notna(threshold):
        raise ValueError("E3 validation threshold is not finite.")

    output_dir = paths["outputs"] / "experiments" / "e4" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "e3_frozen_2000.joblib"

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing model: {output_path}"
        )

    joblib.dump(
        {"detector": detector, "threshold": threshold},
        output_path,
    )

    print("E3 training samples:", len(train_data))
    print("E3 validation nominal windows:", len(validation_windows))
    print("E3 scorable validation windows:", validation_scores.notna().sum())
    print("E3 frozen threshold:", round(threshold, 9))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
