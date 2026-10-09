"""Train and freeze M4 Autoencoder using ESA 2000 telemetry."""

import joblib
import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.data.telemetry_npz import load_telemetry_npz
from satellite_health.models.autoencoder import AutoencoderDetector


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["interim"] / "mission1" / "group8"

    telemetry = load_telemetry_npz(base / "telemetry_2000.npz")
    windows = pd.read_csv(
        base / "windows_2000.csv",
        parse_dates=["window_start", "window_end"],
    )

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

    validation_windows = windows[
        (windows["window_start"] >= train_end)
        & (windows["window_start"] < validation_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    validation_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(validation_windows["window_start"])
    )
    validation_data = telemetry.loc[validation_mask]

    if train_data.empty or validation_data.empty:
        raise ValueError("Training or validation data is empty.")

    output_dir = paths["outputs"] / "experiments" / "m4"
    model_dir = output_dir / "models"
    model_path = model_dir / "m4_frozen_2000.joblib"
    history_path = model_dir / "m4_training_history.csv"

    if model_path.exists() or history_path.exists():
        raise FileExistsError(
            "M4 outputs already exist. Refusing to overwrite."
        )

    detector = AutoencoderDetector(
        epochs=20,
        batch_size=1024,
        learning_rate=0.001,
    ).fit(train_data)

    validation_scores = (
        detector.score_samples(validation_data)
        .resample("30min")
        .max()
    )

    threshold = float(validation_scores.quantile(0.99))

    if not np.isfinite(threshold):
        raise ValueError("M4 validation threshold is not finite.")

    history = pd.DataFrame({
        "epoch": np.arange(1, len(detector.loss_history_) + 1),
        "training_mse": detector.loss_history_,
    })

    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"detector": detector, "threshold": threshold},
        model_path,
    )
    history.to_csv(history_path, index=False)

    print("M4 training samples:", len(train_data))
    print("M4 validation nominal windows:", len(validation_windows))
    print("M4 scorable validation windows:", validation_scores.notna().sum())
    print("M4 frozen threshold:", round(threshold, 9))
    print("M4 initial training MSE:", round(history.training_mse.iloc[0], 6))
    print("M4 final training MSE:", round(history.training_mse.iloc[-1], 6))
    print("Saved model:", model_path)
    print("Saved history:", history_path)


if __name__ == "__main__":
    main()
