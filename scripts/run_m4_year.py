"""Evaluate frozen M4 Autoencoder on one ESA telemetry year."""

import argparse

import joblib
import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths
from satellite_health.data.telemetry_npz import load_telemetry_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["interim"] / "mission1" / "group8"

    model_path = (
        paths["outputs"] / "experiments" / "m4"
        / "models" / "m4_frozen_2000.joblib"
    )

    artifact = joblib.load(model_path)
    detector = artifact["detector"]
    threshold = artifact["threshold"]

    telemetry = load_telemetry_npz(
        base / f"telemetry_{args.year}.npz"
    )
    windows = pd.read_csv(
        base / f"windows_{args.year}.csv",
        parse_dates=["window_start", "window_end"],
    )

    windows["window_start"] = (
        windows["window_start"].dt.tz_localize("UTC")
    )
    windows["window_end"] = (
        windows["window_end"].dt.tz_localize("UTC")
    )

    scores = (
        detector.score_samples(telemetry)
        .resample("30min")
        .max()
    )

    result = windows[
        ["window_start", "window_end", "training_label"]
    ].copy()

    result["m4_score"] = result["window_start"].map(scores)

    result["m4_alert"] = (
        result["m4_score"] > threshold
    ).astype("boolean")

    result.loc[
        result["m4_score"].isna(), "m4_alert"
    ] = pd.NA

    output_dir = (
        paths["outputs"] / "experiments"
        / "m4" / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"m4_{args.year}.csv"

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {output_path}"
        )

    result.to_csv(output_path, index=False)

    nominal = result["training_label"] == "NOMINAL"
    nominal_alert_fraction = result.loc[
        nominal, "m4_alert"
    ].mean()

    print(
        f"{args.year}: "
        f"{len(result)} windows, "
        f"{result['m4_score'].isna().sum()} missing, "
        f"NOMINAL alerts {nominal_alert_fraction:.2%}"
    )


if __name__ == "__main__":
    main()
