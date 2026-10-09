"""Evaluate frozen Isolation Forest on one ESA telemetry year."""

import argparse

import joblib
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2001)
    args = parser.parse_args()

    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["interim"] / "mission1" / "group8"

    model_path = (
        paths["outputs"] / "experiments" / "e4"
        / "models" / "e3_frozen_2000.joblib"
    )
    artifact = joblib.load(model_path)
    detector = artifact["detector"]
    threshold = artifact["threshold"]

    telemetry = pd.read_pickle(base / f"telemetry_{args.year}.pkl")
    windows = pd.read_csv(
        base / f"windows_{args.year}.csv",
        parse_dates=["window_start", "window_end"],
    )

    telemetry.index = telemetry.index.tz_localize("UTC")
    windows["window_start"] = windows["window_start"].dt.tz_localize("UTC")
    windows["window_end"] = windows["window_end"].dt.tz_localize("UTC")

    scores = (
        detector.score_samples(telemetry)
        .resample("30min")
        .max()
    )

    result = windows[
        ["window_start", "window_end", "training_label"]
    ].copy()

    result["e3_score"] = result["window_start"].map(scores)
    result["e3_alert"] = (
        result["e3_score"] > threshold
    ).astype("boolean")

    result.loc[
        result["e3_score"].isna(), "e3_alert"
    ] = pd.NA

    output_dir = (
        paths["outputs"] / "experiments"
        / "e3" / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"e3_{args.year}.csv"

    if output_path.exists():
        raise FileExistsError(f"Refusing overwrite: {output_path}")

    result.to_csv(output_path, index=False)

    nominal = result["training_label"] == "NOMINAL"
    nominal_alert_fraction = result.loc[
        nominal, "e3_alert"
    ].mean()

    print("Year:", args.year)
    print("Windows:", len(result))
    print("Missing E3 scores:", result["e3_score"].isna().sum())
    print("NOMINAL alert fraction:", f"{nominal_alert_fraction:.2%}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
