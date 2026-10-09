
"""Export the frozen E2 detector trained on 2000 nominal data."""

import pandas as pd
import joblib
from satellite_health.data.paths import load_data_paths
from satellite_health.models.pca_detector import PCAAnomalyDetector


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    base = paths["interim"] / "mission1" / "group8"

    telemetry = pd.read_pickle(
        base / "telemetry_2000.pkl"
    )

    windows = pd.read_csv(
        base / "windows_2000.csv",
        parse_dates=["window_start", "window_end"],
    )

    windows["window_start"] = (
        windows["window_start"].dt.tz_localize("UTC")
    )

    telemetry.index = telemetry.index.tz_localize("UTC")

    train_end = pd.Timestamp("2000-06-01", tz="UTC")

    train_windows = windows[
        (windows["window_start"] < train_end)
        & (windows["training_label"] == "NOMINAL")
    ]

    train_mask = telemetry.index.floor("30min").isin(
        pd.DatetimeIndex(train_windows["window_start"])
    )

    train_data = telemetry.loc[train_mask]

    detector = PCAAnomalyDetector(
        n_components=2
    ).fit(train_data)

    print("Training nominal windows:", len(train_windows))
    print("Training samples:", len(train_data))
    print(
        "Explained variance ratio:",
        detector.pca_.explained_variance_ratio_,
    )

    # Verify reconstructed scores against the original E2 experiment.
    original = pd.read_csv(
        paths["outputs"]
        / "experiments"
        / "e2"
        / "window_scores_2000.csv",
        parse_dates=["window_start"],
    )

    reconstructed_scores = (
        detector.score_samples(telemetry)
        .resample("30min")
        .max()
    )

    print(
        "Reconstructed score dtype:",
        reconstructed_scores.dtype,
    )

    print(
        "PCA components dtype:",
        detector.pca_.components_.dtype,
    )

    reconstructed = original["window_start"].map(
        reconstructed_scores
    )

    import numpy as np

    if reconstructed.isna().any():
        raise ValueError("Missing reconstructed E2 scores.")

    original_scores = original["e2_score"].to_numpy()
    new_scores = reconstructed.to_numpy()

    max_difference = np.max(
        np.abs(original_scores - new_scores)
    )

    equivalent = np.allclose(
        original_scores,
        new_scores,
        rtol=1e-6,
        atol=1e-10,
    )

    print("Maximum score difference:", max_difference)
    print("E2 scores equivalent:", equivalent)

    differences = np.abs(original_scores - new_scores)

    relative_differences = differences / np.maximum(
        np.abs(original_scores),
        1.0,
    )

    print(
        "Maximum relative difference:",
        relative_differences.max(),
    )

    print(
        "Windows with difference > 1e-6:",
        int((differences > 1e-6).sum()),
    )

    print(
        "Original maximum score:",
        original_scores.max(),
    )

    summary = pd.read_csv(
        paths["outputs"]
        / "experiments"
        / "e2"
        / "summary_2000.csv"
    )

    threshold = float(summary.loc[0, "threshold"])

    original_alerts = original_scores > threshold
    reconstructed_alerts = new_scores > threshold

    disagreements = int(
        (original_alerts != reconstructed_alerts).sum()
    )

    print("Frozen E2 threshold:", threshold)
    print("Alert disagreements:", disagreements)

    # Check whether CSV rounding explains score differences.
    print(
        "Equivalent at 6 significant digits:",
        np.array_equal(
            original_scores,
            np.array([
                float(f"{score:.6g}")
                for score in new_scores
            ]),
        ),
    )

    if not equivalent:
        raise ValueError(
            "Reconstructed E2 scores exceed float32 tolerance."
        )

    if disagreements != 0:
        raise ValueError(
            "Reconstructed E2 alert decisions differ."
        )

    print("E2 reconstruction verified.")


    output_dir = paths["outputs"] / "experiments" / "e4" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "e2_frozen_2000.joblib"

    if output_path.exists():
        raise FileExistsError(
            f"Frozen model already exists: {output_path}"
        )

    artifact = {
        "detector": detector,
        "threshold": threshold,
        "training_period": "2000-01-01/2000-06-01",
        "validation_period": "2000-06-01/2000-09-01",
        "n_components": 2,
        "score_aggregation": "30min_max",
        "threshold_quantile": 0.99,
        "score_comparison_rtol": 1e-6,
        "score_comparison_atol": 1e-6,
        "alert_disagreements_2000": disagreements,
    }

    joblib.dump(artifact, output_path)

    print("Frozen E2 model saved:", output_path)

if __name__ == "__main__":
    main()
