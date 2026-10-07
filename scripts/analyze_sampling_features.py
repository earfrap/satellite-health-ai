import pandas as pd

from satellite_health.data.paths import load_data_paths


WINDOW = "30min"


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry = pd.read_pickle(
        paths["interim"]
        / "mission1"
        / "group8"
        / "telemetry_2000.pkl"
    )

    index_series = pd.Series(
        telemetry.index,
        index=telemetry.index,
    )

    rows = []

    for window_start, timestamps in index_series.resample(WINDOW):
        if len(timestamps) == 0:
            continue

        dt = timestamps.diff().dt.total_seconds().dropna()

        rows.append(
            {
                "window_start": window_start,
                "sample_count": len(timestamps),
                "median_dt_s": dt.median(),
                "mean_dt_s": dt.mean(),
                "max_gap_s": dt.max(),
            }
        )

    features = pd.DataFrame(rows).set_index("window_start")

    print("=== TEMPORAL SAMPLING FEATURES ===")

    print("\nSummary:")
    print(features.describe().round(3))

    print("\nNormal-density windows (sample_count = 63):")
    print(
        features.loc[
            features["sample_count"] == 63
        ].describe().round(3)
    )

    print("\nLow-density windows (sample_count < 55):")
    print(
        features.loc[
            features["sample_count"] < 55
        ].describe().round(3)
    )

    output_dir = (
        paths["outputs"]
        / "eda"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "sampling_features_2000.csv"
    )

    features.to_csv(output_path)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
