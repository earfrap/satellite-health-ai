"""Summarize E4 temporal generalization across 2001–2013."""

import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    base = paths["outputs"] / "experiments" / "eval"
    comparison_dir = base / "comparisons"

    rows = []

    for year in range(2001, 2014):
        file = comparison_dir / f"comparison_{year}.csv"
        df = pd.read_csv(file)

        if df["window_start"].duplicated().any():
            raise ValueError(f"Duplicate windows in {year}")

        for name in ("e0", "e1", "e2"):
            score = df[f"{name}_score"]
            nominal = df["training_label"] == "NOMINAL"
            valid = score.notna()
            nominal_valid = nominal & valid

            alerts = df[f"{name}_alert"].astype("boolean")

            rows.append({
                "year": year,
                "detector": name.upper(),
                "total_windows": len(df),
                "valid_windows": int(valid.sum()),
                "unscorable_windows": int((~valid).sum()),
                "coverage_pct": round(100 * valid.mean(), 2),
                "nominal_alert_pct": round(
                    100 * alerts.loc[nominal_valid].mean(), 2
                ),
            })

    summary = pd.DataFrame(rows)

    output = base / "e4_yearly_summary.csv"

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output}"
        )

    summary.to_csv(output, index=False)

    print("E4 yearly summary saved:")
    print(output)
    print("\nYears:", summary["year"].nunique())
    print("Detectors:", summary["detector"].unique().tolist())
    print("Rows:", len(summary))


if __name__ == "__main__":
    main()
