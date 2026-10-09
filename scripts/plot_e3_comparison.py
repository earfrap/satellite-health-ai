"""Plot nominal alert fractions for E0-E3, 2001-2013."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    root = paths["outputs"] / "experiments"
    years = range(2001, 2014)

    records = []

    for year in years:
        baseline = pd.read_csv(
            root / "e4" / "comparisons" / f"comparison_{year}.csv"
        )
        e3 = pd.read_csv(
            root / "e3" / "comparisons" / f"e3_{year}.csv"
        )

        for name, frame in [
            ("E0 — Robust threshold", baseline),
            ("E1 — Temporal residual", baseline),
            ("E2 — PCA", baseline),
            ("E3 — Isolation Forest", e3),
        ]:
            nominal = frame["training_label"] == "NOMINAL"
            scores = frame.loc[nominal, f"{name[:2].lower()}_score"]
            alerts = frame.loc[nominal, f"{name[:2].lower()}_alert"]

            valid = scores.notna()
            fraction = alerts[valid].astype(bool).mean() * 100

            records.append({
                "year": year,
                "detector": name,
                "nominal_alert_pct": fraction,
            })

    summary = pd.DataFrame(records)

    output_dir = root / "e3" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for name, group in summary.groupby("detector", sort=False):
        ax.plot(
            group["year"],
            group["nominal_alert_pct"],
            marker="o",
            markersize=3.5,
            linewidth=2,
            label=name,
        )

    ax.set(
        title="Long-term detector behaviour | ESA Mission 1, Group 8",
        xlabel="Year",
        ylabel="Alert fraction in NOMINAL windows (%)",
        ylim=(0, 105),
    )
    ax.set_xticks(list(years))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    for extension in ("png", "pdf"):
        path = output_dir / f"e3_nominal_alert_comparison.{extension}"
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Saved:", path)

    plt.close(fig)

    csv_path = root / "e3" / "nominal_alert_summary.csv"
    if csv_path.exists():
        raise FileExistsError(f"Refusing overwrite: {csv_path}")
    summary.to_csv(csv_path, index=False)
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
