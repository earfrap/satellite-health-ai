"""Generate publication-ready E4 temporal evaluation figures."""

import matplotlib.pyplot as plt
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    base = paths["outputs"] / "experiments" / "e4"
    data = pd.read_csv(base / "e4_yearly_summary.csv")

    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    png = figures / "e4_nominal_alert_fraction.png"
    pdf = figures / "e4_nominal_alert_fraction.pdf"

    for output in (png, pdf):
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing figure: {output}"
            )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    styles = {
        "E0": ("#2563eb", "o", "E0 — Robust threshold"),
        "E1": ("#16a34a", "s", "E1 — Temporal residual"),
        "E2": ("#d97706", "^", "E2 — PCA"),
    }

    for detector, (color, marker, label) in styles.items():
        subset = data[data["detector"] == detector].sort_values("year")

        ax.plot(
            subset["year"],
            subset["nominal_alert_pct"],
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )

    ax.set(
        title="Long-term detector behavior on ESA Mission 1 telemetry",
        xlabel="Evaluation year",
        ylabel="Alert fraction in ESA NOMINAL windows (%)",
        ylim=(0, 105),
    )

    ax.set_xticks(range(2001, 2014))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    plt.close(fig)

    print("Figures saved:")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()

