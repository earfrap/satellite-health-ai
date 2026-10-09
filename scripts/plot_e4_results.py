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

def plot_event_detection(base):
    data = pd.read_csv(
        base / "e4_event_detection_with_latency_2001_2013.csv"
    )

    anomalies = data[data["category"] == "Anomaly"]

    summary = (
        anomalies.groupby("detector")
        .agg(
            events=("event_id", "nunique"),
            detected=("detected", "sum"),
            new_episode=("new_episode_during_event", "sum"),
        )
        .reindex(["E0", "E1", "E2"])
    )

    coverage = 100 * summary["detected"] / summary["events"]
    new = 100 * summary["new_episode"] / summary["events"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(3)
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        coverage,
        width,
        label="Overlapping alert",
        color="#2563eb",
    )

    ax.bar(
        [i + width / 2 for i in x],
        new,
        width,
        label="New alert episode",
        color="#16a34a",
    )

    ax.set_xticks(list(x), ["E0", "E1", "E2"])
    ax.set_ylim(0, 110)
    ax.set_ylabel("ESA anomaly events (%)")
    ax.set_title("Anomaly coverage vs new alert episodes")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    figures = base / "figures"

    for extension in ("png", "pdf"):
        output = figures / f"e4_anomaly_detection.{extension}"

        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing figure: {output}"
            )

        fig.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
        print("Saved:", output)

    plt.close(fig)


if __name__ == "__main__":
    paths = load_data_paths("configs/data/paths.local.yaml")
    base = paths["outputs"] / "experiments" / "e4"
    plot_event_detection(base)
