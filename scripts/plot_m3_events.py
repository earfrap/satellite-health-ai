"""Compare M0-M3 detection of ESA Anomaly events."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    root = paths["outputs"] / "experiments" / "m3"

    events = pd.read_csv(root / "e3_event_detection.csv")
    anomalies = events[events["Category"] == "Anomaly"]

    if len(anomalies) != 46:
        raise ValueError("Expected 46 unique ESA Anomaly events.")

    names = ["M0 Threshold", "M1 Temporal", "M2 PCA", "M3 Isolation Forest"]
    overlap = [45, 42, 46, int(anomalies["overlap"].sum())]
    new = [6, 13, 2, int(anomalies["new_episode"].sum())]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, overlap, width, label="Event overlap")
    bars2 = ax.bar(x + width / 2, new, width, label="New alert episode")

    ax.bar_label(bars1, padding=3)
    ax.bar_label(bars2, padding=3)

    ax.set_xticks(x, names)
    ax.set_ylim(0, 52)
    ax.set_ylabel("ESA Anomaly events (out of 46)")
    ax.set_title("Anomaly event detection | ESA Mission 1, Group 8")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
    )
    fig.subplots_adjust(top=0.82)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for extension in ("png", "pdf"):
        path = output_dir / f"m3_anomaly_detection.{extension}"
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Saved:", path)

    plt.close(fig)


if __name__ == "__main__":
    main()
