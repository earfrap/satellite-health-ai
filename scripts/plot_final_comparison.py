"""Generate final M0-M4 scientific comparison figures from saved CSVs."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from satellite_health.data.paths import load_data_paths


MODELS = {
    "M0": ("e0", "Robust threshold", "#2563eb", "o"),
    "M1": ("e1", "Temporal residual", "#16a34a", "s"),
    "M2": ("e2", "PCA", "#d97706", "^"),
    "M3": ("e3", "Isolation Forest", "#9333ea", "D"),
    "M4": ("m4", "Autoencoder", "#dc2626", "v"),
}


def load_results(paths):
    root = paths["outputs"] / "experiments"
    annual_rows = []
    event_rows = []

    for year in range(2001, 2014):
        shared = pd.read_csv(
            root / "eval" / "comparisons" / f"comparison_{year}.csv"
        )

        for model, (key, _, _, _) in MODELS.items():
            if key in ("e0", "e1", "e2"):
                frame = shared
            else:
                frame = pd.read_csv(
                    root / ("m3" if key == "e3" else key) / "comparisons" / f"{key}_{year}.csv"
                )

            nominal = frame["training_label"] == "NOMINAL"
            valid = nominal & frame[f"{key}_score"].notna()
            alerts = frame.loc[valid, f"{key}_alert"].astype("boolean")

            if not valid.any():
                raise ValueError(f"No scorable NOMINAL windows: {model}, {year}")

            annual_rows.append({
                "year": year,
                "model": model,
                "nominal_alert_pct": 100 * alerts.mean(),
                "nominal_windows": int(valid.sum()),
                "nominal_alerts": int(alerts.sum()),
            })

    for model, (key, _, _, _) in MODELS.items():
        if key in ("e0", "e1", "e2"):
            path = (
                root / "eval"
                / f"{key}_event_detection_continuous.csv"
            )
        else:
            path = (
                root / ("m3" if key == "e3" else key)
                / f"{key}_event_detection_continuous.csv"
            )

        frame = pd.read_csv(path)

        if len(frame) != 108 or frame["ID"].duplicated().any():
            raise ValueError(f"Unexpected ESA event set for {model}")

        frame["model"] = model
        event_rows.append(frame)

    annual = pd.DataFrame(annual_rows)
    events = pd.concat(event_rows, ignore_index=True)

    reference_ids = set(events.loc[events["model"] == "M0", "ID"])
    for model in MODELS:
        ids = set(events.loc[events["model"] == model, "ID"])
        if ids != reference_ids:
            raise ValueError(f"ESA event IDs differ for {model}")

    return annual, events


def save_figure(fig, folder, name):
    targets = [folder / f"{name}.{ext}" for ext in ("png", "pdf")]

    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite: {target}")

    for target in targets:
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print("Saved:", target)

    plt.close(fig)


def temporal_stability(annual, folder):
    fig, ax = plt.subplots(figsize=(11, 5.8))

    for model, (_, label, color, marker) in MODELS.items():
        subset = annual[annual["model"] == model].sort_values("year")
        ax.plot(
            subset["year"],
            subset["nominal_alert_pct"],
            label=f"{model} — {label}",
            color=color,
            marker=marker,
            linewidth=2,
            markersize=5,
        )

    ax.set(
        title="Long-term detector behavior | ESA Mission 1, Group 8",
        xlabel="Evaluation year",
        ylabel="NOMINAL windows with alert (%)",
        ylim=(0, 105),
    )
    ax.set_xticks(range(2001, 2014))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    save_figure(fig, folder, "fig01_temporal_stability")


def event_detection(events, folder):
    categories = [
        ("All events", events),
        ("Anomaly", events[events["Category"] == "Anomaly"]),
        ("Rare Event", events[events["Category"] == "Rare Event"]),
        (
            "Communication Gap",
            events[events["Category"] == "Communication Gap"],
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(MODELS))
    width = 0.36

    for ax, (title, subset) in zip(axes.flat, categories):
        grouped = subset.groupby("model").agg(
            n=("ID", "count"),
            overlap=("overlap", "sum"),
            new=("new_episode", "sum"),
        ).reindex(MODELS)

        overlap = 100 * grouped["overlap"] / grouped["n"]
        new = 100 * grouped["new"] / grouped["n"]

        ax.bar(
            x - width / 2, overlap, width,
            color="#2563eb", label="Overlapping alert",
        )
        ax.bar(
            x + width / 2, new, width,
            color="#16a34a", label="New alert episode",
        )

        ax.set_title(f"{title} (n={int(grouped['n'].iloc[0])})")
        ax.set_xticks(x, list(MODELS))
        ax.set_ylim(0, 110)
        ax.set_ylabel("Events (%)")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    axes[0, 0].legend(frameon=False)
    fig.suptitle("ESA event coverage: overlap vs new alert episodes")
    fig.tight_layout()

    save_figure(fig, folder, "fig02_event_detection")


def operational_tradeoff(annual, events, folder):
    # Weighted across all scorable NOMINAL windows, not an
    # unweighted average of annual percentages.
    summary = annual.groupby("model").agg(
        nominal_windows=("nominal_windows", "sum"),
        nominal_alerts=("nominal_alerts", "sum"),
    )
    summary["alert_pct"] = (
        100 * summary["nominal_alerts"] / summary["nominal_windows"]
    )

    event_summary = events.groupby("model").agg(
        n=("ID", "count"),
        overlap=("overlap", "sum"),
        new=("new_episode", "sum"),
    )

    summary["overlap_pct"] = (
        100 * event_summary["overlap"] / event_summary["n"]
    )
    summary["new_pct"] = (
        100 * event_summary["new"] / event_summary["n"]
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for model, (_, _, color, _) in MODELS.items():
        row = summary.loc[model]

        ax.scatter(
            row["alert_pct"], row["overlap_pct"],
            s=130, color=color, marker="o",
        )
        ax.scatter(
            row["alert_pct"], row["new_pct"],
            s=130, color=color, marker="^",
        )
        ax.annotate(
            model,
            (row["alert_pct"], row["overlap_pct"]),
            xytext=(6, 5),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="None",
                   color="black", label="Event overlap"),
            Line2D([], [], marker="^", linestyle="None",
                   color="black", label="New alert episode"),
        ],
        frameon=False,
    )

    ax.set(
        title="Detection coverage versus NOMINAL alert burden",
        xlabel="NOMINAL windows with alert, 2001–2013 (%)",
        ylabel="ESA events covered (%)",
        xlim=(0, 105),
        ylim=(0, 105),
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    save_figure(fig, folder, "fig03_operational_tradeoff")
    return summary


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")
    folder = (
        paths["outputs"] / "experiments"
        / "eval" / "final_comparison" / "figures"
    )

    annual, events = load_results(paths)

    folder.mkdir(parents=True, exist_ok=True)

    temporal_stability(annual, folder)
    event_detection(events, folder)
    summary = operational_tradeoff(annual, events, folder)

    annual.to_csv(folder.parent / "annual_nominal_summary.csv", index=False)
    summary.to_csv(folder.parent / "model_tradeoff_summary.csv")

    print("\nFinal comparison summary:")
    print(
        summary[
            ["alert_pct", "overlap_pct", "new_pct"]
        ].round(2).to_string()
    )


if __name__ == "__main__":
    main()
