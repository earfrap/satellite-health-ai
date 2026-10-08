
"""Create comparison figures for E3."""

import matplotlib.pyplot as plt
import pandas as pd

from satellite_health.data.paths import load_data_paths


def main():
    paths = load_data_paths("configs/data/paths.local.yaml")

    output_dir = paths["outputs"] / "experiments" / "e3"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    operational = pd.read_csv(
        output_dir / "operational_comparison_2000.csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        operational["experiment"],
        operational["unassociated_episodes_per_day"],
    )

    ax.set_title("Operational alert burden — ESA Mission 1, Group 8")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Unassociated alert episodes per day")
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    figure_path = figures_dir / "unassociated_alert_burden_2000.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    print("Figure saved:", figure_path)



    # Compare detection latency for ESA events.
    events = pd.read_csv(
        output_dir / "event_comparison_2000.csv"
    )

    latency_columns = [
        "E0_latency_min",
        "E1_latency_min",
        "E2_latency_min",
    ]

    detected_events = events.dropna(
        subset=latency_columns,
        how="all",
    ).copy()

    ax = detected_events.plot(
        x="ID",
        y=latency_columns,
        kind="bar",
        figsize=(9, 5),
        rot=0,
    )

    ax.set_title("ESA event detection latency — E0 vs E1 vs E2")
    ax.set_xlabel("ESA event")
    ax.set_ylabel("Detection latency (minutes)")
    ax.set_ylim(bottom=0)
    ax.legend(["E0", "E1", "E2"])

    ax.text(
        0.99,
        0.97,
        "Undetected by all: id_35",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig = ax.figure
    fig.tight_layout()

    latency_path = figures_dir / "event_latency_comparison_2000.png"
    fig.savefig(latency_path, dpi=200)
    plt.close(fig)

    print("Figure saved:", latency_path)


    # Compare normalized detector scores around ESA event id_73.
    score_tables = []

    for experiment in ("e0", "e1", "e2"):
        scores = pd.read_csv(
            paths["outputs"]
            / "experiments"
            / experiment
            / "window_scores_2000.csv",
            parse_dates=["window_start"],
        )

        summary = pd.read_csv(
            paths["outputs"]
            / "experiments"
            / experiment
            / "summary_2000.csv",
        )

        threshold = float(summary.loc[0, "threshold"])
        score_column = f"{experiment}_score"

        scores = scores[["window_start", score_column]].copy()
        scores[f"{experiment.upper()}_normalized"] = (
            scores[score_column] / threshold
        )

        score_tables.append(
            scores[["window_start", f"{experiment.upper()}_normalized"]]
        )

    normalized = score_tables[0]

    for scores in score_tables[1:]:
        normalized = normalized.merge(
            scores,
            on="window_start",
            how="inner",
            validate="one_to_one",
        )


    # Verify that all detectors use identical evaluation windows.
    reference_windows = pd.DatetimeIndex(
        score_tables[0]["window_start"]
    )

    if reference_windows.has_duplicates:
        raise ValueError("Duplicate windows in E0 scores.")

    for experiment, scores in zip(
        ("E0", "E1", "E2"), score_tables
    ):
        current_windows = pd.DatetimeIndex(
            scores["window_start"]
        )

        if current_windows.has_duplicates:
            raise ValueError(
                f"Duplicate windows in {experiment} scores."
            )

        if not current_windows.equals(reference_windows):
            raise ValueError(
                f"Window alignment mismatch for {experiment}."
            )

    if len(normalized) != len(reference_windows):
        raise ValueError("Windows were lost during score merging.")

    print("Score window alignment: OK")

    start = pd.Timestamp("2000-12-22 12:00:00", tz="UTC")
    end = pd.Timestamp("2000-12-22 23:00:00", tz="UTC")
    event_start = pd.Timestamp(
        "2000-12-22 16:58:58.314", tz="UTC"
    )

    selected = normalized.loc[
        (normalized["window_start"] >= start)
        & (normalized["window_start"] < end)
    ].copy()

    if selected.empty:
        raise ValueError("No score windows found around event id_73.")

    fig, ax = plt.subplots(figsize=(11, 5))

    for experiment in ("E0", "E1", "E2"):
        ax.plot(
            selected["window_start"],
            selected[f"{experiment}_normalized"],
            marker="o",
            markersize=3,
            label=experiment,
        )

    ax.axhline(
        y=1.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Alert threshold",
    )

    ax.axvline(
        x=event_start,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label="ESA event start",
    )

    ax.set_title("Normalized detector scores around ESA event id_73")
    ax.set_xlabel("Window start (UTC)")
    ax.set_ylabel("Window score / validation threshold")
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    score_path = figures_dir / "event73_normalized_scores_2000.png"
    fig.savefig(score_path, dpi=200)
    plt.close(fig)

    print("Figure saved:", score_path)


if __name__ == "__main__":
    main()
