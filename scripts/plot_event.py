import matplotlib.pyplot as plt
import pandas as pd

from satellite_health.data.paths import load_data_paths


EVENT_ID = "id_35"
CONTEXT_HOURS = 6


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    telemetry_path = (
        paths["interim"]
        / "mission1"
        / "group8"
        / "telemetry_2000.pkl"
    )

    mission_root = paths["raw"] / "ESA-Mission1"

    telemetry = pd.read_pickle(telemetry_path)

    labels = pd.read_csv(
        mission_root / "labels.csv",
        parse_dates=["StartTime", "EndTime"],
    )

    event_rows = labels[labels["ID"] == EVENT_ID]
   
    event_rows = event_rows[
    event_rows["Channel"].isin(telemetry.columns)
    ]
   
    event_start = event_rows["StartTime"].min().tz_localize(None)
    event_end = event_rows["EndTime"].max().tz_localize(None)

    plot_start = event_start - pd.Timedelta(hours=CONTEXT_HOURS)
    plot_end = event_end + pd.Timedelta(hours=CONTEXT_HOURS)

    window = telemetry.loc[plot_start:plot_end]

    print(f"Event: {EVENT_ID}")
    print(f"Event start: {event_start}")
    print(f"Event end:   {event_end}")
    print(f"Window samples: {len(window):,}")

    axes = window.plot(
        subplots=True,
        figsize=(14, 10),
        sharex=True,
        legend=True,
    )

    for ax in axes:
        ax.axvspan(
            event_start,
            event_end,
            alpha=0.2,
        )
        ax.grid(True, alpha=0.3)

    axes[0].set_title(
        f"Mission1 Group 8 — {EVENT_ID}"
    )

    axes[-1].set_xlabel("Time")

    plt.tight_layout()

    output_dir = paths["outputs"] / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{EVENT_ID}.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved figure to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
