import pandas as pd

from satellite_health.data.paths import load_data_paths


GROUP_8_CHANNELS = [
    "channel_41",
    "channel_42",
    "channel_43",
    "channel_44",
    "channel_45",
    "channel_46",
]

START = pd.Timestamp("2000-01-01")
END = pd.Timestamp("2001-01-01")


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    source_root = (
        paths["raw"]
        / "ESA-Mission1"
        / "extracted_channels"
    )

    output_dir = paths["interim"] / "mission1" / "group8"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []

    for channel in GROUP_8_CHANNELS:
        print(f"Loading {channel}...")

        df = pd.read_pickle(source_root / channel)

        subset = df.loc[
            (df.index >= START) & (df.index < END)
        ]

        frames.append(subset)

    # Safe because we already verified that Group 8 indices are identical.
    telemetry = pd.concat(frames, axis=1)

    print("\n=== DEVELOPMENT SLICE ===")
    print("Shape:", telemetry.shape)
    print("Start:", telemetry.index.min())
    print("End:", telemetry.index.max())
    print("Columns:", telemetry.columns.tolist())
    print("Missing values:")
    print(telemetry.isna().sum())

    print(
        f"Memory: "
        f"{telemetry.memory_usage(deep=True).sum() / 1024**2:.1f} MiB"
    )

    output_path = output_dir / "telemetry_2000.pkl"
    telemetry.to_pickle(output_path)

    print(f"\nSaved to: {output_path}")
    print(
        f"File size: "
        f"{output_path.stat().st_size / 1024**2:.1f} MiB"
    )


if __name__ == "__main__":
    main()
