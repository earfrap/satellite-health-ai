from pathlib import Path

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


def main() -> None:
    paths = load_data_paths("configs/data/paths.local.yaml")

    channel_root = (
        paths["raw"]
        / "ESA-Mission1"
        / "extracted_channels"
    )

    reference_path = channel_root / GROUP_8_CHANNELS[0]
    reference_index = pd.read_pickle(reference_path).index

    print(f"Reference: {GROUP_8_CHANNELS[0]}")
    print(f"Timestamps: {len(reference_index):,}")

    for channel in GROUP_8_CHANNELS[1:]:
        print(f"Checking {channel}...")

        current_index = pd.read_pickle(
            channel_root / channel
        ).index

        identical = reference_index.equals(current_index)

        print(f"  identical index: {identical}")

        if not identical:
            common = reference_index.intersection(current_index)
            print(f"  common timestamps: {len(common):,}")
            print(
                f"  reference-only: "
                f"{len(reference_index.difference(current_index)):,}"
            )
            print(
                f"  channel-only: "
                f"{len(current_index.difference(reference_index)):,}"
            )


if __name__ == "__main__":
    main()
