from pathlib import Path

import pandas as pd

from satellite_health.data.audit import audit_channel
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

    summaries = []

    for channel in GROUP_8_CHANNELS:
        print(f"Auditing {channel}...")
        summaries.append(
            audit_channel(channel_root / channel)
        )

    audit = pd.DataFrame(summaries)

    print("\n=== GROUP 8 TELEMETRY AUDIT ===")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
