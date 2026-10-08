
import pandas as pd
from pathlib import Path


def load_group_events(
    labels_path: str | Path,
    types_path: str | Path,
    channels: list[str],
) -> pd.DataFrame:
    """Load and aggregate ESA events for selected channels."""

    labels = pd.read_csv(labels_path)
    types = pd.read_csv(types_path)

    group = labels[labels["Channel"].isin(channels)].copy()

    group["StartTime"] = pd.to_datetime(
        group["StartTime"], utc=True
    )
    group["EndTime"] = pd.to_datetime(
        group["EndTime"], utc=True
    )

    events = (
        group.groupby("ID", as_index=False)
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
            channel_count=("Channel", "nunique"),
        )
        .merge(
            types[["ID", "Category"]],
            on="ID",
            how="left",
            validate="one_to_one",
        )
    )

    if events["Category"].isna().any():
        raise ValueError("Some events have no category.")

    return events.sort_values("event_start").reset_index(drop=True)

