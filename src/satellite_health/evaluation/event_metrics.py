
import pandas as pd


def extract_alert_episodes(windows: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive alerted windows into episodes."""

    required = {"window_start", "window_end", "alert"}
    if not required.issubset(windows.columns):
        raise ValueError("Missing required columns.")

    data = windows.sort_values("window_start").reset_index(drop=True)

    episodes = []
    active_start = None
    active_end = None
    count = 0

    for row in data.itertuples(index=False):
        if row.alert:
            if active_start is None:
                active_start = row.window_start
                active_end = row.window_end
                count = 1
            elif row.window_start == active_end:
                active_end = row.window_end
                count += 1
            else:
                episodes.append((active_start, active_end, count))
                active_start = row.window_start
                active_end = row.window_end
                count = 1

        elif active_start is not None:
            episodes.append((active_start, active_end, count))
            active_start = None
            active_end = None
            count = 0

    if active_start is not None:
        episodes.append((active_start, active_end, count))

    return pd.DataFrame(
        episodes,
        columns=["episode_start", "episode_end", "window_count"],
    )







def classify_alert_episodes(
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Classify alert episodes by temporal overlap with events."""

    result = episodes.copy()
    categories = []

    for episode in result.itertuples(index=False):
        overlapping = events[
            (events["event_start"] < episode.episode_end)
            & (events["event_end"] > episode.episode_start)
        ]

        if (overlapping["Category"] == "Anomaly").any():
            category = "ANOMALY"
        elif (overlapping["Category"] == "Rare Event").any():
            category = "RARE_EVENT"
        elif (overlapping["Category"] == "Communication Gap").any():
            category = "COMMUNICATION_GAP"
        else:
            category = "UNASSOCIATED"

        categories.append(category)

    result["association"] = categories
    return result

