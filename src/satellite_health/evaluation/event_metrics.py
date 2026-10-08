
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






def compute_event_detection(
    events: pd.DataFrame,
    episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Determine whether each event overlaps at least one alert episode."""

    required_events = {"event_start", "event_end"}
    required_episodes = {"episode_start", "episode_end"}

    if not required_events.issubset(events.columns):
        raise ValueError("Missing required event columns.")

    if not required_episodes.issubset(episodes.columns):
        raise ValueError("Missing required episode columns.")

    result = events.copy().reset_index(drop=True)

    detected = []
    first_episode_starts = []
    preexisting_alerts = []

    for event in result.itertuples(index=False):
        overlapping = episodes[
            (episodes["episode_start"] < event.event_end)
            & (episodes["episode_end"] > event.event_start)
        ]

        if overlapping.empty:
            detected.append(False)
            first_episode_starts.append(pd.NaT)
            preexisting_alerts.append(False)
        else:
            first_start = overlapping["episode_start"].min()

            detected.append(True)
            first_episode_starts.append(first_start)
            preexisting_alerts.append(
                bool(first_start < event.event_start)
            )

    result["detected"] = detected
    result["first_episode_start"] = first_episode_starts
    result["preexisting_alert"] = preexisting_alerts

    return result




def compute_detection_latency(
    events: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Compute event detection latency at alert-window closure."""

    required_events = {"event_start", "event_end"}
    required_windows = {"window_start", "window_end", "alert"}

    if not required_events.issubset(events.columns):
        raise ValueError("Missing required event columns.")

    if not required_windows.issubset(windows.columns):
        raise ValueError("Missing required window columns.")

    result = events.copy().reset_index(drop=True)
    alerted = windows.loc[windows["alert"]].copy()

    detected = []
    first_alert_times = []
    latencies = []

    for event in result.itertuples(index=False):
        overlapping = alerted[
            (alerted["window_start"] < event.event_end)
            & (alerted["window_end"] > event.event_start)
        ]

        if overlapping.empty:
            detected.append(False)
            first_alert_times.append(pd.NaT)
            latencies.append(float("nan"))
        else:
            first_alert_time = overlapping["window_end"].min()

            latency = (
                first_alert_time - event.event_start
            ).total_seconds() / 60.0

            detected.append(True)
            first_alert_times.append(first_alert_time)
            latencies.append(max(0.0, latency))

    result["detected"] = detected
    result["first_alert_time"] = first_alert_times
    result["latency_minutes"] = latencies

    return result
