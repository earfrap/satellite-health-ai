
import pandas as pd

from satellite_health.evaluation.event_metrics import extract_alert_episodes


def make_windows(alerts, starts=None):
    if starts is None:
        starts = pd.date_range(
            "2000-01-01", periods=len(alerts), freq="30min"
        )

    return pd.DataFrame({
        "window_start": starts,
        "window_end": starts + pd.Timedelta(minutes=30),
        "alert": alerts,
    })


def test_consecutive_alerts_form_one_episode():
    windows = make_windows([False, True, True, True, False])

    episodes = extract_alert_episodes(windows)

    assert len(episodes) == 1
    assert episodes.iloc[0]["window_count"] == 3
    assert episodes.iloc[0]["episode_start"] == pd.Timestamp(
        "2000-01-01 00:30:00"
    )
    assert episodes.iloc[0]["episode_end"] == pd.Timestamp(
        "2000-01-01 02:00:00"
    )


def test_nominal_window_separates_episodes():
    windows = make_windows([True, True, False, True, False])

    episodes = extract_alert_episodes(windows)

    assert len(episodes) == 2
    assert episodes["window_count"].tolist() == [2, 1]


def test_no_alerts_produce_no_episodes():
    windows = make_windows([False, False, False])

    episodes = extract_alert_episodes(windows)

    assert episodes.empty
    assert list(episodes.columns) == [
        "episode_start", "episode_end", "window_count"
    ]


def test_time_gap_separates_episodes():
    starts = pd.to_datetime([
        "2000-01-01 00:00:00",
        "2000-01-01 00:30:00",
        "2000-01-01 02:00:00",
    ])

    windows = make_windows([True, True, True], starts=starts)

    episodes = extract_alert_episodes(windows)

    assert len(episodes) == 2
    assert episodes["window_count"].tolist() == [2, 1]






from satellite_health.evaluation.event_metrics import classify_alert_episodes


def make_events():
    return pd.DataFrame({
        "event_start": pd.to_datetime([
            "2000-01-01 01:00",
            "2000-01-01 03:00",
        ]),
        "event_end": pd.to_datetime([
            "2000-01-01 02:00",
            "2000-01-01 04:00",
        ]),
        "Category": ["Anomaly", "Rare Event"],
    })


def test_episode_associated_with_anomaly():
    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 01:30")],
        "episode_end": [pd.Timestamp("2000-01-01 02:30")],
        "window_count": [2],
    })

    result = classify_alert_episodes(episodes, make_events())

    assert result.iloc[0]["association"] == "ANOMALY"


def test_episode_associated_with_rare_event():
    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 03:30")],
        "episode_end": [pd.Timestamp("2000-01-01 04:30")],
        "window_count": [2],
    })

    result = classify_alert_episodes(episodes, make_events())

    assert result.iloc[0]["association"] == "RARE_EVENT"


def test_episode_without_event_is_unassociated():
    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 05:00")],
        "episode_end": [pd.Timestamp("2000-01-01 05:30")],
        "window_count": [1],
    })

    result = classify_alert_episodes(episodes, make_events())

    assert result.iloc[0]["association"] == "UNASSOCIATED"


def test_anomaly_takes_priority_over_rare_event():
    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 01:30")],
        "episode_end": [pd.Timestamp("2000-01-01 03:30")],
        "window_count": [4],
    })

    result = classify_alert_episodes(episodes, make_events())

    assert result.iloc[0]["association"] == "ANOMALY"

