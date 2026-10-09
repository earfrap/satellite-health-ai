import pandas as pd


from satellite_health.evaluation.event_metrics import (
    compute_event_detection,
    compute_detection_latency,
)

def test_event_detected_only_once_with_multiple_episodes():
    events = pd.DataFrame({
        "ID": [101],
        "event_start": [pd.Timestamp("2000-01-01 10:00", tz="UTC")],
        "event_end": [pd.Timestamp("2000-01-01 11:00", tz="UTC")],
        "Category": ["Anomaly"],
    })

    episodes = pd.DataFrame({
        "episode_start": [
            pd.Timestamp("2000-01-01 10:15", tz="UTC"),
            pd.Timestamp("2000-01-01 10:45", tz="UTC"),
        ],
        "episode_end": [
            pd.Timestamp("2000-01-01 10:30", tz="UTC"),
            pd.Timestamp("2000-01-01 11:00", tz="UTC"),
        ],
        "window_count": [1, 1],
    })

    result = compute_event_detection(events, episodes)

    assert len(result) == 1
    assert bool(result.loc[0, "detected"])
    assert result.loc[0, "first_episode_start"] == pd.Timestamp(
        "2000-01-01 10:15", tz="UTC"
    )




def test_event_not_detected():
    events = pd.DataFrame({
        "ID": [102],
        "event_start": [pd.Timestamp("2000-01-01 10:00", tz="UTC")],
        "event_end": [pd.Timestamp("2000-01-01 11:00", tz="UTC")],
    })

    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 12:00", tz="UTC")],
        "episode_end": [pd.Timestamp("2000-01-01 12:30", tz="UTC")],
    })

    result = compute_event_detection(events, episodes)

    assert not bool(result.loc[0, "detected"])
    assert pd.isna(result.loc[0, "first_episode_start"])


def test_touching_boundaries_are_not_detection():
    events = pd.DataFrame({
        "ID": [103],
        "event_start": [pd.Timestamp("2000-01-01 10:00", tz="UTC")],
        "event_end": [pd.Timestamp("2000-01-01 11:00", tz="UTC")],
    })

    episodes = pd.DataFrame({
        "episode_start": [pd.Timestamp("2000-01-01 09:30", tz="UTC")],
        "episode_end": [pd.Timestamp("2000-01-01 10:00", tz="UTC")],
    })

    result = compute_event_detection(events, episodes)

    assert not bool(result.loc[0, "detected"])




def test_operational_detection_latency():
    events = pd.DataFrame({
        "ID": [104],
        "event_start": [pd.Timestamp("2000-01-01 10:00", tz="UTC")],
        "event_end": [pd.Timestamp("2000-01-01 11:00", tz="UTC")],
    })

    windows = pd.DataFrame({
        "window_start": [
            pd.Timestamp("2000-01-01 10:15", tz="UTC"),
        ],
        "window_end": [
            pd.Timestamp("2000-01-01 10:30", tz="UTC"),
        ],
        "alert": [True],
    })

    result = compute_detection_latency(events, windows)

    assert bool(result.loc[0, "detected"])
    assert result.loc[0, "latency_minutes"] == 30.0



def test_latency_when_event_starts_inside_window():
    events = pd.DataFrame({
        "ID": [105],
        "event_start": [pd.Timestamp("2000-01-01 10:10", tz="UTC")],
        "event_end": [pd.Timestamp("2000-01-01 11:00", tz="UTC")],
    })

    windows = pd.DataFrame({
        "window_start": [
            pd.Timestamp("2000-01-01 10:00", tz="UTC")
        ],
        "window_end": [
            pd.Timestamp("2000-01-01 10:30", tz="UTC")
        ],
        "alert": [True],
    })

    result = compute_detection_latency(events, windows)

    assert bool(result.loc[0, "detected"])
    assert result.loc[0, "latency_minutes"] == 20.0




def test_latency_for_undetected_event():
    events = pd.DataFrame({
        "ID": [106],
        "event_start": [
            pd.Timestamp("2000-01-01 10:00", tz="UTC")
        ],
        "event_end": [
            pd.Timestamp("2000-01-01 11:00", tz="UTC")
        ],
    })

    windows = pd.DataFrame({
        "window_start": [
            pd.Timestamp("2000-01-01 12:00", tz="UTC")
        ],
        "window_end": [
            pd.Timestamp("2000-01-01 12:30", tz="UTC")
        ],
        "alert": [True],
    })

    result = compute_detection_latency(events, windows)

    assert not bool(result.loc[0, "detected"])
    assert pd.isna(result.loc[0, "first_alert_time"])
    assert pd.isna(result.loc[0, "latency_minutes"])



def test_latency_with_preexisting_alert_window():
    events = pd.DataFrame({
        "ID": [107],
        "event_start": [
            pd.Timestamp("2000-01-01 10:10", tz="UTC")
        ],
        "event_end": [
            pd.Timestamp("2000-01-01 11:00", tz="UTC")
        ],
    })

    windows = pd.DataFrame({
        "window_start": [
            pd.Timestamp("2000-01-01 10:00", tz="UTC")
        ],
        "window_end": [
            pd.Timestamp("2000-01-01 10:30", tz="UTC")
        ],
        "alert": [True],
    })

    result = compute_detection_latency(events, windows)

    assert bool(result.loc[0, "detected"])
    assert result.loc[0, "latency_minutes"] == 20.0


def test_preexisting_alert_episode():
    events = pd.DataFrame({
        "ID": [108],
        "event_start": [
            pd.Timestamp("2000-01-01 10:10", tz="UTC")
        ],
        "event_end": [
            pd.Timestamp("2000-01-01 11:00", tz="UTC")
        ],
    })

    episodes = pd.DataFrame({
        "episode_start": [
            pd.Timestamp("2000-01-01 09:30", tz="UTC")
        ],
        "episode_end": [
            pd.Timestamp("2000-01-01 10:30", tz="UTC")
        ],
        "window_count": [2],
    })

    result = compute_event_detection(events, episodes)

    assert bool(result.loc[0, "detected"])
    assert bool(result.loc[0, "preexisting_alert"])



def test_alert_episode_starts_after_event():
    events = pd.DataFrame({
        "ID": [109],
        "event_start": [
            pd.Timestamp("2000-01-01 10:00", tz="UTC")
        ],
        "event_end": [
            pd.Timestamp("2000-01-01 11:00", tz="UTC")
        ],
    })

    episodes = pd.DataFrame({
        "episode_start": [
            pd.Timestamp("2000-01-01 10:15", tz="UTC")
        ],
        "episode_end": [
            pd.Timestamp("2000-01-01 10:45", tz="UTC")
        ],
        "window_count": [1],
    })

    result = compute_event_detection(events, episodes)

    assert bool(result.loc[0, "detected"])
    assert not bool(result.loc[0, "preexisting_alert"])



def test_detection_latency_ignores_unscorable_window():
    start = pd.Timestamp("2020-01-01", tz="UTC")

    windows = pd.DataFrame({
        "window_start": pd.date_range(
            start, periods=3, freq="30min"
        ),
        "alert": pd.array(
            [pd.NA, False, True],
            dtype="boolean",
        ),
    })

    windows["window_end"] = (
        windows["window_start"] + pd.Timedelta(minutes=30)
    )

    events = pd.DataFrame({
        "event_start": [start],
        "event_end": [start + pd.Timedelta(hours=2)],
    })

    result = compute_detection_latency(events, windows)

    assert bool(result.iloc[0]["detected"])
    assert result.iloc[0]["first_alert_time"] == (
        start + pd.Timedelta(minutes=90)
    )
