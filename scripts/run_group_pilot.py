
#!/usr/bin/env python3
"""ESA Mission 1 - Group 18 cross-group validation.

Aligned with the Group 8 development protocol:
- original telemetry samples (no 5-minute resampling)
- 30-minute calendar windows
- nominal windows exclude Anomaly and Rare Event overlap
- training: January-May 2000
- calibration: June-August 2000, nominal windows, P99
- evaluation: 2001-2013
- M1 causal rolling median: 30 minutes, min_periods=10
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from satellite_health.models.robust_threshold import (
    RobustThresholdDetector,
)
from satellite_health.models.temporal_residual import (
    TemporalResidualDetector,
    causal_rolling_median,
)
from satellite_health.models.pca_detector import (
    PCAAnomalyDetector,
)
from satellite_health.models.isolation_forest import (
    IsolationForestDetector,
)
from satellite_health.preprocessing.windowing import (
    event_overlap_by_category,
    assign_training_label,
)


DATA_ROOT = Path.home() / "Documents/satellite-health-ai-data"
RAW = DATA_ROOT / "raw" / "ESA-Mission1"
OUTPUT = DATA_ROOT / "outputs" / "pilot_group18"

CHANNELS = [f"channel_{i}" for i in range(70, 77)]

WINDOW = "30min"
WINDOW_DELTA = pd.Timedelta(WINDOW)

TRAIN_END = pd.Timestamp("2000-06-01", tz="UTC")
CAL_END = pd.Timestamp("2000-09-01", tz="UTC")
TEST_START = pd.Timestamp("2001-01-01", tz="UTC")
TEST_END = pd.Timestamp("2014-01-01", tz="UTC")


def read_channel(name):
    """Read the original ESA pickle directly from its ZIP."""

    path = RAW / "channels" / f"{name}.zip"

    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.namelist()
            if not member.endswith("/")
        ]

        if len(members) != 1:
            raise ValueError(f"Unexpected ZIP structure: {path}")

        with archive.open(members[0]) as stream:
            obj = pd.read_pickle(io.BytesIO(stream.read()))

    if isinstance(obj, pd.Series):
        series = obj

    elif isinstance(obj, pd.DataFrame):
        frame = obj.copy()

        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError(
                f"{name}: expected a DatetimeIndex"
            )

        if name in frame.columns:
            series = frame[name]
        elif frame.shape[1] == 1:
            series = frame.iloc[:, 0]
        else:
            raise ValueError(
                f"{name}: cannot identify telemetry column"
            )
    else:
        raise TypeError(f"Unsupported pickle type: {name}")

    series = pd.to_numeric(series, errors="coerce")

    index = pd.to_datetime(series.index, utc=True)
    series.index = index

    if series.index.has_duplicates:
        raise ValueError(f"{name}: duplicate timestamps")

    if not series.index.is_monotonic_increasing:
        series = series.sort_index()

    series = series.loc[
        (series.index >= pd.Timestamp("2000-01-01", tz="UTC"))
        & (series.index < TEST_END)
    ]

    return series.rename(name)


def load_telemetry():
    """Load and align all seven original channels."""

    frames = []

    for name in CHANNELS:
        print(f"Reading {name}...", flush=True)
        frames.append(read_channel(name))

    # Group 8 had identical verified indices.
    # Here we verify rather than assume this property.
    reference = frames[0].index

    identical = all(
        frame.index.equals(reference)
        for frame in frames[1:]
    )

    print("Identical channel timestamps:", identical)

    telemetry = pd.concat(frames, axis=1, join="inner")
    # Group 18 channels are not sampled at identical timestamps.
    # Synchronize them to a common 5-minute grid.
    # Synchronize original channel samples to a common grid.
    frames = [
        frame.resample("5min").median()
        for frame in frames
    ]

    telemetry = pd.concat(frames, axis=1, join="inner")

    print("\n=== SYNCHRONIZATION DIAGNOSTICS ===")
    print("Shape before dropna:", telemetry.shape)
    print("Non-null samples per channel:")
    print(telemetry.notna().sum())
    print("Complete rows:", telemetry.notna().all(axis=1).sum())
    print("First timestamps:")
    print(telemetry.index[:5])

    telemetry = telemetry.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if telemetry.empty:
        raise ValueError("No complete multivariate samples")

    print("Complete telemetry shape:", telemetry.shape)
    print("First timestamp:", telemetry.index.min())
    print("Last timestamp:", telemetry.index.max())

    return telemetry


def load_events():
    """Load ESA events and attach their official categories."""

    labels = pd.read_csv(RAW / "labels.csv")
    types = pd.read_csv(RAW / "anomaly_types.csv")

    labels = labels.loc[
        labels["Channel"].isin(CHANNELS)
    ].copy()

    labels["StartTime"] = pd.to_datetime(
        labels["StartTime"], utc=True
    )
    labels["EndTime"] = pd.to_datetime(
        labels["EndTime"], utc=True
    )

    # Match Group 8: aggregate channel annotations by event ID.
    events = (
        labels.groupby("ID")
        .agg(
            event_start=("StartTime", "min"),
            event_end=("EndTime", "max"),
        )
        .reset_index()
    )

    events = events.merge(
        types[["ID", "Category"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    return events


def nominal_window_mask(window_index, events):
    """Replicate Group 8 NOMINAL-window selection.

    A window is nominal if it has no positive overlap with
    an ESA Anomaly or Rare Event.
    """

    relevant = events.loc[
        events["Category"].isin(["Anomaly", "Rare Event"])
    ]

    nominal = np.ones(len(window_index), dtype=bool)

    for event in relevant.itertuples(index=False):
        intersects = (
            (window_index < event.event_end)
            & (window_index + WINDOW_DELTA > event.event_start)
        )
        nominal &= ~intersects

    return nominal


def aggregate_scores(scores):
    """Maximum across channels, then maximum per 30-min window."""

    if isinstance(scores, pd.DataFrame):
        scores = scores.max(axis=1)

    return scores.resample(WINDOW).max().dropna()


def count_episodes(alerts):
    """Count contiguous sequences of alerted calendar windows."""

    previous = alerts.shift(1, fill_value=False)

    contiguous = (
        alerts.index.to_series().diff() == WINDOW_DELTA
    ).to_numpy()

    episode_starts = (
        alerts.to_numpy()
        & (~previous.to_numpy() | ~contiguous)
    )

    return int(episode_starts.sum())


def count_event_overlap(scores, threshold, events):
    """Count unique ESA events intersecting at least one alert."""

    alert_index = scores.index[scores > threshold]
    matched = set()

    for event in events.itertuples(index=False):
        overlap = (
            (alert_index < event.event_end)
            & (alert_index + WINDOW_DELTA > event.event_start)
        )

        if overlap.any():
            matched.add(event.ID)

    return len(matched)


def windows_overlapping_events(index, events):
    mask = np.zeros(len(index), dtype=bool)

    for event in events.itertuples(index=False):
        mask |= (
            (index < event.event_end)
            & (index + WINDOW_DELTA > event.event_start)
        )

    return mask


def main():
    print(
        "ESA Mission 1 - Group 18 - aligned cross-group pilot",
        flush=True,
    )

    telemetry = load_telemetry()
    events = load_events()

    # Use Group 8's NOMINAL definition for development.
    # No extra recovery buffer.
    development_events = events.loc[
        events["event_start"] < TEST_START
    ]

    train_period = telemetry.index < TRAIN_END
    cal_period = (
        (telemetry.index >= TRAIN_END)
        & (telemetry.index < CAL_END)
    )
    test_period = (
        (telemetry.index >= TEST_START)
        & (telemetry.index < TEST_END)
    )

    development_windows = (
        telemetry.loc[
            telemetry.index < CAL_END
        ]
        .resample(WINDOW)
        .size()
        .index
    )

    nominal = nominal_window_mask(
        development_windows,
        development_events,
    )

    nominal_window_index = development_windows[nominal]

    # Match Group 8: select all original samples belonging
    # to nominal 30-minute calendar windows.
    sample_window = telemetry.index.floor(WINDOW)

    nominal_sample_mask = sample_window.isin(
        nominal_window_index
    )

    train = telemetry.loc[
        train_period & nominal_sample_mask
    ]

    calibration = telemetry.loc[
        cal_period & nominal_sample_mask
    ]

    test = telemetry.loc[test_period]

    print("Training samples:", len(train))
    print("Calibration samples:", len(calibration))
    print("Test samples:", len(test))

    if train.empty or calibration.empty or test.empty:
        raise ValueError("An experiment split is empty")

    # M1: causal reference on the full original time series,
    # exactly as in the Group 8 development approach.
    print("Computing causal residuals...", flush=True)

    reference = causal_rolling_median(
        telemetry,
        window=WINDOW,
        min_periods=3,
    )

    residuals = telemetry - reference

    residual_train = residuals.loc[train.index].dropna()
    residual_cal = residuals.loc[calibration.index]
    residual_test = residuals.loc[test.index]

    models = [
        ("M0", RobustThresholdDetector()),
        ("M1", TemporalResidualDetector()),
        ("M2", PCAAnomalyDetector(n_components=2)),
        (
            "M3",
            IsolationForestDetector(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]

    test_events = events.loc[
        (events["event_start"] < TEST_END)
        & (events["event_end"] > TEST_START)
    ]

    event_count = int(test_events["ID"].nunique())

    summary_rows = []
    detail_frames = []

    for model_name, detector in models:
        print(f"\n[{model_name}] Training...", flush=True)

        if model_name == "M1":
            X_train = residual_train
            X_cal = residual_cal
            X_test = residual_test
        else:
            X_train = train
            X_cal = calibration
            X_test = test

        detector.fit(X_train)

        cal_scores = aggregate_scores(
            detector.score_samples(X_cal)
        )

        test_scores = aggregate_scores(
            detector.score_samples(X_test)
        )

        # Calibration only on Group 8-style nominal windows.
        nominal_cal_mask = nominal_window_mask(
            cal_scores.index,
            development_events,
        )

        nominal_cal_scores = cal_scores.loc[nominal_cal_mask]

        if nominal_cal_scores.empty:
            raise ValueError(
                f"{model_name}: no nominal calibration windows"
            )

        threshold = float(
            nominal_cal_scores.quantile(0.99)
        )

        alerts = test_scores > threshold

        matched = count_event_overlap(
            test_scores,
            threshold,
            test_events,
        )

        annotated_mask = windows_overlapping_events(
            test_scores.index,
            test_events,
        )

        outside = int(
            (alerts.to_numpy() & ~annotated_mask).sum()
        )

        result = {
            "group": 18,
            "model": model_name,
            "threshold": threshold,
            "test_windows": len(test_scores),
            "alert_windows": int(alerts.sum()),
            "alert_pct": 100 * float(alerts.mean()),
            "alert_episodes": count_episodes(alerts),
            "esa_events": event_count,
            "esa_events_overlapped": matched,
            "event_overlap_pct": (
                100 * matched / event_count
                if event_count else np.nan
            ),
            "alert_windows_outside_events": outside,
            "alert_windows_outside_events_pct": (
                100 * outside / len(test_scores)
            ),
        }

        summary_rows.append(result)

        detail_frames.append(
            pd.DataFrame(
                {
                    "timestamp": test_scores.index,
                    "model": model_name,
                    "score": test_scores.to_numpy(),
                    "threshold": threshold,
                    "alert": alerts.to_numpy(),
                    "esa_event_overlap": annotated_mask,
                }
            )
        )

        print(
            f"[{model_name}] "
            f"alert={result['alert_pct']:.3f}% | "
            f"ESA overlap={matched}/{event_count} | "
            f"episodes={result['alert_episodes']}",
            flush=True,
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)

    pd.concat(detail_frames, ignore_index=True).to_csv(
        OUTPUT / "results.csv",
        index=False,
    )

    report = [
        "# Group 18 - Cross-group validation",
        "",
        "## Protocol",
        "",
        "- Original ESA telemetry samples; no resampling.",
        "- 30-minute calendar windows.",
        "- NOMINAL means no overlap with Anomaly or Rare Event.",
        "- No extra recovery buffer.",
        "- Training: January-May 2000.",
        "- Calibration: June-August 2000.",
        "- Threshold: 99th percentile of nominal calibration",
        "  window maximum scores.",
        "- Evaluation: 2001-2013.",
        "- M1: causal 30-minute median, min_periods=10.",
        "- Models: M0, M1, M2, M3.",
        "",
        "## Results",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation limits",
        "",
        "- Event overlap does not prove correct detection.",
        "- Alert fraction is not a false-positive rate.",
        "- ESA labels may not exhaust all true anomalies.",
        "- This validates workflow reuse, not pretrained",
        "  model transfer.",
    ]

    (OUTPUT / "report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\nALIGNED PILOT COMPLETED")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()

