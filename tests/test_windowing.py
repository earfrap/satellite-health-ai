import pandas as pd
import pytest


from satellite_health.preprocessing.windowing import (
    event_overlap_by_category,
    temporal_overlap_fraction,
    assign_training_label,
)




def test_full_overlap():
    assert temporal_overlap_fraction(
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
    ) == pytest.approx(1.0)


def test_partial_overlap():
    assert temporal_overlap_fraction(
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
        pd.Timestamp("2000-01-01 00:10"),
        pd.Timestamp("2000-01-01 00:20"),
    ) == pytest.approx(1 / 3)


def test_no_overlap():
    assert temporal_overlap_fraction(
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
        pd.Timestamp("2000-01-01 01:00"),
        pd.Timestamp("2000-01-01 02:00"),
    ) == pytest.approx(0.0)


def test_invalid_window():
    with pytest.raises(ValueError):
        temporal_overlap_fraction(
            pd.Timestamp("2000-01-01 00:30"),
            pd.Timestamp("2000-01-01 00:00"),
            pd.Timestamp("2000-01-01 00:00"),
            pd.Timestamp("2000-01-01 00:10"),
        )



def test_event_overlap_by_category():
    events = pd.DataFrame(
        {
            "event_start": [
                pd.Timestamp("2000-01-01 00:05"),
                pd.Timestamp("2000-01-01 00:20"),
            ],
            "event_end": [
                pd.Timestamp("2000-01-01 00:15"),
                pd.Timestamp("2000-01-01 00:30"),
            ],
            "Category": [
                "Anomaly",
                "Rare Event",
            ],
        }
    )

    overlaps = event_overlap_by_category(
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
        events,
    )

    assert overlaps["Anomaly"] == pytest.approx(1 / 3)
    assert overlaps["Rare Event"] == pytest.approx(1 / 3)


def test_event_overlap_by_category_no_events():
    events = pd.DataFrame(
        columns=[
            "event_start",
            "event_end",
            "Category",
        ]
    )

    overlaps = event_overlap_by_category(
        pd.Timestamp("2000-01-01 00:00"),
        pd.Timestamp("2000-01-01 00:30"),
        events,
    )

    assert overlaps == {}


def test_assign_training_label():
    assert assign_training_label(0.0, 0.0) == "NOMINAL"
    assert assign_training_label(0.8, 0.0) == "ANOMALY"
    assert assign_training_label(0.0, 0.8) == "RARE_EVENT"
    assert assign_training_label(0.2, 0.0) == "BOUNDARY"
    assert assign_training_label(0.0, 0.2) == "BOUNDARY"


def test_assign_training_label_threshold():
    assert (
        assign_training_label(
            anomaly_overlap=0.5,
            rare_event_overlap=0.0,
            threshold=0.5,
        )
        == "ANOMALY"
    )


def test_assign_training_label_invalid_threshold():
    with pytest.raises(ValueError):
        assign_training_label(
            anomaly_overlap=0.0,
            rare_event_overlap=0.0,
            threshold=1.5,
        )


