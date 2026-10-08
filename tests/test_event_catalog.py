
import pandas as pd

from satellite_health.data.event_catalog import load_group_events


def test_load_group_events(tmp_path):
    labels = pd.DataFrame({
        "ID": ["id_1", "id_1", "id_2", "id_3"],
        "Channel": [
            "channel_41",
            "channel_42",
            "channel_41",
            "channel_10",
        ],
        "StartTime": [
            "2000-01-01T01:00:00Z",
            "2000-01-01T00:30:00Z",
            "2000-01-02T03:00:00Z",
            "2000-01-03T04:00:00Z",
        ],
        "EndTime": [
            "2000-01-01T02:00:00Z",
            "2000-01-01T02:30:00Z",
            "2000-01-02T04:00:00Z",
            "2000-01-03T05:00:00Z",
        ],
    })

    types = pd.DataFrame({
        "ID": ["id_1", "id_2", "id_3"],
        "Category": ["Anomaly", "Rare Event", "Anomaly"],
    })

    labels_path = tmp_path / "labels.csv"
    types_path = tmp_path / "anomaly_types.csv"

    labels.to_csv(labels_path, index=False)
    types.to_csv(types_path, index=False)

    result = load_group_events(
        labels_path,
        types_path,
        ["channel_41", "channel_42"],
    )

    assert len(result) == 2
    assert set(result["ID"]) == {"id_1", "id_2"}

    event_1 = result.set_index("ID").loc["id_1"]

    assert event_1["channel_count"] == 2
    assert event_1["Category"] == "Anomaly"
    assert event_1["event_start"] == pd.Timestamp(
        "2000-01-01 00:30:00", tz="UTC"
    )
    assert event_1["event_end"] == pd.Timestamp(
        "2000-01-01 02:30:00", tz="UTC"
    )


def test_missing_category_raises_error(tmp_path):
    labels = pd.DataFrame({
        "ID": ["id_1"],
        "Channel": ["channel_41"],
        "StartTime": ["2000-01-01T01:00:00Z"],
        "EndTime": ["2000-01-01T02:00:00Z"],
    })

    types = pd.DataFrame({
        "ID": ["id_2"],
        "Category": ["Anomaly"],
    })

    labels_path = tmp_path / "labels.csv"
    types_path = tmp_path / "anomaly_types.csv"

    labels.to_csv(labels_path, index=False)
    types.to_csv(types_path, index=False)

    import pytest

    with pytest.raises(ValueError, match="no category"):
        load_group_events(
            labels_path,
            types_path,
            ["channel_41"],
        )

