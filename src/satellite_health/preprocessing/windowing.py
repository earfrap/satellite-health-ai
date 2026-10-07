import pandas as pd


def temporal_overlap_fraction(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
) -> float:
    """Return the fraction of a window covered by an event."""

    overlap_start = max(window_start, event_start)
    overlap_end = min(window_end, event_end)

    overlap_seconds = max(
        0.0,
        (overlap_end - overlap_start).total_seconds(),
    )

    window_seconds = (
        window_end - window_start
    ).total_seconds()

    if window_seconds <= 0:
        raise ValueError("Window duration must be positive.")

    return overlap_seconds / window_seconds






def event_overlap_by_category(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    events: pd.DataFrame,
) -> dict[str, float]:
    """Return maximum window overlap fraction for each event category."""

    overlaps: dict[str, float] = {}

    for event in events.itertuples(index=False):
        fraction = temporal_overlap_fraction(
            window_start,
            window_end,
            event.event_start,
            event.event_end,
        )

        if fraction <= 0:
            continue

        category = event.Category

        overlaps[category] = max(
            overlaps.get(category, 0.0),
            fraction,
        )

    return overlaps



def assign_training_label(
    anomaly_overlap: float,
    rare_event_overlap: float,
    threshold: float = 0.5,
) -> str:
    """Assign a conservative training label from event overlaps."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if anomaly_overlap >= threshold:
        return "ANOMALY"

    if rare_event_overlap >= threshold:
        return "RARE_EVENT"

    if anomaly_overlap > 0 or rare_event_overlap > 0:
        return "BOUNDARY"

    return "NOMINAL"
