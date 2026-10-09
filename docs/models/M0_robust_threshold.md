# M0 — Robust Static Threshold Baseline

## 1. Research Objective

Establish a reproducible, interpretable baseline for anomaly detection in spacecraft telemetry, against which temporal and multivariate AI methods can be evaluated.

The baseline uses independent, channel-wise robust deviation scores. It does not model temporal dependencies or cross-channel relationships.

## 2. Dataset and Experimental Scope

- **Dataset:** ESA Anomaly Detection Benchmark (ESA-ADB), Mission 1.
- **Subsystem selection:** Group 8, channels 41–46.
- **Development period:** January–December 2000.
- **Telemetry:** 1,102,444 observations across six channels.
- **Windowing:** Non-overlapping 30-minute windows, preserving the original irregular sampling.
- **Event annotations:** Original ESA event intervals and categories.

The Group 8 catalog contains 118 unique annotated events across the available mission history.

## 3. Methodology

For each channel, the detector estimates the median and median absolute deviation (MAD) from nominal training samples.

The robust deviation score is:

\[
z_{t,c} = \frac{|x_{t,c} - \mathrm{median}_c|}{1.4826\,\mathrm{MAD}_c}
\]

The multichannel sample score is the maximum across channels:

\[
s_t = \max_c z_{t,c}
\]

The window-level score is the maximum sample score within each 30-minute window:

\[
S_w = \max_{t \in w} s_t
\]

Although multiple channels contribute to the aggregation, M0 remains a channel-independent baseline: it does not learn cross-channel dependencies.

## 4. Experimental Protocol

The 2000 development period is divided chronologically:

| Split | Period | Windows |
|---|---|---:|
| Training | January–May | 7,296 |
| Validation | June–August | 4,416 |
| Test | September–December | 5,856 |

The detector is fitted using 445,722 samples belonging to 7,079 nominal training windows.

The decision threshold is calibrated at the 99th percentile of scores from 4,385 nominal validation windows.

**Calibrated threshold:** 2.408636.

An alert is raised when a window score strictly exceeds this threshold.

## 5. Development Test Results

| Metric | Result |
|---|---:|
| Test windows | 5,856 |
| Alerted windows | 137 |
| Alert episodes | 108 |
| Episodes associated with Anomaly | 33 |
| Episodes associated with Rare Event | 1 |
| Unassociated episodes | 74 |
| Unassociated episodes per day | 0.607 |

Consecutive alerted windows are grouped into alert episodes. Episodes are associated with ESA annotations through temporal interval overlap.

Both annotated Anomaly events in the test period (`id_200` and `id_73`) were intersected by at least one alert episode. One of the two Rare Events (`id_59`) was also associated with an alert episode.

These are preliminary development results, not estimates of generalization performance.

### 5.1 Event-Level Detection and Latency

Event-level detection is evaluated using the original ESA event
intervals. An event is considered detected when at least one alert
episode has positive temporal overlap with its annotated interval.

Operational detection latency is measured from the annotated event
start to the end of the first overlapping alerted 30-minute window.
This represents the earliest availability of a window-level alert,
not necessarily the physical onset of the underlying deviation.

| ESA Event | Category | Detected | Preexisting Alert | Latency (min) |
|---|---|---|---|---:|
| id_200 | Anomaly | Yes | No | 36.53 |
| id_59 | Rare Event | Yes | No | 48.53 |
| id_35 | Rare Event | No | No | N/A |
| id_73 | Anomaly | Yes | No | 211.03 |

Three of the four annotated test events were detected:
two of two Anomaly events and one of two Rare Events.

No detected event was associated with an alert episode that
started before its annotated onset.

The undetected Rare Event (id_35) lasts approximately 8.5 minutes.
Its short duration motivates further investigation of temporal
aggregation and detection sensitivity.

These results are descriptive only. The small number of events,
permissive temporal-overlap criterion, and prior inspection of
the development year prevent reliable generalization claims.
## 6. Interpretation

The baseline detects deviations in individual channels but produces a substantial number of short, isolated alert episodes.

The median alert episode duration is 30 minutes, indicating considerable alert fragmentation.

The association of an alert episode with a Rare Event suggests that deviation magnitude alone does not reliably distinguish anomalous behavior from unusual operational behavior.

The 74 unassociated episodes should not automatically be interpreted as confirmed false positives because the reference annotations may be incomplete.

## 7. Limitations

1. **Limited evaluation sample:** The development test contains only two annotated Anomaly events.
2. **No temporal modeling:** The detector does not explicitly model trends, dynamics or temporal dependencies.
3. **No learned cross-channel relationships:** Channel scores are calculated independently.
4. **Sampling sensitivity:** Maximum-based window aggregation may be influenced by the number of samples per window.
5. **Simplified event association:** Any temporal overlap between an alert episode and an annotated event is considered an association.
6. **Alert fragmentation:** A single annotated event may produce multiple distinct alert episodes.
7. **Training selection:** Nominal training windows are selected using reference annotations, without an additional temporal embargo around event boundaries.
8. **Development-set reuse:** The year 2000 has been inspected during methodology development and must not serve as the final untouched benchmark.
9. **Event aggregation:** Channel-specific annotation intervals are collapsed into an event-level interval, which may include periods not annotated on individual channels.

## 8. Next Steps

- Validate event-level detection and latency metrics on additional mission years.
- Evaluate alert burden and event associations under stricter temporal matching rules.
- Introduce a univariate temporal baseline (M1).
- Evaluate multivariate methods, including PCA-based reconstruction or residual scoring (M2).
- Compare methods using a common evaluation protocol and held-out mission periods.

## 9. Reproducibility

**Implementation:** `scripts/run_m0_baseline.py`

**Detector:** `src/satellite_health/models/robust_threshold.py`

**Evaluation:** `src/satellite_health/evaluation/event_metrics.py`

**Event catalog:** `src/satellite_health/data/event_catalog.py`

**Development outputs:** `outputs/experiments/m0/` under the external data root.

**Reference implementation commit:** `3de6614`.

The experiment uses local paths configured through `configs/data/paths.local.yaml`. Raw telemetry and generated outputs are stored outside the Git repository.
