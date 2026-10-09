# M1 — Causal Temporal-Residual Baseline

## 1. Objective

Experiment M1 evaluates whether a causal, time-local reference can improve spacecraft telemetry anomaly monitoring relative to the static robust-threshold baseline M0.

M0 measures deviations from a fixed nominal distribution. M1 instead measures deviations from the recent history of each telemetry channel.

The central question is whether temporal adaptation can reduce operational alert burden without compromising the detection and timeliness of annotated spacecraft events.

M1 is a univariate temporal baseline applied independently to each channel, followed by multichannel score aggregation. It does not learn cross-channel dependencies.

## 2. Dataset and Development Scope

The experiment uses the ESA-ADB Mission 1 dataset, restricted to telemetry channels 41–46 (Group 8) and calendar year 2000.

- Telemetry observations: 1,102,444 synchronized timestamps
- Channels: 6
- Evaluation windows: 17,568 non-overlapping 30-minute intervals
- Group 8 events in the complete event catalog: 118
- ESA events overlapping the 2000 test period: 4

The dataset contains irregular sampling intervals. For this reason, M1 uses a time-based rolling reference rather than a fixed number of preceding observations.

All timestamps are interpreted as UTC.

### 2.1 Chronological Split

| Partition | Period | Purpose |
|---|---|---|
| Training | January–May 2000 | Estimate nominal residual distributions |
| Validation | June–August 2000 | Calibrate alert threshold |
| Test | September–December 2000 | Evaluate detection and operational alert burden |

Only windows labeled `NOMINAL` are used for fitting and validation threshold calibration.

Training includes 7,079 nominal windows and 445,722 telemetry observations belonging to those windows. The initial observations lacking sufficient rolling history are excluded from the residual statistics through missing-value handling.

Validation includes 4,385 nominal windows. The test partition contains 5,856 windows and four annotated ESA events.

The test period is a development benchmark, not an independent final generalization assessment.

## 3. Methodology

### 3.1 Causal Temporal Reference

For each channel \(c\) and timestamp \(t\), M1 computes a rolling median using observations from the preceding 30 minutes:

\[
m_c(t)=\operatorname{median}\{x_c(s):t-30\text{ min}\leq s<t\}.
\]

The current observation is excluded using `closed="left"`.

At least 10 preceding observations are required to produce a valid reference.

This ensures that the reference uses past observations only. The reference is computed over the continuous chronological telemetry sequence, including prior observations across the training, validation, and test boundaries.

### 3.2 Temporal Residuals

The residual for each channel is defined as:

\[
r_c(t)=x_c(t)-m_c(t).
\]

The residual measures deviation from recent local behavior, rather than deviation from a fixed nominal channel level.

### 3.3 Robust Residual Normalization

Using only nominal training residuals, M1 estimates a per-channel residual median and median absolute deviation (MAD).

The normalized score is:

\[
z_c(t)=\frac{|r_c(t)-\widetilde{r}_c|}
{1.4826\operatorname{MAD}_c}.
\]

Here, \(\widetilde{r}_c\) and \(\operatorname{MAD}_c\) are estimated exclusively from nominal training residuals.

The implementation checks for invalid training inputs and zero-MAD channels.

### 3.4 Window-Level Score

For each 30-minute evaluation window \(W_k\), M1 aggregates scores by taking the maximum across channels and observations:

\[
S_k=\max_{t\in W_k}\max_c z_c(t).
\]

Thus, an extreme residual in any channel can trigger a window-level alert.

This aggregation does not explicitly model multivariate dependencies.

### 3.5 Threshold Calibration

The alert threshold is set to the 99th percentile of scores from nominal validation windows.

For M1:

\[
\tau_{M1}=2.173327.
\]

An alert is generated when:

\[
S_k>\tau_{M1}.
\]

The threshold is calibrated independently from M0. Consequently, the numerical values of M0 and M1 scores are not directly comparable without reference to their respective thresholds.

## 4. Evaluation Protocol

M1 uses the same evaluation functions and chronological partitions as M0.

Adjacent alerted 30-minute windows are grouped into alert episodes. Episodes overlapping the test period are retained, with validation-period context used to avoid artificially truncating episodes at the test boundary.

Alert episodes are classified according to overlap with ESA annotations:

- `ANOMALY`: overlaps an annotated anomaly
- `RARE_EVENT`: overlaps an annotated rare event, without anomaly priority
- `UNASSOCIATED`: overlaps neither category

These are annotation associations, not confirmed causal explanations.

Event-level detection is evaluated using the original ESA event intervals rather than only the window training labels. Detection latency is measured from event onset to the closing time of the first overlapping alerted window, with the evaluation implementation also checking whether an alert was already active at event onset.

Unassociated episodes must not automatically be interpreted as false positives, because the annotation catalog is not guaranteed to represent every operationally relevant deviation.

## 5. M1 Results

### 5.1 Alert Activity

| Metric | M1 |
|---|---:|
| Test windows | 5,856 |
| Alerted windows | 59 |
| Alert episodes | 56 |
| Anomaly-associated episodes | 5 |
| Rare-event-associated episodes | 1 |
| Unassociated episodes | 50 |
| Unassociated episodes per day | 0.410 |

### 5.2 Event-Level Detection and Latency

| ESA Event | Category | Detected | Latency (minutes) |
|---|---|---|---:|
| `id_200` | Anomaly | Yes | 36.528 |
| `id_59` | Rare Event | Yes | 48.528 |
| `id_35` | Rare Event | No | — |
| `id_73` | Anomaly | Yes | 751.028 |

M1 detects three of the four annotated test events: both anomalies and one of the two rare events.

These fractions describe only the four-event development test sample and should not be interpreted as reliable estimates of operational detection performance.

## 6. Comparison with M0

M0 and M1 use the same dataset, chronological partitions, validation quantile, window definitions, event catalog, and evaluation procedure.

| Metric | M0 | M1 |
|---|---:|---:|
| Validation threshold | 2.408636 | 2.173327 |
| Alerted test windows | 137 | 59 |
| Test alert episodes | 108 | 56 |
| Anomaly-associated episodes | 33 | 5 |
| Rare-event-associated episodes | 1 | 1 |
| Unassociated episodes | 74 | 50 |
| Unassociated episodes/day | 0.607 | 0.410 |
| Detected ESA events | 3/4 | 3/4 |

M1 reduces unassociated episodes from 74 to 50, a reduction of approximately 32% in this test period, while preserving the same event-level detection count.

However, fewer episodes associated with anomalies do not necessarily mean worse event-level detection. Episode counts can be affected by alert fragmentation and duration.

### 6.1 Detection Latency Comparison

| ESA Event | M0 Latency | M1 Latency | Difference |
|---|---:|---:|---:|
| `id_200` | 36.528 min | 36.528 min | 0 min |
| `id_59` | 48.528 min | 48.528 min | 0 min |
| `id_35` | Not detected | Not detected | — |
| `id_73` | 211.028 min | 751.028 min | +540 min |

The largest difference concerns anomaly `id_73`: M1 generates its first event-associated alert nine hours later than M0.

This is an important operational disadvantage despite the reduction in unassociated alert episodes.

### 6.2 Inspection of Event `id_73`

The annotated event begins on 22 December 2000 at 16:58:58 UTC.

M0 first exceeds its threshold in the 20:00–20:30 UTC window:

- M0 score: 2.505138
- M0 threshold: 2.408636

At the same time, M1 remains below threshold:

- M1 score: 2.023467
- M1 threshold: 2.173327

M1 first exceeds its threshold in the 05:00–05:30 UTC window on 23 December:

- M1 score: 2.173637
- M1 threshold: 2.173327

The first M1 exceedance is marginal, only approximately 0.00031 score units above threshold.

One plausible explanation is that the rolling temporal reference adapts to relatively persistent signal deviations, reducing their residual magnitude. This mechanism is consistent with the observed score behavior but has not yet been demonstrated through channel-level signal and rolling-reference inspection.

## 7. Scientific Interpretation

The M0–M1 comparison reveals a possible trade-off between alert burden and detection timeliness.

M1 produces fewer alerted windows and fewer unassociated episodes, while detecting the same three ESA events in this limited development test.

However, its response to `id_73` is substantially slower.

These results suggest that static nominal references and local temporal references may capture complementary forms of deviation:

- Static references can remain sensitive to sustained departures from nominal operating levels.
- Temporal residuals can emphasize departures from recent behavior but may attenuate persistent changes.

This complementarity is a hypothesis for future investigation, not a demonstrated benefit of combining the two approaches.

Neither M0 nor M1 learns correlations among telemetry channels. Both aggregate independently computed channel scores using a maximum operator.

## 8. Limitations and Threats to Validity

1. **Small event sample:** Only four ESA events occur in the development test period. Performance cannot be generalized from this sample.
2. **Single development year:** The experiment has not yet been validated across the full 14-year Mission 1 history.
3. **Annotation incompleteness:** Unassociated episodes cannot automatically be classified as false alarms.
4. **Permissive episode association:** Any temporal overlap may associate an episode with an event; association is not proof of an operationally useful detection.
5. **Alert fragmentation:** Episode counts depend on temporal continuity and do not independently quantify alert duration or operational workload.
6. **Sampling irregularity:** Although the rolling window is time-based, its median weights observations rather than elapsed time, so irregular sampling can affect the reference.
7. **Maximum aggregation:** The maximum over samples and channels is sensitive to isolated extremes and sampling density.
8. **Adaptation to persistent deviations:** A local rolling reference may reduce sensitivity to sustained changes.
9. **No learned multivariate structure:** Cross-channel dependencies are not modeled.
10. **Development-stage evaluation:** The test results have already been inspected during model development; later final assessment requires a separate, protected evaluation protocol.

## 9. Reproducibility and Next Steps

The M1 implementation is located in:

- `src/satellite_health/models/temporal_residual.py`
- `scripts/run_m1_baseline.py`
- `tests/test_temporal_residual.py`

The experiment can be reproduced from the repository root with the configured local data paths:

```bash
python scripts/run_m1_baseline.py
```

Outputs are written to the external data directory under:

```text
outputs/experiments/m1/
```

The experiment generates window-level scores, classified test episodes, event-level detection and latency metrics, and a summary CSV.

The next research steps are to inspect the channel-level behavior of `id_73`, assess the stability of the observed trade-off across additional periods, and introduce a genuinely multivariate method that can learn dependencies among telemetry channels.

M1 provides a temporal univariate reference point against which later multivariate approaches can be compared.