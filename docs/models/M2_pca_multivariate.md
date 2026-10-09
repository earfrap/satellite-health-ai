# M2 — Multivariate PCA Anomaly Detection

## 1. Objective

M2 investigates whether modeling cross-channel relationships in spacecraft telemetry can improve anomaly detection compared with the univariate baselines M0 and M1.

The detector uses Principal Component Analysis (PCA) to learn a low-dimensional representation of nominal telemetry and assigns anomaly scores using the Squared Prediction Error (SPE).

The experiment evaluates three operational dimensions:

- Detection of ESA-annotated events.
- Detection latency relative to event onset.
- Alert burden, approximated by the number of alert episodes not associated with annotated ESA events.

M2 is designed as an interpretable linear multivariate baseline, not as a causal fault-diagnosis system.

## 2. Experimental Setup

- Dataset: ESA-ADB Mission 1, Group 8.
- Channels: 41–46 (six telemetry channels).
- Development period: January–December 2000.
- Training: January–May, nominal windows only.
- Validation: June–August, nominal windows for threshold calibration.
- Test: September–December.
- Preprocessing: per-channel median and MAD scaling fitted on nominal training samples.
- Model: PCA with two retained components.
- Score: sum of squared reconstruction residuals in robust-scaled channel space (SPE).
- Window aggregation: maximum SPE within each 30-minute window.
- Alert threshold: 99th percentile of nominal validation-window scores.

All chronological splits and event-level evaluation procedures follow the M0/M1 development protocol.


## 3. Results

### 3.1 Training and calibration

The M2 detector was fitted on 445,722 nominal telemetry samples from 7,079 training windows.

The 99th-percentile alert threshold was calibrated on 4,385 nominal validation windows, yielding an SPE threshold of **2.331042**.

### 3.2 Test-period performance

The September–December 2000 test period contains 5,856 evaluation windows and four ESA-annotated events.

| Metric | M0 | M1 | M2 |
|---|---:|---:|---:|
| Alerted windows | 137 | 59 | 270 |
| Alert episodes | 108 | 56 | 104 |
| Anomaly-associated episodes | 33 | 5 | 3 |
| Rare-event-associated episodes | 1 | 1 | 1 |
| Unassociated episodes | 74 | 50 | 100 |
| Unassociated episodes per day | 0.607 | 0.410 | 0.820 |
| ESA events detected | 3/4 | 3/4 | 3/4 |

All three methods detected the same three ESA events in this development test period.

### 3.3 Event-level detection latency

| ESA event | Category | M0 latency | M1 latency | M2 latency |
|---|---|---:|---:|---:|
| id_200 | Anomaly | 36.53 min | 36.53 min | 36.53 min |
| id_59 | Rare Event | 48.53 min | 48.53 min | 48.53 min |
| id_35 | Rare Event | Not detected | Not detected | Not detected |
| id_73 | Anomaly | 211.03 min | 751.03 min | 31.03 min |

For event `id_73`, M2 produced its first alert window substantially earlier than both univariate baselines: 180 minutes earlier than M0 and 720 minutes earlier than M1.

This result is consistent with the possibility that multivariate relationships provide useful early detection signals. However, the mechanism has not yet been established through channel-level analysis.

## 4. Interpretation

The M2 PCA detector achieved the same event-level detection coverage as M0 and M1 while substantially reducing detection latency for one annotated anomaly.

However, M2 also produced more alerted windows and more alert episodes unassociated with ESA annotations.

These results reveal a potential trade-off between detection timeliness and operational alert burden.

The increased number of unassociated episodes must not be interpreted directly as an increased false-positive count, because the ESA event catalog may not exhaustively describe all unusual operational behavior.

The results do not establish overall superiority of PCA. They motivate further investigation into whether event `id_73` disrupts learned cross-channel relationships and whether alternative PCA configurations can improve the trade-off.

## 5. Limitations

- **Small test event sample:** only four ESA events occur in the 2000 development test period.
- **Development-slice evaluation:** the results do not establish performance across the full 14-year mission.
- **Single PCA configuration:** two retained components were selected based on nominal training explained variance, without a systematic model-selection study.
- **Threshold sensitivity:** results depend on the 99th-percentile validation threshold and maximum-within-window aggregation.
- **Incomplete event annotations:** unassociated alerts are not necessarily false positives.
- **Limited temporal modeling:** PCA captures linear cross-channel structure but does not explicitly learn temporal dynamics.
- **Limited explanatory evidence:** reconstruction residuals could support channel-level explanations, but these have not yet been investigated.
- **Evaluation protocol:** potential temporal dependence near chronological split boundaries requires further study.

## 6. Preliminary conclusion

M2 demonstrates that a linear multivariate detector can identify deviations from nominal spacecraft telemetry relationships and can, for some events, provide earlier alerts than univariate approaches.

The improvement observed for event `id_73` is accompanied by a higher unassociated-alert burden and requires further investigation before drawing broader conclusions.

The next analysis will examine PCA reconstruction residuals and channel-level contributions around `id_73` to assess whether its early detection is linked to a disruption of multivariate relationships.






## Case study: Explainability of ESA event id_73

### Objective

Investigate why the M2 PCA-based multivariate detector identifies ESA event `id_73` earlier than the M0 robust univariate threshold baseline.

The analysis uses the 2000 development slice of ESA Mission 1, Group 8, and reproduces the original M2 training configuration and validation-calibrated thresholds.

### Temporal detection behavior

ESA event `id_73` begins on 2000-12-22 at 16:58:58.314 UTC.

The first M2 alert occurs in the 17:00–17:30 UTC evaluation window, corresponding to a window-level detection latency of approximately 31.03 minutes.

M0 first alerts in the 20:00–20:30 UTC window, corresponding to approximately 211.03 minutes. M1 detects the event later, with a latency of approximately 751.03 minutes.

During the five hours preceding the event, M2 remains below its calibrated alert threshold. From the 17:00 window onward, M2 exceeds its threshold in every examined window through 22:00.

### Channel-level explanation

At 17:16:58.314 UTC, the maximum PCA squared prediction error (SPE) within the first M2 alert window is 3.529074, above the validation-calibrated threshold of approximately 2.331042.

The channel contributions to this SPE are:

| Channel | SPE contribution | Percentage |
|---|---:|---:|
| channel_46 | 0.938 | 26.59% |
| channel_44 | 0.609 | 17.25% |
| channel_42 | 0.576 | 16.32% |
| channel_45 | 0.568 | 16.10% |
| channel_41 | 0.505 | 14.30% |
| channel_43 | 0.333 | 9.44% |

The contributions sum to the total SPE. Channel 46 has the largest contribution, but the reconstruction error is distributed across all six channels.

At the same timestamp, the maximum M0 robust per-channel deviation is approximately 1.252, below its calibrated threshold of approximately 2.408636.

Thus, this selected observation exceeds the multivariate reconstruction-error threshold without exceeding the univariate maximum-deviation threshold.

### Scientific interpretation

The result is consistent with the hypothesis that multivariate telemetry structure can reveal deviations that independent univariate thresholds do not detect as promptly.

PCA detects deviations from a nominal linear subspace. Channel-level squared reconstruction residuals provide a mathematical explanation of the score, but do not establish physical causality or identify a faulty spacecraft component.

The earlier detection of `id_73` does not establish global superiority of M2. In the 2000 development test, M2 also generates more alert episodes unassociated with ESA annotations than M0 or M1.

### Limitations

- The analysis focuses on one selected event in a development dataset.
- The first alert window is identified by overlap with the ESA annotation, not by causal attribution.
- Thresholds are calibrated on 30-minute window maxima; sample-level scores are used for explanation, not as independent operational alert decisions.
- The selected peak is not representative of every observation in the event.
- The 2000 test period has been inspected repeatedly and cannot be treated as an untouched final evaluation set.
- Physical root-cause attribution requires additional engineering context and domain validation.

### Reproducibility

The analysis is implemented in `scripts/analyze_m2_event73.py`.

The script reconstructs the nominal PCA training, reads the original M0 and M2 thresholds from their experiment summaries, identifies the first M2 alert window overlapping `id_73`, calculates channel-level contributions, and checks consistency between contributions and the total SPE.

The channel-level results are saved to `experiments/eval/development_2000/event73_channel_explanation.csv` under the configured external outputs directory.
