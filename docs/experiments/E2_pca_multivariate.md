# E2 — Multivariate PCA Anomaly Detection

## 1. Objective

E2 investigates whether modeling cross-channel relationships in spacecraft telemetry can improve anomaly detection compared with the univariate baselines E0 and E1.

The detector uses Principal Component Analysis (PCA) to learn a low-dimensional representation of nominal telemetry and assigns anomaly scores using the Squared Prediction Error (SPE).

The experiment evaluates three operational dimensions:

- Detection of ESA-annotated events.
- Detection latency relative to event onset.
- Alert burden, approximated by the number of alert episodes not associated with annotated ESA events.

E2 is designed as an interpretable linear multivariate baseline, not as a causal fault-diagnosis system.

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

All chronological splits and event-level evaluation procedures follow the E0/E1 development protocol.


## 3. Results

### 3.1 Training and calibration

The E2 detector was fitted on 445,722 nominal telemetry samples from 7,079 training windows.

The 99th-percentile alert threshold was calibrated on 4,385 nominal validation windows, yielding an SPE threshold of **2.331042**.

### 3.2 Test-period performance

The September–December 2000 test period contains 5,856 evaluation windows and four ESA-annotated events.

| Metric | E0 | E1 | E2 |
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

| ESA event | Category | E0 latency | E1 latency | E2 latency |
|---|---|---:|---:|---:|
| id_200 | Anomaly | 36.53 min | 36.53 min | 36.53 min |
| id_59 | Rare Event | 48.53 min | 48.53 min | 48.53 min |
| id_35 | Rare Event | Not detected | Not detected | Not detected |
| id_73 | Anomaly | 211.03 min | 751.03 min | 31.03 min |

For event `id_73`, E2 produced its first alert window substantially earlier than both univariate baselines: 180 minutes earlier than E0 and 720 minutes earlier than E1.

This result is consistent with the possibility that multivariate relationships provide useful early detection signals. However, the mechanism has not yet been established through channel-level analysis.

## 4. Interpretation

The E2 PCA detector achieved the same event-level detection coverage as E0 and E1 while substantially reducing detection latency for one annotated anomaly.

However, E2 also produced more alerted windows and more alert episodes unassociated with ESA annotations.

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

E2 demonstrates that a linear multivariate detector can identify deviations from nominal spacecraft telemetry relationships and can, for some events, provide earlier alerts than univariate approaches.

The improvement observed for event `id_73` is accompanied by a higher unassociated-alert burden and requires further investigation before drawing broader conclusions.

The next analysis will examine PCA reconstruction residuals and channel-level contributions around `id_73` to assess whether its early detection is linked to a disruption of multivariate relationships.
