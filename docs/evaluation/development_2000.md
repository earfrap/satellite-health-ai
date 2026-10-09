# Development-Period Comparison (2000) — M0–M2

## 1. Objective

Compare three anomaly detection approaches on the ESA Mission 1, Group 8 telemetry development dataset:

- **M0:** Robust univariate threshold detector.
- **M1:** Causal temporal residual detector.
- **M2:** PCA-based multivariate reconstruction-error detector.

The evaluation investigates event detection coverage, detection latency, operational alert burden, and interpretability.

The objective is not to identify a universally superior model, but to characterize the trade-offs between the approaches.

## 2. Experimental protocol

All three experiments use the same chronological split of the year 2000:

| Partition | Period |
|---|---|
| Training | January–May |
| Validation | June–August |
| Test | September–December |

Training uses nominal observations. Alert thresholds are calibrated independently for each detector at the 99th percentile of nominal validation-window scores.

Scores are evaluated over 30-minute windows. Operational detection latency is measured from the ESA event onset to the end of the first overlapping alerted window.

The comparison script verifies consistency of the experimental periods, calibration quantile, number of test windows, and ESA event identities.

**Evaluation status:** The 2000 test period is an inspected development test, not an untouched independent holdout.

## 3. Operational alert burden

| Metric | M0 | M1 | M2 |
|---|---:|---:|---:|
| Test windows | 5,856 | 5,856 | 5,856 |
| Alerted windows | 137 | 59 | 270 |
| Alerted windows (%) | 2.339 | 1.008 | 4.611 |
| Total alert episodes | 108 | 56 | 104 |
| Anomaly-associated episodes | 33 | 5 | 3 |
| Rare-event-associated episodes | 1 | 1 | 1 |
| Unassociated episodes | 74 | 50 | 100 |
| Unassociated episodes/day | 0.607 | 0.410 | 0.820 |

M1 generates the lowest operational alert burden among the three approaches.

M2 produces the largest proportion of alerted windows, although its total number of alert episodes is slightly lower than M0.

This distinction reflects differences in the temporal distribution and persistence of alerts.

Unassociated episodes must not automatically be interpreted as false positives: the ESA annotation catalog may not describe every operationally relevant deviation.

## 4. Event-level detection

Four ESA events overlap the development test period: two Anomalies and two Rare Events.

| ESA event | Category | M0 latency (min) | M1 latency (min) | M2 latency (min) |
|---|---|---:|---:|---:|
| id_200 | Anomaly | 36.53 | 36.53 | 36.53 |
| id_59 | Rare Event | 48.53 | 48.53 | 48.53 |
| id_35 | Rare Event | Not detected | Not detected | Not detected |
| id_73 | Anomaly | 211.03 | 751.03 | 31.03 |

All three detectors identify three of the four events.

- Overall event coverage: 3/4.
- Anomaly coverage: 2/2.
- Rare Event coverage: 1/2.

The median latency among detected events is approximately 48.53 minutes for M0 and M1, and 36.53 minutes for M2.

These conditional median latencies exclude the undetected event and should not be interpreted independently of event coverage.

## 5. Case study: ESA event id_73

Event `id_73` begins on 2000-12-22 at 16:58:58.314 UTC.

M2 first exceeds its validation-calibrated window-score threshold in the 17:00–17:30 UTC window.

M0 first alerts in the 20:00–20:30 UTC window, while M1 first alerts in the 05:00–05:30 UTC window on the following day.

Consequently, M2 detects this event approximately three hours earlier than M0 and twelve hours earlier than M1.

The normalized-score comparison uses each detector's window score divided by its own validation-calibrated threshold. A normalized value above 1 indicates an alerted window.

This normalization supports visual comparison of threshold crossings, but does not make the detector scores statistically equivalent.

The separate M2 explainability analysis attributes the PCA reconstruction error across all six telemetry channels. It supports an interpretation in terms of deviations from learned nominal multivariate structure, without establishing physical root cause.

## 6. Interpretation

The development results indicate a trade-off:

**M1** produces fewer alerts not associated with ESA annotations, but responds considerably later to event `id_73`.

**M2** responds earliest to `id_73`, but generates more unassociated alert episodes and alerted windows.

**M0** provides a simpler, interpretable reference with intermediate operational alert burden and intermediate latency for `id_73`.

None of the three detectors improves event coverage over the others in this development test.

The findings are consistent with the potential value of multivariate modeling for selected events, but do not demonstrate general superiority of PCA-based detection.

## 7. Limitations

- Only four annotated ESA events occur in the development test period.
- The 2000 test period has already been inspected during model development.
- Unassociated alert episodes are not confirmed false positives.
- Alert-episode association uses temporal overlap with ESA annotations and does not establish causality.
- A single annotated event can overlap multiple alert episodes.
- The current analysis does not establish operational acceptability of the alert burden.
- Statistical generalization requires evaluation on additional, previously uninspected temporal data.
- Explainability based on reconstruction residuals does not identify physical root causes.

## 8. Reproducibility

The comparison is generated by:

`scripts/compare_m0_m1_m2.py`

The figures are generated by:

the historical development-period plotting scripts

Outputs are stored under the configured external directory:

`experiments/eval/development_2000/`

The generated CSV files are:

- `operational_comparison_2000.csv`
- `event_comparison_2000.csv`
- `event_summary_2000.csv`

The generated figures are:

- `figures/unassociated_alert_burden_2000.png`
- `figures/event_latency_comparison_2000.png`
- `figures/event73_normalized_scores_2000.png`

All results derive from the previously saved M0, M1, and M2 experiments without retraining the detectors.

## 9. Conclusion

The 2000 development study establishes a comparative baseline of three complementary spacecraft telemetry anomaly detectors.

The results highlight why event coverage alone is insufficient for operational evaluation: detectors with identical event coverage can differ substantially in detection latency and alert burden.

This development-stage comparison was subsequently extended through the frozen-model 2001–2013 evaluation of M0–M4, documented in [EVAL](EVAL_comparative_evaluation.md). The 2000 test period remains an inspected development dataset rather than an independent holdout.
