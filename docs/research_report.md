# Satellite Health AI

## Explainable Anomaly Detection for Long-Term Spacecraft Health Monitoring

**Technical Research Report — M0–M4 Comparative Study | EVAL: 2001–2013**

### Abstract

This study compares five anomaly-detection approaches on six telemetry channels from ESA Mission 1, Group 8: a robust statistical threshold (M0), a causal temporal residual detector (M1), principal component analysis (M2), Isolation Forest (M3), and a feed-forward autoencoder (M4). Models and alert thresholds were established using data from 2000 and evaluated without retraining or recalibration over 2001–2013. The evaluation distinguishes the fraction of scorable ESA-NOMINAL windows flagged, overlap with 108 annotated ESA events, and the initiation of new alert episodes during events. Continuous episode reconstruction on the common 227,904-window evaluation timeline confirmed that year-boundary handling did not alter event-level classifications for any of the five detectors.

M1 produces alerts in 1.48% of scorable NOMINAL windows and new alert episodes during 36/108 annotated events, while M2 overlaps 106/108 events but flags 83.45% of scorable NOMINAL windows. M4 has lower nominal alert activity than M0, M2, and M3, but also lower event coverage. These results illustrate why event overlap, taken alone, can misrepresent operational usefulness, and why model complexity does not guarantee a favorable monitoring trade-off. NOMINAL alert fraction is not assumed to equal false-positive rate, and temporal coincidence is not treated as proof of event causation.

## 1. Research objective

Spacecraft health monitoring must identify unusual telemetry behavior without generating persistent, difficult-to-interpret alert states. Detectors calibrated on a historical operating period may behave differently years later even if their parameters remain fixed.

**Research question:** How do statistical, temporal, classical machine-learning, and neural reconstruction-based detectors compare under a common frozen-model, long-term telemetry evaluation protocol?

The study examines temporal stability, event overlap versus newly initiated alerts, and the trade-off between event coverage and nominally labeled alert activity. The objective is comparative understanding, not maximization of a single score.

## 2. Dataset and scope

The dataset is the selected **ESA Mission 1, Group 8** telemetry subset: `channel_41` through `channel_46`. Observations are aggregated into 30-minute evaluation windows. Model development uses data from 2000; the held-out chronological evaluation covers **2001–2013**, comprising **227,904 windows**.

The event analysis includes **108 unique ESA event IDs** intersecting the evaluation period and associated with the selected channels. Multiple annotated channels for the same ID are not counted as separate events.

| ESA annotation category | Unique events |
|---|---:|
| Anomaly | 46 |
| Rare Event | 58 |
| Communication Gap | 4 |
| **Total** | **108** |

The labels provide an imperfect reference, not exhaustive proof of nominal or anomalous physical conditions. Findings are limited to this mission subset and should not be generalized to other channels or spacecraft without further testing.

## 3. Methods

All five models produce sample-level anomaly scores that are aggregated to the **maximum score in each 30-minute window**, then compared with a detector-specific frozen threshold.

### 3.1 M0 — Robust statistical threshold

M0 estimates per-channel reference location and scale from nominal training observations using the median and median absolute deviation (MAD). Its score reflects robust deviations from these channel references. This simple baseline is interpretable but potentially sensitive to long-term shifts in operating level or variability.

### 3.2 M1 — Causal temporal residual

M1 compares observations with a rolling 30-minute median computed strictly from preceding observations. A minimum of ten prior observations is required. Robustly scaled residuals are aggregated into window-level scores. The causal construction avoids future-observation leakage, while its local reference can respond differently to gradual shifts and transient deviations.

### 3.3 M2 — Principal component analysis

M2 fits PCA to nominal multivariate telemetry, retaining **two principal components**. Squared prediction error (SPE) measures the residual variation outside the learned subspace. Reconstruction-error growth is a statistical observation and does not by itself identify a physical fault.

### 3.4 M3 — Isolation Forest

M3 is a tree-based unsupervised anomaly detector trained on nominal telemetry. Its score summarizes how unusual an observation appears under the fitted isolation structure. It provides a nonlinear comparison with the robust and PCA baselines, but its scores and threshold are specific to the fitted model and calibration procedure.

### 3.5 M4 — Feed-forward autoencoder

M4 uses robust channel scaling and a fully connected reconstruction network with architecture **6 → 4 → 2 → 4 → 6**, ReLU hidden activations, and a linear output. It is trained on nominal observations with mean-squared reconstruction loss using Adam for **20 epochs** (batch size **1,024**, learning rate **0.001**, seed **42**). The anomaly score is the sum of squared channel reconstruction residuals. The fitted network is a nonlinear reconstruction baseline, not a physical simulator of spacecraft subsystems.

## 4. Experimental protocol

### 4.1 Chronological training and validation

The nominal training period is January–May 2000, with **445,722 samples** corresponding to **7,079 nominal 30-minute windows**. A separate nominal validation period in June–August 2000 contains **4,385 windows**. Each model's decision threshold is the **99th percentile of validation window scores**, using the maximum sample score within each window.

| Model | Frozen threshold |
|---|---:|
| M0 | 2.408636332 |
| M1 | 2.173326984 |
| M2 | 2.331041565 |
| M3 | 0.603551727 |
| M4 | 16.572053909 |

Threshold magnitudes are **not directly comparable** across models because the scoring functions differ. The numerical M3 threshold is deliberately not reproduced here without checking its saved artifact.

### 4.2 Frozen-model evaluation

Models and thresholds remain fixed throughout 2001–2013: no retraining or recalibration on evaluation years. Every detector is evaluated against the same 30-minute window timeline and the same selected ESA event IDs. Some historical artifact filenames and CSV columns retain legacy identifiers (`e0`–`e3` for earlier detectors and `e4` for the former evaluation stage). The current scientific nomenclature is **M0–M4** for the five models and **EVAL** for the evaluation protocol. Legacy identifiers are preserved where changing them would break compatibility with frozen results.

### 4.3 Missing scores

Unscorable windows are represented as missing, not as non-alerts. The earlier M0–M2 evaluation documented **11**, **35**, and **11** unscorable windows respectively across the evaluation period. All nominal alert percentages in the final comparison use only scorable NOMINAL windows for each detector; denominators can therefore differ slightly.

## 5. Evaluation criteria

**NOMINAL alert fraction:** The number of scorable windows labeled NOMINAL that produce an alert, divided by all scorable NOMINAL windows for that model. The overall percentage is weighted by the number of eligible windows across years, not the unweighted mean of annual percentages. This is a descriptive alert-burden proxy, **not** an established false-positive rate.

**Event overlap:** At least one alert episode intersects an ESA event interval. An episode may have begun before the event; therefore overlap is permissive and can be inflated by persistent alert states.

**New episode during event:** At least one alert episode begins within the annotated event interval. This is a stricter timing criterion, but it does not prove causal association or guarantee that the alert would be operationally actionable.

**Episode definition:** A consecutive run of alerted evaluation windows. Missing scores are not automatically treated as negative alerts. Episode counts depend on the windowing and continuity conventions and should not be interpreted as independent faults.

**Latency:** For an event with a new episode, latency is measured from the ESA event start to the end of the first qualifying 30-minute alert window. Available legacy latency results are reported for M0–M2 only; their qualifying event subsets differ.

## 6. Results

### 6.1 Temporal behavior

![Annual NOMINAL alert fraction for M0–M4](figures/fig01_temporal_stability.png)

*Figure 1. Annual percentage of scorable ESA-NOMINAL windows flagged by each frozen detector, 2001–2013.*

M1 maintains approximately 1–2% annual NOMINAL alert activity, whereas M2 and M3 become strongly saturated during later years. M0 also exhibits a substantial later-period increase. M4 is less saturated than M0, M2, or M3 overall but shows elevated activity in 2009–2012. These are changes in detector behavior; their physical causes have not been established.

### 6.2 Aggregate comparison

| Model | Scorable NOMINAL windows with alert | Events with overlap | Events with new episode | Continuous alert episodes |
|---|---:|---:|---:|---:|
| M0 — Robust threshold | 37.34% | 95/108 (87.96%) | 34/108 (31.48%) | 30,059 |
| M1 — Temporal residual | **1.48%** | 80/108 (74.07%) | **36/108 (33.33%)** | **3,555** |
| M2 — PCA | 83.45% | **106/108 (98.15%)** | 28/108 (25.93%) | 14,251 |
| M3 — Isolation Forest | 51.06% | 101/108 (93.52%) | 33/108 (30.56%) | 20,408 |
| M4 — Autoencoder | 5.24% | 63/108 (58.33%) | 22/108 (20.37%) | 10,808 |

These metrics capture different aspects of detector behavior. A high event-overlap percentage should not be interpreted as event-detection recall under exhaustive ground truth, and the NOMINAL alert percentage should not be interpreted as false-positive rate.

### 6.3 ESA event categories

![Event overlap and new episode fractions by category](figures/fig02_event_detection.png)

*Figure 2. Event overlap and new-episode fractions for all ESA events and separately for Anomaly, Rare Event, and Communication Gap annotations. The Communication Gap subset contains only four events.*

For the **46 ESA Anomaly events**, the following counts are available from the verified per-event evaluations:

| Model | Overlap | New episode during event |
|---|---:|---:|
| M0 | 45/46 | 6/46 |
| M1 | 42/46 | 13/46 |
| M2 | 46/46 | 2/46 |
| M3 | 46/46 | 3/46 |
| M4 | 41/46 | 4/46 |

M2 and M3 overlap every annotated Anomaly event, yet comparatively few such events contain the onset of a new alert episode. M1 has fewer overlapping Anomaly events but more newly initiated episodes within them. This distinction matters when a detector is already in an alert state before an event begins.

### 6.4 Operational trade-off

![Event coverage versus NOMINAL alert burden](figures/fig03_operational_tradeoff.png)

*Figure 3. Event-overlap and new-episode coverage plotted against the overall fraction of scorable NOMINAL windows flagged. The horizontal axis measures alert burden, not a validated false-positive rate.*

Under these metrics, M1 offers a particularly favorable balance of low NOMINAL alert activity and new-episode coverage. M2's very high overlap coincides with widespread NOMINAL alerting; M4 reduces alert burden relative to several fixed-reference methods but also has lower event coverage. No universal operational ranking is implied: different missions may prioritize different costs, event types, and alert timing.

### 6.5 Latency for M0–M2

Legacy results for ESA **Anomaly** events with new alert episodes are:

| Model | Qualifying events | Median latency | Mean latency |
|---|---:|---:|---:|
| M0 | 6 | 109.1 min | 322.3 min |
| M1 | 13 | 498.6 min | 1,095.5 min |
| M2 | 2 | 718.3 min | 718.3 min |

These values describe **different subsets of events** and are not a controlled same-event speed comparison. Equivalent summary statistics for M3 and M4 are not claimed here.

### 6.6 Cross-year consistency check

Alert episodes were reconstructed on the **full concatenated 2001–2013 timeline** for all five models, then event-level overlap and new-episode classifications were compared with the earlier year-by-year evaluation. Across the **108 common ESA event IDs**, **zero event classifications changed** for either metric in any model. This verifies the reported event-level counts against the tested year-boundary artifact; it does not validate the completeness of ESA labels or the physical interpretation of alerts.

## 7. Discussion

### 7.1 High overlap can conceal persistent alert states

An alert that starts before an event and continues through it satisfies the overlap criterion without representing a newly triggered response. The contrast between M2's 106/108 overlaps and 28/108 new-episode events illustrates why overlap alone is an insufficient operational metric, especially when nominally labeled windows are frequently flagged.

### 7.2 Temporal adaptation and fixed-reference sensitivity

M1's local causal baseline is associated with substantially more stable NOMINAL alert activity than several fixed-reference approaches in this dataset. However, local adaptation may also reduce sensitivity to sustained shifts. The results do not identify which physical operating changes explain the long-term differences.

### 7.3 Model complexity is not an operational guarantee

The neural autoencoder provides a nonlinear reconstruction alternative, but its event coverage is lower than that of the simpler M1 detector under the current evaluation. This comparison is conditional on the chosen architecture, calibration, training slice, and event definitions; it is not evidence that neural methods are generally inferior.

### 7.4 Explainability and scientific interpretation

M0 links scores to robust channel deviations; M1 to departures from recent channel behavior; M2 to variation outside a learned multivariate subspace. M3 and M4 offer additional nonlinear perspectives but require careful score-level or feature-level interpretation. None of these statistical descriptions establishes spacecraft-level root cause.

## 8. Limitations

- **Restricted scope:** One ESA mission, one telemetry group, six channels, and one chronological training period.
- **Annotation uncertainty:** NOMINAL does not prove absence of anomalies; ESA event labels are not necessarily exhaustive.
- **Temporal coincidence versus causality:** An alert beginning during an event is not proof that the event caused it.
- **Episode dependence:** Adjacent alert windows and overlapping event intervals are temporally dependent.
- **Calibration and distribution shift:** All thresholds remain frozen by design; adaptive recalibration was not assessed.
- **Unequal score availability:** Missing observations can affect detector-specific denominators.
- **Latency selection:** Reported M0–M2 latency summaries are conditional on different qualifying events, and M3–M4 are not yet included in a matched latency analysis.
- **Limited model search:** The M4 architecture is one specified configuration, not a comprehensive neural architecture or hyperparameter search.
- **External validity:** Operational cost, telemetry semantics, and mission conditions may differ substantially elsewhere.

## 9. Conclusions

A common frozen-model evaluation over 2001–2013 reveals substantial differences in the long-term behavior of five spacecraft telemetry anomaly detectors. M1 shows the lowest NOMINAL alert fraction (1.48%) and the highest number of ESA events containing newly initiated alert episodes (36/108). M2 achieves the highest event overlap (106/108) while flagging 83.45% of scorable NOMINAL windows. The autoencoder does not outperform simpler alternatives on the evaluated operational proxies.

The central methodological finding is that **event overlap, alert initiation, temporal stability, and score availability must be considered together**. None of these metrics alone establishes reliable physical anomaly detection or operational superiority. The work provides a documented comparative foundation for further mission-specific evaluation, with full reproducibility dependent on access to the ESA dataset and compatible software environments.

## 10. Future work

Promising extensions include evaluation of additional ESA channel groups or missions; characterization of operating regimes and distribution shift; controlled adaptive recalibration; event-matched latency and alert-burden analysis for all five models; feature-level explanations; and uncertainty analysis for event annotations. These are proposed extensions, **not completed experiments**.

---

**Project:** Satellite Health AI  
**Focus:** Applied AI, explainable anomaly detection, long-term generalization, reproducible evaluation, and scientific R&D methodology.
