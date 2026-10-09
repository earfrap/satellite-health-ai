# EVAL — Long-Term Comparative Evaluation (2001–2013)

**Satellite Health AI | ESA Mission 1, Group 8 | Frozen models M0–M4**

## 1. Objective and scope

EVAL compares five spacecraft-telemetry anomaly detectors under a common chronological, frozen-model evaluation protocol. The purpose is to assess **temporal stability, nominal alert burden, ESA-event overlap, and initiation of new alert episodes**—not to identify a universally optimal detector.

The study uses ESA Mission 1, Group 8, channels 41–46, aggregated into non-overlapping **30-minute windows**. The independent chronological evaluation covers **2001–2013** on a common timeline of **227,904 windows**. It includes **108 unique annotated ESA events** intersecting the selected channels and period: **46 Anomaly**, **58 Rare Event**, and **4 Communication Gap** events. Event IDs, rather than channel-specific annotation records, define unique events.

The preceding **development-period comparison (September–December 2000)** involved M0–M2 and four annotated events. Its results are retained separately in [Development-period comparison (2000)](development_2000.md). They must not be confused with the long-term EVAL results.

## 2. Compared models

| Model | Method | Anomaly-score principle |
|---|---|---|
| **M0** | Robust statistical threshold | Robust per-channel deviation from nominal references |
| **M1** | Causal temporal residual | Robust deviation from a rolling baseline using preceding observations only |
| **M2** | PCA, two components | Squared prediction error outside the nominal principal subspace |
| **M3** | Isolation Forest | Unusualness under the fitted isolation model |
| **M4** | Feed-forward autoencoder, 6 → 4 → 2 → 4 → 6 | Squared multichannel reconstruction residual |

Each method produces sample-level scores; the **maximum score within each 30-minute window** determines the window score. Thresholds are model-specific and their numerical values are not directly comparable across scoring functions.

Implementation details are documented in [M0](../models/M0_robust_threshold.md), [M1](../models/M1_temporal_residual.md), [M2](../models/M2_pca_multivariate.md), [M3](../models/M3_isolation_forest.md), and [M4](../models/M4_autoencoder.md).

## 3. Frozen-model protocol

| Stage | Period | Purpose |
|---|---|---|
| Nominal training | January–May 2000 | Fit detector parameters using 445,722 nominal samples |
| Nominal validation | June–August 2000 | Calibrate each threshold at the 99th percentile of 30-minute maximum scores, using 4,385 nominal windows |
| Long-term evaluation | January 2001–December 2013 | Apply fitted models and thresholds without retraining or recalibration |

The model-specific frozen thresholds recorded in the research report are:

| Model | Threshold |
|---|---:|
| M0 | 2.408636332 |
| M1 | 2.173326984 |
| M2 | 2.331041565 |
| M3 | 0.603551727 |
| M4 | 16.572053909 |

The M3 value was checked against the stored `e3_frozen_2000.joblib` artifact. The study does **not** recalibrate thresholds on the evaluation period. M1's rolling reference is causal; it does not use future observations.

## 4. Evaluation metrics and interpretation

**Scorable NOMINAL alert fraction.** Among evaluation windows annotated NOMINAL and admitting a valid model score, the proportion producing an alert. The aggregate is weighted by the number of eligible windows across years, rather than averaging annual percentages equally. Missing scores are not counted as negative alerts. This quantity describes **nominally labeled alert burden**, **not** a verified false-positive rate.

**ESA-event overlap.** An event is counted if at least one alert episode intersects its annotated interval. An alert that began before the event can qualify. Persistent alerts can therefore increase overlap without providing a new event-onset signal.

**New alert episode during an event.** An event is counted if an alert episode starts inside its annotated interval. This stricter temporal criterion does not establish physical causality or operational actionability.

**Continuous alert episode.** A consecutive run of alerted windows, under the implemented continuity and missing-score conventions. Episode counts measure alert fragmentation and persistence; they are **not** counts of independent faults or confirmed false alarms.

The ESA event catalog is an imperfect reference. Neither NOMINAL labels nor annotated events provide exhaustive physical ground truth.

## 5. Long-term results

### 5.1 Aggregate comparison, 2001–2013

| Model | Scorable NOMINAL windows alerted | ESA events overlapped | ESA events with new episode | Continuous alert episodes |
|---|---:|---:|---:|---:|
| **M0 — Robust threshold** | 37.34% | 95/108 (87.96%) | 34/108 (31.48%) | 30,059 |
| **M1 — Temporal residual** | **1.48%** | 80/108 (74.07%) | **36/108 (33.33%)** | **3,555** |
| **M2 — PCA** | 83.45% | **106/108 (98.15%)** | 28/108 (25.93%) | 14,251 |
| **M3 — Isolation Forest** | 51.06% | 101/108 (93.52%) | 33/108 (30.56%) | 20,408 |
| **M4 — Autoencoder** | 5.24% | 63/108 (58.33%) | 22/108 (20.37%) | 10,808 |

**Interpretation.** M2 overlaps the largest number of ESA events, but its high overlap coincides with alerts in 83.45% of scorable NOMINAL windows. M1 produces substantially fewer nominally labeled alerts and the largest count of events containing a newly initiated episode, although it overlaps fewer events overall. M4 reduces nominal alert burden relative to M0, M2, and M3, but has lower event coverage. These descriptive metrics do not establish an absolute operational ranking.

### 5.2 Temporal stability

![Annual NOMINAL alert fraction, M0–M4](../figures/fig01_temporal_stability.png)

*Figure 1. Annual fraction of scorable ESA-NOMINAL windows flagged by each frozen detector, 2001–2013.*

M1 remains comparatively stable at approximately 1–2% nominal alert activity across the evaluation. M0, M2, and M3 show much greater long-term alert burden, with M2 and M3 approaching saturation in some later years. M4 shows a smaller but noticeable later-period increase. These patterns indicate changes in detector behavior; their physical causes have not been established.

### 5.3 Event categories and alert initiation

![Event overlap and new alert episodes by ESA category](../figures/fig02_event_detection.png)

*Figure 2. Event-overlap and new-episode fractions across all 108 events and by annotation category. The Communication Gap subset contains only four events.*

For the **46 ESA Anomaly events**:

| Model | Events with alert overlap | Events with new episode |
|---|---:|---:|
| M0 | 45/46 | 6/46 |
| M1 | 42/46 | 13/46 |
| M2 | 46/46 | 2/46 |
| M3 | 46/46 | 3/46 |
| M4 | 41/46 | 4/46 |

The difference between overlap and newly initiated episodes is particularly pronounced for M2 and M3. Event overlap alone cannot distinguish an alert that starts in response to changing telemetry from an alert state already active before the annotation interval.

### 5.4 Operational trade-off

![ESA event coverage versus NOMINAL alert burden](../figures/fig03_operational_tradeoff.png)

*Figure 3. Event-overlap and new-episode fractions versus the proportion of scorable NOMINAL windows flagged. The horizontal axis is not a validated false-positive rate.*

The joint view makes the trade-off visible: maximizing overlap can lead to a persistently alerted detector, while reducing alert burden can also reduce event coverage. On these selected measures, M1 provides a useful operating compromise, **not** a universally optimal solution.

## 6. Continuous-timeline consistency check

Alert episodes were reconstructed across the **concatenated 2001–2013 timeline** and compared with year-by-year episode extraction. For the **108 common ESA event IDs**, **zero event-level classifications changed** for either overlap or new-episode occurrence, for **all five models**.

This specifically checks the effect of year boundaries on the reported event-level classifications. It does not independently validate the ESA annotations, model scores, or physical meaning of alerts.

## 7. Historical artifacts and reproducibility

The repository uses **M0–M4** for detectors and **EVAL** for the evaluation protocol. Some previously generated CSV columns and filenames retain historical prefixes (`e0`–`e3` for earlier detectors and `e4` for the former evaluation stage). These identifiers are retained for compatibility with frozen artifacts; they do not indicate an additional E4 model.

The external experiment-output structure includes:

```text
outputs/experiments/
  m0/                          # Development-period M0 outputs
  m1/                          # Development-period M1 outputs
  m2/                          # Development-period M2 outputs
  m3/comparisons/              # Annual M3 outputs, historical e3_YYYY.csv names
  m4/comparisons/              # Annual M4 outputs
  m4/models/                   # Frozen M4 artifact
  eval/comparisons/            # Historical M0–M2 annual comparison outputs
  eval/models/                 # Frozen M0–M3 artifacts
  eval/development_2000/      # Historical M0–M2 development comparison
  eval/final_comparison/      # Five-model aggregate tables and figures
```

Key scripts include `scripts/run_eval_multiyear.py`, `scripts/run_m3_year.py`, `scripts/run_m4_year.py`, `scripts/verify_continuous_events.py`, `scripts/plot_final_comparison.py`, and `scripts/plot_eval_results.py`. Their execution and dependencies should be checked against the current configuration before claiming complete end-to-end reproducibility.

M0–M3 frozen artifacts were produced in the `satellite-health` environment with NumPy 2; the PyTorch-based M4 workflow uses `satellite-health-ae` with NumPy 1.26. Serialized NumPy-major-version-dependent artifacts should be loaded in their original compatible environment. Do not assume that historical joblib pickles can be transferred directly between these environments.

Raw ESA telemetry, event annotations, and generated artifacts are stored outside the Git repository. Access to those inputs and the appropriate local paths is required to reproduce the experiments.

## 8. Limitations

- **Restricted external validity:** one mission, one telemetry group, six channels, and a single historical model-development period.
- **Incomplete annotation ground truth:** ESA-NOMINAL windows are not proven free of meaningful deviations; event overlap is not an exhaustive detection-recall estimate.
- **Temporal dependence:** adjacent windows and alert episodes are not independent observations.
- **Alert timing versus causality:** a new episode within an annotated interval is not proof that the event caused the alert.
- **Frozen calibration:** no adaptive thresholds, regime-specific recalibration, or systematic retraining were evaluated.
- **Detector-specific missing scores:** eligible nominal-window denominators may differ.
- **Model selection:** the M4 network represents one fixed architecture and training setup, not a comprehensive neural search.
- **Operational costs:** acceptable alert burden and missed-event consequences require mission-specific definitions.

## 9. Conclusion

Across thirteen evaluation years, the five frozen detectors show substantially different balances of alert persistence, nominal alert burden, and ESA-event coverage. M2 attains the highest overlap (106/108), but at 83.45% nominal alert activity. M1 attains the lowest nominal alert fraction (1.48%) and the highest number of ESA events containing new alert episodes (36/108). The autoencoder does not improve on M1 under these selected operational proxies.

The central evaluation lesson is that **high event overlap cannot be interpreted independently of alert persistence and initiation timing**. These results support comparative R&D decision-making, but not claims of physical root-cause identification or mission-ready operational superiority.

For detailed scientific interpretation, conditional M0–M2 latency results, and future work, see the [Technical Research Report](../research_report.md).
