# Satellite Health AI

**Explainable Multivariate Anomaly Detection for Spacecraft Health Monitoring**

An applied AI / R&D project investigating how interpretable anomaly detection methods behave when deployed on long-term spacecraft telemetry, with particular emphasis on temporal generalization, alert persistence, and event-level evaluation.

## Research question

Can anomaly detectors calibrated on an initial period of spacecraft telemetry remain useful over subsequent years without retraining or threshold recalibration?

This project examines an important distinction between detecting unusual measurements and producing operationally meaningful alerts.

## Dataset and experimental scope

- Source: ESA Mission 1 spacecraft telemetry and event annotations.
- Scope: Group 8, channels 41–46.
- Analysis unit: non-overlapping 30-minute windows.
- Initial model calibration: 2000.
- Long-term evaluation: 2001–2013.
- Reference annotations: ESA Anomaly, Rare Event, and Communication Gap events.

The evaluation covers 108 unique annotated ESA events across 2001–2013, including 46 events classified as Anomaly.

Raw telemetry and generated experiment artifacts are stored outside the Git repository.

## Methods

Three interpretable anomaly detectors are compared.

| Detector | Approach | Window-level score |
|---|---|---|
| E0 | Robust per-channel median/MAD deviations | Maximum robust deviation |
| E1 | Causal rolling-median residuals | Maximum robust temporal residual |
| E2 | Principal Component Analysis (PCA) | Maximum squared prediction error |

All models are fitted using an initial nominal training period in 2000.

Decision thresholds are calibrated using a separate validation period in 2000 and subsequently frozen. No retraining or threshold recalibration is performed during the 2001–2013 evaluation.

The E1 rolling baseline uses only preceding observations to avoid future-data leakage.

## Long-term temporal generalization

The detectors exhibit markedly different behavior over time.

**E0 — Robust threshold detector**

The proportion of scorable NOMINAL windows generating alerts rises substantially in later years, especially around 2009–2012, before declining in 2013.

**E1 — Temporal residual detector**

The NOMINAL alert fraction remains comparatively stable, generally around 1–2% across the evaluation years.

**E2 — PCA detector**

The NOMINAL alert fraction increases strongly over time and approaches 100% in several later years.

These observations indicate different sensitivities to long-term changes in telemetry behavior. They do not, by themselves, establish the physical causes of those changes.

The NOMINAL alert fraction must not be interpreted as a definitive false-positive rate: ESA annotations do not establish that every nominally labeled window is free of anomalous behavior.

## Event-level evaluation

Two complementary criteria are reported:

1. **Alert overlap:** whether any detector alert overlaps an ESA event.
2. **New alert episode:** whether an alert episode starts during an ESA event.

An alert that began before an ESA event may satisfy the overlap criterion without representing a newly triggered response to that event.

### Results on ESA Anomaly events

| Detector | Alert overlap | New alert episode |
|---|---:|---:|
| E0 | 45/46 (97.8%) | 6/46 (13.0%) |
| E1 | 42/46 (91.3%) | 13/46 (28.3%) |
| E2 | 46/46 (100%) | 2/46 (4.3%) |

E2 achieves the highest event-overlap count but produces persistent alerts across much of the long-term evaluation period. Its high overlap rate therefore does not demonstrate superior operational detection.

E1 produces more newly initiated alert episodes during ESA Anomaly events than either E0 or E2, while maintaining a substantially more stable NOMINAL alert fraction.

These comparisons are descriptive and do not establish a universally optimal detector.

### Detection latency

For ESA Anomaly events with a new alert episode, latency is measured from the annotated event start to the end of the first qualifying 30-minute alert window.

| Detector | Events with new episodes | Median latency |
|---|---:|---:|
| E0 | 6 | 109.1 min |
| E1 | 13 | 498.6 min |
| E2 | 2 | 718.3 min |

Latency statistics are conditional on a new episode occurring and are based on small, unequal event subsets. They should not be interpreted as direct, like-for-like rankings of response speed.

## Key results at a glance

### Long-term alert behavior (2001–2013)

![Long-term NOMINAL alert fraction for E0, E1 and E2](docs/figures/e4_nominal_alert_fraction.png)

*Fraction of scorable NOMINAL windows producing alerts. E1 remains comparatively stable, while E0 increases in later years and E2 approaches alert saturation.*

### ESA Anomaly event evaluation

![ESA Anomaly event detection comparison](docs/figures/e4_anomaly_detection.png)

*Comparison of alert overlap and newly initiated alert episodes across 46 ESA Anomaly events. High event overlap does not necessarily imply effective event-onset detection.*

## Scientific interpretation

The central finding is that high event overlap does not necessarily imply useful anomaly detection.

A detector that remains continuously in an alert state can overlap many reference events without distinguishing their onset.

Long-term monitoring performance must therefore consider:

- Stability of alert behavior over time.
- The distinction between persistent alerts and newly triggered episodes.
- Coverage and windows that cannot be scored.
- The limitations of available reference annotations.
- The trade-off between sensitivity and operational interpretability.

## Reproducibility

The repository contains scripts for:

- Preparing annual Group 8 telemetry.
- Building annual window-level label tables.
- Exporting frozen E0, E1, and E2 models.
- Running annual and multi-year E4 comparisons.
- Summarizing long-term and event-level results.
- Generating scientific figures.
- Testing event-level evaluation logic.

Local data paths are configured through `configs/data/paths.local.yaml`.

The project uses Python and its scientific computing ecosystem. Raw ESA telemetry and locally generated model/output artifacts are not distributed in this repository.

For implementation details, refer to the `scripts/`, `src/satellite_health/`, `configs/`, and `tests/` directories.

## Limitations

- Results are limited to ESA Mission 1 Group 8 and six telemetry channels.
- ESA annotations are an imperfect reference for determining whether every alert is meaningful.
- A new alert episode during an event does not establish that the event caused the alert.
- The long-term analysis is observational and does not establish the physical causes of detector drift or alert saturation.
- No claim is made that the compared methods are optimal for operational spacecraft deployment.
- The results do not demonstrate generalization to other spacecraft or missions.

## Project positioning

This project demonstrates applied research capabilities in:

- Multivariate telemetry analysis.
- Explainable and interpretable anomaly detection.
- PCA-based machine learning.
- Temporal validation and generalization analysis.
- Event-based evaluation design.
- Scientific Python, reproducible experiments, and critical interpretation of model behavior.

**Research focus:** developing and evaluating AI methods as part of a scientifically defensible R&D workflow, rather than optimizing a single performance metric in isolation.
