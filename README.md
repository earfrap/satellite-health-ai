# Satellite Health AI

**Long-term anomaly detection and operational evaluation on spacecraft telemetry**

An applied AI / R&D study of how anomaly detectors behave when their training data and decision thresholds are frozen and then applied to **13 subsequent years** of spacecraft telemetry. The project compares simple statistical baselines, classical machine learning, and a neural-network autoencoder, with an emphasis on temporal generalization, alert persistence, and event-level interpretation.

> **Central finding:** High coverage of annotated events does not necessarily mean operationally useful detection. A detector that is almost continuously in alert can overlap many events without identifying their onset.

## Research question

**Can anomaly detectors calibrated on spacecraft telemetry from 2000 remain useful during 2001–2013 without retraining or threshold recalibration?**

Rather than selecting a model using event overlap alone, this study examines the trade-off among long-term nominal alert burden, annotated-event overlap, and alerts that *start* during annotated events.

## Data and experimental design

| Item | Scope |
|---|---|
| Source | ESA Mission 1 spacecraft telemetry and event annotations |
| Telemetry subset | Group 8, channels 41–46 (six channels) |
| Analysis unit | Non-overlapping 30-minute windows |
| Nominal model training | January–May 2000 |
| Threshold validation | June–August 2000 |
| Out-of-period evaluation | 2001–2013, with frozen models and thresholds |
| Evaluation set | 227,904 windows; 108 intersecting ESA annotated events |
| Event categories | Anomaly (46), Rare Event (58), Communication Gap (4) |

Models are fitted on an initial nominal training period; thresholds are selected on separate nominal validation windows using the 99th percentile of 30-minute maximum scores. The subsequent evaluation does **not** use 2001–2013 data to refit models or recalibrate thresholds. The temporal residual baseline uses preceding observations rather than future measurements.

This is a **focused case study**, not a benchmark across ESA missions or spacecraft. The raw ESA data and locally generated experiment artifacts are kept outside the Git repository.

## Five anomaly-detection models

| ID | Method | Anomaly signal |
|---|---|---|
| **M0** | Robust per-channel threshold | Robust deviations from nominal channel behavior |
| **M1** | Causal temporal residual | Deviations from a preceding rolling baseline |
| **M2** | Principal Component Analysis (PCA) | Multivariate reconstruction / squared prediction error |
| **M3** | Isolation Forest | Multivariate isolation-based anomaly score |
| **M4** | Feed-forward Autoencoder | Neural-network reconstruction error |

M4 uses a compact **6 → 4 → 2 → 4 → 6** architecture. Its inclusion tests whether greater model complexity yields better long-term operational behavior; it is not assumed to outperform the simpler baselines.

Detailed model notes: [M0](docs/models/M0_robust_threshold.md) · [M1](docs/models/M1_temporal_residual.md) · [M2](docs/models/M2_pca_multivariate.md) · [M3](docs/models/M3_isolation_forest.md) · [M4](docs/models/M4_autoencoder.md).

## Evaluation: three complementary measures

1. **Nominal alert fraction:** Percentage of scorable ESA-NOMINAL windows containing an alert. This measures *alert burden*, **not** a verified false-positive rate.
2. **Event overlap:** Percentage of annotated ESA events intersected by at least one alert, including alerts that started before the event.
3. **New alert episode:** Percentage of annotated events during which a new alert episode begins. This helps distinguish new responses from persistent alerts, but does not establish causal event detection.

All event-level percentages below use the **108 annotated events** intersecting 2001–2013. The nominal alert fraction is aggregated across evaluation windows.

### Long-term results (2001–2013)

| Model | Nominal windows with alert | ESA event overlap | Events with new alert episode |
|---|---:|---:|---:|
| M0 — Robust threshold | 37.34% | 95/108 (87.96%) | 34/108 (31.48%) |
| **M1 — Temporal residual** | **1.48%** | **80/108 (74.07%)** | **36/108 (33.33%)** |
| M2 — PCA | 83.45% | 106/108 (98.15%) | 28/108 (25.93%) |
| M3 — Isolation Forest | 51.06% | 101/108 (93.52%) | 33/108 (30.56%) |
| M4 — Autoencoder | 5.24% | 63/108 (58.33%) | 22/108 (20.37%) |

**Interpretation.** M2 and M3 achieve high event overlap but exhibit substantial nominal alert burden; overlap is therefore inflated by persistent alert states. M1 provides a comparatively favorable compromise **under these specific metrics**, maintaining a low nominal alert fraction while starting new alert episodes during 36 of 108 events. M4 reduces nominal alert burden relative to M0/M2/M3 but also has lower event overlap than M1. These observations are descriptive, not evidence that one method is universally optimal.

### 1. Temporal stability

![Yearly nominal alert fraction, M0–M4](docs/figures/fig01_temporal_stability.png)

*Fraction of scorable ESA-NOMINAL 30-minute windows containing an alert in each evaluation year. The pronounced increases for M0, M2 and M3 contrast with the relative stability of M1.*

### 2. ESA event coverage

![ESA event overlap and new alert episode coverage by event category](docs/figures/fig02_event_detection.png)

*Overlap and newly initiated alert episodes are shown separately for all 108 events and for Anomaly (46), Rare Event (58) and Communication Gap (4). Small category counts, particularly Communication Gap, require caution.*

### 3. Operational trade-off

![Event coverage versus nominal alert burden for M0–M4](docs/figures/fig03_operational_tradeoff.png)

*The horizontal axis measures nominal alert burden, not false-positive rate. Circles represent event overlap; triangles represent events with a newly initiated alert episode. Both event metrics use the same 108-event denominator.*

## Scientific conclusions and limitations

- **Event overlap alone is insufficient:** persistent alerts can overlap events without signaling their onset.
- **Temporal generalization matters:** models calibrated on one period can produce very different alert burdens years later.
- **Complexity is not a guarantee of operational value:** the PCA, Isolation Forest and Autoencoder models do not uniformly outperform the causal temporal baseline under the selected metrics.
- **ESA annotations are an incomplete reference:** windows labeled nominal are not guaranteed to be free of anomalies; nominal alert fraction must not be reported as a true false-positive rate.
- **New episodes are not causal attribution:** an episode starting during an event does not prove the event triggered it, nor does overlap establish diagnosis.
- **Scope is limited:** one mission, one telemetry group, six channels, and a fixed historical evaluation period. No claims are made about transfer to other missions or operational readiness.

For methodological detail, including historical development comparisons and evaluation caveats, see the [Technical Research Report](docs/research_report.md) and [EVAL documentation](docs/evaluation/EVAL_comparative_evaluation.md).

## Repository organization

```text
src/satellite_health/   # Core data, modeling and evaluation code
scripts/                # Preparation, model export, evaluation and plotting
configs/data/           # Example/local paths to external ESA data and outputs
docs/models/            # M0–M4 model documentation
docs/evaluation/        # Long-term evaluation methodology
docs/figures/           # Selected figures displayed on GitHub
tests/                  # Automated tests
```

Historical CSV filenames and column keys may retain `e0`, `e1`, `e2`, `e3` or `e4` prefixes for backward compatibility. **M0–M4** denote the five models; **EVAL** denotes the evaluation workflow. These legacy data fields do not indicate additional models.

## Installation and reproducibility

The project uses **Python 3.12** and local ESA telemetry files that are **not distributed** with the repository. After cloning, configure the external data and output directories:

```bash
python -m pip install -e ".[dev]"
cp configs/data/paths.example.yaml configs/data/paths.local.yaml
# Edit paths.local.yaml for your own local ESA dataset and output directories.
```

**Environment compatibility is important.** Historical M0–M3 joblib artifacts were serialized in an environment using NumPy 2.x. The M4 Autoencoder workflow uses PyTorch 2.2.2 with NumPy 1.26.4. In the original research setup these are separate Conda environments, named `satellite-health` and `satellite-health-ae`; a single fresh environment should **not** be assumed to load every historical artifact safely. In particular, do not try to work around pickle incompatibility by aliasing NumPy internal modules.

Validated test commands in the respective research environments:

```bash
# In the satellite-health environment (historical M0–M3 stack):
python -m pytest -q --ignore=tests/test_autoencoder.py

# In the satellite-health-ae environment (compatible M4/PyTorch stack):
python -m pytest -q
```

The recorded results were **55 passed** and **58 passed**, respectively. These test results validate the tested code paths; they do not imply that the full ESA evaluation can be reproduced without the external source data, preprocessing outputs and frozen artifacts.

### Workflow overview

1. Prepare Group 8 annual telemetry and ESA event/window annotations using the data-preparation scripts.
2. Fit and export M0–M4 using the designated **2000 training and validation splits**.
3. Apply frozen models and thresholds to **2001–2013** telemetry, without recalibration.
4. Aggregate annual nominal alert burden and event-level overlap/new-episode results in the **EVAL** workflow.
5. Generate comparative figures from the saved evaluation tables.

Representative scripts include `scripts/prepare_group8_year.py`, `scripts/build_year_window_table.py`, `scripts/export_m0_m1_models.py`, `scripts/export_m2_model.py`, `scripts/export_m3_model.py`, `scripts/export_m4_model.py`, `scripts/run_eval_multiyear.py`, `scripts/run_m3_year.py`, `scripts/run_m4_year.py`, and `scripts/plot_final_comparison.py`.

**Reproduction note:** These are workflow entry points, not a claim of a one-command, clean-room reproduction. The model-export scripts, external data dependencies and cross-NumPy artifact compatibility must be audited before publishing a fully automated end-to-end reproduction recipe. Plotting saved evaluation CSVs is distinct from retraining or reevaluating frozen models.

## R&D relevance

This project demonstrates an end-to-end applied research workflow: defining an operationally meaningful question, designing a leakage-aware temporal protocol, comparing statistical/ML/neural approaches, analyzing failure modes and metric limitations, preserving frozen experiments, and communicating evidence-based trade-offs rather than optimizing a single headline score.
