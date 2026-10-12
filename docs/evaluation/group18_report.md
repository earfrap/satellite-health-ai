# Group 18 — Cross-Group Pilot

## Objective

Evaluate whether the Satellite Health AI workflow can be applied to a second ESA Mission 1 telemetry group associated with a different spacecraft subsystem.

This experiment tests workflow portability through group-specific training and calibration. It does not evaluate transfer of pretrained Group 8 models.

## Experimental protocol

| Component | Configuration |
|---|---|
| Dataset | ESA Mission 1, Group 18 |
| Channels | 70–76 (7 channels) |
| Synchronization | Five-minute median resampling |
| Analysis windows | Non-overlapping 30-minute calendar windows |
| Training | January–May 2000 |
| Calibration | June–August 2000 |
| Training/calibration selection | NOMINAL windows with no Anomaly or Rare Event overlap |
| Recovery buffer | None |
| Threshold | 99th percentile of nominal calibration window maximum scores |
| Evaluation | 2001–2013 |
| Models | M0, M1, M2, M3 |
| M1 temporal baseline | Causal 30-minute rolling median, `min_periods=3` |

The models and their decision thresholds are fitted and calibrated separately for Group 18.

## Results

| Model | Evaluation windows | Alert windows | Alert fraction | Alert episodes | ESA events overlapped |
|---|---:|---:|---:|---:|---:|
| M0 — Robust Threshold | 226,928 | 2,476 | 1.091% | 1,947 | 7/24 (29.17%) |
| M1 — Temporal Residual | 226,920 | 5,434 | 2.395% | 4,797 | 10/24 (41.67%) |
| M2 — PCA | 226,928 | 4,039 | 1.780% | 3,498 | 14/24 (58.33%) |
| M3 — Isolation Forest | 226,928 | 174 | 0.077% | 166 | 0/24 (0%) |

The alert fraction represents the percentage of scorable evaluation windows containing an alert.

## Interpretation

M2 exhibits the highest overlap with annotated ESA events in Group 18, while M3 generates very few alerts and overlaps none of the 24 evaluated events.

The experiment demonstrates that the same modeling and calibration workflow can be executed on a second telemetry group, despite different channel synchronization requirements.

## Methodological limitations

- ESA event overlap is not equivalent to verified anomaly detection recall.
- The alert fraction is not a verified false-positive rate.
- ESA annotations may not cover all genuine anomalous behavior.
- Group 18 uses five-minute resampling, whereas the primary Group 8 analysis uses its original synchronized telemetry.
- Group 18 reports overall evaluation alert fraction; Group 8 reports ESA-NOMINAL alert fraction.
- Event-overlap and episode accounting are not fully identical across the two workflows.
- Results therefore support a descriptive cross-group assessment, not a controlled benchmark or claims of mission-wide generalization.

## Reproducibility

The pilot is implemented in `scripts/run_group_pilot.py`.

The detailed window-level results are stored outside the Git repository. The compact numerical summary is available in [group18_summary.csv](group18_summary.csv).
