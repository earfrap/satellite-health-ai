# M4 — Feed-forward Autoencoder for Spacecraft Telemetry Anomaly Detection

## 1. Objective

M4 investigates whether a compact nonlinear reconstruction model can identify deviations from nominal spacecraft telemetry behavior.

The detector is trained exclusively on NOMINAL telemetry from the year 2000 and evaluated without retraining over 2001–2013.

M4 complements the robust threshold (M0), temporal residual (M1), PCA reconstruction (M2), and Isolation Forest (M3) approaches.

## 2. Data and experimental protocol

The experiment uses ESA Mission 1, Group 8 telemetry, comprising six channels (41–46).

The temporal protocol is:

- **Training:** January–May 2000, NOMINAL samples only.
- **Validation:** June–August 2000, NOMINAL windows only.
- **Frozen evaluation:** 2001–2013.

The training dataset contains approximately 445,722 nominal samples. Validation uses 4,385 nominal 30-minute windows.

The scaler and neural network are fitted on the training subset only. Validation data are reserved for threshold calibration.

## 3. Model architecture

M4 implements a fully connected autoencoder:

**6 → 4 → 2 → 4 → 6**

The encoder compresses six telemetry measurements into a two-dimensional latent representation. The decoder reconstructs the original six-dimensional input.

ReLU activations follow the first three linear layers. The final reconstruction layer is linear.

Before training, input telemetry is transformed using the `RobustChannelScaler` also employed by the PCA detector.

## 4. Training

The implementation uses PyTorch with:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 20 |
| Batch size | 1024 |
| Training loss | Mean squared error |
| PyTorch random seed | 42 |
| Execution | CPU, single PyTorch thread |

Training minimizes the mean squared reconstruction error across samples and channels.

The training procedure does not use ESA event annotations as supervised targets.

## 5. Anomaly scoring

For a scaled telemetry vector \(z_t\), let \(\hat z_t\) denote the autoencoder reconstruction.

The sample-level anomaly score is:

\[
s_t = \sum_{j=1}^{6}(z_{t,j}-\hat z_{t,j})^2
\]

Larger scores indicate greater reconstruction error relative to the representation learned from nominal training telemetry.

**Implementation detail:** the anomaly score is the *sum* of squared channel errors, whereas the training loss is their *mean*. These are different numerical quantities, although both measure reconstruction discrepancy.

## 6. Threshold calibration

Sample-level validation scores are aggregated into non-overlapping 30-minute windows using the maximum score in each window:

\[
S_w = \max_{t \in w}s_t
\]

The frozen threshold is the empirical 99th percentile of the NOMINAL validation-window scores:

\[
\tau_{M4}=Q_{0.99}(\{S_w\}_{\mathrm{validation}})
\]

A window is flagged when its score exceeds the frozen threshold, according to the evaluation implementation.

The threshold is not adjusted using telemetry or ESA event labels from 2001–2013.

## 7. Longitudinal evaluation

The frozen M4 detector is evaluated over 13 years (2001–2013) alongside M0–M3.

The common evaluation contains 108 ESA events.

| Metric | M4 result |
|---|---:|
| NOMINAL alerted-window fraction | 5.24% |
| ESA event overlap | 63/108 (58.33%) |
| ESA events with newly starting alert episode | 22/108 (20.37%) |
| Alert episodes | 10,808 |

The NOMINAL alerted-window fraction measures alert burden within windows labeled NOMINAL. It must not be interpreted as a conventional false-positive rate, because NOMINAL labels do not guarantee the absence of every unannotated operational deviation.

Likewise, ESA event overlap indicates temporal coincidence, not necessarily successful causal detection or diagnosis.

The newly starting alert episode metric is more restrictive than simple temporal overlap because it distinguishes alerts initiated during an event from alerts already active beforehand.

## 8. Comparison with other detectors

M4 produces a substantially lower NOMINAL alert fraction than M0, M2, and M3, but a higher fraction than M1.

Its ESA event overlap and newly starting episode coverage are lower than those of M1.

Under the operational metrics adopted in this study, M1 offers a more favorable compromise between alert burden and event coverage.

This conclusion is specific to the selected dataset, temporal protocol, event annotations, and evaluation definitions. It does not establish that linear methods universally outperform autoencoders.

## 9. Temporal consistency

The evaluation includes a consistency check comparing event classification under year-by-year processing and continuous processing across year boundaries.

For M4, no event classification changes were observed for either event overlap or newly starting alert episodes.

This check addresses an evaluation-boundary artifact; it does not establish general robustness to mission drift or distribution shift.

## 10. Limitations

- Only one spacecraft mission and one telemetry group are included.
- The autoencoder uses a compact architecture without a hyperparameter search.
- The network processes individual telemetry samples and does not explicitly model temporal sequences.
- Reconstruction error does not provide a physical diagnosis of the underlying spacecraft subsystem.
- The ESA event catalogue may not capture every operationally meaningful deviation.
- A frozen threshold calibrated in 2000 may become less representative as the telemetry distribution evolves.
- The experiment does not establish uncertainty bounds or statistical superiority between models.

## 11. Reproducibility and artifacts

Implementation:

- `src/satellite_health/models/autoencoder.py`
- `scripts/export_m4_model.py`
- `scripts/run_m4_year.py`
- `tests/test_autoencoder.py`

Frozen outputs:

- `outputs/experiments/m4/models/m4_frozen_2000.joblib`
- `outputs/experiments/m4/models/m4_training_history.csv`

The model artifact contains the fitted detector and its frozen threshold.

The experiment was executed with Python 3.12, NumPy 1.26.4, and PyTorch 2.2.2.

**Reproducibility caveat:** the current training script contains a compatibility workaround for telemetry serialized with NumPy 2.x. This dependency must be resolved or documented through a supported data-conversion workflow before claiming fully reproducible training from the original serialized inputs.

## 12. Scientific interpretation

M4 demonstrates that a nonlinear reconstruction detector can be integrated into the same frozen-training and longitudinal evaluation framework as classical statistical and machine-learning methods.

Its value in this project is both methodological and comparative: it establishes a neural-network baseline and demonstrates that additional model complexity does not automatically improve operational anomaly-detection performance.

The comparison supports selecting models based on measurable operational trade-offs rather than model complexity alone.
