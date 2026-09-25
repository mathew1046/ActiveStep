# ActiveStep — FOG Detection Model

On-device freezing-of-gait detection for Parkinson's patients. A tiny 1D CNN
runs on the ESP32 shank band and fuses its output with the classical Freeze
Index to drive real-time cueing.

## At a Glance

| | |
|---|---|
| Architecture | 1D CNN (2 conv blocks) + dense head |
| Parameters | **~3,700** |
| On-device size | **13.9 KB** (int8 TFLite Micro) |
| Input | 2 s window of 3-axis shank acceleration @ 100 Hz |
| Inference rate | every 250 ms (hop), real-time on ESP32 |
| Output | `pFOG` — freeze probability per window |
| Training data | Daphnet FOG dataset, leave-one-subject-out CV |

## Architecture

```
Input (200 × 3)  ── shank accelerometer, per-axis standardized
  └─ Conv1D(16, k=7) → BatchNorm → MaxPool(2)
  └─ Conv1D(32, k=5) → BatchNorm → GlobalAveragePooling
  └─ Dense(16, ReLU) → Dropout(0.2)
  └─ Dense(1, sigmoid)  →  p_cnn
```

Deliberately small: large receptive-field kernels (7, then 5) capture the
3–8 Hz freeze tremor; global pooling keeps it cheap and shift-tolerant. The
entire net is ~3.7k parameters — small enough for int8 quantization with room
to spare in the ESP32 latency budget.

## Hybrid Scoring — CNN + Freeze Index

The CNN doesn't decide alone. Its probability is fused with the classical
**Freeze Index** (Moore et al. 2008: 3–8 Hz freeze-band ÷ 0.5–3 Hz locomotor-band
power), each calibrated to [0, 1] first:

```
pFOG = 0.6 · norm(p_cnn) + 0.4 · norm(FI)
```

The combined score feeds a **hysteresis state machine** (separate on/off
thresholds, minimum cue duration, refractory period) that prevents cue
flicker — learned detection with a classical, explainable safety net.

## Training

- **Leave-one-subject-out** cross-validation — the model is always tested on a
  patient it has never seen (the clinically honest setting)
- Class-weighted loss (freeze windows are rare), early stopping,
  ReduceLROnPlateau
- Per-subject decision threshold + hysteresis tuned on inner folds only —
  no test-set leakage
- Final deployment model trained on all data, quantized to int8

## Results (nested CV, 10 subjects, event-level)

Per-window accuracy understates performance — what matters clinically is
whether a freeze *event* is caught, how fast, and how often the device
cries wolf:

| Metric | Result | Target |
|---|---|---|
| Event sensitivity | **~40%** of freeze events detected | ≥ 65% |
| Detection delay | **~1.8 s** mean after onset | ≤ 1.5 s |
| False cue starts | ~25 per non-freeze hour | ≤ 10 |
| Window PR-AUC | 0.44 | — |

Honest read: event detection works, but cross-patient generalization and
false-alarm rate are the known bottlenecks — the same gap the published
literature reports for Daphnet-trained models. Our roadmap targets exactly
this: adversarial patient-invariance training, multi-dataset augmentation,
and per-patient threshold personalization.

## Why This Design

- **Latency budget:** <100 ms inference target on ESP32 — a 3.7k-param CNN
  meets it; an LSTM/attention stack would not
- **Explainability:** the Freeze Index half of the score is a clinically
  validated biomarker, not a black box
- **Upgrade path:** the architecture accepts richer inputs (gyro axes,
  past-window embeddings) without restructuring the fusion layer
