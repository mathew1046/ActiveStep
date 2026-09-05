# ActiveStep FOG detection assessment report

Generated: 2026-09-05T09:43:40+00:00
Source summary: `models/nested/baseline_v1/summary.json`

## Protocol

- Nested leave-one-participant-out; recipe (weights + cue threshold) selected on inner participant-grouped out-of-fold streams, refit on all outer-train participants, evaluated once on the held-out participant.
- Window 2.0 s, training hop 0.5 s, evaluation (deployment) hop 0.25 s, hysteresis 0.15.
- Operating point: max pooled event sensitivity s.t. false-cue starts <= 1.5/h (inner OOF); training label: any-freeze-in-window (current recipe).
- Threshold grid: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
- Weight grid: [[0.5, 0.5], [0.6, 0.4]]

## Dataset audit

- 17 recordings, 10 participants, 8.32 h total / 4.95 h valid experimental.
- Freeze time 28.9 min over 237 events; non-freeze experimental time 4.47 h.
- Participants without freezes (kept in evaluation for false alarms): [4, 10].
- Max timestamp gap 0.02 s; duplicate timestamps 0; non-monotonic 0; NaN samples 0.

## Candidate: `cnn_fi`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 95 | 0.401 | [0.269, 0.518] | 24.828 | [7.331, 48.005] |
| tol 0.3 s (annotation jitter) | 237 | 100 | 0.422 | n/a | 23.710 | n/a |

- Macro event sensitivity: 0.346; macro false cues/h: 24.435.
- Pooled window PR-AUC (endpoint label): 0.438.
- Mean onset-to-cue delay across folds: min 1.01 s, max 4.09 s (per-fold means).

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 11 | 0.478 | 28 | 56.14 | 0.50 | 0.95 | (0.6,0.4) |
| S02 | 24 | 9 | 0.375 | 4 | 11.67 | 0.34 | 0.95 | (0.6,0.4) |
| S03 | 49 | 30 | 0.612 | 49 | 102.35 | 0.48 | 0.95 | (0.6,0.4) |
| S04 | 0 | 0 | n/a | 0 | 0.00 | 0.58 | 0.95 | (0.6,0.4) |
| S05 | 66 | 19 | 0.288 | 8 | 17.83 | 0.45 | 0.95 | (0.6,0.4) |
| S06 | 10 | 0 | 0.000 | 2 | 3.87 | 0.52 | 0.95 | (0.6,0.4) |
| S07 | 24 | 12 | 0.500 | 4 | 9.42 | 0.42 | 0.95 | (0.6,0.4) |
| S08 | 14 | 0 | 0.000 | 1 | 6.33 | 0.16 | 0.95 | (0.6,0.4) |
| S09 | 27 | 14 | 0.519 | 15 | 36.74 | 0.41 | 0.95 | (0.6,0.4) |
| S10 | 0 | 0 | n/a | 0 | 0.00 | 0.62 | 0.95 | (0.6,0.4) |

## Candidate: `fi_only`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 38 | 0.160 | [0.055, 0.259] | 213.837 | [167.138, 257.774] |
| tol 0.3 s (annotation jitter) | 237 | 46 | 0.194 | n/a | 212.048 | n/a |

- Macro event sensitivity: 0.158; macro false cues/h: 214.164.
- Pooled window PR-AUC (endpoint label): 0.243.
- Mean onset-to-cue delay across folds: min 0.80 s, max 11.63 s (per-fold means).

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 0 | 0.000 | 38 | 76.19 | 0.50 | 0.30 | (0.0,1.0) |
| S02 | 24 | 8 | 0.333 | 112 | 326.84 | 0.34 | 0.30 | (0.0,1.0) |
| S03 | 49 | 0 | 0.000 | 76 | 158.75 | 0.48 | 0.30 | (0.0,1.0) |
| S04 | 0 | 0 | n/a | 128 | 222.61 | 0.58 | 0.30 | (0.0,1.0) |
| S05 | 66 | 18 | 0.273 | 111 | 247.34 | 0.45 | 0.30 | (0.0,1.0) |
| S06 | 10 | 3 | 0.300 | 89 | 172.39 | 0.52 | 0.30 | (0.0,1.0) |
| S07 | 24 | 6 | 0.250 | 140 | 329.72 | 0.42 | 0.30 | (0.0,1.0) |
| S08 | 14 | 0 | 0.000 | 24 | 151.81 | 0.16 | 0.30 | (0.0,1.0) |
| S09 | 27 | 3 | 0.111 | 86 | 210.62 | 0.41 | 0.30 | (0.0,1.0) |
| S10 | 0 | 0 | n/a | 152 | 245.38 | 0.62 | 0.30 | (0.0,1.0) |

## Candidate: `logistic_fi`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 106 | 0.447 | [0.278, 0.569] | 55.025 | [22.343, 91.413] |
| tol 0.3 s (annotation jitter) | 237 | 113 | 0.477 | n/a | 53.459 | n/a |

- Macro event sensitivity: 0.419; macro false cues/h: 61.332.
- Pooled window PR-AUC (endpoint label): 0.434.
- Mean onset-to-cue delay across folds: min 1.30 s, max 4.40 s (per-fold means).

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 11 | 0.478 | 60 | 120.30 | 0.50 | 0.95 | (0.6,0.4) |
| S02 | 24 | 0 | 0.000 | 2 | 5.84 | 0.34 | 0.95 | (0.6,0.4) |
| S03 | 49 | 30 | 0.612 | 53 | 110.70 | 0.48 | 0.95 | (0.6,0.4) |
| S04 | 0 | 0 | n/a | 4 | 6.96 | 0.58 | 0.95 | (0.6,0.4) |
| S05 | 66 | 29 | 0.439 | 52 | 115.87 | 0.45 | 0.95 | (0.6,0.4) |
| S06 | 10 | 0 | 0.000 | 3 | 5.81 | 0.52 | 0.95 | (0.6,0.4) |
| S07 | 24 | 9 | 0.375 | 14 | 32.97 | 0.42 | 0.95 | (0.6,0.4) |
| S08 | 14 | 13 | 0.929 | 19 | 120.18 | 0.16 | 0.95 | (0.6,0.4) |
| S09 | 27 | 14 | 0.519 | 38 | 93.06 | 0.41 | 0.95 | (0.6,0.4) |
| S10 | 0 | 0 | n/a | 1 | 1.61 | 0.62 | 0.95 | (0.6,0.4) |

## Claim scope

These are patient-independent, retrospective, offline detection results on the Daphnet dataset. They do not establish real-world wearable accuracy on our hardware, cueing effectiveness, fall reduction, or any clinical benefit. False-cue burden is reported per annotated non-FOG experimental hour (label 1 includes standing, walking and turning). The FP budget used for threshold selection is an engineering choice, not a clinical acceptance threshold.
