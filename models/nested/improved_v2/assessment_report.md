# ActiveStep FOG detection assessment report

Generated: 2026-09-05T17:42:20+00:00
Source summary: `models/nested/improved_v2/summary.json`

## Protocol

- Nested leave-one-participant-out; recipe (weights + cue threshold) selected on inner participant-grouped out-of-fold streams, refit on all outer-train participants, evaluated once on the held-out participant.
- Window 2.0 s, training hop 0.5 s, evaluation (deployment) hop 0.25 s; hysteresis grid [0.1, 0.15, 0.2].
- Operating point: max pooled event sensitivity with false-cue starts <= 5.0/h and unnecessary cueing <= 2.0 min/h (inner OOF); training label: freeze-at-window-end.
- Latency timeline: window availability time (includes resampling lookahead).
- Threshold grid: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995]
- Weight grid: [[1.0, 0.0]]

## Dataset audit

- 17 recordings, 10 participants, 8.32 h total / 4.95 h valid experimental.
- Freeze time 28.9 min over 237 events; non-freeze experimental time 4.47 h.
- Participants without freezes (kept in evaluation for false alarms): [4, 10].
- Max timestamp gap 0.02 s; duplicate timestamps 0; non-monotonic 0; NaN samples 0.

## Candidate: `gated_fi`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 91 | 0.384 | [0.271, 0.497] | 19.236 | [7.134, 34.022] |
| tol 0.3 s (annotation jitter) | 237 | 96 | 0.405 | n/a | 18.118 | n/a |

- Macro event sensitivity: 0.385; macro false cues/h: 18.778.
- Pooled window PR-AUC (endpoint label): 0.444; mean onset-to-cue delay: 3.55 s; unnecessary cueing: 0.97 min/h.
- Minimum viable engineering criteria: FAIL.
- Per-fold mean onset-to-cue delay range: 2.30–6.61 s.

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 10 | 0.435 | 22 | 44.11 | 0.50 | 0.99 | (1.0,0.0) |
| S02 | 24 | 9 | 0.375 | 2 | 5.84 | 0.34 | 0.99 | (1.0,0.0) |
| S03 | 49 | 26 | 0.531 | 29 | 60.57 | 0.48 | 0.99 | (1.0,0.0) |
| S04 | 0 | 0 | n/a | 0 | 0.00 | 0.58 | 0.99 | (1.0,0.0) |
| S05 | 66 | 14 | 0.212 | 10 | 22.28 | 0.45 | 0.99 | (1.0,0.0) |
| S06 | 10 | 1 | 0.100 | 1 | 1.94 | 0.52 | 0.99 | (1.0,0.0) |
| S07 | 24 | 9 | 0.375 | 9 | 21.20 | 0.42 | 0.99 | (1.0,0.0) |
| S08 | 14 | 7 | 0.500 | 0 | 0.00 | 0.16 | 0.99 | (1.0,0.0) |
| S09 | 27 | 15 | 0.556 | 13 | 31.84 | 0.41 | 0.99 | (1.0,0.0) |
| S10 | 0 | 0 | n/a | 0 | 0.00 | 0.62 | 0.99 | (1.0,0.0) |

## Candidate: `logistic_endpoint`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 5 | 0.021 | [0.008, 0.036] | 3.132 | [0.000, 6.324] |
| tol 0.3 s (annotation jitter) | 237 | 6 | 0.025 | n/a | 2.908 | n/a |

- Macro event sensitivity: 0.021; macro false cues/h: 2.772.
- Pooled window PR-AUC (endpoint label): 0.382; mean onset-to-cue delay: 8.83 s; unnecessary cueing: 0.11 min/h.
- Minimum viable engineering criteria: FAIL.
- Per-fold mean onset-to-cue delay range: 5.03–14.43 s.

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 0 | 0.000 | 0 | 0.00 | 0.50 | 0.99 | (1.0,0.0) |
| S02 | 24 | 0 | 0.000 | 0 | 0.00 | 0.34 | 0.97 | (1.0,0.0) |
| S03 | 49 | 2 | 0.041 | 2 | 4.18 | 0.48 | 0.97 | (1.0,0.0) |
| S04 | 0 | 0 | n/a | 0 | 0.00 | 0.58 | 0.99 | (1.0,0.0) |
| S05 | 66 | 1 | 0.015 | 0 | 0.00 | 0.45 | 0.99 | (1.0,0.0) |
| S06 | 10 | 0 | 0.000 | 0 | 0.00 | 0.52 | 0.97 | (1.0,0.0) |
| S07 | 24 | 0 | 0.000 | 0 | 0.00 | 0.42 | 0.97 | (1.0,0.0) |
| S08 | 14 | 1 | 0.071 | 0 | 0.00 | 0.16 | 0.95 | (1.0,0.0) |
| S09 | 27 | 1 | 0.037 | 5 | 12.25 | 0.41 | 0.95 | (1.0,0.0) |
| S10 | 0 | 0 | n/a | 7 | 11.30 | 0.62 | 0.95 | (1.0,0.0) |

## Candidate: `cnn_endpoint`

| Matching | Events | Detected | Event sensitivity | 95% CI | False cues / non-FOG h | 95% CI |
|---|---|---|---|---|---|---|
| tol 0.0 s | 237 | 41 | 0.173 | [0.070, 0.295] | 6.263 | [0.740, 15.320] |
| tol 0.3 s (annotation jitter) | 237 | 43 | 0.181 | n/a | 5.816 | n/a |

- Macro event sensitivity: 0.206; macro false cues/h: 5.960.
- Pooled window PR-AUC (endpoint label): 0.410; mean onset-to-cue delay: 3.56 s; unnecessary cueing: 0.86 min/h.
- Minimum viable engineering criteria: FAIL.
- Per-fold mean onset-to-cue delay range: 1.17–28.90 s.

| Subject | Events | Detected | Sensitivity | False starts | False/h | non-FOG h | Thr | Weights |
|---|---|---|---|---|---|---|---|---|
| S01 | 23 | 8 | 0.348 | 4 | 8.02 | 0.50 | 0.95 | (1.0,0.0) |
| S02 | 24 | 4 | 0.167 | 2 | 5.84 | 0.34 | 0.90 | (1.0,0.0) |
| S03 | 49 | 15 | 0.306 | 20 | 41.78 | 0.48 | 0.95 | (1.0,0.0) |
| S04 | 0 | 0 | n/a | 0 | 0.00 | 0.58 | 0.97 | (1.0,0.0) |
| S05 | 66 | 1 | 0.015 | 0 | 0.00 | 0.45 | 0.99 | (1.0,0.0) |
| S06 | 10 | 1 | 0.100 | 0 | 0.00 | 0.52 | 0.95 | (1.0,0.0) |
| S07 | 24 | 5 | 0.208 | 1 | 2.36 | 0.42 | 0.97 | (1.0,0.0) |
| S08 | 14 | 7 | 0.500 | 0 | 0.00 | 0.16 | 0.95 | (1.0,0.0) |
| S09 | 27 | 0 | 0.000 | 0 | 0.00 | 0.41 | 0.95 | (1.0,0.0) |
| S10 | 0 | 0 | n/a | 1 | 1.61 | 0.62 | 0.97 | (1.0,0.0) |

## Claim scope

These are patient-independent, retrospective, offline detection results on the Daphnet dataset. They do not establish real-world wearable accuracy on our hardware, cueing effectiveness, fall reduction, or any clinical benefit. False-cue burden is reported per annotated non-FOG experimental hour (label 1 includes standing, walking and turning). The FP budget used for threshold selection is an engineering choice, not a clinical acceptance threshold.
