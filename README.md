# ActiveStep — Full Software Stack

End-to-end software for the ActiveStep Parkinson's FOG (Freezing of Gait) assistive prototype.

## What's included

| Layer | Path | What it does |
|---|---|---|
| **Config** | `activestep/config.py` | Central configuration, general-purpose pin maps for Pi / ESP32 / UNO Q |
| **Hardware abstraction** | `activestep/hardware/` | `CueOutput`, `SwitchInput`, `IMUReader`, `StatusLED` with mock / Pi / UNO Q backends |
| **Real-time runtime** | `activestep/runner.py` | 100 Hz IMU → Freeze Index + CNN → cueing + UDP telemetry |
| **Model training** | `src/` | Daphnet loader, CNN training, LOO eval, TFLite export |
| **UNO Q Linux services** | `unoq/` | ingest, metronome, bandit, fall FSM, Tier-2 features, replay |
| **Dashboard** | `dashboard/` | FastAPI + WebSocket + Chart.js live UI |
| **UNO Q App Lab** | `firmware/unoq_app_lab/` | MCU sketch + Linux `app.py` + App Lab manifest |
| **ESP32 firmware** | `firmware/esp32/` | TFLite Micro tier-1 detector with WiFi/UDP |
| **Tests** | `tests/` | pytest unit tests |
| **Scripts** | `scripts/` | training, simulation, systemd, UNO Q setup |

## Quick start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate activestep
```

### 2. Train and export the model

```bash
bash scripts/run_training.sh
```

Artifacts appear in `models/final/`:
- `fog_cnn.keras` — full Keras model
- `model_quantized.tflite` — float32 TFLite (primary, 22 KB)
- `model_quantized_int8.tflite` — int8 TFLite (optional, 14 KB)
- `model_quantized.h` — C header for the ESP32
- `scaler.json` — z-score constants

### 3. Run the full software simulation

```bash
bash scripts/simulate.sh
```

This starts:
- `unoq.ingest`
- `unoq.features`
- `unoq.fall`
- `dashboard.main`
- `activestep.runner` in **mock** mode replaying Daphnet subject 01

Open `http://<ip>:8000` for the dashboard.

### 4. Run on real hardware

#### Raspberry Pi (general Pi pinouts)

```bash
ACTIVESTEP_PLATFORM=pi python -m activestep.runner
```

Edit `activestep/config.py` `PI_PINS` if needed.

#### UNO Q (App Lab)

1. Flash `firmware/unoq_app_lab/sketch/sketch.ino` to the UNO Q MCU.
2. Run `firmware/unoq_app_lab/app.py` from Arduino App Lab or standalone.
3. Or start services with `bash scripts/start_unoq.sh`.

#### ESP32

1. Generate `firmware/esp32/model_quantized.h` from `models/final/model_quantized.tflite`:
   ```bash
   xxd -i models/final/model_quantized.tflite > firmware/esp32/model_quantized.h
   ```
2. Build `firmware/esp32/main.ino` with the tflite-micro Arduino library.
3. It connects to the `ActiveStep` WiFi AP and streams telemetry.

## Configuration

All tunables live in `activestep/config.py`:
- `PLATFORM`
- pin maps (`PI_PINS`, `ESP32_PINS`, `UNOQ_MCU_PINS`)
- network (`UNOQ_IP`, `ESP32_IP`, ports)
- thresholds (`DEFAULT_PFOG_THRESHOLD`, `FREEZE_INDEX_THRESHOLD`)
- metronome (`METRONOME_RATIO = 1.10`)

Environment overrides:
- `ACTIVESTEP_PLATFORM=simulator|pi|unoq`
- `ACTIVESTEP_SSID`, `ACTIVESTEP_PASS`
- `ACTIVESTEP_DASHBOARD_PORT`

## Tests

```bash
python -m pytest tests/ -q
```

## Model assessment pipeline (use this for any reported numbers)

`scripts/run_assessment.sh` runs the leakage-free nested assessment
(the old `run_training.sh` path is superseded — see note below):

```bash
bash scripts/run_assessment.sh audit    # dataset inventory -> models/dataset_audit.json
bash scripts/run_assessment.sh cache    # window cache -> .cache/windows
bash scripts/run_assessment.sh smoke    # 1-epoch mechanics check on subjects 1-3
bash scripts/run_assessment.sh full     # full nested LOPO assessment (all subjects)
bash scripts/run_assessment.sh report   # markdown report from summary.json
```

What it does (see `plan.md` and module docstrings for the full protocol):

- **Corrected data pipeline** (`src/data.py`): annotations are never
  interpolated; recordings are windowed independently; windows never cross
  run boundaries or non-experimental segments; every window carries
  provenance (subject, run, decision time, labels).
- **Nested leave-one-participant-out** (`src/train_nested.py`): the recipe
  (combine weights + cue threshold) is selected on inner participant-grouped
  out-of-fold streams, the model is refit on all outer-train participants,
  and evaluated exactly once on the held-out participant.
- **Streaming benchmark** (`src/benchmark.py`): event sensitivity,
  false cues per non-FOG hour, onset-to-cue and cue-stop delays,
  unnecessary cue time, window metrics, bootstrap CIs, and a 0.3 s
  annotation-jitter sensitivity — all through the same cue FSM the device
  runtime uses (`src/detector.py`).
- **Candidates**: `fi_only` (Freeze Index baseline), `logistic_fi`
  (feature baseline), `cnn_fi` (current CNN + FI).

Outputs land in `models/nested/<tag>/` (per-fold recipes, sweeps,
predictions, results, `summary.json`) and `assessment_report.md`.

**Claim scope:** results are patient-independent, retrospective, offline
detection results on Daphnet. They do not establish real-world wearable
accuracy, cueing effectiveness, or clinical benefit.

## Model notes

- Input: 2 s @ 100 Hz, 3-axis shank acceleration (mg), z-scored
- Architecture: Conv1D(16) → Conv1D(32) → GAP → Dense(16) → Sigmoid
- Training data: Daphnet (10 subjects, resampled to 100 Hz)
- The **float32 TFLite** is the recommended artifact; the int8 variant is smaller but degrades the small model.
- The historical S01 fold metrics in `models/fold_s01/` predate the
  corrected data pipeline and an audit showed its window set was inconsistent
  with the raw recordings; do not cite them. Use the nested assessment
  (`bash scripts/run_assessment.sh full`) for any reported numbers.

## Project structure

```
ActiveStep/
├── activestep/              # config, hardware abstraction, runtime
│   ├── config.py
│   ├── hardware/
│   └── runner.py
├── src/                     # data, features, model, train, quantize, infer
├── unoq/                    # UNO Q Linux services
├── dashboard/               # FastAPI dashboard
├── firmware/                # ESP32, UNO Q MCU, App Lab
│   ├── common/pins.h
│   ├── esp32/
│   ├── unoq_mcu/
│   └── unoq_app_lab/
├── scripts/                 # run_training.sh, simulate.sh, start_unoq.sh, systemd/
├── tests/                   # pytest tests
├── models/                  # generated artifacts
└── README.md
```

## Safety / legal

This is a hackathon prototype, not a medical device. Do not make diagnostic claims. The pitch should say "early detection of freeze onset / assistive prototype".
