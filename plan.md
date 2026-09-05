# ActiveStep — Build Plan

Companion to `context.md`. That file says *what* and *why*; this one says *how*: wiring, software architecture, integration phases, demo strategy, and verification.

---

## 1. System Overview

```
 SHANK BAND (Unit 1)                          WAIST UNIT (Unit 2)
 ┌─────────────────────────┐                  ┌──────────────────────────────────────────┐
 │ ESP32 DevKit V1         │   WiFi (UDP)     │ Arduino UNO Q                            │
 │  IMU-A (MPU-6050) ──I2C─┤ ───────────────► │  ┌─ Linux (Cortex-A53 x4, Debian) ─────┐ │
 │  Freeze Index + TFLite  │ ◄─────────────── │  │ ingest → SQLite                     │ │
 │  cue FSM                │  threshold /     │  │ tier2 features + context model       │ │
 │  motor-1  laser-1       │  modality /      │  │ metronome (numpy → PipeWire/BlueZ)  │ │
 │  SW TRUE  SW FALSE      │  tempo / OTA     │  │ bandit · personalize · fall FSM     │ │
 └─────────────────────────┘                  │  │ FastAPI + WebSocket dashboard        │ │
                                              │  │ WiFi AP "ActiveStep"                 │ │
        ┌────────────────┐    WiFi (HTTP/WS)  │  └──────────────▲──────────────────────┘ │
        │ Caregiver phone│ ◄────────────────► │            Bridge RPC                    │
        │ activestep.local│                   │  ┌──────────────▼──────────────────────┐ │
        └────────────────┘                    │  │ STM32U585 MCU (Arduino/Zephyr)      │ │
                                              │  │ IMU-B (MPU-6050) @100 Hz            │ │
        ┌────────────────┐    Bluetooth A2DP  │  │ fall primitives · motor-2 · laser-2 │ │
        │ BT speaker     │ ◄──────────────────│  │ SW "I'm OK"                          │ │
        └────────────────┘                    │  └─────────────────────────────────────┘ │
                                              │  USB-C ◄── 5V/3A power bank              │
                                              └──────────────────────────────────────────┘
```

Time-critical loop (IMU-A → pFOG → cue) lives entirely inside the ESP32. Everything crossing WiFi is advisory or logging.

---

## 2. Hardware Connections

### 2.1 Common driver circuit (used 4×: 2 motors, 2 lasers)

Low-side N-channel switch with the AO3400A:

```
      V_LOAD (3V3 for motor, 3V3 or 5V for laser)
         │
        LOAD ──┐
         │     │ 1N5819 (motor only; cathode/stripe to V_LOAD side)
         ├─────┘
         │ Drain
   GPIO ─100Ω─┤ Gate      AO3400A
         │     Source
        10kΩ    │
         │      │
        GND ───GND
```

- 100 Ω in series with the gate limits switching current from the GPIO.
- 10 kΩ from gate to GND keeps the load off while the MCU is booting / pins are floating.
- 1N5819 across the motor only (cathode to +). Lasers are resistive/regulated modules — no diode needed, but harmless if added.
- AO3400A Vgs(th) is ~1.05 V typical, Rds(on) low at 2.5 V — fully on from 3.3 V logic on both boards.
- Power lasers from 5 V (`VIN` on ESP32 / `5V` pin on UNO Q) only if the module is rated for it; otherwise 3V3. Check the module label.

### 2.2 Unit 1 — ESP32 DevKit V1 (30-pin)

| Function | ESP32 pin | Notes |
|---|---|---|
| IMU-A SDA | GPIO21 | Default `Wire` SDA |
| IMU-A SCL | GPIO22 | Default `Wire` SCL |
| IMU-A VCC / GND | 3V3 / GND | MPU-6050 breakout has an onboard LDO; 3V3 works on the GY-521 |
| IMU-A AD0 | GND | I2C address 0x68 |
| IMU-A INT | GPIO19 | Optional; data-ready interrupt (can poll instead) |
| Motor 1 MOSFET gate | GPIO25 | via 100 Ω, 10 kΩ pull-down |
| Laser 1 MOSFET gate | GPIO26 | via 100 Ω, 10 kΩ pull-down |
| SW TRUE-freeze | GPIO32 → GND | `INPUT_PULLUP`, debounce in software (50 ms) |
| SW FALSE-alarm | GPIO33 → GND | `INPUT_PULLUP` |
| Status LED | GPIO2 | Onboard LED: blink = link OK, solid = cueing |
| Power | micro-USB from small power bank | ~5 V; board LDO gives 3V3 |

Pin rules: avoid GPIO 0, 2 (as an input), 12, 15 for anything load-related (boot strapping); GPIO 34–39 are input-only with no pull-ups — don't use for the switches unless adding external 10 kΩ pull-ups; GPIO 6–11 are flash — never use.

IMU-A mounting: board flat against the lateral shank, X axis pointing forward (walking direction), Z axis up. Record the orientation and keep it fixed — the model is trained in this frame.

### 2.3 Unit 2 — Arduino UNO Q (MCU side, 3.3 V logic)

| Function | UNO Q pin | Notes |
|---|---|---|
| IMU-B | Qwiic connector (SDA/SCL/3V3/GND) or header `SDA`/`SCL` | 3.3 V I2C, address 0x68. Qwiic needs a JST-SH cable → dupont adapter |
| Motor 2 MOSFET gate | D5 | via 100 Ω, 10 kΩ pull-down |
| Laser 2 MOSFET gate | D6 | via 100 Ω, 10 kΩ pull-down |
| SW "I'm OK" | D2 → GND | `INPUT_PULLUP` |
| Load supply | 3V3 / 5V header pins | 5V pin is bus power from USB-C — check current headroom |
| Power | USB-C from 5 V / 3 A power bank | Use a bank that supports ≥3 A; some banks sleep on low draw — pick one with an always-on mode |
| Speaker | Bluetooth (no wiring) | Pair once via `bluetoothctl`, mark trusted |

The UNO Q header is **3.3 V only** — do not feed 5 V into any digital pin. On-board 8×13 LED matrix is driven from the MCU and is free to use as a status display (pFOG bar / link state).

IMU-B mounting: waist unit on the belt at the lower back or hip, X forward, Z up, same convention as IMU-A.

### 2.4 Perfboard layout (2× 9 cm × 15 cm)

- **Board A (shank):** ESP32 on female headers at one end; two MOSFET driver cells in a row; screw terminals or JST for motor, laser, IMU lead, two switches. Keep the IMU on a short lead (<10 cm) and strain-relieve all leads with a zip tie through the perfboard.
- **Board B (waist):** UNO Q mounts *beside* the perfboard in the pouch (its headers are UNO-style, so a small shield-style perfboard on top is also an option). Two MOSFET driver cells, IMU-B lead, one switch.
- Lasers must be aimed at the floor ~40–60 cm ahead of the foot; mount laser 1 on the front of the shank band angled down, laser 2 on the front of the belt angled down.

### 2.5 BOM sanity check

| Part | Have | Need | OK |
|---|---|---|---|
| AO3400A | 4 | 4 (2 motors + 2 lasers) | ✓ no spares — solder carefully |
| 1N5819 | 4 | 2 (motors) | ✓ |
| 100 Ω | 5 | 4 | ✓ |
| 10 kΩ | 5 | 4 (gate pull-downs; switches use internal pull-ups) | ✓ |
| Microswitches | ≥3 | 3 | ✓ |
| Perfboard | 2 | 2 | ✓ |

To buy/borrow: USB power banks (1× 10,000 mAh 5V/3A, 1× small), Qwiic cable or dupont wires, straps/pouch, USB power meter (for the power-budget measurement).

---

## 3. Software Architecture

### 3.1 ESP32 firmware (Arduino framework, PlatformIO)

```
src/
  main.cpp          setup: WiFi STA join "ActiveStep", Wire, pins; loop: tasks below
  imu.cpp           MPU-6050 @100 Hz → ring buffer (400 samples = 4 s), low-pass, gravity removal
  features.cpp      every 250 ms on a 2 s window: FFT (arduinoFFT) → FI = P(3–8 Hz)/P(0.5–3 Hz),
                    locomotor power, step detection for cadence hint
  model.cpp         TFLite Micro interpreter; int8 1D-CNN on 200×3 window → p_model
  detector.cpp      pFOG = w1·norm(FI) + w2·p_model; hysteresis (on 0.6 / off 0.4, adjustable);
                    threshold = base + offset_from_unoq
  cue.cpp           FSM: IDLE → CUEING → RECOVERING → IDLE; modality mask from bandit;
                    motor pulsed at metronome tempo (PWM), laser on, request audio
  labels.cpp        switches with 50 ms debounce; 10 s label window after each cue
  link.cpp          UDP tx: {seq, t_ms, imu[20×6], FI, pFOG, state, events[]} @ 5 Hz
                    TCP/UDP rx: {threshold_offset, modality_mask, tempo_bpm, model_ota}
  ota_model.cpp     receive model blob → write to LittleFS → re-init interpreter; keep previous
```

Defaults are baked in so the band cues correctly with no UNO Q present.

### 3.2 UNO Q — MCU sketch (Arduino App Lab, Zephyr core)

```
sketch/
  sketch.ino        MPU-6050 @100 Hz; fall primitives (|a|<0.5g → |a|>2.5g within 500 ms →
                    2 s stillness/orientation change → FALL_SUSPECT); D5/D6 drivers;
                    D2 switch; LED matrix status.
                    Bridge RPC: imuBatch(ts, samples[]) → Linux; onCue(mask, tempo), onLedStatus(...) ← Linux
```

### 3.3 UNO Q — Linux services (Python 3, systemd)

```
activestep/
  ingest.py         UDP listener (ESP32) + Bridge subscriber (MCU) → in-memory ring + SQLite writer
  features.py       1 s features per sensor: cadence (autocorr of vertical acc), step amplitude,
                    step-time asymmetry (shank vs trunk step timing), festination index,
                    turn detection (integrated gyro yaw on IMU-B > 45° in 2 s)
  tier2.py          context risk → threshold_offset (e.g. −0.1 during turn / gait initiation)
  metronome.py      cadence → tempo = 1.10 × cadence; numpy click synth → sounddevice/PipeWire
                    → BT A2DP sink; sends tempo to ESP32 and MCU for haptic sync
  bandit.py         Thompson sampling, arms = {vib, laser, audio, vib+audio, all};
                    reward = 1/(time_to_recovery); persists posteriors in SQLite
  personalize.py    calibrate() 30 s baseline; fit_threshold() from labels (logistic on pFOG);
                    finetune() optional last-layer on labeled windows → tflite int8 → OTA push;
                    model_versions table + rollback on FP-rate regression
  fall.py           FSM: SUSPECT → play clip → 20 s countdown → OK (D2) | ALARM
                    (speaker alarm + dashboard alert + optional ntfy/Telegram)
  replay.py         streams Daphnet files through ingest at real-time speed
  api.py            FastAPI: REST (events, labels, meds, calibrate, replay, export.csv)
                    + WebSocket /live (pFOG, FI, cadence, spectrogram frames, alerts)
  static/           index.html, app.js (Chart.js), style.css
  db.py             SQLite schema (below)
  esp_link.py       command channel to ESP32 (threshold_offset, modality, tempo, model_ota)
```

System setup: NetworkManager hotspot `ActiveStep` (2.4 GHz, WPA2), avahi mDNS `activestep.local`, BlueZ + PipeWire for A2DP, `sounddevice`/`pyaudio` for playback, systemd units for `activestep-ingest`, `activestep-api`, `activestep-metronome`.

### 3.4 Data schema (SQLite)

| Table | Columns |
|---|---|
| `samples` | t_ms, sensor (A/B), ax, ay, az, gx, gy, gz |
| `features` | t_ms, sensor, cadence, step_amp, fi, asymmetry, festination, turning |
| `events` | id, t_start, t_end, pfog_peak, fi_peak, modality_mask, recovery_ms, model_version, context |
| `labels` | event_id, label (TRUE/FALSE), t_ms, source (patient/caregiver/dashboard) |
| `falls` | t_ms, outcome (OK/ALARM/CANCELLED), response_ms |
| `meds` | t_ms, state (ON/OFF), note |
| `model_versions` | id, created, source (base/finetune), fp_rate, tp_rate, active |
| `bandit_state` | arm, alpha, beta |

### 3.5 Model training (laptop, once)

1. Download Daphnet FOG; use shank + trunk channels; resample to 100 Hz.
2. 2 s windows, 50% overlap; label = any freeze annotation in window; leave-one-patient-out split.
3. Model: 1D-CNN (Conv 16 → Conv 32 → GAP → Dense), <20 k params; also compute FI per window and report the FI-only baseline.
4. Quantize int8, export `.tflite`, convert to C array for the ESP32; also keep the Keras model on the UNO Q for last-layer fine-tuning.
5. Record held-out precision / recall / detection latency for the slide.

---

## 4. Integration Plan

| Phase | Work | Done when |
|---|---|---|
| **0 — Bench-up** | Flash both boards; read both IMUs to serial; blink each motor/laser via its MOSFET; UNO Q hotspot up; SSH to UNO Q over the AP | IMU values scroll on both; all 4 loads switch; `ssh activestep.local` works |
| **1 — Data path** | ESP32 joins AP, sends UDP packets; `ingest.py` logs to SQLite; Bridge RPC streams IMU-B; minimal dashboard live chart | Both IMU traces animate in the browser, packet loss <1% |
| **2 — Tier-1 loop** | Train model on laptop; FI + TFLite on ESP32; cue FSM; label switches; `replay.py` with Daphnet | Simulated shuffle-in-place triggers motor + laser within 100 ms; labels appear in DB; replay reproduces Daphnet freezes on the dashboard |
| **3 — Audio** | Pair speaker; cadence estimation; metronome synth; tempo → ESP32 haptic sync | Walking faster/slower changes tempo audibly; buzz pulses land on clicks |
| **4 — Tier-2 + Fall** | Bilateral features, turn detection, threshold offset push; MCU fall primitives; `fall.py` FSM with speech clips + D2 cancel + dashboard alert | Turning lowers threshold visibly on dashboard; dropping the waist unit onto a cushion triggers clip → countdown → alarm/cancel |
| **5 — Personalization** | Calibration flow; threshold fit from labels; bandit; model versions; optional fine-tune + OTA | After ~10 labels the threshold moves and FP count drops in replay; bandit preference chart updates; (stretch) new model version shows on ESP32 |
| **6 — Polish + demo** | Spectrogram, trends, medication overlay, CSV export; power measurement; band/pouch assembly; demo script; pitch copy review | Full run-through twice without touching a laptop; power numbers on the slide |

Suggested ordering priority if time runs short: 0 → 1 → 2 → 3 → 6, then 4, then 5. Phases 0–3 + 6 already give a complete, demoable closed loop.

---

## 5. Demo Script & Risk Mitigations

**Script (~4 min):**
1. Caregiver phone joins `ActiveStep` WiFi, opens `activestep.local` — live traces visible.
2. Presenter walks normally: cadence shows, metronome silent, pFOG low.
3. Presenter simulates a freeze (feet planted, rapid small shuffles / knee trembling): pFOG spikes, FI spectrogram lights up 3–8 Hz, motor + laser fire, metronome starts at ~1.1× the previous cadence.
4. Presenter steps over the laser line and walks — cues stop immediately (phasic).
5. Press TRUE switch → label logged; dashboard shows event with modality and recovery time.
6. Turn in place → threshold offset visibly drops (Tier-2 context).
7. Fall demo: drop the waist unit onto a cushion → "Are you okay?" clip → press "I'm OK" → logged. Second time, don't press → alarm + dashboard alert.
8. Show trends, medication overlay, bandit preference, model-version panel.

**Mitigations:**
- Own WiFi AP → no venue-network dependency.
- Replay mode on the dashboard → if live sensors misbehave, replay a Daphnet patient with real annotated freezes.
- Speaker pre-paired and trusted; fallback: UNO Q USB-C audio or laptop speaker via HDMI/USB.
- Spare charged power banks; USB power meter to show draw.
- Motor/laser tested with a jumper before demo; lasers are class 2 — never aim at faces.
- All claims in copy reviewed: "assistive prototype", "early detection", no diagnosis language.

---

## 6. Verification Checklist

- [ ] IMU-A → cue latency <100 ms (timestamp in ESP32 log: window end → GPIO set)
- [ ] UDP packet loss <1% over 10 min at 3 m
- [ ] ESP32 keeps cueing when UNO Q is powered off (link independence)
- [ ] Daphnet leave-one-patient-out: precision, recall, latency recorded
- [ ] Freeze Index alone vs FI+CNN: both numbers on the slide
- [ ] Fall FSM unit tests: OK path, ALARM path, cancel during countdown
- [ ] Bandit converges in simulation with a biased reward
- [ ] Threshold fit moves in the right direction with synthetic labels
- [ ] Dashboard fully functional on a phone with mobile data off
- [ ] Metronome has no dropouts during 5 min of continuous UDP ingest
- [ ] 2 h battery soak test both units; report measured mA
- [ ] CSV export opens in a spreadsheet with sensible columns
- [ ] Copy review: no diagnostic/prediction-horizon claims
