# ActiveStep — Predictive Wearable Aid for Parkinson's Freezing of Gait (FOG)

## Hackathon Track

**Care Beyond Labels — Overcoming Overlooked Health Challenges That Hinder Human Growth and Well-Being**

The track asks for sustainable systems/technologies addressing overlooked or inadequately addressed health challenges, focused on improving quality of life, accessibility, prevention/recovery, and long-term independence.

## The Problem

Freezing of Gait (FOG) is a disabling, episodic motor symptom of Parkinson's disease: a sudden, involuntary inability to walk, often described as feeling like the feet are "glued to the floor." It affects over 50% of advanced Parkinson's patients, severely compromises postural stability, and dramatically raises fall risk. Levodopa and other pharmacological treatments are frequently ineffective for FOG — and can sometimes worsen it. Existing sensory cueing devices (e.g., continuous laser shoes) lose effectiveness over time due to brain habituation and cognitive fatigue. There is a real need for a device that predicts freezing before it happens and delivers on-demand, non-habituating, multi-modal cueing to restore walking — while giving caregivers/clinicians visibility into what's actually happening.

## What We're Building

A closed-loop wearable system that:
1. Continuously monitors gait using IMUs on the shank and trunk
2. Detects freeze onset and pre-freeze signatures (shuffling/trembling in the 3–8 Hz "freeze band", festination, turning) early enough to cue — expressed as a real-time pFOG index
3. Immediately triggers on-demand (phasic, not continuous) multi-modal cueing to help the patient resume walking
4. Detects falls independently and can alert a caregiver
5. Learns from real patient feedback over time, personalizing thresholds and cue selection to the individual
6. Gives caregivers/clinicians a simple dashboard into the patient's mobility patterns, without requiring constant internet connectivity
7. Chooses the cue modality that has worked best for this patient (a bandit over vibration / light / audio) instead of always firing everything, further reducing habituation

This is a prototype/demo build for a hackathon, not a certified medical device — framing and claims should stay appropriately scoped (assistive prototype, not diagnostic or clinically validated).

## Core Design Philosophy

- **Phasic, not continuous, stimulation.** Cues fire only when freeze risk is detected and stop the instant gait resumes. This preserves cue salience and avoids the sensory habituation that plagues continuous cueing devices.
- **Multi-modal cueing**, so the system remains effective across environments and patient states:
  - Vibration cue (haptic)
  - Light cue (visual)
  - Audio metronome cue (auditory rhythmic cueing, played over a Bluetooth speaker)
- **Personalization through real-world feedback.** Patients (or caregivers) confirm whether a detected freeze was real or a false alarm via physical buttons. Personalization is layered so that it works even with very few labels: (1) a 30-second baseline calibration of the patient's normal gait, (2) adaptive detection thresholds fitted from true/false labels, (3) a bandit that learns which cue modality recovers gait fastest, and (4) optionally, a last-layer fine-tune of the model on the UNO Q that is pushed over-the-air to the ESP32. We do not claim full on-device retraining from a handful of labels.
- **Explainable two-tier detection.** Tier 1 on the ESP32 combines a physics-based Freeze Index (freeze-band vs locomotor-band power ratio) with a small learned model, so every cue can be explained by a visible signal. Tier 2 on the UNO Q fuses both IMUs (bilateral asymmetry, festination, turning) and adjusts Tier 1's sensitivity — it never sits in the time-critical path.
- **Offline-first / privacy-conscious.** The system should be fully functional without an internet connection. Cloud sync (for remote clinician monitoring) is a nice-to-have layer on top, not a dependency for core operation. Patient mobility data should stay on-device by default.
- **Caregiver visibility.** Caregivers and clinicians currently rely on unreliable patient self-reported diaries for FOG frequency. A dashboard giving objective event logs, fall alerts, and trends over time is a meaningful improvement on its own.

## Physical Form Factor

The system is built as **two body-worn units**, not a single wearable + stationary hub:

- **Unit 1 — Shank band (ESP32):** worn on the lower leg / ankle. Houses the ESP32, IMU-A (MPU-6050), vibration motor 1, line laser 1 (aimed at the floor ahead of the foot), and the two labelling microswitches (TRUE freeze / FALSE alarm). This is the real-time sensing + cueing unit.
- **Unit 2 — Waist unit (UNO Q):** worn on a belt / hip pouch. Houses the Arduino UNO Q, IMU-B (MPU-6050, read by the UNO Q's STM32 MCU), vibration motor 2, line laser 2, the fall "I'm OK" microswitch, and a USB-C power bank. This is the compute / logging / audio / dashboard unit, and also a second cueing unit.

Why the waist and not a second leg band: (a) the UNO Q needs 5 V at up to 2–3 A and a power bank to match — too heavy for a leg band; (b) the trunk/hip is the standard sensor location in the FOG literature (e.g., the Daphnet dataset) and the standard location for fall detection; (c) trunk + shank gives two complementary views of gait rather than two of the same thing.

**Inter-unit communication is WiFi, not BLE.** The UNO Q runs a WiFi access point (`ActiveStep`, reachable as `activestep.local`); the ESP32 joins it and streams data over UDP; the caregiver's phone/laptop joins the same AP for the dashboard. Bluetooth on the UNO Q is reserved exclusively for the speaker (A2DP) — sharing one radio between BLE data and A2DP audio causes metronome dropouts, and running BLE + WiFi together on the ESP32 causes coexistence stalls. Running our own AP also means the demo does not depend on venue WiFi.

**Role of the two IMUs (decided):**
- IMU-A (shank) feeds the Tier-1 low-latency freeze detector on the ESP32. Shank acceleration has the strongest freeze-band signature.
- IMU-B (trunk) is used on the UNO Q for: fall detection (independent of FOG), bilateral/trunk-vs-shank asymmetry, festination trend (cadence rising while step amplitude falls), turn detection (gyro yaw), and as a second input to the Tier-2 context model that tunes Tier-1's threshold.

Power, weight, and battery life remain real constraints for both units (see Power Budget).

## Hardware We Have

| Component | Qty | Role |
|---|---|---|
| ESP32 DevKit V1 (30-pin) | 1 | Tier-1 real-time controller: samples IMU-A at 100 Hz, computes Freeze Index + runs TFLite Micro model, fires cue set 1 with <100 ms latency, streams to UNO Q over WiFi UDP, accepts threshold/model updates |
| Arduino UNO Q (2GB RAM) | 1 | **MCU side (STM32U585):** reads IMU-B at 100 Hz, runs fall-detection primitives, drives cue set 2, reads "I'm OK" switch. **Linux side (quad Cortex-A53):** WiFi AP, data ingest + SQLite logging, Tier-2 bilateral/context model, cadence + adaptive metronome (Bluetooth audio), cue-modality bandit, personalization (calibration, threshold fitting, optional fine-tune + OTA), fall escalation, caregiver dashboard (FastAPI + HTML/JS), Daphnet replay mode |
| MPU-6050 (tri-axial IMU) | 2 | IMU-A on the shank band (ESP32), IMU-B on the waist unit (UNO Q MCU via Qwiic/I2C) |
| Coin vibration motor (1027, 3V) | 2 | Haptic cueing — one per unit; pulsed in sync with the metronome tempo, not continuous |
| Line laser module (3–5V, line not dot) | 2 | Visual cueing — projects a line on the floor as a visual "obstacle" to step over; one per unit |
| AO3400A (N-channel MOSFET) | 4 | Driving vibration motors / laser modules from digital GPIO |
| 1N5819 (Schottky diode) | 4 | Flyback/protection diodes for inductive/motor loads |
| 100 Ω resistor | 5 | Gate/current-limiting resistors |
| 10 kΩ resistor | 5 | Pull-down/pull-up resistors (e.g., for MOSFET gates, microswitches) |
| Perfboard (9cm x 15cm) | 2 | Circuit assembly |
| Bluetooth speaker | 1 | Plays the adaptive metronome, fall check-in speech clips, and alarm (A2DP from UNO Q Linux) |
| Microswitches | 3 | Shank band: TRUE-freeze and FALSE-alarm labels. Waist unit: "I'm OK" fall cancel |
| USB power bank, 5V/3A, ~10,000 mAh | 1 | Powers the UNO Q waist unit via USB-C |
| USB power bank, small (~2,000–5,000 mAh) | 1 | Powers the ESP32 shank band via micro-USB |
| USB microphone (optional, stretch) | 0–1 | Only if offline voice reply to the fall check-in is attempted |

**Note:** The FSR insole sensors described in the original abstract have been removed from this build. Freeze detection relies solely on the IMU data (no plantar pressure sensing in this version).

## Functional Requirements

### Tier-1 FOG Detection & Cueing (ESP32, shank band)
- Sample IMU-A at 100 Hz into a ring buffer; evaluate a 2–4 s sliding window every 250 ms
- Compute the **Freeze Index** (power in 3–8 Hz freeze band ÷ power in 0.5–3 Hz locomotor band) and a **TFLite Micro 1D-CNN** on the same window; combine into a real-time pFOG
- When pFOG crosses the (adaptive) threshold with hysteresis, immediately fire the cue set selected by the bandit (vibration and/or laser locally; audio requested from the UNO Q)
- Stop cueing as soon as normal gait resumes (locomotor-band power returns, cadence detected)
- Read the TRUE / FALSE microswitches to label each detected event; a label window stays open ~10 s after each cue
- Stream IMU batches, pFOG, cue events, and labels to the UNO Q over WiFi UDP (JSON or compact binary), with sequence numbers for loss tracking
- Accept commands from the UNO Q: threshold offset, cue-modality selection, metronome tempo (for haptic sync), and OTA model blob (stored in flash, hot-swapped into the interpreter)
- Run fully standalone with defaults if the UNO Q is unreachable — the cueing loop never depends on the link

### Tier-2 Bilateral / Context Model (UNO Q, Linux)
- Fuse IMU-A (streamed) and IMU-B (local, via Bridge RPC from the MCU) into 1 s features: cadence per sensor, step-time asymmetry, trunk-vs-shank phase, festination index (cadence rising while step amplitude falls), turn detection (integrated gyro yaw on IMU-B)
- Output a context risk score; push a **threshold offset** to the ESP32 (e.g., lower threshold during turns and gait initiation, where most freezes occur)
- Runs at ~1 Hz — it informs Tier 1 but is never in the time-critical path

### Adaptive Audio Metronome (UNO Q)
- Estimate the patient's real-time stride cadence from the IMU streams (autocorrelation of vertical acceleration)
- Set metronome tempo dynamically (target: ~10% faster than the patient's freely-chosen cadence, per rhythmic auditory cueing literature) rather than using a fixed BPM
- Synthesize the click train in Python (numpy) and play through the Bluetooth speaker via PipeWire/BlueZ A2DP
- Send the current tempo to the ESP32 so vibration pulses are delivered in sync with the audio beat (haptic metronome, not a continuous buzz)
- Recompute tempo periodically to track changes (fatigue, medication wearing off, etc.)

### Personalization & Cue Selection (UNO Q)
- **Baseline calibration:** 30 s guided normal walk on first use → per-patient cadence, Freeze Index baseline, step amplitude; stored and used to normalize features
- **Adaptive threshold:** fit the pFOG decision threshold from accumulated TRUE / FALSE labels (works from ~10 labels); push to ESP32
- **Cue-modality bandit:** Thompson sampling over {vibration, laser, audio, combinations}, rewarded by time-to-gait-recovery after each cue; the chosen arm is sent to the ESP32 before the next event
- **Optional fine-tune:** retrain the last layer of the CNN on the UNO Q using labeled windows, export TFLite, push OTA to the ESP32; keep model versions and roll back if false-positive rate worsens
- Store all labeled events locally; track precision / false-positive trend per model version so improvement is visible on the dashboard

### Fall Detection & Alerting (UNO Q)
- Fall primitives run on the MCU on IMU-B: free-fall (|a| < 0.5 g) → impact (|a| > 2.5 g) → ~2 s stillness / orientation change; distinct from FOG detection
- Linux escalation state machine: suspected fall → pause metronome → play a **pre-recorded speech clip** ("Are you okay? Press your button.") → 20 s countdown → if the "I'm OK" switch is pressed, log and resume; otherwise play alarm + raise a dashboard alert
- Speech clips are pre-recorded WAV files by default (unambiguous vs the metronome, and simplest); offline Piper TTS is an optional upgrade if dynamic phrases (caregiver name, medication reminders) are wanted
- Caregiver notification: dashboard banner + speaker alarm always (offline); ntfy/Telegram push only when internet is available (opt-in)

### Caregiver Dashboard (UNO Q, local-first web app, FastAPI + HTML/JS)
- Live status: current pFOG, Freeze Index, cadence, last cue fired and which modality, unit connection state, packet loss
- **Live freeze-band spectrogram** of IMU-A so a caregiver can see *why* a cue fired
- Session timeline: FOG events, fall alerts, true/false label ratio over time, model version markers
- Cue-effectiveness view: recovery time per modality and the bandit's current preference
- Gait trends: cadence, step-time asymmetry, festination index over the session/day
- **Medication overlay (headline view):** manual ON/OFF entries drawn over the FOG-frequency timeline
- Calibration and replay controls; CSV export of events and gait features for clinicians
- Should work fully on the local network without depending on internet access; optional cloud sync for remote/clinician access is a secondary layer

## Datasets & Validation

- **Daphnet Freezing of Gait dataset (UCI):** 10 PD patients, 3 accelerometers (shank, thigh, trunk), annotated freeze episodes — matches our shank + trunk placement. Primary source for training the Tier-1 model and for tuning Freeze Index thresholds
- **Kaggle "Parkinson's Freezing of Gait Prediction" (tlvmc, 2023):** larger lower-back accelerometer dataset with Start Hesitation / Turn / Walking labels — useful for the Tier-2 turn/initiation context features
- Report held-out-patient precision / recall / detection latency on Daphnet on a slide; do not claim clinical validation
- **Replay mode:** the UNO Q can stream Daphnet recordings through the entire pipeline (features → detection → cues → dashboard) as the demo fallback and for repeatable testing

## Power Budget

- UNO Q waist unit: ~5 V at 1–2 A under load (WiFi AP + Bluetooth audio + Python services) → a 10,000 mAh power bank gives roughly 4–6 h; measure actual draw with a USB power meter and report it
- ESP32 shank band: ~150–250 mA peak (WiFi TX) plus motor (~75 mA) and laser (~30 mA) when cueing → a 2,000–5,000 mAh bank lasts many hours
- Motor/laser loads are switched via AO3400A MOSFETs, never directly from GPIO
- Include the measured power budget in the pitch — judges of wearable projects ask

## Constraints & Framing for the Build

- This is a hackathon prototype: prioritize a working, demoable closed loop over production-grade robustness.
- Keep the low-latency cue-triggering path entirely on the ESP32 — don't route time-critical decisions through the UNO Q/WiFi link, since that adds latency. The ESP32 must keep cueing with default settings if the link drops.
- The UNO Q should own everything that benefits from more compute and doesn't need sub-second response: Tier-2 context model, cadence analysis, metronome audio, personalization, fall escalation, and the dashboard. Use both halves of the UNO Q: the STM32 MCU for IMU-B, fall primitives, and cue set 2; the Linux side for everything else, connected via Bridge RPC.
- Avoid making unsubstantiated clinical/diagnostic claims in the app's copy or dashboard language — this is an assistive/monitoring prototype, not a diagnosed medical device. In particular, say "early detection of freeze onset", not "predicts freezes seconds in advance"; the datasets we train on support detection, not long-horizon prediction.
- Prefer simple, reliable mechanisms over impressive-sounding ones when they conflict: pre-recorded speech over TTS, threshold adaptation over full retraining, WiFi UDP over BLE.
- Data privacy matters: this system handles a Parkinson's patient's continuous movement data. Default to on-device storage, explicit/opt-in cloud sync only.