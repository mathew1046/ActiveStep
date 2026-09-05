# Deep Research: Model Architectures for Closed-Loop Predictive Wearable FOG Aids

A survey of model architectures used by researchers tackling the same problem
ActiveStep addresses — closed-loop, wearable, real-time detection/cueing of
Freezing of Gait (FOG) in Parkinson's disease — with concrete ideas we could
borrow. Each entry notes the architecture, results, deployment target, and the
specific idea worth stealing.

ActiveStep's current baseline (for comparison):
- **Tier-1 (ESP32):** 1D-CNN, 2× Conv1D (16→32) + BN + GAP + Dense(16) + sigmoid,
  ~<20k params, int8 TFLite Micro, 2 s / 200-sample windows, hop 250 ms.
- **Score fusion:** `pFOG = 0.6·norm(p_cnn) + 0.4·norm(FI)` with hysteresis FSM.
- **Tier-2 (UNO Q):** hand-crafted bilateral features (cadence, asymmetry,
  festination, turn) → threshold offset pushed to Tier-1.

---

## 1. PhysioGait Predictive Network (PhysioGPN) — IEEE JBHI 2025
*Wang et al., "Addressing Multiple Challenges in Early Gait Freezing Prediction
for Parkinson's Disease: A Practical Deep Learning Approach."*

The most directly relevant paper — it explicitly targets **prediction ≥2 s before
onset** (not just detection), generalization across patients, and few-sensor
convenience, the same three-way tradeoff ActiveStep navigates.

**Architecture (four strategies):**
1. **Large convolutional kernels** to capture *progressive motion changes* in the
   pre-freeze transition (the slow drift toward freeze, not just the freeze
   itself). Small kernels see the tremor; large kernels see the festination
   trend leading into it.
2. **Multi-dimensional + multi-scale convolution** — parallel conv branches at
   different kernel widths over ACC and GYRO, fused. Captures both fast
   trembling (3–8 Hz) and slow gait dynamics (0.5–3 Hz) in one pass.
3. **Twin-tower structure** — two parallel encoders, one per leg/sensor,
   sharing an ACC/GYRO embedding, to model gait **self-similarity and
   asymmetry**. This is the learned analogue of ActiveStep's hand-crafted
   `step_time_asymmetry` / bilateral features.
4. **Multi-domain attention** for cross-domain (time ↔ frequency) information
   exchange.

**Knowledge distillation (KD) framework:** a multi-sensor teacher is distilled
into a **single-sensor student**, recovering ~5.1% AUC when sensors are reduced.
This is the single most portable idea for ActiveStep: train a rich
shank+trunk+gyro teacher on the UNO Q, distill into the shank-only int8 CNN on
the ESP32, so the edge model benefits from bilateral context without paying for
it at inference time.

**Results:** 85.8% AUC for *prediction* (2 s horizon), subject-independent.

**Borrow:** (a) large-kernel + multi-scale conv branch in the Tier-1 CNN;
(b) KD from a Tier-2 teacher to the Tier-1 student — formalizes our existing
two-tier split and gives a principled OTA fine-tune story;
(c) twin-tower as a future Tier-2 upgrade once we have bilateral IMU data.

---

## 2. CNN + RNN + Past-Sample (PS) on microcontroller — PMC11505507, 2024
*"Wearable Online Freezing of Gait Detection and Cueing System."*

A full closed-loop wearable, deployed on an **Arduino Portenta H7 (STM32H7,
Cortex-M7 480 MHz)** — the same class of MCU as our ESP32, just faster. This is
the closest published analog to ActiveStep's shank band.

**Architecture:** CNN → RNN (LSTM/GRU) → "Past Sample" (PS) preprocessing.
The PS trick feeds the **previous window's features** back in as extra input,
giving the network one-hop memory without a full recurrent stack — cheap on a
microcontroller. Runs at 40 Hz on-device.

**Results:** 95.1% detection accuracy, **261 ms average detection delay**,
on-demand vibratory cueing reduced average FOG episode duration by **45%** in
real PD patients. 6 h battery, 57 g, 550 mW.

**Borrow:**
- The **PS (past-sample) input trick**: concatenate the previous window's
  embedding (or its pFOG) into the current window's input. Near-free recurrent
  behavior for ~zero extra params — fits the ESP32 budget and our 250 ms hop
  naturally. This is a cheap upgrade to `build_fog_cnn`.
- Their 261 ms latency is a useful benchmark for our <100 ms target — we're
  already more aggressive, which is defensible to cite.
- Confirms phasic/on-demand cueing (their 45% episode reduction) as the right
  cueing philosophy — direct evidence for our core design claim.

---

## 3. CNN-BiLSTM + Self-Attention (U-Net inspired) — Frontiers in Physiology 2025
*Al-Adhaileh et al., "Deep learning techniques for detecting freezing of gait
episodes in Parkinson's disease using wearable sensors."*

**Architecture:** U-Net-style CNN encoder for spatial feature extraction →
BiLSTM for temporal modeling → **self-attention** for interpretability and
focus on critical gait segments. Trained on tDCS FOG, DeFOG, Daily Living, and
Hantao multimodal datasets.

**Edge deployment:** post-training **quantization + pruning** → Raspberry Pi /
Coral TPU, inference latency **<350 ms**.

**Results:** 92.5% accuracy, F1 89.3%, AUC 0.91, cross-dataset.

**Borrow:**
- **Self-attention over temporal segments** is the cheapest interpretability
  win — it produces a per-timestep weight that maps cleanly onto our dashboard's
  "live freeze-band spectrogram / why did the cue fire" view. Even a tiny
  single-head attention over the post-conv sequence would give a saliency map
  for free.
- **Multi-dataset training** (tDCS FOG + DeFOG + Daily Living) for robustness —
  we currently only use Daphnet. DeFOG and the Kaggle tlvmc set are worth adding
  to the training pipeline for cross-dataset generalization, which the
  deployment-gap paper below shows is the real bottleneck.
- Pruning + quantization pipeline is exactly our int8 path; their <350 ms on a
  Pi validates that the architecture class is edge-viable.

---

## 4. Multi-Stage Temporal Convolutional Network (MS-TCN) — medRxiv 2023 / Frontiers 2024
*"Freezing of gait assessment with IMUs and deep learning"* and *"Detection of
FOG from foot-pressure insoles using a temporal convolutional neural network."*

**Architecture:** MS-TCN, originally from video frame-by-frame segmentation.
Stage 1: stacked temporal conv layers produce an initial per-timestep
prediction; subsequent **refinement stages** smooth the output and reduce
**over-segmentation error** (the tendency to fragment one freeze into many
short on/off flickers). Dilated causal convolutions → large receptive field,
fully parallel (no recurrence), so it's fast and edge-friendly.

**Borrow — high value for our cue FSM:**
- Our `CueFSM` already fights over-segmentation with hysteresis. MS-TCN's
  refinement-stage idea is the *learned* version of hysteresis: a second cheap
  conv head that smooths the per-hop pFOG sequence to avoid cue flicker. Could
  replace/augment the hand-tuned hysteresis band.
- **TCN over LSTM** argument: parallel, dilated, lower latency, easier to
  quantize — aligns with our ESP32 latency constraint better than an LSTM would.
  A small TCN (3–4 dilated 1D-conv blocks) is a strong alternative to the
  current 2-conv CNN and gives a much larger receptive field at similar param
  count.

---

## 5. CBA-BiLSTM (Conv–Bottleneck-Attention–BiLSTM) — PMC11688057, 2024
*"Deep Learning–Based Prediction of FOG… With the Ensemble Channel Selection
Approach."*

**Architecture:** BiLSTM with a **bottleneck attention module** inserted, plus
an **ensemble channel-selection** front end (k-NN-based supervised channel
weighting) to pick the best 2 of 3 sensor channels, then attention-based
feature reduction.

**Results:** 99.88% accuracy with only **two channels** — i.e., they
explicitly show you can drop a sensor and keep performance via channel
selection + attention. Reduced compute → real-time.

**Borrow:**
- **Learned channel selection** justifies our shank-only Tier-1 design
  quantitatively (their two-channel result). Useful citation for the pitch.
- Bottleneck attention = cheap parameter reduction before the recurrent head;
  relevant if we ever grow the Tier-2 model.

---

## 6. CNN + LSTM + Adversarial Training — Neural Computing & Apps 2025
*"Enhancing patient-independent detection of FOG… with deep adversarial network."*

**Architecture:** CNN → LSTM → attention, made **patient-independent** via
**adversarial training** — a gradient-reversal head tries to predict the patient
ID from the embedding, forcing the shared encoder to be patient-invariant.

**Results:** +7.6% sensitivity without adversarial, **+8.36% with adversarial**
over prior SOTA, little specificity compromise.

**Borrow — directly addresses our hardest problem:**
- Our Daphnet leave-one-patient-out numbers are the weakest part of the model
  (inter-patient generalization). **Adversarial domain adaptation** is the
  single most effective published technique for patient-independent FOG
  detection. A small gradient-reversal patient-ID head added during training
  (removed at inference) is a low-cost, high-impact upgrade to `train.py`. This
  is probably the highest-leverage idea in this whole survey for improving our
  held-out-patient metrics.

---

## 7. Hybrid LSTM-Attention Framework — IEEE SoutheastCon 2026
*"Prediction of Freezing of Gait… Using Hybrid LSTM Attention Framework."*

**Architecture:** LSTM + **self-attention + spatial attention** + FC classifier.
Rigorously cross-validated, ensemble on holdout. Benchmarked against LSTM,
1D-CNN, 1D-CNN+self-attn, TCN, TCN+self-attn.

**Borrow:** Their benchmark ladder (LSTM / 1D-CNN / 1D-CNN+attn / TCN / TCN+attn)
is a ready-made **ablation table** for our slide. Replicating that exact
comparison on Daphnet would give us a credible "we tried the options" story
with minimal extra work, and TCN+self-attn is a likely winner on our
latency budget.

---

## 8. GRU-LSTM Network Bank — IEEE ICMI 2025
*"A GRU-LSTM Network Bank for Efficient Detection of FOG."*

**Architecture:** a **bank** of GRU and LSTM subnets, each on a different
component of the 3-axis acceleration; final prediction by **agreement/voting**
across the bank.

**Results:** 88% subject-independent.

**Borrow:** The "ensemble-by-agreement" idea is a cheap robustness lever —
instead of one CNN, train 3 tiny per-axis CNNs and fire a cue only when ≥2
agree. Trades a little latency for a marked false-positive reduction, which is
exactly the failure mode that annoys patients and erodes cue salience. Pairs
well with our existing FI+CNN fusion (we already combine two scorers; adding a
third voter generalizes the pattern).

---

## 9. FAF-Self-Attention-Bi-GRU (Fuzzy Activation) — IEEE ODICON 2024
*"Risk Prediction of Freezing of Gaits… Using Gated Recurrent Unit."*

**Architecture:** Bi-GRU with **fuzzy activation functions** and self-attention,
for *prediction* (not just detection). Fuzzy activations smooth gradient
behavior on noisy wearable signals.

**Results:** 98.1% (accel) / 99.2% (gyro) on the prediction task.

**Borrow:** marginal — fuzzy activations are niche and add quantization
complexity. Mainly useful as evidence that **gyroscope channels carry strong
predictive signal** (their gyro model beat their accel model). Since our
MPU-6050 gives gyro on IMU-B but we currently feed only accel into the model,
**adding gyro axes to the Tier-2 feature set** (turn rate already uses it) is a
free signal we're underusing.

---

## 10. Intelligent Wearable System (IWS) — plantar pressure + cueing — PMC10936378
*"Intelligent wearable system with accurate detection of abnormal gait and timely
cueing for mobility enhancement."*

Not a neural architecture, but a **complete closed-loop system** comparable to
ActiveStep: plantar-pressure insoles → fast gait algorithm → wirelessly
controlled cueing. 97% offline / **94% real-time** FoG detection, **0.37 s
latency** onset→cue, 88% of patients said it enhanced walking, 70% overcame FoG.

**Borrow:** Their 0.37 s onset→cue latency and 94% real-time (vs 97% offline)
numbers are the right comparison points for our <100 ms / replay-vs-live claims.
Their **patient questionnaire** (comfort, convenience, did-it-help) is a cheap,
judge-friendly validation method we should steal for the demo/pitch.

---

## 11. CNN-LSTM + sEMG + multi-modal cueing — IEEE EWDTs 2025
*"Intelligent Multimodal Cueing Wearable Device for Gait Rehabilitation."*

**Architecture:** hybrid **CNN-LSTM** for gait-phase + movement-intent
classification (90% accuracy), plus a **Random Forest on sEMG** for
dorsiflexor/plantar-flexor activity, driving **adaptive visual + haptic** cues
(laser projection + vibration) — almost exactly our cue modality set.

**Borrow:** The **gait-phase classification** framing (heel-strike / toe-off /
swing) is a useful intermediate representation we don't currently model
explicitly. A tiny gait-phase head on the CNN would let cues be *phase-locked*
(e.g., laser fires at heel-strike, not mid-swing) — a known cueing-effectiveness
lever and a nice differentiator from "fire whenever pFOG high."

---

## Cross-Cutting Findings & The Deployment Gap
*PMC12944384, "Deep Learning for FOG Detection: Cross-Dataset Validation
Reveals Critical Deployment Gaps."*

**TCN** trained on Daphnet (lab) vs Figshare (daily living): lab F1 0.9999 vs
real-world F1 0.55 — an **83% performance gap**. Key training findings:
- **F1-based early stopping** beats AUC-based by 47%.
- **Stacking imbalance corrections** (focal loss + weighting + sampling
  together) *hurts* — ~60× over-weighting of the minority class crashed
  precision to 33%. Use one, not all three.

This is the most important cautionary result: our Daphnet numbers will look
great and mean little for real-world use. It directly motivates our
personalization layer (per-patient threshold fitting) as the bridge from lab
to daily living.

---

## Prioritized Recommendations for ActiveStep

Ranked by leverage × fit to our constraints (ESP32 latency budget, hackathon
timeline, existing two-tier architecture):

| # | Idea | Source | Effort | Impact |
|---|------|--------|--------|--------|
| 1 | **Adversarial patient-invariance head** at training time (removed at inference) to fix leave-one-patient-out | §6 | Med | High — directly attacks our weakest metric |
| 2 | **Knowledge distillation**: Tier-2 (shank+trunk+gyro) teacher → Tier-1 (shank-only) student | §1 | Med-High | High — formalizes the two-tier split, principled OTA story |
| 3 | **Past-Sample (PS) input trick**: feed previous window's embedding into current input | §2 | Low | Med — near-free temporal memory on ESP32 |
| 4 | **TCN** (dilated causal 1D-convs) replacing/augmenting the 2-conv CNN — larger receptive field, parallel, quantization-friendly | §4, §7 | Med | Med-High — better pre-freeze trend capture |
| 5 | **Self-attention** over the post-conv sequence → free saliency map for the dashboard "why did it fire" view | §3, §7 | Low-Med | Med — interpretability + slight accuracy |
| 6 | **MS-TCN refinement stage** as a learned hysteresis to cut cue flicker | §4 | Med | Med — replaces hand-tuned hysteresis |
| 7 | **Multi-dataset training** (add DeFOG + Kaggle tlvmc to Daphnet) | §3 | Low-Med | Med — robustness, the real bottleneck per §deployment-gap |
| 8 | **Per-axis ensemble / voting bank** for false-positive reduction | §8 | Med | Med — cue salience preservation |
| 9 | **Gyro axes into Tier-2 features** (we already read them, only use yaw for turns) | §9 | Low | Low-Med — free predictive signal |
| 10 | **Gait-phase head** for phase-locked cueing (laser at heel-strike) | §11 | Med | Low-Med — cueing-effectiveness differentiator |
| 11 | **F1 early-stopping + single imbalance correction** (not stacked) | §deployment-gap | Low | Med — training hygiene |

**Quickest wins for the hackathon:** #3 (PS input), #5 (tiny self-attention for
dashboard saliency), #11 (F1 early stopping), #9 (gyro features) — all low
effort, all defensible on the slide. #1 (adversarial) is the biggest accuracy
win available and is training-only, so it doesn't touch the firmware latency
budget.

---

## References
1. Wang et al., PhysioGPN, IEEE JBHI 2025 — https://ieeexplore.ieee.org/document/10816191
2. Wearable Online FOG Detection & Cueing, PMC11505507 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11505507/
3. Al-Adhaileh et al., CNN-BiLSTM+attention, Front. Physiol. 2025 — https://doi.org/10.3389/fphys.2025.1581699
4. MS-TCN for FOG, medRxiv 2023 — https://www.medrxiv.org/content/10.1101/2023.05.05.23289387v1 ; TCNN on insoles, Front. Aging Neurosci. 2024 — https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2024.1437707/full
5. CBA-BiLSTM, PMC11688057 — https://pmc.ncbi.nlm.nih.gov/articles/PMC11688057/
6. CNN+LSTM+adversarial, Neural Comput. Appl. 2025 — https://dl.acm.org/doi/10.1007/s00521-025-11068-x
7. Hybrid LSTM-Attention, IEEE SoutheastCon 2026 — https://doi.org/10.1109/southeastcon63549.2026.11476600
8. GRU-LSTM Network Bank, IEEE ICMI 2025 — https://doi.org/10.1109/icmi65310.2025.11141325
9. FAF-Bi-GRU, IEEE ODICON 2024 — https://doi.org/10.1109/odicon62106.2024.10797573
10. IWS plantar-pressure + cueing, PMC10936378 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10936378/
11. CNN-LSTM + sEMG multimodal cueing, IEEE EWDTS 2025 — https://doi.org/10.1109/ewdts67441.2025.11303700
12. Deployment-gap cross-dataset study, PMC12944384 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12944384/
13. Moore et al., original Freeze Index (3–8 Hz / 0.5–3 Hz ratio), J NeuroEng Rehab 2008 — https://doi.org/10.1186/1743-0003-10-19
14. Unified gait freeze index benchmark, Front. Neurol. 2025 — https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1528963/full
