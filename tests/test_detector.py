"""Tests for the shared detector core (src/detector.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import CueFSM, DetectorConfig, StreamDetector, combine_scores
from src.features import freeze_index_components, p_fog_combined


def test_combine_scores_matches_features_impl():
    cfg = DetectorConfig(w_cnn=0.6, w_fi=0.4)
    p, fi = 0.7, 3.3
    assert combine_scores(p, fi, cfg) == pytest.approx(p_fog_combined(p, fi, 0.6, 0.4))
    arr_p, arr_fi = np.array([0.1, 0.9]), np.array([1.0, 4.0])
    np.testing.assert_allclose(combine_scores(arr_p, arr_fi, cfg), p_fog_combined(arr_p, arr_fi, 0.6, 0.4))


def test_fsm_hysteresis_and_recovery():
    fsm = CueFSM(threshold=0.7, hysteresis=0.15)
    ts = list(range(1000, 6000, 250))
    scores = [0.2, 0.75, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    events = []
    for t, s in zip(ts, scores):
        events.extend(fsm.push(t, s))
    starts = [e for e in events if e.type == "cue_start"]
    stops = [e for e in events if e.type == "cue_stop"]
    assert len(starts) == 1 and len(stops) == 1
    assert starts[0].t_ms == 1250  # first score >= 0.7
    # 0.65 and 0.6 stay above off-threshold 0.55; 0.5 crosses below -> stop at t=2000
    assert stops[0].t_ms == 2000
    assert stops[0].recovery_ms == 750
    assert not fsm.cueing


def test_fsm_finish_closes_ongoing_cue_as_censored():
    fsm = CueFSM(threshold=0.5, hysteresis=0.1)
    fsm.push(100, 0.9)
    ev = fsm.finish(t_end_ms=5000)
    assert len(ev) == 1 and ev[0].type == "cue_stop"
    assert ev[0].censored and ev[0].recovery_ms == 4900


def test_stream_detector_fi_only_cues_on_freeze_band_signal():
    """5 Hz shank trembling (freeze band) must cue; 1.5 Hz walking must not."""
    cfg = DetectorConfig(window_samples=200, hop_samples=25, w_cnn=0.0, w_fi=1.0, threshold=0.6)
    det = StreamDetector(cfg, predict_fn=None, scaler=None)

    fs = 100
    t = np.arange(200 * 4) / fs
    quiet = np.zeros((200, 3), dtype=np.float32)
    walk = np.stack([np.sin(2 * np.pi * 1.5 * t[:200]) * 300] * 3, axis=1).astype(np.float32)
    tremor = np.stack([np.sin(2 * np.pi * 5.0 * t[:200]) * 300] * 3, axis=1).astype(np.float32)

    events = []
    cueing = False
    for i in range(200):  # 2 s quiet warm-up
        d = det.push(int(i * 10), quiet[i])
        if d:
            events.extend(d.events)
            cueing = d.cueing
    assert not cueing

    for i in range(200):  # 2 s of 5 Hz trembling
        d = det.push(int((200 + i) * 10), tremor[i])
        if d:
            events.extend(d.events)
            cueing = d.cueing
            if d.events:
                break
    starts = [e for e in events if e.type == "cue_start"]
    assert starts, "expected a cue during freeze-band trembling"
    assert starts[0].t_ms >= 2000  # only after the window is filled with tremor

    # walking band signal alone should not cue
    det2 = StreamDetector(cfg, predict_fn=None, scaler=None)
    ev2 = []
    for i in range(600):
        d = det2.push(int(i * 10), walk[i % 200])
        if d:
            ev2.extend(d.events)
    assert not [e for e in ev2 if e.type == "cue_start"]


def test_stream_detector_threshold_offset():
    cfg = DetectorConfig(window_samples=200, hop_samples=25, w_cnn=0.0, w_fi=1.0, threshold=0.9)
    det = StreamDetector(cfg)
    det.set_threshold_offset(-0.35)
    assert det.effective_threshold == pytest.approx(0.55)
    assert det._fsm.threshold == pytest.approx(0.55)


def test_fsm_causal_confirmation_minimum_duration_and_refractory():
    fsm = CueFSM(
        threshold=0.7, hysteresis=0.1, on_consecutive=2, off_consecutive=2,
        min_cue_ms=500, refractory_ms=500,
    )
    assert not fsm.push(0, 0.8)
    assert fsm.push(250, 0.8)[0].type == "cue_start"
    assert not fsm.push(500, 0.1)
    assert not fsm.push(750, 0.1)
    assert fsm.push(1000, 0.1)[0].type == "cue_stop"
    assert not fsm.push(1250, 0.9)
    assert not fsm.push(1500, 0.9)
    assert fsm.push(1750, 0.9)[0].type == "cue_start"


def test_power_gate_suppresses_low_energy_freeze_ratio():
    t = np.arange(200) / 100.0
    weak = np.stack([np.sin(2 * np.pi * 5.0 * t)] * 3, axis=1).astype(np.float32)
    strong = weak * 300.0
    _, weak_power = freeze_index_components(weak[None], aggregate_axes=True)
    _, strong_power = freeze_index_components(strong[None], aggregate_axes=True)
    gate = float(np.sqrt(weak_power[0] * strong_power[0]))
    cfg = DetectorConfig(
        window_samples=200, hop_samples=25, w_cnn=0.0, w_fi=1.0,
        threshold=0.6, fi_power_threshold=gate,
    )
    weak_detector = StreamDetector(cfg)
    strong_detector = StreamDetector(cfg)
    weak_decision = strong_decision = None
    for i in range(200):
        weak_decision = weak_detector.push(i * 10, weak[i]) or weak_decision
        strong_decision = strong_detector.push(i * 10, strong[i]) or strong_decision
    assert weak_decision.fi == 0.0 and not weak_decision.cueing
    assert strong_decision.fi > 0.0 and strong_decision.cueing
