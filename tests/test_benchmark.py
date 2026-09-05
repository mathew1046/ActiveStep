"""Tests for the streaming benchmark (src/benchmark.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmark import (
    JITTER_TOL_S,
    Cue,
    aggregate,
    evaluate_participant,
    match_run,
    run_cue_fsm,
    window_metrics,
)
from src.data import ANNOT_EXP, ANNOT_FOG, FogEvent, RunData, RunWindows, exposure, resample_run


def _ev(onset, end, sc=False, ec=False):
    return FogEvent(int(onset), int(end), sc, ec, 1)


def _mw(decision_ms, events, nonfog_s, t_end, labels_ep=None, labels_any=None, scores=None):
    """Minimal RunWindows for benchmark tests."""
    k = len(decision_ms)
    return RunWindows(
        subject=1, run=1, fs=100, window_s=2.0, hop_s=0.25,
        decision_ms=np.asarray(decision_ms, dtype=np.int64),
        avail_ms=np.asarray(decision_ms, dtype=np.int64) + 16,
        X_shank=np.empty((k, 200, 3), dtype=np.float32),
        label_endpoint=np.asarray(labels_ep if labels_ep is not None else [False] * k, dtype=bool),
        label_any=np.asarray(labels_any if labels_any is not None else [False] * k, dtype=bool),
        occupancy=np.zeros(k, dtype=np.float32),
        fi=np.zeros(k, dtype=np.float32),
        n_excluded=0, t_start_ms=0, t_end_ms=int(t_end),
        events=events,
        exposure={"total_s": t_end / 1000, "valid_s": t_end / 1000,
                  "fog_s": 0.0, "nonfog_s": nonfog_s},
    )


# --- FSM over a score stream -------------------------------------------------

def test_run_cue_fsm_intervals_and_censoring():
    t = np.arange(0, 3000, 250)
    scores = np.array([0.0, 0.8, 0.8, 0.6, 0.2, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
    cues = run_cue_fsm(scores, t, threshold=0.7, hysteresis=0.15, t_end_ms=2750)
    assert len(cues) == 2
    assert cues[0].start_ms == 250 and cues[0].stop_ms == 1000 and not cues[0].censored
    assert cues[1].start_ms == 1500 and cues[1].stop_ms is None and cues[1].censored


# --- matching rules -----------------------------------------------------------

def test_match_perfect_detection():
    ev = [_ev(1000, 2000)]
    cues = [Cue(1000, 2200, False)]
    m = match_run(ev, cues, nonfog_hours=1.0, tol_s=0.0, t_end_ms=3000)
    assert m["n_events"] == 1 and m["n_detected"] == 1
    assert m["detection_delay"]["n"] == 1 and m["detection_delay"]["mean_s"] == 0.0
    assert m["n_false_cue_starts"] == 0
    assert m["detected_within"]["0.5s"] == 1
    assert m["cue_stop_delay"]["mean_s"] == pytest.approx(0.2)


def test_match_missed_event():
    m = match_run([_ev(1000, 2000)], [], nonfog_hours=1.0, tol_s=0.0, t_end_ms=3000)
    assert m["n_detected"] == 0
    assert m["detection_delay"]["n"] == 0


def test_match_false_cue_and_unnecessary_duration():
    ev = [_ev(10000, 11000)]
    cues = [Cue(1000, 4000, False)]  # 3 s cue far from the event
    m = match_run(ev, cues, nonfog_hours=0.5, tol_s=0.0, t_end_ms=20000)
    assert m["n_false_cue_starts"] == 1
    assert m["unnecessary_cue_ms"] == pytest.approx(3000.0)
    # false cues per hour derived at participant level: 1 / 0.5 h = 2/h
    assert m["nonfog_hours"] == pytest.approx(0.5)


def test_match_spanning_cue_gets_no_credit_and_no_double_credit():
    """One long cue cannot detect two events, nor gain credit by spanning."""
    ev = [_ev(5000, 6000), _ev(8000, 9000)]
    cues = [Cue(1000, 9500, False)]  # starts before both events, spans them
    m = match_run(ev, cues, nonfog_hours=1.0, tol_s=0.0, t_end_ms=10000)
    assert m["n_detected"] == 0          # onset rule: no onset inside either event
    assert m["n_cued_during_event"] == 2  # but the patient WAS cued during both
    assert m["n_false_cue_starts"] == 1   # the onset was never credited
    # unnecessary time excludes the true event intervals
    assert m["unnecessary_cue_ms"] == pytest.approx(8500 - 1000 - 1000)


def test_match_second_event_needs_new_onset():
    ev = [_ev(1000, 2000), _ev(8000, 9000)]
    cues = [Cue(1000, 2500, False), Cue(8100, 9500, False)]
    m = match_run(ev, cues, nonfog_hours=1.0, tol_s=0.0, t_end_ms=10000)
    assert m["n_detected"] == 2
    assert m["n_false_cue_starts"] == 0
    assert m["detection_delay"]["mean_s"] == pytest.approx(0.05)


def test_match_jitter_tolerance():
    ev = [_ev(1000, 2000)]
    cues = [Cue(800, 2100, False)]  # cue starts 200 ms before annotated onset
    strict = match_run(ev, cues, 1.0, tol_s=0.0, t_end_ms=3000)
    jitter = match_run(ev, cues, 1.0, tol_s=JITTER_TOL_S, t_end_ms=3000)
    assert strict["n_detected"] == 0 and strict["n_false_cue_starts"] == 1
    assert jitter["n_detected"] == 1 and jitter["n_false_cue_starts"] == 0
    assert jitter["detection_delay"]["mean_s"] == pytest.approx(-0.2)


def test_match_cue_ended_before_event_end():
    ev = [_ev(1000, 3000)]
    cues = [Cue(1000, 1500, False)]
    m = match_run(ev, cues, 1.0, tol_s=0.0, t_end_ms=5000)
    assert m["n_detected"] == 1
    assert m["n_cues_ended_before_event_end"] == 1
    assert m["cue_stop_delay"]["n"] == 0


def test_match_no_events_participant():
    cues = [Cue(1000, 2000, False), Cue(4000, None, True)]
    m = match_run([], cues, nonfog_hours=2.0, tol_s=0.0, t_end_ms=10000)
    assert m["n_events"] == 0
    assert m["n_false_cue_starts"] == 2
    # censored cue's active time counts up to the stream end
    assert m["unnecessary_cue_ms"] == pytest.approx(1000 + 6000)


# --- window metrics -----------------------------------------------------------

def test_window_metrics_exact_counts():
    scores = np.array([0.9, 0.8, 0.3, 0.2])
    y = np.array([True, False, True, False])
    m = window_metrics(scores, y, threshold=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    # AP = mean of precision at each positive rank: (1/1 + 2/3) / 2
    assert m["pr_auc"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_window_metrics_single_class():
    m = window_metrics(np.array([0.1, 0.2]), np.array([False, False]), threshold=0.5)
    assert m["pr_auc"] is None


# --- participant aggregation and bootstrap -------------------------------------

def test_evaluate_participant_pools_runs():
    b1 = _mw([1000, 1250], [_ev(900, 1500)], nonfog_s=1800, t_end=2000,
             labels_ep=[True, True], labels_any=[True, True])
    b2 = _mw([5000], [], nonfog_s=1800, t_end=6000, labels_ep=[False])
    scores1 = np.array([0.9, 0.9])
    scores2 = np.array([0.1])
    res = evaluate_participant([(b1, scores1), (b2, scores2)], threshold=0.7, hysteresis=0.15)
    assert res["n_events"] == 1 and res["n_detected"] == 1
    assert res["event_sensitivity"] == pytest.approx(1.0)
    assert res["nonfog_hours"] == pytest.approx(1.0)  # 3600 s of non-FOG exposure
    assert res["n_false_cue_starts"] == 0
    assert res["exposure"]["nonfog_s"] == pytest.approx(3600)


def test_aggregate_pooled_macro_and_bootstrap():
    def part(det, ev, false, nonfog_h):
        return {
            "n_events": ev, "n_detected": det, "nonfog_hours": nonfog_h,
            "n_false_cue_starts": false,
            "event_sensitivity": (det / ev) if ev else None,
            "false_cue_starts_per_nonfog_hour": (false / nonfog_h) if nonfog_h else None,
        }

    parts = [part(8, 10, 2, 1.0), part(4, 10, 0, 1.0), part(0, 0, 6, 2.0)]
    agg = aggregate(parts, n_boot=500, seed=1)
    assert agg["pooled"]["event_sensitivity"] == pytest.approx(12 / 20)
    assert agg["pooled"]["false_cue_starts_per_nonfog_hour"] == pytest.approx(8 / 4)
    assert agg["macro"]["event_sensitivity"] == pytest.approx(0.6)
    ci = agg["bootstrap"]["event_sensitivity_ci95"]
    assert ci[0] <= 0.6 <= ci[1]
    fci = agg["bootstrap"]["false_cues_per_hour_ci95"]
    assert fci[0] <= 2.0 <= fci[1]
