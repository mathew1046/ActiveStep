"""Streaming benchmark: match cue timelines against ground-truth FOG events.

Predefined matching rules (fixed before looking at results):
- An event is DETECTED if a not-yet-credited cue ONSET falls within
  [onset - tol, end + tol]. Each cue onset can be credited with at most one
  event, so one continuously active cue cannot take credit for several
  episodes (a new cue onset is required for each new event).
- A cue onset that is never credited is a FALSE cue start, regardless of
  whether the cue happens to still be active during a later event. Such
  spanning cues are visible separately as "cued coverage".
- Unnecessary cue duration is cue-active time outside all true event
  intervals (censored cues are counted up to the stream end).
- Delays are measured in real time against the ORIGINAL annotation onsets.

Default tolerance is 0.0 s (exact boundaries); the annotation-jitter
sensitivity uses 0.3 s (dataset documentation: up to a couple of 100 ms of
onset jitter).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from src.data import FogEvent, RunWindows
    from src.detector import CueFSM
except ImportError:  # imported as a top-level module
    from data import FogEvent, RunWindows
    from detector import CueFSM

DEFAULT_TOL_S = 0.0
JITTER_TOL_S = 0.3
WITHIN_DELAYS_S = (0.5, 1.0, 2.0)


@dataclass
class Cue:
    start_ms: int
    stop_ms: int | None  # None -> still active at stream end (censored)
    censored: bool

    def stop_or(self, t_end_ms: int) -> int:
        return self.stop_ms if self.stop_ms is not None else t_end_ms


def run_cue_fsm(
    scores: np.ndarray,
    decision_ms: np.ndarray,
    threshold: float,
    hysteresis: float,
    t_end_ms: int,
) -> list[Cue]:
    """Run the shared hysteresis FSM over one recording's score stream."""
    fsm = CueFSM(threshold, hysteresis)
    cues: list[Cue] = []
    start: int | None = None
    for t, s in zip(decision_ms, scores):
        for ev in fsm.push(int(t), float(s)):
            if ev.type == "cue_start":
                start = ev.t_ms
            elif ev.type == "cue_stop" and start is not None:
                cues.append(Cue(start, ev.t_ms, False))
                start = None
    for ev in fsm.finish(t_end_ms):
        if ev.type == "cue_stop" and start is not None:
            cues.append(Cue(start, None, True))
    return cues


def _summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean_s": None, "median_s": None, "p90_s": None, "max_s": None}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(len(a)),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "p90_s": float(np.percentile(a, 90)),
        "max_s": float(a.max()),
    }


def match_run(events: list[FogEvent], cues: list[Cue], nonfog_hours: float, tol_s: float, t_end_ms: int) -> dict:
    """Apply the predefined matching rules to one recording."""
    tol_ms = tol_s * 1000.0
    credited: set[int] = set()
    n_events = len(events)
    delays_ms: list[float] = []
    n_detected = 0
    repeated_onsets = 0

    for ev in events:
        lo, hi = ev.onset_ms - tol_ms, ev.end_ms + tol_ms
        credited_this = None
        for i, c in enumerate(cues):
            if i in credited:
                continue
            if lo <= c.start_ms <= hi:
                credited_this = i
                break
        if credited_this is not None:
            credited.add(credited_this)
            n_detected += 1
            delays_ms.append(cues[credited_this].start_ms - ev.onset_ms)
            # extra (uncredited) onsets inside this event window
            for i, c in enumerate(cues):
                if i != credited_this and i not in credited and lo <= c.start_ms <= hi:
                    repeated_onsets += 1

    false_starts = [i for i in range(len(cues)) if i not in credited]

    # events during which any cue was active (assistive coverage, no credit)
    n_cued = sum(
        1
        for ev in events
        if any(c.start_ms <= ev.end_ms and c.stop_or(t_end_ms) > ev.onset_ms for c in cues)
    )

    # stop behaviour around event ends (credited cues only)
    stop_delays_ms: list[float] = []
    n_stops_censored = 0
    n_ended_early = 0
    for ev in events:
        i = next((j for j in credited if cues[j].start_ms <= ev.end_ms and cues[j].stop_or(t_end_ms) > ev.onset_ms), None)
        if i is None:
            continue
        c = cues[i]
        if c.stop_ms is None:
            n_stops_censored += 1
        elif c.stop_ms > ev.end_ms:
            stop_delays_ms.append(c.stop_ms - ev.end_ms)
        else:
            n_ended_early += 1

    # unnecessary (non-FOG) cue-active time
    unnecessary_ms = 0.0
    for c in cues:
        cs, ce = c.start_ms, c.stop_or(t_end_ms)
        remaining = max(0.0, float(ce - cs))
        for ev in events:
            o, e = ev.onset_ms, ev.end_ms
            remaining -= max(0.0, min(ce, e) - max(cs, o))
        unnecessary_ms += max(0.0, remaining)

    within = {f"{d}s": int(sum(1 for x in delays_ms if x <= d * 1000.0)) for d in WITHIN_DELAYS_S}

    return {
        "n_events": int(n_events),
        "n_detected": int(n_detected),
        "n_cued_during_event": int(n_cued),
        "detected_within": within,
        "detection_delay": _summarize([x / 1000.0 for x in delays_ms]),
        "detection_delay_raw_s": [x / 1000.0 for x in delays_ms],
        "n_cue_starts": int(len(cues)),
        "n_false_cue_starts": int(len(false_starts)),
        "repeated_cue_onsets": int(repeated_onsets),
        "cue_stop_delay": _summarize([x / 1000.0 for x in stop_delays_ms]),
        "cue_stop_delay_raw_s": [x / 1000.0 for x in stop_delays_ms],
        "n_cue_stops_censored": int(n_stops_censored),
        "n_cues_ended_before_event_end": int(n_ended_early),
        "unnecessary_cue_ms": float(unnecessary_ms),
        "nonfog_hours": float(nonfog_hours),
    }


def window_metrics(scores: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    """Window-level precision/recall/F1 at the operating threshold + PR-AUC."""
    out: dict = {}
    if len(scores) == 0:
        return {"n": 0}
    pred = np.asarray(scores) >= threshold
    y = np.asarray(label, dtype=bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fp) else None)
    out.update({
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
    })
    if len(np.unique(y)) > 1:
        try:
            from sklearn.metrics import average_precision_score
            out["pr_auc"] = float(average_precision_score(y, np.asarray(scores)))
        except Exception:
            out["pr_auc"] = None
    else:
        out["pr_auc"] = None
    return out


def evaluate_run(bundle: RunWindows, scores: np.ndarray, threshold: float, hysteresis: float, tol_s: float) -> dict:
    """Full evaluation of one recording's score stream against its ground truth."""
    cues = run_cue_fsm(scores, bundle.decision_ms, threshold, hysteresis, bundle.t_end_ms)
    nonfog_h = bundle.exposure.get("nonfog_s", 0.0) / 3600.0
    m = match_run(bundle.events, cues, nonfog_h, tol_s, bundle.t_end_ms)
    m["window_endpoint"] = window_metrics(scores, bundle.label_endpoint, threshold)
    m["window_any"] = window_metrics(scores, bundle.label_any, threshold)
    m["exposure"] = {
        "total_s": bundle.exposure.get("total_s", 0.0),
        "valid_s": bundle.exposure.get("valid_s", 0.0),
        "fog_s": bundle.exposure.get("fog_s", 0.0),
        "nonfog_s": bundle.exposure.get("nonfog_s", 0.0),
        "n_windows": int(len(bundle.decision_ms)),
        "n_windows_excluded_invalid": int(bundle.n_excluded),
    }
    m["cues"] = [
        {"start_ms": c.start_ms, "stop_ms": c.stop_ms, "censored": c.censored} for c in cues
    ]
    m["subject"] = bundle.subject
    m["run"] = bundle.run
    m["events"] = [
        {
            "onset_ms": e.onset_ms, "end_ms": e.end_ms,
            "duration_s": e.duration_s, "start_censored": e.start_censored,
            "end_censored": e.end_censored,
        }
        for e in bundle.events
    ]
    return m


def evaluate_participant(
    run_evals: list[tuple[RunWindows, np.ndarray]],
    threshold: float,
    hysteresis: float,
    tol_s: float = DEFAULT_TOL_S,
) -> dict:
    """Aggregate run evaluations into one participant result (counts kept raw)."""
    per_run = [evaluate_run(b, s, threshold, hysteresis, tol_s) for b, s in run_evals]

    n_events = sum(r["n_events"] for r in per_run)
    n_detected = sum(r["n_detected"] for r in per_run)
    nonfog_h = sum(r["nonfog_hours"] for r in per_run)
    n_false = sum(r["n_false_cue_starts"] for r in per_run)
    unnecessary_ms = sum(r["unnecessary_cue_ms"] for r in per_run)
    delays = [d for r in per_run for d in r.get("detection_delay_raw_s", [])]
    stop_delays = [d for r in per_run for d in r.get("cue_stop_delay_raw_s", [])]
    within = {
        k: int(sum(r["detected_within"][k] for r in per_run)) for k in per_run[0]["detected_within"]
    } if per_run else {}

    result = {
        "n_runs": len(per_run),
        "n_events": int(n_events),
        "n_detected": int(n_detected),
        "event_sensitivity": (n_detected / n_events) if n_events else None,
        "detected_within": within,
        "detection_delay": _summarize(delays),
        "n_false_cue_starts": int(n_false),
        "nonfog_hours": float(nonfog_h),
        "false_cue_starts_per_nonfog_hour": (n_false / nonfog_h) if nonfog_h > 0 else None,
        "unnecessary_cue_min_per_nonfog_hour": (unnecessary_ms / 60000.0 / nonfog_h) if nonfog_h > 0 else None,
        "n_cue_starts": int(sum(r["n_cue_starts"] for r in per_run)),
        "repeated_cue_onsets": int(sum(r["repeated_cue_onsets"] for r in per_run)),
        "n_cued_during_event": int(sum(r["n_cued_during_event"] for r in per_run)),
        "cue_stop_delay": _summarize(stop_delays),
        "n_cue_stops_censored": int(sum(r["n_cue_stops_censored"] for r in per_run)),
        "n_cues_ended_before_event_end": int(sum(r["n_cues_ended_before_event_end"] for r in per_run)),
        "exposure": {
            k: float(sum(r["exposure"][k] for r in per_run))
            for k in ("total_s", "valid_s", "fog_s", "nonfog_s", "n_windows", "n_windows_excluded_invalid")
        },
        "window_endpoint": _merge_window_metrics([r["window_endpoint"] for r in per_run], threshold),
        "window_any": _merge_window_metrics([r["window_any"] for r in per_run], threshold),
        "per_run": per_run,
    }
    return result


def _merge_window_metrics(metrics: list[dict], threshold: float) -> dict:
    """Recompute window metrics from pooled tp/fp/fn; PR-AUC needs raw scores, so pooled from parts is approximate -> None."""
    n = sum(m.get("n", 0) for m in metrics)
    if n == 0:
        return {"n": 0}
    tp = sum(m.get("tp", 0) for m in metrics)
    fp = sum(m.get("fp", 0) for m in metrics)
    fn = sum(m.get("fn", 0) for m in metrics)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if (tp + fp) else None)
    return {
        "n": int(n), "n_pos": int(sum(m.get("n_pos", 0) for m in metrics)),
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "pr_auc": None,  # pooled across runs requires raw scores; see summary-level computation
    }


def aggregate(participant_results: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    """Pooled + macro aggregation with participant-level bootstrap CIs."""
    with_events = [r for r in participant_results if r["n_events"] > 0]
    tot_events = sum(r["n_events"] for r in participant_results)
    tot_detected = sum(r["n_detected"] for r in participant_results)
    tot_nonfog_h = sum(r["nonfog_hours"] for r in participant_results)
    tot_false = sum(r["n_false_cue_starts"] for r in participant_results)

    pooled = {
        "n_participants": len(participant_results),
        "n_participants_with_events": len(with_events),
        "total_events": int(tot_events),
        "total_detected": int(tot_detected),
        "event_sensitivity": (tot_detected / tot_events) if tot_events else None,
        "total_nonfog_hours": float(tot_nonfog_h),
        "total_false_cue_starts": int(tot_false),
        "false_cue_starts_per_nonfog_hour": (tot_false / tot_nonfog_h) if tot_nonfog_h > 0 else None,
    }
    macro = {
        "event_sensitivity": float(np.mean([r["event_sensitivity"] for r in with_events])) if with_events else None,
        "false_cue_starts_per_nonfog_hour": (
            float(np.mean([r["false_cue_starts_per_nonfog_hour"] for r in participant_results
                           if r["false_cue_starts_per_nonfog_hour"] is not None]))
            if participant_results else None
        ),
    }

    out = {"pooled": pooled, "macro": macro, "bootstrap": None}
    if len(participant_results) > 1 and n_boot > 0:
        rng = np.random.default_rng(seed)
        n = len(participant_results)
        sens, fph = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sample = [participant_results[i] for i in idx]
            te = sum(r["n_events"] for r in sample)
            td = sum(r["n_detected"] for r in sample)
            th = sum(r["nonfog_hours"] for r in sample)
            tf = sum(r["n_false_cue_starts"] for r in sample)
            sens.append(td / te if te else np.nan)
            fph.append(tf / th if th > 0 else np.nan)
        out["bootstrap"] = {
            "n_boot": int(n_boot),
            "seed": int(seed),
            "event_sensitivity_ci95": _ci(sens),
            "false_cues_per_hour_ci95": _ci(fph),
        }
    return out


def _ci(values: list[float]) -> list | None:
    a = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return None
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def to_jsonable(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items() if not str(k).startswith("__")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
