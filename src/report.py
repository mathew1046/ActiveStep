"""Render a human-readable assessment report from a nested-run summary.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CLAIM_SCOPE_NOTE = (
    "These are patient-independent, retrospective, offline detection results on the "
    "Daphnet dataset. They do not establish real-world wearable accuracy on our hardware, "
    "cueing effectiveness, fall reduction, or any clinical benefit. False-cue burden is "
    "reported per annotated non-FOG experimental hour (label 1 includes standing, walking "
    "and turning). The FP budget used for threshold selection is an engineering choice, "
    "not a clinical acceptance threshold."
)


def _f(v, nd=3):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _r(ci, nd=3):
    if not ci:
        return "n/a"
    return f"[{_f(ci[0], nd)}, {_f(ci[1], nd)}]"


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def render(summary_path: Path, audit_path: Path | None = None, out_path: Path | None = None) -> str:
    data = json.loads(Path(summary_path).read_text())
    lines = []
    lines.append("# ActiveStep FOG detection assessment report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Source summary: `{summary_path}`")
    lines.append("")

    protocol = next(iter(data.values())).get("protocol", {}) if data else {}
    if protocol:
        lines.append("## Protocol")
        lines.append("")
        lines.append(
            f"- Nested leave-one-participant-out; recipe (weights + cue threshold) selected on "
            f"inner participant-grouped out-of-fold streams, refit on all outer-train participants, "
            f"evaluated once on the held-out participant."
        )
        lines.append(
            f"- Window {protocol.get('window_s')} s, training hop {protocol.get('train_hop_s')} s, "
            f"evaluation (deployment) hop {protocol.get('eval_hop_s')} s, "
            f"hysteresis {protocol.get('hysteresis')}."
        )
        lines.append(
            f"- Operating point: max pooled event sensitivity s.t. false-cue starts "
            f"<= {protocol.get('fp_budget_per_hour')}/h (inner OOF); training label: "
            f"{protocol.get('label')}."
        )
        lines.append(f"- Threshold grid: {protocol.get('threshold_grid')}")
        lines.append(f"- Weight grid: {protocol.get('weight_grid')}")
        lines.append("")

    if audit_path and Path(audit_path).exists():
        audit = json.loads(Path(audit_path).read_text())
        s = audit.get("summary", {})
        lines.append("## Dataset audit")
        lines.append("")
        lines.append(
            f"- {s.get('n_recordings')} recordings, {s.get('n_subjects')} participants, "
            f"{_f(s.get('total_duration_h'), 2)} h total / {_f(s.get('total_valid_h'), 2)} h valid experimental."
        )
        lines.append(
            f"- Freeze time {_f(s.get('total_fog_min'), 1)} min over {s.get('total_events')} events; "
            f"non-freeze experimental time {_f(s.get('total_nonfog_h'), 2)} h."
        )
        lines.append(
            f"- Participants without freezes (kept in evaluation for false alarms): "
            f"{s.get('subjects_without_freeze')}."
        )
        lines.append(
            f"- Max timestamp gap {_f(s.get('max_gap_s'), 2)} s; duplicate timestamps "
            f"{s.get('total_dup_timestamps')}; non-monotonic {s.get('total_nonmono')}; "
            f"NaN samples {s.get('total_nan_acc')}."
        )
        lines.append("")

    for cand, s in data.items():
        p = s["tol_0s"]["pooled"]
        b = s["tol_0s"].get("bootstrap") or {}
        m = s["tol_0s"]["macro"]
        j = s["jitter_tol_0.3s"]["pooled"]
        lines.append(f"## Candidate: `{cand}`")
        lines.append("")
        rows = [
            [
                "tol 0.0 s", p.get("total_events"), p.get("total_detected"),
                _f(p.get("event_sensitivity")), _r(b.get("event_sensitivity_ci95")),
                _f(p.get("false_cue_starts_per_nonfog_hour")),
                _r(b.get("false_cues_per_hour_ci95")),
            ],
            [
                "tol 0.3 s (annotation jitter)", j.get("total_events"), j.get("total_detected"),
                _f(j.get("event_sensitivity")), "n/a",
                _f(j.get("false_cue_starts_per_nonfog_hour")), "n/a",
            ],
        ]
        lines.append(
            _table(
                ["Matching", "Events", "Detected", "Event sensitivity", "95% CI",
                 "False cues / non-FOG h", "95% CI"],
                rows,
            )
        )
        lines.append("")
        det = s["folds"][0].get("delay_mean_s") if s["folds"] else None
        delays = [f["delay_mean_s"] for f in s["folds"] if f.get("delay_mean_s") is not None]
        lines.append(
            f"- Macro event sensitivity: {_f(m.get('event_sensitivity'))}; "
            f"macro false cues/h: {_f(m.get('false_cue_starts_per_nonfog_hour'))}."
        )
        lines.append(
            f"- Pooled window PR-AUC (endpoint label): {_f(s.get('pooled_window_pr_auc_endpoint'))}."
        )
        if delays:
            lines.append(
                f"- Mean onset-to-cue delay across folds: min {_f(min(delays), 2)} s, "
                f"max {_f(max(delays), 2)} s (per-fold means)."
            )
        lines.append("")
        fold_rows = [
            [
                f"S{f['subject']:02d}", f.get("n_events"), f.get("n_detected"),
                _f(f.get("event_sensitivity")), f.get("n_false_cue_starts"),
                _f(f.get("false_cue_starts_per_nonfog_hour"), 2), _f(f.get("nonfog_hours"), 2),
                _f((f.get("recipe") or {}).get("threshold"), 2),
                f"({(f.get('recipe') or {}).get('w_cnn')},{(f.get('recipe') or {}).get('w_fi')})",
            ]
            for f in s["folds"]
        ]
        lines.append(
            _table(
                ["Subject", "Events", "Detected", "Sensitivity", "False starts",
                 "False/h", "non-FOG h", "Thr", "Weights"],
                fold_rows,
            )
        )
        lines.append("")

    lines.append("## Claim scope")
    lines.append("")
    lines.append(CLAIM_SCOPE_NOTE)
    lines.append("")

    text = "\n".join(lines)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(text)
        print(f"Report written to {out_path}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Render assessment report")
    parser.add_argument("--summary", default="models/nested/baseline_v1/summary.json", type=Path)
    parser.add_argument("--audit", default="models/dataset_audit.json", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()

    out = args.out or args.summary.parent / "assessment_report.md"
    render(args.summary, args.audit, out)


if __name__ == "__main__":
    main()
