"""Daphnet Freezing-of-Gait dataset: audit, corrected resampling, windowing, cache.

Assessment-pipeline rules (see plan.md):
- Annotations are categorical and are never numerically interpolated.
- Recordings (runs) are processed independently: windows never cross run
  boundaries and never overlap non-experimental (annot==0) samples.
- Resampling is streaming-compatible: linear interpolation for acceleration
  (one-sample lookahead, charged to decision availability time), zero-order
  hold for annotations.
- Every window carries provenance: subject, run, decision time, endpoint and
  any-freeze labels, freeze occupancy.
- Ground-truth freeze events are derived from the original 64 Hz annotations
  before any windowing, so matching and latency are measured in real time.

Units: acceleration in mg. Axis order per Daphnet: forward, vertical, lateral
for each of shank (ankle), thigh, trunk.
"""

from __future__ import annotations

import glob
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:
    from src.features import compute_freeze_index
except ImportError:  # imported as a top-level module
    from features import compute_freeze_index

ORIGINAL_FS = 64
TARGET_FS = 100

# Column layout of each S<ss>R<rr>.txt file (0-based):
#   0: time (ms), 1-3: shank f/v/l, 4-6: thigh f/v/l, 7-9: trunk f/v/l, 10: annotation
SHANK_COLS = slice(1, 4)
THIGH_COLS = slice(4, 7)
TRUNK_COLS = slice(7, 10)
ANNOT_COL = 10

ANNOT_NONE = 0  # not part of the experiment
ANNOT_EXP = 1   # experiment, no freeze (stand / walk / turn)
ANNOT_FOG = 2   # freeze

# Freeze runs separated by less than this much *experimental* non-freeze time
# are treated as one event (guards against annotation flicker).
MERGE_GAP_S = 0.5

# Linear interpolation of acceleration needs the next original 64 Hz sample.
LOOKAHEAD_MS = 1000.0 / ORIGINAL_FS  # ~15.6 ms, charged to decision availability

WINDOW_S = 2.0
TRAIN_HOP_S = 0.5
EVAL_HOP_S = 0.25

ACC_SLICES = {"shank": slice(0, 3), "thigh": slice(3, 6), "trunk": slice(6, 9)}


@dataclass
class RunData:
    """One raw recording, as stored on disk."""

    subject: int
    run: int
    t_ms: np.ndarray  # (n,) int64, original timestamps
    acc: np.ndarray   # (n, 9) float32, mg, shank+thigh+trunk f/v/l
    annot: np.ndarray # (n,) int8, in {0, 1, 2}


@dataclass
class FogEvent:
    """One ground-truth freeze episode, in original recording time (ms).

    onset_ms: time of the first annotated freeze sample.
    end_ms:   time one median sample after the last annotated freeze sample
              (i.e. when the freeze annotation stops).
    """

    onset_ms: int
    end_ms: int
    start_censored: bool
    end_censored: bool
    n_samples: int

    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.onset_ms) / 1000.0


@dataclass
class RunWindows:
    """All evaluated windows of one recording, plus its ground truth."""

    subject: int
    run: int
    fs: int
    window_s: float
    hop_s: float
    decision_ms: np.ndarray      # (k,) window END (= decision time), ms
    avail_ms: np.ndarray         # (k,) when the decision is available, ms
    X_shank: np.ndarray          # (k, W, 3) float32, raw mg
    label_endpoint: np.ndarray   # (k,) bool, annot==2 at the window end
    label_any: np.ndarray        # (k,) bool, any annot==2 inside the window
    occupancy: np.ndarray        # (k,) float32, fraction of annot==2 samples
    fi: np.ndarray               # (k,) float32, Freeze Index of the window
    n_excluded: int              # windows skipped for overlapping annot==0
    t_start_ms: int
    t_end_ms: int
    events: list[FogEvent] = field(default_factory=list)
    exposure: dict = field(default_factory=dict)
    X_trunk: np.ndarray | None = None  # only populated for the legacy loader

    def truncate_for_eval(self, max_windows: int) -> "RunWindows":
        """Return a copy keeping only the first `max_windows` windows (smoke runs)."""
        if max_windows is None or len(self.decision_ms) <= max_windows:
            return self
        k = max_windows
        return RunWindows(
            self.subject, self.run, self.fs, self.window_s, self.hop_s,
            self.decision_ms[:k], self.avail_ms[:k], self.X_shank[:k],
            self.label_endpoint[:k], self.label_any[:k], self.occupancy[:k],
            self.fi[:k], self.n_excluded, self.t_start_ms, self.t_end_ms,
            self.events, self.exposure,
        )


def subject_from_filename(path: str | Path) -> int:
    """Extract the subject id from a Daphnet file name like 'S01R01.txt'."""
    m = re.search(r"S(\d+)R", str(path))
    if not m:
        raise ValueError(f"Cannot parse subject from {path}")
    return int(m.group(1))


def run_from_filename(path: str | Path) -> int:
    m = re.search(r"R(\d+)\.txt$", str(path))
    if not m:
        raise ValueError(f"Cannot parse run from {path}")
    return int(m.group(1))


def list_subjects(dataset_dir: str | Path) -> list[int]:
    """Return the list of subject numbers present in the dataset."""
    files = sorted(glob.glob(str(Path(dataset_dir) / "S*R*.txt")))
    subjects = {subject_from_filename(f) for f in files}
    return sorted(subjects)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(path: str | Path) -> RunData:
    """Load one Daphnet recording. Values are kept exactly as on disk."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 11:
        raise ValueError(f"{path}: expected 11 columns, got {df.shape[1]}")
    t = df.iloc[:, 0].to_numpy(dtype=np.int64)
    acc = df.iloc[:, 1:10].to_numpy(dtype=np.float32)
    annot = df.iloc[:, ANNOT_COL].to_numpy(dtype=np.float32)
    annot = np.where(np.isnan(annot), ANNOT_NONE, annot).astype(np.int8)
    return RunData(subject_from_filename(path), run_from_filename(path), t, acc, annot)


# ---------------------------------------------------------------------------
# Ground-truth events and audit
# ---------------------------------------------------------------------------

def fog_events(annot: np.ndarray, t_ms: np.ndarray, merge_gap_s: float = MERGE_GAP_S) -> list[FogEvent]:
    """Derive freeze episodes from the ORIGINAL annotations.

    Contiguous annot==2 runs are grouped; runs separated by a stretch of
    experimental (annot==1) non-freeze time shorter than `merge_gap_s` are
    merged into one event. Separations involving annot==0 (experiment break)
    are never merged. An event is boundary-censored when it touches the
    recording edge or a non-experimental region.
    """
    idx = np.flatnonzero(annot == ANNOT_FOG)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])  # inclusive

    if len(t_ms) > 1:
        dt_med = float(np.median(np.diff(t_ms)))
    else:
        dt_med = 1000.0 / ORIGINAL_FS
    gap_ms = merge_gap_s * 1000.0

    merged: list[tuple[int, int]] = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        ps, pe = merged[-1]
        stretch = annot[pe + 1 : s]
        stretch_ok = bool(np.all(stretch == ANNOT_EXP))
        stretch_ms = float(t_ms[s] - t_ms[pe]) - dt_med
        if stretch_ok and stretch_ms < gap_ms:
            merged[-1] = (ps, int(e))
        else:
            merged.append((int(s), int(e)))

    n = len(annot)
    events: list[FogEvent] = []
    for s, e in merged:
        start_censored = s == 0 or annot[s - 1] == ANNOT_NONE
        end_censored = e == n - 1 or annot[e + 1] == ANNOT_NONE
        events.append(
            FogEvent(
                onset_ms=int(t_ms[s]),
                end_ms=int(t_ms[e]) + int(round(dt_med)),
                start_censored=bool(start_censored),
                end_censored=bool(end_censored),
                n_samples=int(e - s + 1),
            )
        )
    return events


def exposure(annot: np.ndarray, t_ms: np.ndarray) -> dict:
    """Durations (seconds) by annotation class, from the original stream."""
    if len(t_ms) < 2:
        return {"total_s": 0.0, "valid_s": 0.0, "fog_s": 0.0, "nonfog_s": 0.0}
    dt = np.diff(t_ms) / 1000.0
    a = annot[:-1]
    total = float(dt.sum())
    valid = float(dt[a != ANNOT_NONE].sum())
    fog = float(dt[a == ANNOT_FOG].sum())
    return {
        "total_s": total,
        "valid_s": valid,
        "fog_s": fog,
        "nonfog_s": valid - fog,
        "fog_fraction_valid": fog / valid if valid > 0 else 0.0,
    }


def audit_run(run: RunData, merge_gap_s: float = MERGE_GAP_S) -> dict:
    """Per-recording quality and content inventory."""
    dt = np.diff(run.t_ms) if len(run.t_ms) > 1 else np.array([])
    ev = fog_events(run.annot, run.t_ms, merge_gap_s)
    exp = exposure(run.annot, run.t_ms)
    durations = [e.duration_s for e in ev]
    return {
        "subject": run.subject,
        "run": run.run,
        "n_samples": int(len(run.t_ms)),
        "duration_s": exp["total_s"],
        "dt_median_ms": float(np.median(dt)) if len(dt) else None,
        "n_dup_timestamps": int((dt == 0).sum()),
        "n_nonmono": int((dt < 0).sum()),
        "max_gap_s": float(dt.max() / 1000.0) if len(dt) else 0.0,
        "n_nan_acc": int(np.isnan(run.acc).sum()),
        "valid_s": exp["valid_s"],
        "fog_s": exp["fog_s"],
        "nonfog_s": exp["nonfog_s"],
        "fog_fraction_valid": exp.get("fog_fraction_valid", 0.0),
        "n_events_raw": int(np.sum(np.diff(np.concatenate([[0], (run.annot == ANNOT_FOG).astype(int), [0]])) == 1)),
        "n_events": len(ev),
        "n_events_merged_away": int(
            np.sum(np.diff(np.concatenate([[0], (run.annot == ANNOT_FOG).astype(int), [0]])) == 1) - len(ev)
        ),
        "event_duration_min_s": float(min(durations)) if durations else None,
        "event_duration_median_s": float(np.median(durations)) if durations else None,
        "event_duration_max_s": float(max(durations)) if durations else None,
        "n_events_short_lt_1s": int(sum(d < 1.0 for d in durations)),
        "n_start_censored": int(sum(e.start_censored for e in ev)),
        "n_end_censored": int(sum(e.end_censored for e in ev)),
    }


def audit_dataset(dataset_dir: str | Path, out_json: str | Path | None = None) -> dict:
    """Audit every recording; returns per-run rows plus a summary."""
    files = sorted(glob.glob(str(Path(dataset_dir) / "S*R*.txt")))
    runs = []
    for f in files:
        a = audit_run(load_run(f))
        a["file"] = Path(f).name
        a["sha256"] = file_sha256(f)
        runs.append(a)

    events_per_subject: dict[int, int] = {}
    for r in runs:
        events_per_subject[r["subject"]] = events_per_subject.get(r["subject"], 0) + r["n_events"]
    fog_free = sorted(s for s, n in events_per_subject.items() if n == 0)
    summary = {
        "n_recordings": len(runs),
        "n_subjects": len({r["subject"] for r in runs}),
        "total_duration_h": float(sum(r["duration_s"] for r in runs) / 3600.0),
        "total_valid_h": float(sum(r["valid_s"] for r in runs) / 3600.0),
        "total_fog_min": float(sum(r["fog_s"] for r in runs) / 60.0),
        "total_nonfog_h": float(sum(r["nonfog_s"] for r in runs) / 3600.0),
        "total_events": int(sum(r["n_events"] for r in runs)),
        "subjects_without_freeze": fog_free,
        "max_gap_s": float(max(r["max_gap_s"] for r in runs)),
        "total_dup_timestamps": int(sum(r["n_dup_timestamps"] for r in runs)),
        "total_nonmono": int(sum(r["n_nonmono"] for r in runs)),
        "total_nan_acc": int(sum(r["n_nan_acc"] for r in runs)),
    }
    out = {"runs": runs, "summary": summary}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# Resampling and windowing
# ---------------------------------------------------------------------------

@dataclass
class ResampledRun:
    subject: int
    run: int
    fs: int
    t_ms: np.ndarray     # (m,) int64, uniform target-fs grid
    acc: np.ndarray      # (m, 9) float32, mg
    annot: np.ndarray    # (m,) int8, zero-order-hold of the original labels
    avail_ms: np.ndarray # (m,) decision availability (t + LOOKAHEAD_MS)
    events: list[FogEvent]
    exposure: dict


def resample_run(run: RunData, target_fs: int = TARGET_FS) -> ResampledRun:
    """Resample one recording to `target_fs` on a uniform grid.

    Acceleration: linear interpolation between neighbouring original samples
    (streaming-compatible; the one-sample lookahead is charged to `avail_ms`).
    Annotations: zero-order hold — categorical values are never interpolated.
    Events and exposure are computed on the original stream.
    """
    t0, t1 = int(run.t_ms[0]), int(run.t_ms[-1])
    n = int((t1 - t0) * target_fs / 1000.0) + 1
    t_new = t0 + (np.arange(n) * (1000.0 / target_fs)).astype(np.int64)

    acc_new = np.empty((n, 9), dtype=np.float32)
    t_f = run.t_ms.astype(np.float64)
    for c in range(9):
        col = run.acc[:, c]
        if np.isnan(col).any():
            ok = ~np.isnan(col)
            acc_new[:, c] = np.interp(t_new, t_f[ok], col[ok]).astype(np.float32)
        else:
            acc_new[:, c] = np.interp(t_new, t_f, col).astype(np.float32)

    hold_idx = np.clip(np.searchsorted(t_f, t_new, side="right") - 1, 0, len(t_f) - 1)
    annot_new = run.annot[hold_idx]

    return ResampledRun(
        subject=run.subject,
        run=run.run,
        fs=target_fs,
        t_ms=t_new,
        acc=acc_new,
        annot=annot_new,
        avail_ms=t_new + int(round(LOOKAHEAD_MS)),
        events=fog_events(run.annot, run.t_ms),
        exposure=exposure(run.annot, run.t_ms),
    )


def extract_windows(
    rr: ResampledRun,
    window_s: float = WINDOW_S,
    hop_s: float = EVAL_HOP_S,
    with_trunk: bool = False,
    compute_fi: bool = True,
) -> RunWindows:
    """Extract trailing windows from one resampled recording.

    A window [end-W+1 .. end] is included iff every sample is experimental
    (annot != 0). Windows overlapping annot==0 are excluded and counted.
    Decision time = window end; availability = end sample availability.
    """
    fs = rr.fs
    win = int(round(window_s * fs))
    hop = int(round(hop_s * fs))
    n = len(rr.t_ms)

    base = RunWindows(
        subject=rr.subject, run=rr.run, fs=fs, window_s=window_s, hop_s=hop_s,
        decision_ms=np.empty(0, dtype=np.int64),
        avail_ms=np.empty(0, dtype=np.int64),
        X_shank=np.empty((0, win, 3), dtype=np.float32),
        label_endpoint=np.empty(0, dtype=bool),
        label_any=np.empty(0, dtype=bool),
        occupancy=np.empty(0, dtype=np.float32),
        fi=np.empty(0, dtype=np.float32),
        n_excluded=0,
        t_start_ms=int(rr.t_ms[0]) if n else 0,
        t_end_ms=int(rr.t_ms[-1]) if n else 0,
        events=list(rr.events),
        exposure=dict(rr.exposure),
    )
    if n < win:
        return base

    valid = (rr.annot != ANNOT_NONE).astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(valid)])
    ends = np.arange(win - 1, n, hop)
    ok = (csum[ends + 1] - csum[ends + 1 - win]) == win
    ends_ok = ends[ok]
    n_excluded = int((~ok).sum())
    starts = ends_ok - win + 1

    sw_annot = sliding_window_view(rr.annot, win)[starts]
    shank_view = sliding_window_view(rr.acc[:, ACC_SLICES["shank"]], win, axis=0)[starts]  # (k, 3, W)
    X_shank = np.ascontiguousarray(np.transpose(shank_view, (0, 2, 1)))

    labels_any = (sw_annot == ANNOT_FOG).any(axis=1)
    occupancy = (sw_annot == ANNOT_FOG).mean(axis=1).astype(np.float32)

    if with_trunk:
        trunk_view = sliding_window_view(rr.acc[:, ACC_SLICES["trunk"]], win, axis=0)[starts]
        base.X_trunk = np.ascontiguousarray(np.transpose(trunk_view, (0, 2, 1)))

    fi = (
        compute_freeze_index(X_shank, fs=fs)
        if compute_fi and len(X_shank)
        else np.zeros(len(X_shank), dtype=np.float32)
    )

    base.decision_ms = rr.t_ms[ends_ok]
    base.avail_ms = rr.avail_ms[ends_ok]
    base.X_shank = X_shank
    base.label_any = labels_any
    base.label_endpoint = rr.annot[ends_ok] == ANNOT_FOG
    base.occupancy = occupancy
    base.fi = np.asarray(fi, dtype=np.float32)
    base.n_excluded = n_excluded
    return base


# ---------------------------------------------------------------------------
# Participant loading (direct and cached)
# ---------------------------------------------------------------------------

def load_participant(
    dataset_dir: str | Path,
    subject: int,
    window_s: float = WINDOW_S,
    hop_s: float = EVAL_HOP_S,
    with_trunk: bool = False,
    compute_fi: bool = True,
) -> list[RunWindows]:
    """Load every recording of one subject through the corrected pipeline."""
    files = sorted(glob.glob(str(Path(dataset_dir) / f"S{subject:02d}R*.txt")))
    if not files:
        raise FileNotFoundError(f"No files found for subject {subject:02d} in {dataset_dir}")
    out = []
    for f in files:
        rr = resample_run(load_run(f))
        out.append(extract_windows(rr, window_s, hop_s, with_trunk, compute_fi))
    return out


def _cache_run_dir(cache_dir: str | Path, subject: int, run: int, hop_s: float) -> Path:
    return Path(cache_dir) / f"S{subject:02d}R{run:02d}_hop{int(round(hop_s * 1000)):03d}"


def _build_cache_run(
    run: RunData,
    cache_dir: str | Path,
    window_s: float,
    hop_s: float,
    with_trunk: bool = False,
) -> str | None:
    """Build one recording's cache entry for one hop. Returns dir name or None if present."""
    d = _cache_run_dir(cache_dir, run.subject, run.run, hop_s)
    if (d / "X_shank.npy").exists():
        return None
    rw = extract_windows(resample_run(run), window_s, hop_s, with_trunk, compute_fi=True)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "X_shank.npy", rw.X_shank)
    if with_trunk and rw.X_trunk is not None:
        np.save(d / "X_trunk.npy", rw.X_trunk)
    ev = rw.events
    np.savez(
        d / "meta.npz",
        decision_ms=rw.decision_ms, avail_ms=rw.avail_ms,
        label_endpoint=rw.label_endpoint,
        label_any=rw.label_any, occupancy=rw.occupancy, fi=rw.fi,
        event_onset_ms=np.array([e.onset_ms for e in ev], dtype=np.int64),
        event_end_ms=np.array([e.end_ms for e in ev], dtype=np.int64),
        event_start_censored=np.array([e.start_censored for e in ev], dtype=bool),
        event_end_censored=np.array([e.end_censored for e in ev], dtype=bool),
        event_n_samples=np.array([e.n_samples for e in ev], dtype=np.int64),
        t_start_ms=rw.t_start_ms, t_end_ms=rw.t_end_ms,
        n_excluded=rw.n_excluded,
        exposure_json=json.dumps(rw.exposure),
    )
    return d.name


def build_cache(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    window_s: float = WINDOW_S,
    hops: tuple[float, ...] = (EVAL_HOP_S, TRAIN_HOP_S),
    with_trunk: bool = False,
) -> dict:
    """Precompute and store windows (X as .npy for mmap; metadata in .npz)."""
    files = sorted(glob.glob(str(Path(dataset_dir) / "S*R*.txt")))
    built, skipped = [], []
    for f in files:
        run = load_run(f)
        for hop_s in hops:
            name = _build_cache_run(run, cache_dir, window_s, hop_s, with_trunk)
            (built if name else skipped).append(name or _cache_run_dir(cache_dir, run.subject, run.run, hop_s).name)
    return {"built": built, "skipped": skipped}


def load_cached_run(cache_dir: str | Path, subject: int, run: int, hop_s: float, mmap: bool = False) -> RunWindows:
    """Load one cached recording. X_shank is memmapped unless `mmap=False`."""
    d = _cache_run_dir(cache_dir, subject, run, hop_s)
    X = np.load(d / "X_shank.npy", mmap_mode="r" if mmap else None)
    m = np.load(d / "meta.npz")
    events = [
        FogEvent(int(o), int(e), bool(sc), bool(ec), int(ns))
        for o, e, sc, ec, ns in zip(
            m["event_onset_ms"], m["event_end_ms"],
            m["event_start_censored"], m["event_end_censored"], m["event_n_samples"],
        )
    ]
    return RunWindows(
        subject=subject, run=run, fs=TARGET_FS,
        window_s=WINDOW_S, hop_s=hop_s,
        decision_ms=m["decision_ms"], avail_ms=m["avail_ms"],
        X_shank=np.asarray(X), label_endpoint=m["label_endpoint"], label_any=m["label_any"],
        occupancy=m["occupancy"], fi=m["fi"], n_excluded=int(m["n_excluded"]),
        t_start_ms=int(m["t_start_ms"]), t_end_ms=int(m["t_end_ms"]),
        events=events, exposure=json.loads(str(m["exposure_json"])),
    )


def load_participant_cached(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    subject: int,
    hop_s: float,
    window_s: float = WINDOW_S,
    build_if_missing: bool = True,
) -> list[RunWindows]:
    """Load one participant's runs at `hop_s`, building the cache if needed."""
    files = sorted(glob.glob(str(Path(dataset_dir) / f"S{subject:02d}R*.txt")))
    if not files:
        raise FileNotFoundError(f"No files found for subject {subject:02d} in {dataset_dir}")
    out = []
    for f in files:
        run_id = run_from_filename(f)
        d = _cache_run_dir(cache_dir, subject, run_id, hop_s)
        if not (d / "X_shank.npy").exists():
            if not build_if_missing:
                raise FileNotFoundError(f"Missing cache entry {d}")
            _build_cache_run(load_run(f), cache_dir, window_s, hop_s)
        out.append(load_cached_run(cache_dir, subject, run_id, hop_s))
    return out


# ---------------------------------------------------------------------------
# Legacy compatibility API (same keys as the previous implementation)
# ---------------------------------------------------------------------------

def freeze_index_per_window(acc: np.ndarray, fs: int) -> float:
    """Mean Freeze Index across the 3 accelerometer axes of one window."""
    n = acc.shape[0]
    if n < 8:
        return 0.0
    try:
        from src.features import compute_freeze_index
    except ImportError:
        from features import compute_freeze_index
    return float(compute_freeze_index(acc[None, ...], fs=fs)[0])


def load_subject_windows(
    dataset_dir: str | Path,
    subject: int,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> dict:
    """Legacy loader: corrected pipeline, same return keys as before.

    Windows are produced per recording (never crossing run boundaries) and
    only from fully experimental segments; `y` is the any-freeze label.
    """
    runs = load_participant(dataset_dir, subject, window_seconds, step_seconds, with_trunk=True)
    usable = [r for r in runs if len(r.decision_ms) and r.X_shank is not None]
    w = int(round(window_seconds * TARGET_FS))
    if not usable:
        empty3 = np.empty((0, w, 3), np.float32)
        return {"subject": subject, "shank": empty3, "trunk": empty3.copy(),
                "y": np.empty(0, bool), "fi": np.empty(0, np.float32),
                "runs": [f"S{r.subject:02d}R{r.run:02d}" for r in runs]}
    shank = np.concatenate([r.X_shank for r in usable])
    trunk = np.concatenate([r.X_trunk for r in usable if r.X_trunk is not None])
    if len(trunk) != len(shank):
        trunk = np.zeros_like(shank)
    y = np.concatenate([r.label_any for r in usable])
    fi = np.concatenate([r.fi for r in usable])
    return {
        "subject": subject,
        "shank": shank,
        "trunk": trunk,
        "y": y,
        "fi": fi,
        "runs": [f"S{r.subject:02d}R{r.run:02d}" for r in runs],
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daphnet audit / window cache builder")
    parser.add_argument("--dataset", default="dataset_fog_release/dataset", type=Path)
    parser.add_argument("--audit", action="store_true", help="Audit recordings and print/emit JSON")
    parser.add_argument("--build-cache", action="store_true", help="Build the window cache")
    parser.add_argument("--cache-dir", default=".cache/windows", type=Path)
    parser.add_argument("--hops", default=f"{EVAL_HOP_S},{TRAIN_HOP_S}", help="Comma-separated hops in seconds")
    parser.add_argument("--out", default="models/dataset_audit.json", type=Path)
    args = parser.parse_args()

    if args.audit:
        out = audit_dataset(args.dataset, args.out)
        print(json.dumps(out["summary"], indent=2))
        print(f"Per-run audit written to {args.out}")
    if args.build_cache:
        hops = tuple(float(h) for h in args.hops.split(","))
        res = build_cache(args.dataset, args.cache_dir, hops=hops)
        print(f"Cache: built {len(res['built'])} entries, skipped {len(res['skipped'])} existing")


if __name__ == "__main__":
    main()
