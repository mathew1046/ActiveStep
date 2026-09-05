"""Tests for the corrected Daphnet data pipeline (src/data.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import (
    ANNOT_EXP,
    ANNOT_FOG,
    ANNOT_NONE,
    ORIGINAL_FS,
    TARGET_FS,
    FogEvent,
    RunData,
    audit_run,
    extract_windows,
    fog_events,
    load_run,
    load_subject_windows,
    resample_run,
)


def _make_run(subject=99, run=1, n=6400, fog_slices=(), none_slices=(), seed=0):
    """Synthetic Daphnet-style recording: 64 Hz, mg units, 11 columns."""
    rng = np.random.default_rng(seed)
    t = (np.arange(n) * (1000.0 / ORIGINAL_FS)).astype(np.int64)
    acc = rng.normal(0, 50, size=(n, 9)).astype(np.float32)
    acc[:, 1] += 1000.0  # gravity-ish on vertical
    annot = np.full(n, ANNOT_EXP, dtype=np.int8)
    for a, b in none_slices:
        annot[a:b] = ANNOT_NONE
    for a, b in fog_slices:
        annot[a:b] = ANNOT_FOG
    return RunData(subject, run, t, acc, annot)


def _write_run_files(tmp_path: Path, subject=98, runs=2, n_per_run=6400, fog_slices=(), none_slices=()):
    d = tmp_path / "dataset"
    d.mkdir(exist_ok=True)
    for r in range(1, runs + 1):
        rd = _make_run(subject=subject, run=r, n=n_per_run,
                       fog_slices=fog_slices if r == 1 else (), seed=r)
        arr = np.column_stack([rd.t_ms.astype(np.float64), rd.acc, rd.annot.astype(np.float64)])
        np.savetxt(d / f"S{subject:02d}R{r:02d}.txt", arr, fmt="%.1f")
    return d


# --- ground-truth events ----------------------------------------------------

def test_fog_events_basic_and_censoring():
    run = _make_run(n=640, fog_slices=[(100, 200), (400, 500)])
    ev = fog_events(run.annot, run.t_ms)
    assert len(ev) == 2
    assert ev[0].onset_ms == run.t_ms[100]
    assert ev[0].end_ms == pytest.approx(int(run.t_ms[199]) + 1000.0 / ORIGINAL_FS, abs=1)
    assert not ev[0].start_censored and not ev[0].end_censored
    assert ev[0].n_samples == 100


def test_fog_events_merges_short_experimental_gaps_only():
    # two 1 s freezes separated by 0.25 s of walking -> merged
    run = _make_run(n=640, fog_slices=[(100, 164), (180, 244)])  # 16 samples = 0.25 s gap
    ev = fog_events(run.annot, run.t_ms, merge_gap_s=0.5)
    assert len(ev) == 1
    # merged event spans both fog runs plus the gap (n_samples = span length)
    assert ev[0].n_samples == 144
    assert ev[0].duration_s == pytest.approx(144 / ORIGINAL_FS, abs=0.05)

    # separated by 2 s -> not merged
    run = _make_run(n=3200, fog_slices=[(100, 164), (288, 352)])  # 124 samples = ~1.94 s gap
    assert len(fog_events(run.annot, run.t_ms)) == 2

    # separated by non-experimental time -> never merged
    run = _make_run(n=640, fog_slices=[(100, 164), (180, 244)], none_slices=[(165, 179)])
    assert len(fog_events(run.annot, run.t_ms)) == 2


def test_fog_events_censor_flags():
    # fog [0,160) runs into a non-experimental stretch [160,200);
    # fog [200,320) starts right after that stretch and touches the recording end
    run = _make_run(n=320, fog_slices=[(0, 160), (200, 320)], none_slices=[(160, 200)])
    ev = fog_events(run.annot, run.t_ms)
    assert len(ev) == 2  # never merged across non-experimental time
    assert ev[0].start_censored  # touches recording start
    assert ev[0].end_censored    # freeze runs into non-experimental region
    assert ev[1].start_censored  # freeze starts right after non-experimental region
    assert ev[1].end_censored    # touches recording end


# --- resampling --------------------------------------------------------------

def test_resample_keeps_annotations_categorical():
    run = _make_run(n=6400, fog_slices=[(1000, 2000)], none_slices=[(3000, 4000)])
    rr = resample_run(run)
    assert set(np.unique(rr.annot)).issubset({ANNOT_NONE, ANNOT_EXP, ANNOT_FOG})
    # durations preserved to within ~1 original sample on each side
    fog_new = float((rr.annot == ANNOT_FOG).sum()) / TARGET_FS
    assert fog_new == pytest.approx(1000 / ORIGINAL_FS, abs=0.05)
    # availability charges one original sample interval (15.625 ms rounded to 16 ms)
    assert np.all(rr.avail_ms - rr.t_ms == 16)


def test_resample_linear_interpolation_matches_numpy():
    run = _make_run(n=64, seed=3)
    rr = resample_run(run)
    expected = np.interp(rr.t_ms, run.t_ms, run.acc[:, 0])
    assert np.allclose(rr.acc[:, 0], expected, atol=1e-3)


# --- window extraction --------------------------------------------------------

def test_windows_never_cross_invalid_segments():
    run = _make_run(n=6400, none_slices=[(3000, 3400)])  # 6.25 s non-experimental
    rr = resample_run(run)
    rw = extract_windows(rr, window_s=2.0, hop_s=0.25)
    win = int(2.0 * TARGET_FS)
    # no window may overlap the invalid stretch [t(3000), t(3400))
    lo, hi = run.t_ms[3000], run.t_ms[3400]
    for start_t, end_t in zip(rw.decision_ms - win * 10, rw.decision_ms):  # 10 ms per sample
        assert not (start_t < hi and end_t > lo), "window overlaps non-experimental segment"
    assert rw.n_excluded > 0


def test_window_labels_and_occupancy():
    # freeze in the middle of the recording: annot==2 for samples [2000, 2600)
    run = _make_run(n=6400, fog_slices=[(2000, 2600)])
    rr = resample_run(run)
    rw = extract_windows(rr, window_s=2.0, hop_s=0.25)
    assert (rw.label_endpoint & rw.label_any).sum() == rw.label_endpoint.sum()
    assert 0 < rw.label_endpoint.sum() < rw.label_any.sum()
    # a window fully inside the freeze has occupancy 1
    full = rw.occupancy >= 0.999
    assert full.sum() > 0


def test_windows_do_not_cross_runs(tmp_path):
    d = _write_run_files(tmp_path, runs=2, n_per_run=6400, fog_slices=[(1000, 1500)])
    from src.data import load_participant

    bundles = load_participant(d, 98, window_s=2.0, hop_s=0.25)
    assert len(bundles) == 2
    for b in bundles:
        # every decision lies inside its own recording's time span
        assert b.t_start_ms <= b.decision_ms.min() <= b.decision_ms.max() <= b.t_end_ms
        # window count bounded by run duration alone (no cross-run windows)
        max_windows = int((b.t_end_ms - b.t_start_ms) / 1000 / 0.25) + 1
        assert len(b.decision_ms) <= max_windows


def test_legacy_load_subject_windows_compat(tmp_path):
    d = _write_run_files(tmp_path, runs=2, n_per_run=6400, fog_slices=[(1000, 1500)])
    out = load_subject_windows(d, 98, 2.0, 0.5)
    assert out["shank"].ndim == 3 and out["shank"].shape[1] == 200 and out["shank"].shape[2] == 3
    assert out["y"].dtype == bool and len(out["y"]) == out["shank"].shape[0]
    assert out["trunk"].shape == out["shank"].shape
    assert out["y"].any()  # the freeze run contributes positives


def test_audit_run_counts():
    run = _make_run(n=6400, fog_slices=[(1000, 1640), (1800, 2100)], none_slices=[(4000, 4320)])
    a = audit_run(run)
    assert a["n_samples"] == 6400
    assert a["n_events"] == 2
    assert a["duration_s"] == pytest.approx(100.0, abs=0.1)
    assert a["valid_s"] == pytest.approx(95.0, abs=0.2)
    assert a["fog_s"] == pytest.approx(940 / 64, abs=0.05)
    assert a["nonfog_s"] == pytest.approx(a["valid_s"] - a["fog_s"], abs=0.1)


def test_cache_roundtrip(tmp_path):
    d = _write_run_files(tmp_path, runs=1, n_per_run=6400, fog_slices=[(1000, 1500)])
    cache = tmp_path / "cache"
    from src.data import build_cache, load_cached_run

    build_cache(d, cache, hops=(0.25, 0.5))
    b = load_cached_run(cache, 98, 1, 0.25)
    assert b.X_shank.shape[1] == 200 and b.X_shank.shape[2] == 3
    assert len(b.events) == 1
    assert b.exposure["fog_s"] > 0
    # decision times strictly increasing
    assert np.all(np.diff(b.decision_ms) > 0)
