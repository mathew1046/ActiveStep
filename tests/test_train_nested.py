"""Tests for the nested assessment driver (src/train_nested.py).

The smoke test runs the full nested protocol on real Daphnet subjects 1-3
with a tiny budget (1 epoch, capped windows). It verifies MECHANICS and
leakage guards, not model quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.train_nested import (
    THRESHOLD_GRID,
    candidate_weight_grid,
    grouped_inner_splits,
    run_fold,
    subject_balanced_weights,
    summarize,
)

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "dataset_fog_release" / "dataset"


def test_grouped_inner_splits_cover_each_participant_once():
    train = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    splits = grouped_inner_splits(train, 3)
    vals = [s for _, val in splits for s in val]
    assert sorted(vals) == sorted(train)  # every participant is inner-val exactly once
    for itr, iva in splits:
        assert not (set(itr) & set(iva))
        assert set(itr) | set(iva) == set(train)


def test_grouped_inner_splits_two_subjects():
    splits = grouped_inner_splits([2, 3], 3)
    vals = [s for _, val in splits for s in val]
    assert sorted(vals) == [2, 3]


def test_subject_balanced_weights_equalize_participant_mass():
    subjects = np.array([1, 1, 2, 2, 2, 2])
    labels = np.array([0, 1, 0, 0, 0, 1])
    weights = subject_balanced_weights(subjects, labels, positive_weight=3.0)
    assert weights[labels == 1].mean() > weights[labels == 0].mean()
    assert weights[subjects == 1].sum() == pytest.approx(weights[subjects == 2].sum())
    assert weights.mean() == pytest.approx(1.0)


@pytest.fixture(scope="module")
def smoke_ctx(tmp_path_factory):
    if not DATASET.exists():
        pytest.skip("Daphnet dataset not present")
    return {
        "dataset": str(DATASET),
        "cache_dir": str(tmp_path_factory.mktemp("cache")),
        "outdir": str(tmp_path_factory.mktemp("nested")),
        "tag": "smoke",
        "subjects": [1, 2, 3],
        "epochs": 1,
        "batch": 64,
        "seed": 7,
        "fp_budget": 1.5,
        "n_boot": 50,
        "max_windows": 600,
        "force": False,
    }


def test_nested_smoke_all_candidates(smoke_ctx):
    for cand in ("gated_fi", "logistic_endpoint", "cnn_endpoint"):
        results = [run_fold(smoke_ctx, cand, s) for s in smoke_ctx["subjects"]]
        out = Path(smoke_ctx["outdir"]) / "smoke"
        for r in results:
            fold = out / cand / f"fold_s{r['subject']:02d}"
            assert (fold / "result.json").exists()
            assert (fold / "recipe.json").exists()
            assert (fold / "sweep.json").exists()
            assert (fold / "predictions.npz").exists()
            # recipe values come from the predeclared grids
            assert r["recipe"]["threshold"] in THRESHOLD_GRID
            grid = candidate_weight_grid(cand)
            assert (r["recipe"]["w_cnn"], r["recipe"]["w_fi"]) in grid
            assert r["recipe"]["on_consecutive"] >= 1
            assert r["recipe"]["selection_tier"] in {
                "declared_budget", "relaxed_fallback", "minimum_burden_fallback",
            }
            # protocol echo recorded
            assert r["protocol"]["fp_budget_per_hour"] == smoke_ctx["fp_budget"]
            # predictions saved with all channels
            z = np.load(fold / "predictions.npz")
            assert len(z.files) >= 1
            arr = z[z.files[0]]
            assert arr.shape[0] == 6  # t_ms, fi, p_model, score, label_endpoint, label_any

        s = summarize(cand, results, smoke_ctx)
        assert s["tol_0s"]["pooled"]["n_participants"] == 3
        assert s["tol_0s"]["bootstrap"] is not None
        assert len(s["folds"]) == 3
        # subjects without freezes yield undefined sensitivity, not zeros
        per_subject = {f["subject"]: f["event_sensitivity"] for f in s["folds"]}
        assert all(v is None or 0.0 <= v <= 1.0 for v in per_subject.values())
