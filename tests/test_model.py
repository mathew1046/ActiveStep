"""Unit tests for model/data helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import list_subjects, load_subject_windows
from src.model import StandardScaler


def test_list_subjects():
    s = list_subjects("dataset_fog_release/dataset")
    assert 1 in s and 10 in s
    assert len(s) == 10


def test_load_subject_windows():
    d = load_subject_windows("dataset_fog_release/dataset", 1)
    assert d["shank"].ndim == 3
    assert d["shank"].shape[1] == 200
    assert d["shank"].shape[2] == 3
    assert d["y"].dtype == bool
    assert len(d["y"]) == d["shank"].shape[0]


def test_scaler():
    X = np.random.randn(100, 50, 3).astype(np.float32)
    s = StandardScaler().fit(X)
    Xt = s.transform(X)
    np.testing.assert_allclose(Xt.mean(axis=(0, 1)), 0, atol=1e-6)
