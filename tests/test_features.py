"""Unit tests for feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import compute_cadence, compute_freeze_index, festination_index, step_amplitude


def test_freeze_index_high_for_tremor():
    fs = 100
    t = np.arange(fs * 2) / fs
    # 5 Hz tremor in vertical axis (freeze band)
    x = np.sin(2 * np.pi * 5 * t) * 500  # mg
    acc = np.zeros((len(t), 3), dtype=np.float32)
    acc[:, 1] = x
    fi = compute_freeze_index(acc[None, ...], fs)[0]
    assert fi > 3.0, f"expected high FI, got {fi}"


def test_freeze_index_low_for_steady_walk():
    fs = 100
    t = np.arange(fs * 2) / fs
    # 1.5 Hz locomotion in vertical axis
    x = np.sin(2 * np.pi * 1.5 * t) * 500
    acc = np.zeros((len(t), 3), dtype=np.float32)
    acc[:, 1] = x
    fi = compute_freeze_index(acc[None, ...], fs)[0]
    assert fi < 1.0, f"expected low FI, got {fi}"


def test_cadence_estimation():
    fs = 100
    t = np.arange(fs * 3) / fs
    # 100 steps/min -> 1.667 Hz
    x = np.sin(2 * np.pi * 100 / 60 * t) * 500
    cad = compute_cadence(x, fs)
    assert 90 < cad < 110, f"expected cadence near 100, got {cad}"


def test_festination_positive():
    # cadence up, amplitude down
    f = festination_index(110, 300, 100, 500)
    assert f > 0


def test_step_amplitude():
    a = step_amplitude(np.array([0, 100, -50, 200], dtype=np.float32))
    assert a == 250
