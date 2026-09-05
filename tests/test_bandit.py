"""Tests for the cue-modality bandit."""

from __future__ import annotations

import numpy as np

from unoq.bandit import CueBandit


def test_bandit_converges_to_best_arm():
    b = CueBandit()
    # Arm 3 (vibration+laser) is fastest
    true_rec = {1: 4000, 2: 2500, 4: 5000, 3: 1500, 5: 3000, 7: 2000}
    for _ in range(100):
        arm = b.select()
        noisy = true_rec[arm] * np.random.uniform(0.9, 1.1)
        b.reward(arm, noisy)
    pref = b.preference()
    assert pref["vibration+laser"] > pref["vibration"]
