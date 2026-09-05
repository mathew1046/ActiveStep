"""Cue-modality bandit: Thompson sampling over vibration / laser / audio / combos.

Each arm is a bitmask: bit0=vibration, bit1=laser, bit2=audio.
Reward = 1 / time_to_gait_recovery after a cue, normalized to (0,1].
Arms persist in the bandit_state SQLite table so learning survives restarts.
"""

from __future__ import annotations

import json
import time

import numpy as np

from .db import connect

ARMS = {
    1: "vibration",
    2: "laser",
    4: "audio",
    3: "vibration+laser",
    5: "vibration+audio",
    7: "all",
}
MAX_RECOVERY_S = 10.0


class CueBandit:
    def __init__(self):
        self.conn = connect()
        self._init_arms()

    def _init_arms(self):
        cur = self.conn.cursor()
        for arm in ARMS:
            cur.execute(
                "INSERT OR IGNORE INTO bandit_state (arm, alpha, beta) VALUES (?, 1.0, 1.0)",
                (json.dumps(arm),),
            )
        self.conn.commit()

    def select(self) -> int:
        """Thompson sample -> modality bitmask."""
        cur = self.conn.execute("SELECT arm, alpha, beta FROM bandit_state")
        best_arm, best_sample = 7, -1.0
        for row in cur.fetchall():
            sample = np.random.beta(row["alpha"], row["beta"])
            if sample > best_sample:
                best_sample, best_arm = sample, json.loads(row["arm"])
        return int(best_arm)

    def reward(self, arm: int, recovery_ms: float | None):
        """Update the arm posterior after a cue event resolves."""
        if recovery_ms is None:
            r = 0.05  # cue never resolved -> near-zero reward
        else:
            r = float(np.clip(1.0 - (recovery_ms / 1000.0) / MAX_RECOVERY_S, 0.0, 1.0))
            r = max(r, 0.05)
        self.conn.execute(
            "UPDATE bandit_state SET alpha=alpha+?, beta=beta+? WHERE arm=?",
            (r, 1.0 - r, json.dumps(arm)),
        )
        self.conn.commit()

    def preference(self) -> dict:
        """Mean preference per arm for the dashboard."""
        out = {}
        for row in self.conn.execute("SELECT arm, alpha, beta FROM bandit_state"):
            arm = json.loads(row["arm"])
            out[ARMS.get(arm, str(arm))] = row["alpha"] / (row["alpha"] + row["beta"])
        return out


def main():
    """Demo: simulate the bandit converging on a biased best arm."""
    b = CueBandit()
    true_rec = {1: 4000, 2: 2500, 4: 5000, 3: 1500, 5: 3000, 7: 2000}  # ms, vib+laser best
    for i in range(50):
        arm = b.select()
        noisy = true_rec[arm] * np.random.uniform(0.7, 1.4)
        b.reward(arm, noisy)
    print(json.dumps(b.preference(), indent=2))


if __name__ == "__main__":
    main()
