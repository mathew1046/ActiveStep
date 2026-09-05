"""SQLite schema shared by all UNO Q services and the dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "activestep.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    t_ms INTEGER, sensor TEXT,
    ax REAL, ay REAL, az REAL,
    gx REAL, gy REAL, gz REAL
);
CREATE TABLE IF NOT EXISTS features (
    t_ms INTEGER, sensor TEXT,
    cadence REAL, step_amp REAL, fi REAL,
    asymmetry REAL, festination REAL, turning REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    t_start INTEGER, t_end INTEGER,
    pfog_peak REAL, fi_peak REAL,
    modality_mask INTEGER, recovery_ms INTEGER,
    model_version TEXT, context TEXT
);
CREATE TABLE IF NOT EXISTS labels (
    event_id INTEGER, label TEXT,
    t_ms INTEGER, source TEXT
);
CREATE TABLE IF NOT EXISTS falls (
    t_ms INTEGER, outcome TEXT, response_ms INTEGER
);
CREATE TABLE IF NOT EXISTS meds (
    t_ms INTEGER, state TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT DEFAULT (datetime('now')),
    source TEXT, fp_rate REAL, tp_rate REAL, active INTEGER
);
CREATE TABLE IF NOT EXISTS bandit_state (
    arm TEXT PRIMARY KEY, alpha REAL, beta REAL
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
