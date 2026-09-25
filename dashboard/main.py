"""FastAPI + WebSocket dashboard for the UNO Q."""

from __future__ import annotations

import asyncio
import csv
import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from activestep.config import NODE2_HTTP_URL
from unoq.db import connect

DB_PATH = Path(__file__).resolve().parent.parent / "activestep.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connect()
    yield


app = FastAPI(title="ActiveStep Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


try:
    from unoq.ingest import STATE as LIVE_STATE
except ImportError:
    LIVE_STATE = None


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/events")
async def get_events(limit: int = 50):
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM events ORDER BY t_start DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/labels")
async def get_labels():
    conn = connect()
    rows = conn.execute(
        """SELECT l.*, e.t_start FROM labels l
           JOIN events e ON e.id = l.event_id
           ORDER BY t_ms DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/bandit")
async def bandit_preference():
    try:
        from unoq.bandit import CueBandit
        return CueBandit().preference()
    except Exception:
        return {}


@app.post("/api/label/{event_id}")
async def post_label(event_id: int, label: str):
    conn = connect()
    conn.execute(
        "INSERT INTO labels (event_id, label, t_ms, source) VALUES (?, ?, strftime('%s','now')*1000, 'dashboard')",
        (event_id, label),
    )
    conn.commit()
    return {"ok": True}


@app.post("/api/meds")
async def post_med(state: str, note: str = ""):
    conn = connect()
    conn.execute(
        "INSERT INTO meds (t_ms, state, note) VALUES (strftime('%s','now')*1000, ?, ?)",
        (state, note),
    )
    conn.commit()
    return {"ok": True}


@app.post("/api/node2/{command}")
async def node2_command(command: str):
    if command not in {"vibrate", "stop"}:
        raise HTTPException(status_code=404, detail="Unknown Node 2 command")
    if not NODE2_HTTP_URL:
        raise HTTPException(status_code=503, detail="ACTIVESTEP_NODE2_URL is not configured")

    def send():
        with urlopen(f"{NODE2_HTTP_URL}/{command}", timeout=2.0) as response:
            return response.read().decode("utf-8")

    try:
        message = await asyncio.to_thread(send)
    except OSError as exc:
        raise HTTPException(status_code=502, detail=f"Node 2 is unreachable: {exc}") from exc
    return {"ok": True, "message": message}


@app.get("/api/export/csv")
async def export_csv():
    out_path = Path("/tmp/activestep_events.csv")
    conn = connect()
    rows = conn.execute("SELECT * FROM events").fetchall()
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "t_start", "t_end", "pfog_peak", "fi_peak", "modality", "recovery_ms", "model_version", "context"])
        writer.writerows(rows)
    return FileResponse(out_path, filename="activestep_events.csv")


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        import asyncio

        if LIVE_STATE is not None:
            q = asyncio.Queue(maxsize=100)
            LIVE_STATE.subscribers.append(q)
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    msg = {**LIVE_STATE.latest, "status": "standby"}
                await websocket.send_json(msg)
        else:
            # Heartbeat fallback when dashboard is run standalone.
            while True:
                await websocket.send_json(
                    {
                        "pfog": 0.0,
                        "cadence": 0.0,
                        "status": "standby",
                        "fi": 0.0,
                        "asymmetry": 0.0,
                    }
                )
                await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
