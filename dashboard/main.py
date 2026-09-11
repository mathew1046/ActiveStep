"""ActiveStep public dashboard and ESP telemetry bridge."""
from __future__ import annotations

import asyncio
import csv
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

try:
    from unoq.db import connect
except Exception:  # standalone cloud deployment
    connect = None

DB_PATH = Path(__file__).resolve().parent.parent / "activestep.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"
TELEMETRY_TOKEN = os.getenv("ACTIVESTEP_TELEMETRY_TOKEN", "")


class LiveState:
    def __init__(self):
        self.latest: dict = {"status": "waiting", "sensors": {}}
        self.history = deque(maxlen=300)
        self.subscribers: list[asyncio.Queue] = []

    def publish(self, msg: dict):
        sensor = str(msg.get("sensor") or msg.get("unit") or "shank").lower()
        now = int(time.time() * 1000)
        normalized = {
            "sensor": sensor,
            "ts": msg.get("ts", msg.get("t_ms", now)),
            "acc_mag": float(msg.get("acc_mag", msg.get("accMag", 0)) or 0),
            "gyro_mag": float(msg.get("gyro_mag", msg.get("gyroMag", 0)) or 0),
            "fi": float(msg.get("fi", 0) or 0),
            "pfog": float(msg.get("pfog", 0) or 0),
            "state": str(msg.get("state", "IDLE")),
            "event": msg.get("event"),
            "seq": msg.get("seq"),
        }
        self.latest.setdefault("sensors", {})[sensor] = normalized
        self.latest["status"] = "live"
        self.latest["last_seen"] = now
        self.history.append(normalized)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait({"type": "telemetry", **normalized, "sensors": self.latest["sensors"]})
            except asyncio.QueueFull:
                pass


STATE = LiveState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="ActiveStep Public Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    return {"ok": True, "service": "activestep-dashboard", "sensors": list(STATE.latest["sensors"])}


@app.get("/api/state")
async def state():
    return STATE.latest


@app.get("/api/history")
async def history(limit: int = 120):
    return list(STATE.history)[-max(1, min(limit, 300)):]


@app.post("/api/telemetry")
async def telemetry(payload: dict, x_activestep_token: str | None = Header(default=None)):
    if TELEMETRY_TOKEN and x_activestep_token != TELEMETRY_TOKEN:
        raise HTTPException(status_code=401, detail="invalid telemetry token")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    try:
        STATE.publish(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid telemetry: {exc}") from exc
    return {"ok": True, "sensor": STATE.latest["sensors"][str(payload.get("sensor") or payload.get("unit") or "shank").lower()]["sensor"]}


@app.get("/api/events")
async def get_events(limit: int = 50):
    if connect is None:
        return []
    conn = connect()
    rows = conn.execute("SELECT * FROM events ORDER BY t_start DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/export/csv")
async def export_csv():
    out_path = Path("/tmp/activestep_events.csv")
    if connect is None:
        out_path.write_text("sensor,ts,acc_mag,gyro_mag,fi,pfog,state\n")
    else:
        conn = connect()
        rows = conn.execute("SELECT * FROM events").fetchall()
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "t_start", "t_end", "pfog_peak", "fi_peak", "modality", "recovery_ms", "model_version", "context"])
            writer.writerows(rows)
    return FileResponse(out_path, filename="activestep_events.csv")


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    STATE.subscribers.append(queue)
    try:
        await websocket.send_json({"type": "snapshot", **STATE.latest})
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                message = {"type": "heartbeat", **STATE.latest}
            await websocket.send_json(message)
    except WebSocketDisconnect:
        pass
    finally:
        if queue in STATE.subscribers:
            STATE.subscribers.remove(queue)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
