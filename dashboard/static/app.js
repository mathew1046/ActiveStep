/* ActiveStep dashboard — live telemetry UI */

const $ = (id) => document.getElementById(id);

const PFOG_THRESHOLD = 0.7;
const MAX_POINTS = 240;
const MODALITY_BITS = { 1: "vibration", 2: "laser", 4: "audio" };

/* ---------------- helpers ---------------- */

function decodeModality(mask) {
  if (mask === undefined || mask === null) return "--";
  const parts = Object.keys(MODALITY_BITS)
    .filter((b) => mask & Number(b))
    .map((b) => MODALITY_BITS[b]);
  return parts.length ? parts.join(" + ") : `mask ${mask}`;
}

let toastTimer;
function toast(msg) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2600);
}

/* ---------------- event log ---------------- */

const log = $("log");
const badgeFor = {
  cue_start: ["cue", "badge-cue"],
  cue_stop: ["stop", "badge-stop"],
  label: ["label", "badge-label"],
};

function addLog(msg, kind = "info", time = new Date()) {
  const stamp = time instanceof Date ? time.toLocaleTimeString() : String(time);
  const row = document.createElement("div");
  row.className = "log-row";
  const [badgeText, badgeCls] = badgeFor[kind] || ["info", "badge-info"];
  row.innerHTML =
    `<span class="log-time">${stamp}</span>` +
    `<span class="badge ${badgeCls}">${badgeText}</span>` +
    `<span class="log-msg"></span>`;
  row.querySelector(".log-msg").textContent = msg;
  log.prepend(row);
  while (log.children.length > 60) log.lastChild.remove();
}

/* Seed the log from history so a refresh isn't empty. */
fetch("/api/events?limit=15")
  .then((r) => r.json())
  .then((rows) => {
    for (const ev of rows.slice().reverse()) {
      // t_ms may be epoch ms or device uptime; only render real dates.
      const ms = Number(ev.t_start);
      const t = ms > 946684800000 ? new Date(ms) : `t+${(ms / 1000).toFixed(1)}s`;
      const rec = ev.recovery_ms ? ` · recovered ${(ev.recovery_ms / 1000).toFixed(1)}s` : "";
      addLog(
        `cue_start · ${decodeModality(ev.modality_mask)} · peak ${Number(ev.pfog_peak).toFixed(2)}${rec}`,
        "cue_start",
        t
      );
    }
  })
  .catch(() => {});

/* ---------------- gauge ---------------- */

const gaugeArc = $("gaugeArc");
const pfogEl = $("pfog");

function setGauge(v) {
  const clamped = Math.min(Math.max(v, 0), 1);
  gaugeArc.style.strokeDasharray = `${clamped * 100} 100`;
  pfogEl.textContent = v === undefined || v === null ? "--" : v.toFixed(3);
  pfogEl.classList.toggle("hot", clamped >= PFOG_THRESHOLD);
}

/* ---------------- status pill ---------------- */

const statusEl = $("status");
const statusText = $("statusText");
const KNOWN_STATES = ["idle", "walking", "cueing", "recovering", "standby"];

function setStatus(raw) {
  const s = String(raw || "standby").toLowerCase();
  const cls = KNOWN_STATES.includes(s) ? s : "standby";
  statusEl.className = `status status-${cls}`;
  statusText.textContent = s;
}

/* ---------------- chart ---------------- */

Chart.defaults.color = "#8b98ad";
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;

const chartCtx = $("pfogChart").getContext("2d");
const fill = chartCtx.createLinearGradient(0, 0, 0, 300);
fill.addColorStop(0, "rgba(45, 212, 191, 0.30)");
fill.addColorStop(1, "rgba(45, 212, 191, 0.00)");

const chart = new Chart(chartCtx, {
  type: "line",
  data: {
    labels: Array(MAX_POINTS).fill(""),
    datasets: [
      {
        label: "pFOG",
        data: [],
        borderColor: "#2dd4bf",
        backgroundColor: fill,
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.35,
        yAxisID: "y",
      },
      {
        label: "freeze index",
        data: [],
        borderColor: "rgba(251, 191, 36, 0.75)",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0.35,
        yAxisID: "y1",
      },
      {
        label: "threshold",
        data: Array(MAX_POINTS).fill(PFOG_THRESHOLD),
        borderColor: "rgba(248, 113, 113, 0.55)",
        borderWidth: 1,
        borderDash: [6, 6],
        pointRadius: 0,
        fill: false,
        yAxisID: "y",
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { display: false },
      y: {
        min: 0,
        max: 1,
        grid: { color: "rgba(148, 163, 184, 0.08)" },
        ticks: { stepSize: 0.25 },
        title: { display: true, text: "pFOG", color: "#8b98ad", font: { size: 10 } },
      },
      y1: {
        min: 0,
        max: 8,
        position: "right",
        grid: { drawOnChartArea: false },
        title: { display: true, text: "FI", color: "#8b98ad", font: { size: 10 } },
      },
    },
    plugins: {
      legend: {
        display: true,
        labels: {
          usePointStyle: true,
          pointStyle: "line",
          boxWidth: 8,
          boxHeight: 8,
          filter: (item) => item.text !== "threshold",
        },
      },
      tooltip: {
        backgroundColor: "#131b29",
        borderColor: "rgba(148,163,184,0.2)",
        borderWidth: 1,
        callbacks: { label: (c) => `${c.dataset.label}: ${Number(c.parsed.y).toFixed(3)}` },
      },
    },
  },
});

function pushChart(pfog, fi) {
  const ds = chart.data.datasets;
  ds[0].data.push(pfog);
  ds[1].data.push(fi ?? null);
  for (const d of [ds[0], ds[1]]) {
    if (d.data.length > MAX_POINTS) d.data.shift();
  }
  chart.update();
}

/* ---------------- bandit preference ---------------- */

function renderBandit(pref) {
  const box = $("bandit");
  const entries = Object.entries(pref || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    box.innerHTML = '<span class="empty">no data yet…</span>';
    return;
  }
  box.innerHTML = "";
  for (const [name, mean] of entries) {
    const row = document.createElement("div");
    row.className = "bandit-row";
    row.innerHTML =
      `<span class="bandit-name"></span>` +
      `<div class="bandit-track"><div class="bandit-fill"></div></div>` +
      `<span class="bandit-pct"></span>`;
    row.querySelector(".bandit-name").textContent = name;
    row.querySelector(".bandit-fill").style.width = `${Math.round(mean * 100)}%`;
    row.querySelector(".bandit-pct").textContent = `${Math.round(mean * 100)}%`;
    box.appendChild(row);
  }
}

function pollBandit() {
  fetch("/api/bandit")
    .then((r) => r.json())
    .then(renderBandit)
    .catch(() => {});
}
pollBandit();
setInterval(pollBandit, 15000);

/* ---------------- meds ---------------- */

function postMed(state) {
  fetch(`/api/meds?state=${state}`, { method: "POST" })
    .then(() => {
      toast(`Medication state "${state}" logged`);
      addLog(`meds marked ${state}`, "info");
      $("medOn").classList.toggle("active-on", state === "ON");
      $("medOff").classList.toggle("active-off", state === "OFF");
    })
    .catch(() => toast("Failed to log med state"));
}
$("medOn").addEventListener("click", () => postMed("ON"));
$("medOff").addEventListener("click", () => postMed("OFF"));

async function sendCueCommand(command) {
  const start = $("cueStart");
  const stop = $("cueStop");
  start.disabled = true;
  stop.disabled = true;
  try {
    const response = await fetch(`/api/node2/${command}`, { method: "POST" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Command failed");
    const started = command === "vibrate";
    toast(started ? "Node 1 vibration started" : "Node 1 vibration stopped");
    addLog(started ? "manual vibration started" : "manual vibration stopped", started ? "cue_start" : "cue_stop");
    if (started) $("cue").textContent = "vibration";
  } catch (err) {
    toast(err.message);
  } finally {
    start.disabled = false;
    stop.disabled = false;
  }
}

$("cueStart").addEventListener("click", () => sendCueCommand("vibrate"));
$("cueStop").addEventListener("click", () => sendCueCommand("stop"));

/* ---------------- websocket ---------------- */

const connEl = $("conn");
const connText = $("connText");
let wsDelay = 1000;

function setConn(state, text) {
  connEl.className = `conn conn-${state}`;
  connText.textContent = text;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);

  ws.onopen = () => {
    wsDelay = 1000;
    setConn("live", "live");
  };

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);

    if (m.pfog !== undefined) {
      setGauge(m.pfog);
      pushChart(m.pfog, m.fi);
    }
    if (m.cadence !== undefined) $("cadence").textContent = m.cadence.toFixed(1);
    if (m.fi !== undefined) $("fi").textContent = m.fi.toFixed(2);
    if (m.asymmetry !== undefined) $("asymmetry").textContent = (m.asymmetry * 100).toFixed(1);
    const state = m.state || m.status;
    if (state) setStatus(state);
    if (m.node2_connected === true) setConn("live", "node 2 live");
    if (m.node2_connected === false) setConn("retry", "node 2 offline");
    if (m.demo_metrics === true) $("demoBadge").hidden = false;

    if (m.event) {
      const e = m.event;
      const mod = decodeModality(e.modality);
      if (e.type === "cue_start") {
        $("cue").textContent = mod;
        setStatus("cueing");
      } else if (e.type === "cue_stop") {
        if (m.recovery_ms !== undefined && m.recovery_ms !== null) {
          $("recovery").textContent = `${(m.recovery_ms / 1000).toFixed(1)} s`;
        }
        setStatus("recovering");
        pollBandit();
      }
      const extra =
        e.type === "label" ? ` · ${e.label || "?"}` : e.modality !== undefined ? ` · ${mod}` : "";
      const cause = e.cause ? ` · ${e.cause}` : "";
      const pf = m.pfog !== undefined ? ` · pFOG ${m.pfog.toFixed(3)}` : "";
      addLog(`${e.type}${cause}${extra}${pf}`, e.type);
    }
  };

  ws.onclose = () => {
    setConn("retry", "reconnecting");
    setStatus("standby");
    setTimeout(connect, wsDelay);
    wsDelay = Math.min(wsDelay * 1.5, 8000);
  };

  ws.onerror = () => ws.close();
}

setConn("retry", "connecting");
connect();
