function clamp(n, min, max){
  return Math.max(min, Math.min(max, n));
}

function $(id){ return document.getElementById(id); }

const UI = window.FL_UI || {
  MAIN_SERVER_URL: "https://fltalk-system.onrender.com",
  EXPERIMENT_ID: "exp1",
  POLL_STATUS_MS: 1000,
  POLL_LOGS_MS: 700
};

const clientsInput = $("clientsInput");
const clientsList = $("clientsList");
const logsBox = $("logsBox");
const clearLogsBtn = $("clearLogsBtn");
const pauseLogsBtn = $("pauseLogsBtn");
const metricsBox = document.getElementById("metricsBox");
let lastMetricId = 0;
const metricStore = []; // keep received items

let lastLogId = 0;
let logsPaused = false;


function renderMetrics(){
  if (!metricsBox) return;

  // simple table
  let html = `
    <div style="overflow:auto;">
      <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
        <thead>
          <tr>
            <th style="text-align:left; padding:8px; border-bottom:1px solid rgba(15,23,42,0.10);">Round</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid rgba(15,23,42,0.10);">Client</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid rgba(15,23,42,0.10);">Accuracy</th>
            <th style="text-align:left; padding:8px; border-bottom:1px solid rgba(15,23,42,0.10);">Loss</th>
          </tr>
        </thead>
        <tbody>
  `;

  // show latest first
  const rows = [...metricStore].sort((a,b)=> (b.round - a.round) || (String(a.node_id).localeCompare(String(b.node_id))));
  for (const it of rows){
    const acc = it.metrics?.accuracy;
    const loss = it.metrics?.loss;
    html += `
      <tr>
        <td style="padding:8px; border-bottom:1px solid rgba(15,23,42,0.06);">${it.round ?? ""}</td>
        <td style="padding:8px; border-bottom:1px solid rgba(15,23,42,0.06);">${it.node_id ?? ""}</td>
        <td style="padding:8px; border-bottom:1px solid rgba(15,23,42,0.06);">${acc != null ? Number(acc).toFixed(4) : ""}</td>
        <td style="padding:8px; border-bottom:1px solid rgba(15,23,42,0.06);">${loss != null ? Number(loss).toFixed(4) : ""}</td>
      </tr>
    `;
  }

  html += `</tbody></table></div>`;
  metricsBox.innerHTML = html;
}

function ensureStatusPill(el){
  let pill = el.querySelector(".status-pill");
  if (!pill){
    pill = document.createElement("span");
    pill.className = "status-pill";
    pill.textContent = "offline";
    el.appendChild(pill);
  }
  return pill;
}

function setBoxState(el, online, statusText){
  // online/offline class
  el.classList.toggle("is-online", !!online);
  el.classList.toggle("is-offline", !online);

  // remove previous state classes
  el.classList.remove("state-training", "state-waiting", "state-complete", "state-error");

  const s = String(statusText || "").toLowerCase();

  // normalize to broad states
  if (!online || s === "offline") {
    // nothing
  } else if (s.includes("train")) {
    el.classList.add("state-training");
  } else if (s.includes("wait")) {
    el.classList.add("state-waiting");
  } else if (s.includes("complete") || s.includes("done")) {
    el.classList.add("state-complete");
  } else if (s.includes("error") || s.includes("fail")) {
    el.classList.add("state-error");
  } else {
    // default: treat as online but neutral (no extra class)
  }

  // pill text
  const pill = ensureStatusPill(el);
  pill.textContent = online ? (statusText || "online") : "offline";
}

function renderClients(count){
  clientsList.innerHTML = "";

  for (let i = 1; i <= count; i++){
    const div = document.createElement("div");
    div.className = "client-box is-offline";
    div.dataset.nodeType = "client";
    div.dataset.nodeId = `client_${i}`; // aligned with Python
    div.textContent = div.dataset.nodeId;

    ensureStatusPill(div);
    clientsList.appendChild(div);
  }
}

function getServerBox(){
  return document.querySelector('.server-box[data-node-id="fed_server"]');
}

function collectAllNodeElements(){
  const nodes = {};
  document.querySelectorAll("[data-node-id]").forEach((el) => {
    nodes[el.dataset.nodeId] = el;
  });
  return nodes;
}

/* ---------- Logs UI ---------- */

function appendLogLine(item){
  if (!logsBox) return;

  const node = item.node_id || "unknown";
  const type = item.node_type || "";
  const msg  = item.msg || "";
  const ts   = item.ts ? new Date(item.ts * 1000).toLocaleTimeString() : "";

  const line = document.createElement("div");
  line.className = "log-line";

  const meta = document.createElement("span");
  meta.className = "log-meta";
  meta.textContent = `[${ts}] ${type}:${node}`;

  const m = document.createElement("span");
  m.className = "log-msg";
  m.textContent = ` ${msg}`;

  line.appendChild(meta);
  line.appendChild(m);
  logsBox.appendChild(line);

  // autoscroll to bottom
  logsBox.scrollTop = logsBox.scrollHeight;
}

/* ---------- Polling ---------- */


async function pollMetrics(){
  try{
    const url = `${UI.MAIN_SERVER_URL}/ui/metrics?experiment_id=${encodeURIComponent(UI.EXPERIMENT_ID)}&since=${encodeURIComponent(lastMetricId)}&limit=500`;
    const data = await fetchJson(url);
    const items = data.items || [];

    if (items.length){
      metricStore.push(...items);
      if (typeof data.next === "number") lastMetricId = data.next;
      renderMetrics();
    }
  }catch(e){
    // ignore transient failures
  }
}


async function fetchJson(url){
  const res = await fetch(url, { method: "GET" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

async function pollStatus(){
  try {
    const url = `${UI.MAIN_SERVER_URL}/ui/status?experiment_id=${encodeURIComponent(UI.EXPERIMENT_ID)}`;
    const data = await fetchJson(url);

    const nodes = data.nodes || {};
    const elMap = collectAllNodeElements();

    // Update each rendered box
    Object.keys(elMap).forEach((nodeId) => {
      const el = elMap[nodeId];
      const info = nodes[nodeId];

      if (!info){
        // if backend has no record yet, keep offline
        setBoxState(el, false, "offline");
        return;
      }

      setBoxState(el, info.online, info.status);
    });

  } catch (e) {
    // If backend not reachable, mark everything offline (safe)
    const elMap = collectAllNodeElements();
    Object.keys(elMap).forEach((nodeId) => setBoxState(elMap[nodeId], false, "offline"));
  }
}

async function pollLogs(){
  if (logsPaused) return;

  try {
    const url =
      `${UI.MAIN_SERVER_URL}/ui/logs?experiment_id=${encodeURIComponent(UI.EXPERIMENT_ID)}&since=${encodeURIComponent(lastLogId)}&limit=200`;
    const data = await fetchJson(url);

    const logs = data.logs || [];
    for (const item of logs){
      appendLogLine(item);
    }

    if (typeof data.next === "number") {
      lastLogId = data.next;
    } else if (logs.length){
      lastLogId = logs[logs.length - 1].id;
    }
  } catch (e) {
    // ignore transient failures
  }
}


async function postJson(url, body){
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

/* ---------- Controls ---------- */

if (clearLogsBtn && logsBox){
  clearLogsBtn.addEventListener("click", async () => {
    // stop logs while resetting (avoid race with pollLogs)
    const prevPaused = logsPaused;
    logsPaused = true;

    // clear UI immediately
    logsBox.innerHTML = "";
    lastLogId = 0;


    metricStore.length = 0;
    lastMetricId = 0;
    if (metricsBox) metricsBox.innerHTML = "";


    try{
      // clear backend buffers too (so old logs never reappear)
      await postJson(`${UI.MAIN_SERVER_URL}/ui/reset`, {
        experiment_id: UI.EXPERIMENT_ID
      });
    } catch(e){
      console.warn("Backend reset failed. Old logs may reappear.", e);
    } finally {
      logsPaused = prevPaused; // resume previous state
    }
  });
}


if (pauseLogsBtn){
  pauseLogsBtn.addEventListener("click", () => {
    logsPaused = !logsPaused;
    pauseLogsBtn.textContent = logsPaused ? "Resume" : "Pause";
  });
}

/* ---------- Init ---------- */

const initial = Number(clientsInput?.value) || 1;
renderClients(initial);

// Ensure server box has pill too
const srv = getServerBox();
if (srv) {
  srv.classList.add("is-offline");
  ensureStatusPill(srv);
}

clientsInput?.addEventListener("input", () => {
  const raw = Number(clientsInput.value);
  const safe = clamp(isNaN(raw) ? 1 : raw, 1, 50);
  clientsInput.value = safe;
  renderClients(safe);

  // after re-render, next status poll will paint states
});

// Start polling loops
setInterval(pollStatus, UI.POLL_STATUS_MS);
setInterval(pollLogs, UI.POLL_LOGS_MS);
setInterval(pollMetrics, 1000);
pollMetrics();


// First immediate refresh
pollStatus();
pollLogs();
