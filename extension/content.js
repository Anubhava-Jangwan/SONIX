/*
 * On-page side panel for meet.google.com.
 *
 * A half-transparent column pinned to the right of the Meet window, showing
 * the live risk band, the score history as a graph, and a model picker -- so
 * nobody has to watch the extension popup during a call.
 *
 * The graph is the same reading as the Streamlit live-mic timeline: one point
 * per scored 4 s window, the amber/red thresholds drawn as dotted rules, and
 * the line coloured by the band it is currently in. It is a <canvas> rather
 * than a chart library because a content script shares the page's origin and
 * CSP -- pulling a library in here is a needless way to get blocked by a
 * site's script-src.
 *
 * Changing the model does NOT restart the call. The picker sends a message
 * that ends up as a set_model frame on the existing socket, so the call_id,
 * the consent grant and this graph all survive the swap. Points scored before
 * the swap keep the head that actually scored them.
 */

const MAX_POINTS = 240;          // ~2 minutes at one window per 0.5 s

const COLOUR = {
  green: "#39c26a",
  amber: "#e0a92b",
  red: "#f0685f",
  idle: "#7e8d8d",
  accent: "#3fc2ce",
  grid: "rgba(255,255,255,.10)",
  ink3: "#7e8d8d",
};

const AMBER_AT = 0.35;
const RED_AT = 0.65;

let panel = null;
let canvas = null;
let lastModels = "";             // serialised, to avoid rebuilding the <select>

/* ------------------------------------------------------------------ */
/* Band / verdict                                                      */
/* ------------------------------------------------------------------ */

function bandOf(score) {
  if (score === null || score === undefined) return "idle";
  if (score >= RED_AT) return "red";
  if (score >= AMBER_AT) return "amber";
  return "green";
}

/* The call the panel actually makes. "Cloned" and "Authentic" are the words
   the operator needs; the percentage behind them is kept next to the word so
   the verdict is never colour-alone and never a bare number. */
function verdictOf(score, scoringAvailable) {
  if (!scoringAvailable) {
    return ["Monitoring", COLOUR.accent, "No trained head loaded — not scoring."];
  }
  if (score === null || score === undefined) {
    return ["Listening", COLOUR.idle, "Waiting for the first scored window."];
  }
  const band = bandOf(score);
  if (band === "red") {
    return ["Cloned", COLOUR.red,
            "Reads as synthetic. Verify on a number you already have."];
  }
  if (band === "amber") {
    return ["Uncertain", COLOUR.amber,
            "Neither clean nor clearly synthetic. Treat with caution."];
  }
  return ["Authentic", COLOUR.green, "Consistent with a real human voice."];
}

/* ------------------------------------------------------------------ */
/* Panel construction                                                  */
/* ------------------------------------------------------------------ */

function ensurePanel() {
  if (panel && document.body.contains(panel)) return panel;

  panel = document.createElement("div");
  panel.id = "sonix-panel";

  // Built with DOM calls, not innerHTML with interpolated state: model labels
  // come from a checkpoint's own config and must never become markup.
  const head = document.createElement("div");
  head.className = "sonix-head";
  const dot = document.createElement("span");
  dot.className = "sonix-dot";
  const name = document.createElement("span");
  name.className = "sonix-name sonix-hide-collapsed";
  name.textContent = "SONIX";
  const toggle = document.createElement("button");
  toggle.className = "sonix-toggle";
  toggle.textContent = "›";
  toggle.title = "Collapse panel";
  toggle.addEventListener("click", () => {
    const collapsed = panel.classList.toggle("sonix-collapsed");
    toggle.textContent = collapsed ? "‹" : "›";
    toggle.title = collapsed ? "Expand panel" : "Collapse panel";
    if (!collapsed) draw(lastState);
  });
  head.append(dot, name, toggle);

  const verdict = document.createElement("div");
  verdict.className = "sonix-verdict sonix-hide-collapsed";
  const word = document.createElement("div");
  word.className = "sonix-word";
  word.textContent = "—";
  const pct = document.createElement("div");
  pct.className = "sonix-pct";
  const sub = document.createElement("div");
  sub.className = "sonix-sub";
  sub.textContent = "Starting up…";
  verdict.append(word, pct, sub);

  const chartwrap = document.createElement("div");
  chartwrap.className = "sonix-chartwrap sonix-hide-collapsed";
  canvas = document.createElement("canvas");
  chartwrap.appendChild(canvas);
  const axis = document.createElement("div");
  axis.className = "sonix-axis";
  const axL = document.createElement("span");
  axL.textContent = "older";
  const axR = document.createElement("span");
  axR.textContent = "now";
  axis.append(axL, axR);
  chartwrap.appendChild(axis);

  const ctrl = document.createElement("div");
  ctrl.className = "sonix-hide-collapsed";
  const label = document.createElement("div");
  label.className = "sonix-label";
  label.textContent = "Detection head";
  const select = document.createElement("select");
  select.id = "sonix-model";
  select.addEventListener("change", () => {
    chrome.runtime.sendMessage({ type: "overlay:setModel", model: select.value });
  });
  ctrl.append(label, select);

  const err = document.createElement("div");
  err.className = "sonix-err sonix-hide-collapsed";
  err.id = "sonix-err";
  err.hidden = true;

  const meta = document.createElement("div");
  meta.className = "sonix-meta sonix-hide-collapsed";
  meta.id = "sonix-meta";

  panel.append(head, verdict, chartwrap, ctrl, err, meta);
  document.body.appendChild(panel);
  return panel;
}

function removePanel() {
  if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
  panel = null;
  canvas = null;
  lastModels = "";
}

/* ------------------------------------------------------------------ */
/* The graph                                                           */
/* ------------------------------------------------------------------ */

function drawChart(scores) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // Size the backing store to the device pixel ratio, or the line is blurry
  // on every laptop screen made in the last decade.
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 132;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const y = (v) => h - v * h;             // score 0..1 -> pixels, 0 at bottom

  // Threshold bands, drawn behind everything.
  ctx.fillStyle = "rgba(240,104,95,.09)";
  ctx.fillRect(0, 0, w, y(RED_AT));
  ctx.fillStyle = "rgba(224,169,43,.07)";
  ctx.fillRect(0, y(RED_AT), w, y(AMBER_AT) - y(RED_AT));

  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  for (const [v, c] of [[AMBER_AT, COLOUR.amber], [RED_AT, COLOUR.red]]) {
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.moveTo(0, y(v));
    ctx.lineTo(w, y(v));
    ctx.stroke();
  }
  ctx.setLineDash([]);

  if (!scores || scores.length === 0) {
    ctx.fillStyle = COLOUR.ink3;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("waiting for the first window…", 8, h / 2);
    return;
  }

  // x maps the most recent MAX_POINTS across the full width, so the line
  // grows from the left and then scrolls rather than rescaling every tick.
  const n = Math.max(scores.length, 2);
  const x = (i) => (i / (MAX_POINTS - 1)) * w;

  // Filled area under the line, tinted by the CURRENT band.
  const band = bandOf(scores[scores.length - 1]);
  const line = COLOUR[band] || COLOUR.idle;

  ctx.beginPath();
  ctx.moveTo(x(0), y(scores[0]));
  for (let i = 1; i < scores.length; i++) ctx.lineTo(x(i), y(scores[i]));
  ctx.lineTo(x(scores.length - 1), h);
  ctx.lineTo(x(0), h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, line + "44");
  grad.addColorStop(1, line + "05");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(x(0), y(scores[0]));
  for (let i = 1; i < scores.length; i++) ctx.lineTo(x(i), y(scores[i]));
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  // Head of the series, so the eye lands on "now".
  const lx = x(scores.length - 1);
  const ly = y(scores[scores.length - 1]);
  ctx.fillStyle = line;
  ctx.beginPath();
  ctx.arc(lx, ly, 3, 0, Math.PI * 2);
  ctx.fill();
  void n;
}

/* ------------------------------------------------------------------ */
/* Render                                                              */
/* ------------------------------------------------------------------ */

let lastState = null;

function syncModels(state) {
  const select = panel.querySelector("#sonix-model");
  if (!select) return;
  const models = state.models || [];
  const sig = JSON.stringify([models.map((m) => m.key), state.model]);
  if (sig === lastModels) return;          // rebuilding steals focus mid-click
  lastModels = sig;

  select.textContent = "";
  if (!models.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Server default";
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.key;
    opt.textContent = m.label;
    if (m.key === state.model) opt.selected = true;
    select.appendChild(opt);
  }
}

function draw(s) {
  if (!s || !panel) return;
  lastState = s;

  const [wordText, colour, subText] = verdictOf(s.lastScore, s.scoringAvailable);

  panel.querySelector(".sonix-dot").style.background = colour;
  const verdict = panel.querySelector(".sonix-verdict");
  verdict.style.borderLeftColor = colour;

  const word = panel.querySelector(".sonix-word");
  const pct = panel.querySelector(".sonix-pct");
  const sub = panel.querySelector(".sonix-sub");

  if (s.callState === "consent_pending") {
    word.textContent = "Consent";
    word.style.color = COLOUR.amber;
    pct.textContent = `Code ${s.pairingCode || "------"}`;
    sub.textContent = "Approve in the SONIX dashboard to begin scoring.";
  } else {
    word.textContent = wordText;
    word.style.color = colour;
    pct.textContent =
      s.lastScore === null || s.lastScore === undefined
        ? ""
        : `P(AI voice) ${(s.lastScore * 100).toFixed(0)}%`;
    sub.textContent = subText;
  }

  drawChart(s.scores || []);
  syncModels(s);

  const err = panel.querySelector("#sonix-err");
  err.hidden = !s.error;
  err.textContent = s.error || "";

  const meta = panel.querySelector("#sonix-meta");
  meta.textContent = "";
  const rows = [
    ["windows scored", String(s.windows || 0)],
    ["call", s.callState ? s.callState.toUpperCase() : "—"],
  ];
  for (const [k, v] of rows) {
    const row = document.createElement("div");
    row.textContent = k + " ";
    const b = document.createElement("b");
    b.textContent = v;
    row.appendChild(b);
    meta.appendChild(row);
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "sonix:state") return;
  if (!msg.state.capturing) {
    removePanel();
    return;
  }
  ensurePanel();
  draw(msg.state);
});

// The panel is sized in vh/%, so a window resize needs a redraw of the canvas.
window.addEventListener("resize", () => {
  if (panel && lastState) drawChart(lastState.scores || []);
});
