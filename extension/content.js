/*
 * On-page side panel for meet.google.com.
 *
 * A half-transparent column pinned to the right of the Meet window: a live
 * voice-activity visualiser, the risk verdict, the score history as a graph,
 * the band thresholds, and a model picker.
 *
 * Why the visualiser exists. When nothing is being scored there are two very
 * different causes and, until now, one symptom. Either no audio is reaching
 * the extension at all, or audio is arriving but sits under the server's
 * silence gate so no window is ever scored. Both looked like "0 windows" and
 * a frozen graph. The visualiser is driven by an AnalyserNode on the same
 * audio graph that feeds the socket, so it answers that question instantly
 * and locally -- 20 Hz from the offscreen document, eased to 60 fps here.
 *
 * The graph is a <canvas> rather than a chart library because a content script
 * shares the page's origin and CSP, and meet.google.com's script-src is not
 * something we get to negotiate with.
 *
 * Changing the model does NOT restart the call, so the call_id, the consent
 * grant and the graph all survive a swap. Points keep the head that scored
 * them.
 */

const MAX_POINTS = 240;          // ~2 minutes at one window per 0.5 s
const THRESH_KEY = "sonix.thresholds";

const COLOUR = {
  green: "#34D399",
  amber: "#FBBF24",
  red: "#FB7185",
  idle: "#7A8299",
  accent: "#7C8CFF",
  accent2: "#22D3EE",
  ink3: "#7A8299",
};

const GLOW = {
  [COLOUR.green]: "rgba(52,211,153,.16)",
  [COLOUR.amber]: "rgba(251,191,36,.16)",
  [COLOUR.red]: "rgba(251,113,133,.18)",
  [COLOUR.idle]: "rgba(122,130,153,.10)",
  [COLOUR.accent]: "rgba(34,211,238,.12)",
};

/* Display bands. These move the PANEL's reading only -- they are not the
   server's pinned production thresholds in realtime/thresholds.py, and
   nothing here is written back to it. */
let AMBER_AT = 0.35;
let RED_AT = 0.65;

/* Anything under this reads as silence. Matches the server's default energy
   floor so the widget and the gate agree about what counts as audio. */
const SILENCE_RMS = 0.003;

let panel = null;
let canvas = null;          // score history
let orb = null;             // voice activity
let lastModels = "";
let lastState = null;

/* ------------------------------------------------------------------ */
/* Participants, and who is actually speaking                          */
/* ------------------------------------------------------------------ */

/* WHAT THIS CAN AND CANNOT DO.
 *
 * Tab capture gives us ONE mixed stream: every participant summed together.
 * SONIX does no diarisation, so a score belongs to a 4-second window, not to
 * a person. Printing a verdict beside a name on that basis would be inventing
 * a result.
 *
 * What is real is Meet's own active-speaker signal. When Meet says exactly one
 * participant is speaking, the audio in that window is that participant's
 * voice, and the score can be attributed to them honestly. When two or more
 * are speaking, or when we cannot read the signal at all, the window is
 * recorded as unattributed and shown that way.
 *
 * Meet's DOM is not a public API and its class names change without notice, so
 * every selector below is a guess with fallbacks, and failure is reported in
 * the UI rather than silently producing a confident-looking wrong answer.
 */

const participants = new Map();   // id -> {id, name, speaking, scores:[], last}
let selectedId = null;            // the target the operator picked
let domReadable = false;          // did we find any participant tiles at all?
let speakerReadable = false;      // could we read a speaking indicator?

function readParticipants() {
  const tiles = document.querySelectorAll("[data-participant-id]");
  if (!tiles.length) {
    domReadable = false;
    return;
  }
  domReadable = true;

  const seen = new Set();
  let sawSpeakingAttr = false;

  for (const tile of tiles) {
    const id = tile.getAttribute("data-participant-id");
    if (!id) continue;
    seen.add(id);

    // Name: Meet puts it in a few different places depending on layout.
    let name = "";
    const nameEl =
      tile.querySelector("[data-self-name]") ||
      tile.querySelector(".zWGUib") ||
      tile.querySelector("[data-tooltip]");
    if (nameEl) {
      name = (nameEl.getAttribute("data-self-name") ||
              nameEl.textContent || "").trim();
    }
    if (!name) name = (tile.getAttribute("data-tooltip") || "").trim();
    if (!name) name = tile.textContent.trim().split("\n")[0] || "Participant";
    name = name.slice(0, 40);

    // Speaking: Meet marks the tile while a participant has the floor. Several
    // candidate signals, because none of them is guaranteed to survive a
    // Meet update.
    let speaking = false;
    const sig =
      tile.querySelector(".IisKdb") ||          // animated speaking bars
      tile.querySelector("[class*='speaking']") ||
      tile.querySelector("[data-is-speaking]");
    if (sig) {
      sawSpeakingAttr = true;
      const attr = sig.getAttribute("data-is-speaking");
      speaking = attr !== null
        ? attr !== "false"
        : !!(sig.offsetParent !== null && sig.getClientRects().length);
    }

    const avatar = tile.querySelector("img");
    const prev = participants.get(id) || { scores: [], last: null };
    participants.set(id, {
      id, name, speaking,
      avatar: avatar ? avatar.src : null,
      scores: prev.scores,
      last: prev.last,
    });
  }

  speakerReadable = sawSpeakingAttr;
  for (const id of [...participants.keys()]) {
    if (!seen.has(id)) participants.delete(id);
  }
  if (selectedId && !participants.has(selectedId)) selectedId = null;
}

/* The single speaker to credit this window to, or null when the answer is
   genuinely ambiguous. Silence about who spoke is better than a guess. */
function soleSpeaker() {
  if (!speakerReadable) return null;
  const talking = [...participants.values()].filter((p) => p.speaking);
  return talking.length === 1 ? talking[0] : null;
}

/* Called once per scored window. Only credits a participant when Meet says
   exactly one person had the floor. */
function attribute(score) {
  const who = soleSpeaker();
  if (!who) return;
  who.scores.push(score);
  if (who.scores.length > 40) who.scores.shift();
  who.last = score;
}

/* Level state, eased every frame so 20 Hz input looks continuous. */
let targetLevel = 0;
let orbLevel = 0;
let lastLevelAt = 0;
let rafId = null;
let peopleTimer = null;

/* ------------------------------------------------------------------ */
/* Band / verdict                                                      */
/* ------------------------------------------------------------------ */

function bandOf(score) {
  if (score === null || score === undefined) return "idle";
  if (score >= RED_AT) return "red";
  if (score >= AMBER_AT) return "amber";
  return "green";
}

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

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function ensurePanel() {
  if (panel && document.body.contains(panel)) return panel;

  panel = el("div");
  panel.id = "sonix-panel";

  // Built with DOM calls, not innerHTML with interpolated state: model labels
  // come from a checkpoint's own config and must never become markup.
  const head = el("div", "sonix-head");
  const dot = el("span", "sonix-dot");
  const name = el("span", "sonix-name sonix-hide-collapsed", "SONIX");
  const toggle = el("button", "sonix-toggle", "›");
  toggle.title = "Collapse panel";
  toggle.addEventListener("click", () => {
    const collapsed = panel.classList.toggle("sonix-collapsed");
    toggle.textContent = collapsed ? "‹" : "›";
    toggle.title = collapsed ? "Expand panel" : "Collapse panel";
    if (!collapsed && lastState) draw(lastState);
  });
  head.append(dot, name, toggle);

  const verdict = el("div", "sonix-verdict sonix-hide-collapsed");
  verdict.append(el("div", "sonix-rail"),
                 el("div", "sonix-word", "—"),
                 el("div", "sonix-pct"),
                 el("div", "sonix-sub", "Starting up…"));

  // --- voice activity ---
  const voice = el("div", "sonix-voice sonix-hide-collapsed");
  orb = el("canvas");
  orb.className = "sonix-orb";
  const vstat = el("div", "sonix-voice-stat");
  vstat.append(el("span", "sonix-vdot"), el("span", "sonix-vtext", "no audio"));
  voice.append(orb, vstat);

  // --- participants ---
  const people = el("div", "sonix-people sonix-hide-collapsed");
  const phead = el("div", "sonix-people-head");
  phead.append(el("div", "sonix-label", "In this call"),
               el("span", "sonix-count", "—"));
  const plist = el("div", "sonix-plist");
  plist.id = "sonix-plist";
  const pnote = el("div", "sonix-pnote");
  pnote.id = "sonix-pnote";
  people.append(phead, plist, pnote);

  const chartwrap = el("div", "sonix-chartwrap sonix-hide-collapsed");
  canvas = el("canvas");
  chartwrap.appendChild(canvas);
  const axis = el("div", "sonix-axis");
  axis.append(el("span", null, "older"), el("span", null, "now"));
  chartwrap.appendChild(axis);

  // --- thresholds ---
  const th = el("div", "sonix-hide-collapsed");
  th.append(el("div", "sonix-label", "Band thresholds"));
  th.appendChild(makeSlider("amber", "Amber", () => AMBER_AT, (v) => {
    AMBER_AT = Math.min(v, RED_AT - 0.02);
  }));
  th.appendChild(makeSlider("red", "Red", () => RED_AT, (v) => {
    RED_AT = Math.max(v, AMBER_AT + 0.02);
  }));

  const ctrl = el("div", "sonix-hide-collapsed");
  ctrl.append(el("div", "sonix-label", "Detection head"));
  const select = el("select");
  select.id = "sonix-model";
  select.addEventListener("change", () => {
    chrome.runtime.sendMessage({ type: "overlay:setModel", model: select.value });
  });
  ctrl.appendChild(select);

  const btns = el("div", "sonix-btns sonix-hide-collapsed");
  const again = el("button", "sonix-btn", "New code");
  again.title = "End this call and start a new one to get a fresh pairing code";
  again.addEventListener("click", () =>
    chrome.runtime.sendMessage({ type: "overlay:restart" }));
  const stop = el("button", "sonix-btn sonix-btn-stop", "Stop");
  stop.title = "Stop monitoring and close the panel";
  stop.addEventListener("click", () =>
    chrome.runtime.sendMessage({ type: "overlay:stop" }));
  btns.append(again, stop);

  const err = el("div", "sonix-err sonix-hide-collapsed");
  err.id = "sonix-err";
  err.hidden = true;

  const meta = el("div", "sonix-meta sonix-hide-collapsed");
  meta.id = "sonix-meta";

  panel.append(head, verdict, voice, people, chartwrap, th, ctrl, btns, err, meta);
  document.body.appendChild(panel);

  startOrb();

  // Participants join, leave and take the floor between scored windows, which
  // arrive only every half second at best and not at all during silence. This
  // keeps the roster and the speaking pips honest in between.
  clearInterval(peopleTimer);
  peopleTimer = setInterval(() => {
    if (!panel || panel.classList.contains("sonix-collapsed")) return;
    readParticipants();
    renderPeople();
  }, 1000);

  return panel;
}

function makeSlider(key, label, get, set) {
  const row = el("div", "sonix-slider");
  const top = el("div", "sonix-slider-top");
  const name = el("span", null, label);
  const val = el("b", null, `${Math.round(get() * 100)}%`);
  top.append(name, val);

  const input = document.createElement("input");
  input.type = "range";
  input.min = "5";
  input.max = "95";
  input.step = "1";
  input.value = String(Math.round(get() * 100));
  input.className = `sonix-range sonix-range-${key}`;
  input.addEventListener("input", () => {
    set(Number(input.value) / 100);
    input.value = String(Math.round(get() * 100));
    val.textContent = `${Math.round(get() * 100)}%`;
    try {
      chrome.storage.local.set({ [THRESH_KEY]: { amber: AMBER_AT, red: RED_AT } });
    } catch { /* storage unavailable */ }
    if (lastState) draw(lastState);
  });

  row.append(top, input);
  return row;
}

function removePanel() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  clearInterval(peopleTimer);
  peopleTimer = null;
  participants.clear();
  selectedId = null;
  prevScoreCount = 0;
  if (panel && panel.parentNode) panel.parentNode.removeChild(panel);
  panel = null;
  canvas = null;
  orb = null;
  lastModels = "";
}

/* ------------------------------------------------------------------ */
/* Voice activity -- the Siri-style visualiser                         */
/* ------------------------------------------------------------------ */

function fitCanvas(c, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 300;
  const h = cssH || c.clientHeight || 100;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* rms -> 0..1, with a square-root curve so quiet speech is still visible.
   Tab audio from Meet is AGC'd and lands far below close-mic levels, so a
   linear mapping would leave the widget almost flat for normal speech. */
function levelOf(rms) {
  return Math.max(0, Math.min(1, Math.sqrt(rms / 0.06)));
}

function startOrb() {
  if (rafId) cancelAnimationFrame(rafId);

  const tick = (t) => {
    rafId = requestAnimationFrame(tick);
    if (!orb || !panel || panel.classList.contains("sonix-collapsed")) return;

    // If the offscreen document stops reporting, decay to silence rather
    // than freezing mid-wave -- a frozen wave reads as "still listening".
    if (t - lastLevelAt > 900) targetLevel = 0;

    // Asymmetric easing: rise fast so speech feels instant, fall slow so the
    // wave settles instead of stuttering between syllables.
    const k = targetLevel > orbLevel ? 0.32 : 0.09;
    orbLevel += (targetLevel - orbLevel) * k;

    drawOrb(t);
  };
  rafId = requestAnimationFrame(tick);
}

function drawOrb(t) {
  const { ctx, w, h } = fitCanvas(orb, 78);
  ctx.clearRect(0, 0, w, h);

  const mid = h / 2;
  const live = orbLevel > 0.04;
  // Always some motion, so the widget reads as alive even in silence.
  const idle = 0.055 + Math.sin(t * 0.0016) * 0.022;
  const amp = Math.max(idle, orbLevel);

  const grad = ctx.createLinearGradient(0, 0, w, 0);
  if (live) {
    grad.addColorStop(0, COLOUR.accent);
    grad.addColorStop(0.5, COLOUR.accent2);
    grad.addColorStop(1, COLOUR.accent);
  } else {
    grad.addColorStop(0, "rgba(122,130,153,.55)");
    grad.addColorStop(1, "rgba(122,130,153,.55)");
  }

  ctx.globalCompositeOperation = "lighter";

  const WAVES = 4;
  for (let i = 0; i < WAVES; i++) {
    const waveAmp = (mid - 7) * amp * (1 - i * 0.17);
    const freq = 1.25 + i * 0.6;
    const phase = t * (0.0014 + i * 0.00052) + i * 1.9;

    ctx.beginPath();
    for (let x = 0; x <= w; x += 2) {
      const p = x / w;
      // Taper both ends so the waveform floats rather than hitting the edges.
      const env = Math.pow(Math.sin(Math.PI * p), 1.7);
      const y = mid + Math.sin(p * Math.PI * 2 * freq + phase) * waveAmp * env;
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = grad;
    ctx.globalAlpha = 0.85 - i * 0.16;
    ctx.lineWidth = 2.3 - i * 0.38;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (live) { ctx.shadowColor = COLOUR.accent2; ctx.shadowBlur = 10; }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = "source-over";
}

/* ------------------------------------------------------------------ */
/* The score graph                                                     */
/* ------------------------------------------------------------------ */

function drawChart(scores) {
  if (!canvas) return;
  const { ctx, w, h } = fitCanvas(canvas, 146);
  ctx.clearRect(0, 0, w, h);

  const y = (v) => h - v * h;

  ctx.fillStyle = "rgba(251,113,133,.09)";
  ctx.fillRect(0, 0, w, y(RED_AT));
  ctx.fillStyle = "rgba(251,191,36,.07)";
  ctx.fillRect(0, y(RED_AT), w, y(AMBER_AT) - y(RED_AT));

  ctx.setLineDash([3, 4]);
  ctx.lineWidth = 1;
  for (const [v, c] of [[AMBER_AT, COLOUR.amber], [RED_AT, COLOUR.red]]) {
    ctx.strokeStyle = c;
    ctx.globalAlpha = 0.65;
    ctx.beginPath();
    ctx.moveTo(0, y(v));
    ctx.lineTo(w, y(v));
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  if (!scores || scores.length === 0) {
    // Says WHY it is empty. Only windows with audible speech are scored --
    // the silence gate drops the rest before the model sees them, so a quiet
    // call legitimately produces no points and "waiting" read like a fault.
    ctx.fillStyle = COLOUR.ink3;
    ctx.font = "500 11px Inter, system-ui, sans-serif";
    ctx.fillText("no speech scored yet", 10, h / 2 - 5);
    ctx.fillStyle = "rgba(122,130,153,.7)";
    ctx.font = "10px Inter, system-ui, sans-serif";
    ctx.fillText("silent windows are skipped", 10, h / 2 + 11);
    return;
  }

  const x = (i) => (i / (MAX_POINTS - 1)) * w;
  const line = COLOUR[bandOf(scores[scores.length - 1])] || COLOUR.idle;

  /* Smooth curve through the points, drawn as quadratic segments between
     midpoints. A polyline through per-window scores is visually noisy and
     invites reading each vertex as an event; the eye should follow the trend,
     and the exact values are on the axis. */
  const curve = () => {
    ctx.moveTo(x(0), y(scores[0]));
    if (scores.length < 3) {
      for (let i = 1; i < scores.length; i++) ctx.lineTo(x(i), y(scores[i]));
      return;
    }
    for (let i = 1; i < scores.length - 1; i++) {
      const mx = (x(i) + x(i + 1)) / 2;
      const my = (y(scores[i]) + y(scores[i + 1])) / 2;
      ctx.quadraticCurveTo(x(i), y(scores[i]), mx, my);
    }
    const n2 = scores.length - 1;
    ctx.quadraticCurveTo(x(n2), y(scores[n2]), x(n2), y(scores[n2]));
  };

  ctx.beginPath();
  curve();
  ctx.lineTo(x(scores.length - 1), h);
  ctx.lineTo(x(0), h);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, line + "59");
  grad.addColorStop(0.55, line + "1F");
  grad.addColorStop(1, line + "00");
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  curve();
  ctx.strokeStyle = line;
  ctx.lineWidth = 2.25;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowColor = line;
  ctx.shadowBlur = 9;
  ctx.stroke();
  ctx.shadowBlur = 0;

  const lx = x(scores.length - 1);
  const ly = y(scores[scores.length - 1]);
  ctx.fillStyle = line + "33";
  ctx.beginPath();
  ctx.arc(lx, ly, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = line;
  ctx.beginPath();
  ctx.arc(lx, ly, 3.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(10,11,16,.85)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

/* ------------------------------------------------------------------ */
/* Render                                                              */
/* ------------------------------------------------------------------ */

function syncModels(state) {
  const select = panel.querySelector("#sonix-model");
  if (!select) return;
  const models = state.models || [];
  const sig = JSON.stringify([models.map((m) => m.key), state.model]);
  if (sig === lastModels) return;          // rebuilding steals focus mid-click
  lastModels = sig;

  select.textContent = "";
  if (!models.length) {
    const opt = el("option", null, "Server default");
    opt.value = "";
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const m of models) {
    const opt = el("option", null, m.label);
    opt.value = m.key;
    if (m.key === state.model) opt.selected = true;
    select.appendChild(opt);
  }
}

function initials(name) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2)
             .map((w) => w[0].toUpperCase()).join("") || "?";
}

function renderPeople() {
  const list = panel.querySelector("#sonix-plist");
  const note = panel.querySelector("#sonix-pnote");
  const count = panel.querySelector(".sonix-count");
  if (!list) return;

  const people = [...participants.values()];
  count.textContent = people.length ? `${people.length}` : "—";

  list.textContent = "";

  if (!domReadable) {
    note.textContent =
      "Can't read Meet's participant list on this layout — scores below are " +
      "for the whole call.";
    note.className = "sonix-pnote warn";
    return;
  }

  for (const p of people) {
    const row = el("div", "sonix-person" + (p.id === selectedId ? " on" : ""));
    row.tabIndex = 0;
    row.title = "Focus the verdict on this participant";

    const av = el("div", "sonix-av");
    if (p.avatar) {
      const img = document.createElement("img");
      img.src = p.avatar;
      img.alt = "";
      av.appendChild(img);
    } else {
      av.textContent = initials(p.name);
    }
    if (p.speaking) av.classList.add("talking");

    const who = el("div", "sonix-who");
    who.append(el("div", "sonix-pname", p.name));

    // Per-person reading, only from windows Meet attributed to them alone.
    let sub, tone = "idle";
    if (p.last === null || p.last === undefined) {
      sub = p.speaking ? "speaking — listening" : "no attributed audio yet";
    } else {
      tone = bandOf(p.last);
      const label = tone === "red" ? "Cloned"
                  : tone === "amber" ? "Uncertain" : "Authentic";
      sub = `${label} · ${(p.last * 100).toFixed(0)}% · ${p.scores.length} win`;
    }
    const subEl = el("div", `sonix-psub ${tone}`, sub);
    who.appendChild(subEl);

    const pip = el("span", `sonix-pip ${tone}` + (p.speaking ? " live" : ""));

    row.append(av, who, pip);
    const pick = () => {
      selectedId = selectedId === p.id ? null : p.id;
      renderPeople();
      if (lastState) draw(lastState);
    };
    row.addEventListener("click", pick);
    row.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
    });
    list.appendChild(row);
  }

  if (!people.length) {
    note.textContent = "No participant tiles visible yet.";
    note.className = "sonix-pnote";
  } else if (!speakerReadable) {
    note.textContent =
      "Meet's speaking indicator isn't readable here, so windows can't be " +
      "attributed to a person. The verdict above is for the whole call.";
    note.className = "sonix-pnote warn";
  } else {
    note.textContent =
      "Attributed by Meet's active-speaker signal, only while one person " +
      "has the floor. Not voice biometrics.";
    note.className = "sonix-pnote";
  }
}

let prevScoreCount = 0;

function draw(s) {
  if (!s || !panel) return;
  lastState = s;

  readParticipants();

  // Credit any windows that arrived since the last render. The state carries
  // the whole history, so new points are whatever is past the high-water mark;
  // a shorter list means the call was restarted, so attribution starts over.
  const sc = s.scores || [];
  if (sc.length < prevScoreCount) {
    prevScoreCount = 0;
    for (const p of participants.values()) { p.scores = []; p.last = null; }
  }
  for (let i = prevScoreCount; i < sc.length; i++) attribute(sc[i]);
  prevScoreCount = sc.length;

  // A selected participant retargets the verdict onto the windows Meet
  // attributed to them. With nobody selected it stays the whole-call reading.
  const target = selectedId ? participants.get(selectedId) : null;
  const shown = target && target.last !== null && target.last !== undefined
    ? target.last
    : s.lastScore;

  const [wordText, colour, subText] = verdictOf(shown, s.scoringAvailable);

  const dot = panel.querySelector(".sonix-dot");
  dot.style.background = colour;
  dot.style.boxShadow = `0 0 0 4px ${GLOW[colour] || "transparent"}`;

  const verdict = panel.querySelector(".sonix-verdict");
  panel.querySelector(".sonix-rail").style.background = colour;
  verdict.style.setProperty("--band-glow", GLOW[colour] || "transparent");

  const word = panel.querySelector(".sonix-word");
  const pct = panel.querySelector(".sonix-pct");
  const sub = panel.querySelector(".sonix-sub");

  if (s.callState === "consent_pending") {
    word.textContent = s.pairingCode || "------";
    word.style.color = COLOUR.amber;
    word.style.letterSpacing = ".14em";
    word.style.fontVariantNumeric = "tabular-nums";
    pct.textContent = "pairing code";
    sub.textContent =
      "Approve it in the SONIX dashboard. It expires after 120 s — press " +
      "New code below if it has.";
  } else {
    word.style.letterSpacing = "-.02em";
    word.textContent = wordText;
    word.style.color = colour;
    pct.textContent =
      shown === null || shown === undefined
        ? ""
        : `P(AI voice) ${(shown * 100).toFixed(0)}%`;
    // Always says whose reading this is, so a per-person verdict can never be
    // mistaken for the whole call or the other way round.
    sub.textContent = target
      ? `${target.name} — ${target.scores.length} attributed window` +
        `${target.scores.length === 1 ? "" : "s"}. ${subText}`
      : subText;
  }

  // The graph stays the whole-call series when nothing is selected, and
  // switches to the target's attributed windows when something is.
  drawChart(target ? target.scores : (s.scores || []));
  renderPeople();
  syncModels(s);

  const err = panel.querySelector("#sonix-err");
  err.hidden = !s.error;
  err.textContent = s.error || "";

  const meta = panel.querySelector("#sonix-meta");
  meta.textContent = "";
  for (const [k, v] of [
    ["listening to", "the call, not your mic"],
    ["windows scored", String(s.windows || 0)],
    ["call", s.callState ? s.callState.toUpperCase() : "—"],
  ]) {
    const row = el("div");
    row.append(el("span", null, k), el("b", null, v));
    meta.appendChild(row);
  }
}

/* ------------------------------------------------------------------ */
/* Messages                                                            */
/* ------------------------------------------------------------------ */

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "sonix:level") {
    targetLevel = levelOf(msg.rms || 0);
    lastLevelAt = performance.now();
    if (panel) {
      const livedot = panel.querySelector(".sonix-vdot");
      const text = panel.querySelector(".sonix-vtext");
      const heard = (msg.rms || 0) > SILENCE_RMS;
      livedot.classList.toggle("on", heard);
      text.textContent = heard
        ? "voice detected — being scored"
        : "no audio from the call";
      text.classList.toggle("on", heard);
    }
    return;
  }

  if (msg.type !== "sonix:state") return;
  if (!msg.state.capturing) {
    removePanel();
    return;
  }
  ensurePanel();
  draw(msg.state);
});

window.addEventListener("resize", () => {
  if (panel && lastState) drawChart(lastState.scores || []);
});

/* Restore the operator's band thresholds before the first paint. */
try {
  chrome.storage.local.get(THRESH_KEY).then((v) => {
    const t = v && v[THRESH_KEY];
    if (t && typeof t.amber === "number" && typeof t.red === "number") {
      AMBER_AT = t.amber;
      RED_AT = t.red;
      if (lastState) draw(lastState);
    }
  });
} catch { /* storage unavailable */ }
