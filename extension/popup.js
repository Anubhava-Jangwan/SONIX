/* Popup: thin control surface. All state lives in the service worker. */

const $ = (id) => document.getElementById(id);
const SERVER_KEY = "sonix.serverUrl";
const DEFAULT_SERVER = "http://localhost:8000";

function bandFor(score, scoringAvailable) {
  if (!scoringAvailable) return { label: "Monitoring", sub: "Scoring unavailable — no trained head loaded.", colour: "var(--accent)" };
  if (score === null || score === undefined) return { label: "Listening", sub: "Waiting for the first scored window.", colour: "var(--ink-3)" };
  if (score >= 0.65) return { label: `Cloned · ${Math.round(score * 100)}%`, sub: "Reads as synthetic. Verify by another channel.", colour: "var(--crit)" };
  if (score >= 0.35) return { label: `Uncertain · ${Math.round(score * 100)}%`, sub: "Neither clean nor clearly synthetic.", colour: "var(--warn)" };
  return { label: `Authentic · ${Math.round(score * 100)}%`, sub: "Consistent with a real human voice.", colour: "var(--good)" };
}

/* ------------------------------------------------------------------ */
/* Server address                                                      */
/* ------------------------------------------------------------------ */

/* The stored value wins over the field's default forever, so one bad paste
   is permanent until someone notices. A URL with a path or a #fragment is
   always wrong here -- this is a WebSocket origin, not a page -- so strip
   those rather than letting ws://host/page/#x be attempted. */
function normaliseServer(raw) {
  const v = (raw || "").trim();
  if (!v) return DEFAULT_SERVER;
  try {
    const u = new URL(/^https?:\/\//.test(v) ? v : `http://${v}`);
    return `${u.protocol}//${u.host}`;
  } catch {
    return DEFAULT_SERVER;
  }
}

/* Probe it and say so. Pointing the extension at the Streamlit dashboard
   instead of the detection server looks identical from inside the popup --
   the button works, nothing connects, and no error is ever shown. */
async function checkServer() {
  const url = normaliseServer($("server").value);
  const dot = $("serverDot");
  const msg = $("serverMsg");
  dot.className = "dot";
  msg.textContent = "Checking…";
  msg.style.color = "var(--ink-3)";
  $("server").classList.remove("bad");

  try {
    const r = await fetch(`${url}/api/status`, { cache: "no-store" });
    const d = await r.json();
    if (!("scoring_available" in d)) throw new Error("not a SONIX server");
    dot.className = "dot ok";
    msg.textContent = `SONIX server · ${d.mode} · scoring ${d.scoring_available ? "on" : "OFF"}`;
    msg.style.color = "var(--ink-3)";
    return true;
  } catch {
    dot.className = "dot bad";
    $("server").classList.add("bad");
    msg.textContent =
      url === DEFAULT_SERVER
        ? "No SONIX server here. Start realtime.server on port 8000."
        : `Nothing that looks like SONIX at ${url} — press reset for ${DEFAULT_SERVER}.`;
    msg.style.color = "var(--crit)";
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* Render                                                              */
/* ------------------------------------------------------------------ */

function render(s) {
  const capturing = !!s.capturing;
  $("toggle").textContent = capturing ? "Stop monitoring" : "Start monitoring";
  $("toggle").className = capturing ? "on" : "";

  if (!capturing) {
    $("band").style.borderLeftColor = "var(--ink-3)";
    $("bandLabel").textContent = "Not monitoring";
    $("bandLabel").style.color = "var(--ink)";
    $("bandSub").textContent = "Open a Google Meet call, then start.";
  } else {
    const b = bandFor(s.lastScore, s.scoringAvailable);
    $("band").style.borderLeftColor = b.colour;
    $("bandLabel").textContent = b.label;
    $("bandLabel").style.color = b.colour;
    $("bandSub").textContent = b.sub;
  }

  const pending = capturing && s.callState === "consent_pending";
  $("pairing").hidden = !pending;
  if (pending) $("code").textContent = s.pairingCode || "------";

  $("callId").textContent = s.callId || "—";
  $("callState").textContent = s.callState ? s.callState.toUpperCase() : "—";
  $("windows").textContent = s.windows || 0;

  $("err").hidden = !s.error;
  $("err").textContent = s.error || "";
}

async function refresh() {
  const s = await chrome.runtime.sendMessage({ type: "popup:state" });
  if (s) render(s);
}

/* ------------------------------------------------------------------ */
/* Actions                                                             */
/* ------------------------------------------------------------------ */

$("toggle").addEventListener("click", async () => {
  const s = await chrome.runtime.sendMessage({ type: "popup:state" });
  if (s.capturing) {
    await chrome.runtime.sendMessage({ type: "popup:stop" });
  } else {
    const serverUrl = normaliseServer($("server").value);
    $("server").value = serverUrl;
    await chrome.storage.local.set({ [SERVER_KEY]: serverUrl });
    const r = await chrome.runtime.sendMessage({
      type: "popup:start", serverUrl, model: $("model").value || undefined,
    });
    if (r && !r.ok) {
      $("err").hidden = false;
      $("err").textContent = r.error;
    }
  }
  refresh();
});

/* Total control: drop the capture stream and the offscreen document even if
   the service worker thinks nothing is running. Without this, a stream Chrome
   still holds can only be cleared by restarting the browser. */
$("forceStop").addEventListener("click", async () => {
  const btn = $("forceStop");
  btn.disabled = true;
  btn.textContent = "Stopping…";
  const r = await chrome.runtime.sendMessage({ type: "popup:forceStop" });
  btn.disabled = false;
  btn.textContent = "Force-stop capture stream";
  $("err").hidden = false;
  $("err").textContent = (r && r.ok)
    ? "Capture stream released. You can start monitoring again."
    : `Could not release the stream: ${(r && r.error) || "unknown"}`;
  refresh();
});

$("serverReset").addEventListener("click", async () => {
  $("server").value = DEFAULT_SERVER;
  await chrome.storage.local.set({ [SERVER_KEY]: DEFAULT_SERVER });
  if (await checkServer()) loadModels();
});

$("server").addEventListener("change", async () => {
  const v = normaliseServer($("server").value);
  $("server").value = v;
  await chrome.storage.local.set({ [SERVER_KEY]: v });
  if (await checkServer()) loadModels();
});

$("code").addEventListener("click", () => {
  const code = $("code").textContent.trim();
  if (!code || code === "------") return;
  navigator.clipboard.writeText(code).then(() => {
    const hint = $("codeHint");
    hint.textContent = "Copied.";
    setTimeout(() => {
      hint.textContent = "Approve it on the SONIX dashboard — click to copy.";
    }, 1400);
  }).catch(() => {});
});

async function loadModels() {
  try {
    const base = normaliseServer($("server").value);
    const info = await (await fetch(`${base}/api/models`)).json();
    const ready = (info.models || []).filter((m) => m.exists);
    const sel = $("model");
    sel.textContent = "";
    if (info.mock || !ready.length) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = "Mock";
      sel.appendChild(opt);
      return;
    }
    // DOM nodes with textContent, not innerHTML - a model label comes from a
    // checkpoint's own config and must never be interpreted as markup.
    for (const m of ready) {
      const opt = document.createElement("option");
      opt.value = m.key; opt.textContent = m.label;
      if (m.key === info.default) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) {
    // Server unreachable yet - the default "Default" option stays put.
  }
}

chrome.storage.local.get(SERVER_KEY).then(async (v) => {
  if (v[SERVER_KEY]) $("server").value = v[SERVER_KEY];
  if (await checkServer()) loadModels();
});

refresh();
setInterval(refresh, 1000);
