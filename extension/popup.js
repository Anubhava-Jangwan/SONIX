/* Popup: thin control surface. All state lives in the service worker. */

const $ = (id) => document.getElementById(id);
const SERVER_KEY = "sonix.serverUrl";

function bandFor(score, scoringAvailable) {
  if (!scoringAvailable) return { label: "Monitoring", sub: "Scoring unavailable — no trained head loaded.", colour: "var(--accent)" };
  if (score === null || score === undefined) return { label: "Listening", sub: "Waiting for the first scored window.", colour: "var(--ink-3)" };
  if (score >= 0.65) return { label: `Red · ${Math.round(score * 100)}%`, sub: "Likely synthetic. Verify by another channel.", colour: "var(--crit)" };
  if (score >= 0.35) return { label: `Amber · ${Math.round(score * 100)}%`, sub: "Uncertain. Treat with caution.", colour: "var(--warn)" };
  return { label: `Green · ${Math.round(score * 100)}%`, sub: "Consistent with a real voice.", colour: "var(--good)" };
}

function render(s) {
  const capturing = !!s.capturing;
  $("toggle").textContent = capturing ? "Stop monitoring" : "Start monitoring";
  $("toggle").className = capturing ? "on" : "";

  if (!capturing) {
    $("band").style.borderLeftColor = "var(--ink-3)";
    $("bandLabel").textContent = "Not monitoring";
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

$("toggle").addEventListener("click", async () => {
  const s = await chrome.runtime.sendMessage({ type: "popup:state" });
  if (s.capturing) {
    await chrome.runtime.sendMessage({ type: "popup:stop" });
  } else {
    const serverUrl = $("server").value.trim();
    await chrome.storage.local.set({ [SERVER_KEY]: serverUrl });
    const r = await chrome.runtime.sendMessage({ type: "popup:start", serverUrl });
    if (r && !r.ok) {
      $("err").hidden = false;
      $("err").textContent = r.error;
    }
  }
  refresh();
});

chrome.storage.local.get(SERVER_KEY).then((v) => {
  if (v[SERVER_KEY]) $("server").value = v[SERVER_KEY];
});

refresh();
setInterval(refresh, 1000);
