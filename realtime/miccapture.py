"""
Browser microphone capture for SONIX live detection.

Serves a self-contained page at /mic that:
  1. opens the user's microphone with getUserMedia
  2. runs an AudioWorklet at 16 kHz (asked for directly, so no client-side
     resampling is needed on Chrome/Edge)
  3. streams little-endian int16 PCM over the SAME WebSocket the dashboard uses,
     as binary frames
  4. shows the pairing code and waits - audio is buffered by the server but not
     scored until an operator approves the code in the dashboard
  5. once scoring starts, plots P(AI voice) per 4-second window on a risk
     timeline (same Green/Amber/Red bands as the dashboard and the Upload tab)

getUserMedia requires a secure context. http://localhost counts as secure, so
this works for the demo without TLS. Serving it over a LAN IP will NOT work
without https - use localhost, or tunnel.
"""

from aiohttp import web

PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>SONIX - Microphone Capture</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--bg:#0d0d0d;--surface:#1a1a19;--ink:#fff;--muted:#898781;--line:#2c2c2a;
        --good:#0ca30c;--warning:#fab219;--critical:#d03b3b;--blue:#3987e5}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
       display:flex;align-items:center;justify-content:center;min-height:100vh;padding:24px}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;
        padding:28px;max-width:520px;width:100%}
  h1{font-size:18px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin-bottom:22px}
  .row{display:flex;align-items:center;gap:10px;margin:14px 0}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--muted);flex:none}
  .dot.on{background:var(--good)}.dot.err{background:var(--critical)}
  .dot.wait{background:var(--warning)}
  button{font:inherit;font-weight:600;padding:11px 20px;border-radius:8px;border:0;
         cursor:pointer;background:var(--blue);color:#fff}
  button.stop{background:var(--critical)}
  button:disabled{opacity:.45;cursor:not-allowed}
  .meter{height:8px;background:#2c2c2a;border-radius:4px;overflow:hidden;margin-top:6px}
  .meter i{display:block;height:100%;width:0;background:var(--good);transition:width .08s linear}
  .code{font-size:34px;letter-spacing:.14em;font-weight:700;font-variant-numeric:tabular-nums}
  .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.07em}
  .stat{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);
        padding:5px 0;border-bottom:1px solid var(--line)}
  .stat b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
  .warn{background:#2a2410;border:1px solid #6b5518;color:#fab219;padding:10px 12px;
        border-radius:8px;font-size:13px;margin-top:18px}
  .band{margin:18px 0 10px;padding:14px;border-radius:10px;border:1px solid var(--line);
        background:#141413;text-align:center}
  .verdict{font-size:26px;font-weight:700;line-height:1.2;font-variant-numeric:tabular-nums}
  .why{color:var(--muted);font-size:12px;margin-top:5px}
  .metrics{display:flex;gap:8px;margin:14px 0 6px}
  .metrics>div{flex:1;background:#141413;border:1px solid var(--line);border-radius:8px;
               padding:8px 6px;text-align:center}
  .metrics b{display:block;font-size:19px;font-weight:700;font-variant-numeric:tabular-nums;margin-top:2px}
  canvas#risk{width:100%;height:170px;display:block;margin-top:6px;border-radius:8px;
              background:#0a0a0a;border:1px solid var(--line)}
  #scorebox[hidden]{display:none}
</style>
<div class="card">
  <h1>SONIX &mdash; microphone capture</h1>
  <div class="sub">Streams 16&nbsp;kHz mono PCM to the detection server.</div>

  <div class="row"><span class="dot" id="dot"></span><span id="status">Idle</span></div>

  <div class="row" style="gap:14px">
    <button id="btn">Start capture</button>
    <div style="flex:1">
      <div class="label">Input level</div>
      <div class="meter"><i id="level"></i></div>
    </div>
  </div>

  <div id="pairing" style="display:none;margin:18px 0">
    <div class="label">Pairing code &mdash; approve in the dashboard</div>
    <div class="code" id="code">------</div>
  </div>

  <div class="stat"><span>Call ID</span><b id="callid">&mdash;</b></div>
  <div class="stat"><span>Sample rate</span><b id="sr">&mdash;</b></div>
  <div class="stat"><span>Audio sent</span><b id="sent">0.0 s</b></div>
  <div class="stat"><span>State</span><b id="callstate">&mdash;</b></div>

  <div class="band">
    <div class="verdict" id="verdict">&mdash;</div>
    <div class="why" id="why">Waiting for the first 4-second window.</div>
  </div>

  <div id="scorebox">
    <div class="metrics">
      <div><span class="label">Latest</span><b id="mLatest">&mdash;</b></div>
      <div><span class="label">Mean</span><b id="mMean">&mdash;</b></div>
      <div><span class="label">Max</span><b id="mMax">&mdash;</b></div>
      <div><span class="label">Windows</span><b id="mN">0</b></div>
    </div>
    <div class="label">Risk timeline &mdash; P(AI voice) per 4&nbsp;s window</div>
    <canvas id="risk" height="170"></canvas>
  </div>

  <div class="warn" id="warn" style="display:none"></div>
</div>

<script>
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const TARGET_SR = 16000;
const HOP_S = 0.5;   // seconds between window starts (server RingBuffer hop)

let ws = null, ctx = null, node = null, stream = null, running = false;
let sentSamples = 0, callId = null;

// Must match AMBER_AT / RED_AT in realtime/live_ui.py and the extension - every
// view of this decision has to agree.
const AMBER_AT = 0.35, RED_AT = 0.65;
let scoringAvailable = false, lastScore = null;
const scoresByIdx = new Map();   // window_idx -> P(AI) in [0,1]

const $ = id => document.getElementById(id);
function setStatus(text, cls){ $("status").textContent = text; $("dot").className = "dot " + (cls||""); }
function warn(msg){ $("warn").style.display = "block"; $("warn").textContent = msg; }

// [name, css colour, plain-language action] - shared shape with popup.js/content.js
function bandFor(s){
  if (s === null || s === undefined) return ["—", "var(--muted)", "Waiting for the first 4-second window."];
  if (s >= RED_AT)   return ["Red · "   + Math.round(s*100) + "%", "var(--critical)", "Likely synthetic — verify on another channel."];
  if (s >= AMBER_AT) return ["Amber · " + Math.round(s*100) + "%", "var(--warning)",  "Uncertain — treat with caution."];
  return                     ["Green · " + Math.round(s*100) + "%", "var(--good)",     "Consistent with a real voice."];
}

function renderBand(){
  const v = $("verdict"), why = $("why");
  if (!scoringAvailable){
    v.textContent = "Scoring unavailable";
    v.style.color = "var(--muted)";
    why.textContent = "Server is running without a trained head (--ckpt). Capture is live; no verdict is shown.";
    return;
  }
  const [name, colour, action] = bandFor(lastScore);
  v.textContent = name;
  v.style.color = colour;
  why.textContent = lastScore === null
    ? "Waiting for the first 4-second window."
    : action + "  (Amber ≥ 35%, Red ≥ 65% — provisional thresholds)";
}

function renderScores(){
  const vals = [...scoresByIdx.values()];
  const pct = x => (x === null || x === undefined) ? "—" : Math.round(x * 100) + "%";
  if (vals.length){
    const latestIdx = Math.max(...scoresByIdx.keys());
    lastScore = scoresByIdx.get(latestIdx);
    $("mLatest").textContent = pct(lastScore);
    $("mMean").textContent   = pct(vals.reduce((a, b) => a + b, 0) / vals.length);
    $("mMax").textContent    = pct(Math.max(...vals));
    $("mN").textContent      = vals.length;
  } else {
    lastScore = null;
    $("mLatest").textContent = $("mMean").textContent = $("mMax").textContent = "—";
    $("mN").textContent = 0;
  }
  renderBand();
  drawRisk();
}

// Plain 2D line chart of P(AI voice) over time, with the Green/Amber/Red
// decision bands behind it - the "danger level" plot that replaced the
// spectrogram. No charting library: it is one <canvas> and ~30 lines.
function drawRisk(){
  const c = $("risk");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = c.clientWidth || 460, cssH = 170;
  c.width = cssW * dpr; c.height = cssH * dpr;
  const g = c.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cssW, cssH);

  const padL = 34, padR = 12, padT = 10, padB = 20;
  const w = cssW - padL - padR, h = cssH - padT - padB;
  const pts = [...scoresByIdx.entries()].sort((a, b) => a[0] - b[0]);
  const maxIdx = pts.length ? pts[pts.length - 1][0] : 0;
  const spanIdx = Math.max(20, maxIdx);          // at least 10 s of x-axis
  const X = i => padL + (spanIdx ? (i / spanIdx) * w : 0);
  const Y = s => padT + (1 - s) * h;

  // decision bands
  const bands = [[0, AMBER_AT, "rgba(12,163,12,0.10)"],
                 [AMBER_AT, RED_AT, "rgba(250,178,25,0.10)"],
                 [RED_AT, 1, "rgba(208,59,59,0.13)"]];
  for (const [lo, hi, col] of bands){ g.fillStyle = col; g.fillRect(padL, Y(hi), w, Y(lo) - Y(hi)); }

  // threshold lines
  g.lineWidth = 1; g.font = "10px system-ui";
  g.setLineDash([4, 3]);
  for (const [y, col, txt] of [[AMBER_AT, "#fab219", "Amber"], [RED_AT, "#d03b3b", "Red"]]){
    g.strokeStyle = col; g.beginPath(); g.moveTo(padL, Y(y)); g.lineTo(padL + w, Y(y)); g.stroke();
    g.fillStyle = col; g.fillText(txt, padL + w - 32, Y(y) - 3);
  }
  g.setLineDash([]);

  // axes
  g.fillStyle = "#898781";
  g.fillText("100%", 4, Y(1) + 3);
  g.fillText("50%", 8, Y(0.5) + 3);
  g.fillText("0%", 14, Y(0) + 3);
  g.fillText((spanIdx * HOP_S).toFixed(0) + " s", padL + w - 26, cssH - 5);
  g.fillText("time →", padL, cssH - 5);

  // score polyline + dots
  if (pts.length){
    g.strokeStyle = "#3987e5"; g.lineWidth = 2; g.beginPath();
    pts.forEach(([i, s], k) => { const x = X(i), y = Y(s); k ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
    g.fillStyle = "#3987e5";
    pts.forEach(([i, s]) => { g.beginPath(); g.arc(X(i), Y(s), 2.6, 0, Math.PI * 2); g.fill(); });
  }
}

// AudioWorklet: forward raw float32 frames to the main thread.
const WORKLET = `
class Cap extends AudioWorkletProcessor {
  process(inputs){
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('cap', Cap);
`;

function floatToPCM16(f32){
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++){
    let s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

async function start(){
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount:1, echoCancellation:false, noiseSuppression:false, autoGainControl:false }
    });
  } catch (e) {
    setStatus("Microphone denied", "err");
    warn("The browser blocked microphone access. Allow it for this site, and make sure you opened this page on http://localhost (getUserMedia needs a secure context).");
    return;
  }

  // The server decides whether a verdict may be shown at all; a mock number must
  // never reach the screen dressed as one.
  try {
    const st = await (await fetch("/api/status")).json();
    scoringAvailable = !!st.scoring_available;
    if (st.scoring_synthetic){
      warn("UNTRAINED DEV CHECKPOINT. Every score below is random noise and says "
           + "nothing about any voice. Plumbing and latency testing only — "
           + "never show this to anyone.");
    }
  } catch { scoringAvailable = false; }

  scoresByIdx.clear();
  $("scorebox").hidden = !scoringAvailable;
  renderScores();
  renderBand();

  // Ask for 16 kHz directly. Chrome/Edge honour this; if not, we resample below.
  ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: TARGET_SR });
  await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], {type:"application/javascript"})));
  $("sr").textContent = ctx.sampleRate + " Hz" + (ctx.sampleRate === TARGET_SR ? "" : " (resampled)");

  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("Connected — awaiting approval", "wait");
    ws.send(JSON.stringify({ type:"start_mic_call", sample_rate: ctx.sampleRate, caller:"browser-mic" }));
  };

  ws.onmessage = ev => {
    if (typeof ev.data !== "string") return;
    let m; try { m = JSON.parse(ev.data); } catch { return; }

    if (m.type === "mic_call_started"){
      callId = m.call_id;
      $("callid").textContent = callId;
      $("code").textContent = m.pairing_code;
      $("pairing").style.display = "block";
      $("callstate").textContent = "CONSENT_PENDING";
    }
    if (m.type === "pairing_request" && m.call_id === callId){
      $("code").textContent = m.pairing_code;
    }
    if (m.type === "scores" && callId && m.data && m.data[callId]){
      if (!scoringAvailable) return;          // mock scores never reach the plot
      const d = m.data[callId];
      const items = (Array.isArray(d.batch) && d.batch.length)
        ? d.batch
        : [{ window_idx: d.window_idx, score: d.score }];
      for (const it of items){
        if (it && it.window_idx != null && it.score != null){
          scoresByIdx.set(it.window_idx, Math.max(0, Math.min(1, it.score)));
        }
      }
      renderScores();
    }
    if (m.type === "call_state" && m.call_id === callId){
      $("callstate").textContent = m.state.toUpperCase();
      if (m.state === "listening" || m.state === "scoring"){
        setStatus("Approved — streaming audio", "on");
        $("pairing").style.display = "none";
      }
    }
  };

  ws.onerror = () => { setStatus("WebSocket error", "err"); warn("Could not reach the server. Is it running on this port?"); };
  ws.onclose = () => { if (running) stop(); setStatus("Disconnected"); };

  const src = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, "cap");

  node.port.onmessage = ev => {
    const f32 = ev.data;

    let peak = 0;
    for (let i = 0; i < f32.length; i++){ const a = Math.abs(f32[i]); if (a > peak) peak = a; }
    $("level").style.width = Math.min(100, peak * 140) + "%";

    if (ws && ws.readyState === WebSocket.OPEN){
      ws.send(floatToPCM16(f32).buffer);
      sentSamples += f32.length;
      $("sent").textContent = (sentSamples / ctx.sampleRate).toFixed(1) + " s";
    }
  };

  src.connect(node);
  // Keep the graph alive without echoing the mic to the speakers.
  const mute = ctx.createGain(); mute.gain.value = 0;
  node.connect(mute).connect(ctx.destination);

  running = true;
  $("btn").textContent = "Stop capture";
  $("btn").className = "stop";
}

function stop(){
  running = false;
  if (ws && ws.readyState === WebSocket.OPEN){
    if (callId) ws.send(JSON.stringify({ type:"end_call", call_id: callId }));
    ws.close();
  }
  lastScore = null;
  scoresByIdx.clear();
  renderScores();
  renderBand();
  if (node) node.disconnect();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (ctx) ctx.close();
  ws = node = stream = ctx = null;
  $("btn").textContent = "Start capture";
  $("btn").className = "";
  $("pairing").style.display = "none";
  setStatus("Stopped");
}

$("btn").onclick = () => running ? stop() : start();
window.addEventListener("resize", drawRisk);

if (!navigator.mediaDevices || !window.AudioWorkletNode){
  $("btn").disabled = true;
  warn("This browser does not support AudioWorklet capture. Use Chrome or Edge.");
}
</script>
"""


async def mic_page_handler(request):
    """Serve the microphone capture page."""
    return web.Response(text=PAGE, content_type="text/html")
