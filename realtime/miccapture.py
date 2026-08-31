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
        padding:28px;max-width:460px;width:100%}
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
  canvas#spec{width:100%;height:140px;display:block;margin-top:6px;border-radius:8px;
              background:#0a0a0a;border:1px solid var(--line)}
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

  <div class="label">Spectrogram &mdash; tinted by the live risk band</div>
  <canvas id="spec" height="140"></canvas>

  <div class="warn" id="warn" style="display:none"></div>
</div>

<script>
const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
const TARGET_SR = 16000;

let ws = null, ctx = null, node = null, stream = null, running = false;
let sentSamples = 0, callId = null;

// Must match AMBER_AT / RED_AT in realtime/live_ui.py - the dashboard and this
// page are two views of the same decision, and they must not disagree.
const AMBER_AT = 0.35, RED_AT = 0.65;
let analyser = null, specFrame = null, scoringAvailable = false, lastScore = null;

// [name, css colour, rgb triple for the spectrogram tint]
function bandFor(s){
  if (s === null)     return ["—", "var(--muted)",    [137,135,129]];
  if (s >= RED_AT)    return ["Red",     "var(--critical)", [208, 59, 59]];
  if (s >= AMBER_AT)  return ["Amber",   "var(--warning)",  [250,178, 25]];
  return                     ["Green",   "var(--good)",     [ 12,163, 12]];
}

function renderBand(){
  const v = $("verdict"), why = $("why");
  if (!scoringAvailable){
    v.textContent = "Scoring unavailable";
    v.style.color = "var(--muted)";
    why.textContent = "Server is running without a trained head (--ckpt). Capture is live; no verdict is shown.";
    return;
  }
  const [name, colour] = bandFor(lastScore);
  v.style.color = colour;
  if (lastScore === null){
    v.textContent = "—";
    why.textContent = "Waiting for the first 4-second window.";
    return;
  }
  v.textContent = name + " · " + Math.round(lastScore * 100) + "%";
  why.textContent = "P(AI voice) over the last 4 s. Amber ≥ 35%, Red ≥ 65% — provisional thresholds.";
}

// Scrolling spectrogram: one 2px column per frame, newest on the right.
// Tint carries the risk band, so the danger level is readable off the plot itself.
function drawSpec(){
  const c = $("spec"), g = c.getContext("2d"), w = c.width, h = c.height, col = 2;
  const bins = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(bins);

  g.drawImage(c, -col, 0);
  g.clearRect(w - col, 0, col, h);

  const rgb = bandFor(scoringAvailable ? lastScore : null)[2];
  for (let y = 0; y < h; y++){
    const mag = bins[Math.floor((1 - y / h) * (bins.length - 1))] / 255;
    if (mag <= 0.02) continue;
    g.fillStyle = "rgba(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + "," + mag.toFixed(3) + ")";
    g.fillRect(w - col, y, col, 1);
  }
  specFrame = requestAnimationFrame(drawSpec);
}

const $ = id => document.getElementById(id);
function setStatus(text, cls){ $("status").textContent = text; $("dot").className = "dot " + (cls||""); }
function warn(msg){ $("warn").style.display = "block"; $("warn").textContent = msg; }

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
      lastScore = m.data[callId].score;
      renderBand();
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

  analyser = ctx.createAnalyser();
  analyser.fftSize = 512;               // 256 bins over 0-8 kHz at 16 kHz
  analyser.smoothingTimeConstant = 0.6;
  src.connect(analyser);

  const c = $("spec");
  c.width = c.clientWidth || 460;       // match CSS width, else the plot stretches
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
  drawSpec();

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
  if (specFrame) cancelAnimationFrame(specFrame);
  specFrame = analyser = null;
  lastScore = null;
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

if (!navigator.mediaDevices || !window.AudioWorkletNode){
  $("btn").disabled = true;
  warn("This browser does not support AudioWorklet capture. Use Chrome or Edge.");
}
</script>
"""


async def mic_page_handler(request):
    """Serve the microphone capture page."""
    return web.Response(text=PAGE, content_type="text/html")
