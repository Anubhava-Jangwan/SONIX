/*
 * SONIX offscreen document — the only place audio is touched.
 *
 * Flow:
 *   streamId (from chrome.tabCapture in the service worker)
 *     -> getUserMedia({chromeMediaSource:"tab"})           the Meet tab's audio
 *     -> AudioContext at the device's native rate
 *          |-> ctx.destination      so the user still HEARS the meeting
 *          '-> AudioWorklet         so we get a copy of the samples
 *     -> Float32 -> int16 little-endian
 *     -> WebSocket binary frames to the SONIX server
 *
 * Two things worth knowing:
 *
 *  1. Tab capture MUTES the tab. Connecting the source to ctx.destination
 *     re-plays it. Without that line the meeting goes silent the moment you
 *     press Start, which looks exactly like a crash.
 *
 *  2. We do NOT resample here. The context runs at its native rate (usually
 *     48 kHz) and we tell the server that rate in start_mic_call; the server
 *     resamples to the 16 kHz wav2vec2 expects. Resampling in the browser would
 *     also degrade the playback path above.
 *
 * What is captured: the TAB's audio, which is the other participants. Your own
 * microphone is not captured — you are trying to detect whether the person
 * calling YOU is synthetic.
 */

let ws = null;
let ctx = null;
let node = null;
let stream = null;
let callId = null;
let serverBase = "http://localhost:8000";

// Remembered so a call can be restarted on the EXISTING socket without going
// back through chrome.tabCapture -- see restartCall().
let lastCaller = "google-meet";
let lastModel;

// Live input level, measured here rather than asked of the server.
// The server only reports whether a 4 s window cleared the silence gate, which
// arrives seconds late and says nothing while a window is filling. An
// AnalyserNode on the same graph that feeds the socket is the real signal, at
// no cost -- and it answers the only question that matters when nothing is
// being scored: is any audio arriving at all?
let analyser = null;
let levelTimer = null;

function startLevelMeter(source) {
  analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.55;
  source.connect(analyser);

  const buf = new Float32Array(analyser.fftSize);
  clearInterval(levelTimer);
  // 20 Hz: fast enough that the panel's own easing looks continuous, slow
  // enough not to flood the message port.
  levelTimer = setInterval(() => {
    if (!analyser) return;
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = buf[i];
      sum += v * v;
      const a = v < 0 ? -v : v;
      if (a > peak) peak = a;
    }
    toBackground({ type: "level", rms: Math.sqrt(sum / buf.length), peak });
  }, 50);
}

const toBackground = (msg) =>
  chrome.runtime.sendMessage({ target: "background", ...msg });

function floatToPCM16(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

async function start({ streamId, serverUrl, caller, model }) {
  serverBase = (serverUrl || serverBase).replace(/\/+$/, "");
  lastCaller = caller || "google-meet";
  lastModel = model || undefined;
  const wsUrl = serverBase.replace(/^http/, "ws") + "/ws";

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: "tab",
          chromeMediaSourceId: streamId,
        },
      },
    });
  } catch (e) {
    return toBackground({ type: "error", message: "Tab capture failed: " + e.message });
  }

  ctx = new AudioContext();                       // native rate, see note 2
  const source = ctx.createMediaStreamSource(stream);
  source.connect(ctx.destination);                // note 1 — keep it audible
  startLevelMeter(source);

  try {
    await ctx.audioWorklet.addModule(chrome.runtime.getURL("worklet.js"));
  } catch (e) {
    return toBackground({ type: "error", message: "Worklet failed: " + e.message });
  }
  node = new AudioWorkletNode(ctx, "sonix-capture");
  source.connect(node);

  ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";

  ws.onopen = async () => {
    ws.send(
      JSON.stringify({
        type: "start_mic_call",
        sample_rate: ctx.sampleRate,
        caller: caller || "google-meet",
        model: model || undefined,
      })
    );
    // scoring_available only appears on the telemetry endpoint
    try {
      const r = await fetch(`${serverBase}/api/telemetry`);
      const d = await r.json();
      toBackground({ type: "scoring_available", value: !!d.scoring_available });
    } catch {
      toBackground({ type: "scoring_available", value: false });
    }
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") return;
    let m;
    try {
      m = JSON.parse(ev.data);
    } catch {
      return;
    }

    if (m.type === "mic_call_started") {
      callId = m.call_id;
      toBackground({ type: "call_started", callId, pairingCode: m.pairing_code });
    }
    if (m.type === "call_state" && m.call_id === callId) {
      toBackground({ type: "call_state", state: m.state });
    }
    if (m.type === "scores" && m.data && callId in m.data) {
      const d = m.data[callId];
      toBackground({ type: "score", score: d.score, windows: d.window_idx + 1 });
    }
    if (m.type === "model_changed" && m.call_id === callId) {
      toBackground({ type: "model_changed", model: m.model });
    }
    if (m.type === "error") {
      toBackground({ type: "error", message: m.message });
    }
  };

  ws.onerror = () =>
    toBackground({
      type: "error",
      message: `Cannot reach the SONIX server at ${serverBase}. Is it running?`,
    });

  ws.onclose = () => toBackground({ type: "closed" });

  node.port.onmessage = (ev) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(floatToPCM16(ev.data).buffer);
    }
  };
}

function stop() {
  try {
    if (ws && ws.readyState === WebSocket.OPEN) {
      if (callId) ws.send(JSON.stringify({ type: "end_call", call_id: callId }));
      ws.close();
    }
  } catch {}
  clearInterval(levelTimer);
  levelTimer = null;
  analyser = null;
  if (node) node.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
  ws = node = stream = ctx = null;
  callId = null;
}

/* Swap the head scoring THIS call, without touching the audio path.
 *
 * Deliberately not a stop()/start() pair: restarting would mint a new call_id
 * and send the operator back through the consent gate, and the panel's graph
 * would lose the call it is graphing. The server re-points the existing
 * session instead -- see SonicServer._set_call_model.
 */
function setModel(model) {
  if (!ws || ws.readyState !== WebSocket.OPEN || !callId) return;
  lastModel = model;
  ws.send(JSON.stringify({ type: "set_model", call_id: callId, model }));
}

/* End the current call and open a new one on the SAME socket, to mint a fresh
 * pairing code.
 *
 * Deliberately not a stop()/start() pair. Starting over would need a new
 * stream id from chrome.tabCapture.getMediaStreamId(), and that call requires
 * a user gesture in the EXTENSION's own context -- user activation does not
 * survive a sendMessage hop from a content script, so a button in the in-page
 * panel could not satisfy it. It would fail with "Extension has not been
 * invoked for the current page".
 *
 * None of that is necessary: the tab capture, the AudioContext and the
 * worklet are all still live and still feeding this socket. Only the server
 * side needs recycling, and the server mints a pairing code per call.
 */
function restartCall() {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    toBackground({
      type: "error",
      message: "Not connected to the SONIX server — press Stop, then start " +
               "monitoring again from the toolbar popup.",
    });
    return;
  }
  if (callId) ws.send(JSON.stringify({ type: "end_call", call_id: callId }));
  callId = null;
  ws.send(JSON.stringify({
    type: "start_mic_call",
    sample_rate: ctx ? ctx.sampleRate : 48000,
    caller: lastCaller,
    model: lastModel,
  }));
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.target !== "offscreen") return;
  if (msg.type === "start") start(msg);
  if (msg.type === "stop") stop();
  if (msg.type === "set_model") setModel(msg.model);
  if (msg.type === "restart") restartCall();
});
