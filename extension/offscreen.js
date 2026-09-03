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
  if (node) node.disconnect();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
  ws = node = stream = ctx = null;
  callId = null;
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.target !== "offscreen") return;
  if (msg.type === "start") start(msg);
  if (msg.type === "stop") stop();
});
