# SONIX Chrome Extension — live voice-clone detection in Google Meet

Streams the audio of a Google Meet tab to the local SONIX detection server and
shows a live Green / Amber / Red risk band, both in the extension popup and as
an overlay on the Meet page itself.

**Version 0.1.0** · Chrome 116+ or Edge 116+ · Manifest V3

---

## What it captures

**The tab's audio — the other participants.** Not your microphone.

That is the right way round for this problem: you are trying to detect whether
the person *calling you* is a synthetic voice. Your own voice is not sent
anywhere.

One consequence worth understanding: tab audio is the **mixed** output of the
call. With three people speaking you get all three in one stream, and a score
applies to the four-second window, not to a named participant. Per-speaker
attribution would need diarisation, which SONIX does not do.

---

## Before you start

The extension is a client. The server does the work, so it must be running:

```bash
cd B:\SIH_sonix
python -m realtime.server --mock --ws-port 8000 --mode webrtc
```

Optionally open the dashboard in another terminal — you need it to approve the
pairing code:

```bash
streamlit run realtime/live_ui.py
```

---

## Install

1. Open `chrome://extensions`
2. Turn on **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder inside `SIH_sonix`

The SONIX icon appears in the toolbar. Pin it — you will want it visible during
a call.

---

## Use

1. Join or start a Google Meet call.
2. **Tell the other participants they are being monitored.** The overlay is
   visible on your screen only; Meet gives them no indication.
3. Click the SONIX icon → **Start monitoring**.
4. A six-digit pairing code appears in the popup and in the on-page overlay.
5. Approve it in the dashboard's **Live Calls** tab. Until you do, the server
   buffers nothing and scores nothing — the consent gate is enforced
   server-side, not by the extension.
6. The overlay switches to the live band. The first result takes about four
   seconds, because a full 4-second window has to fill.
7. **Stop monitoring** ends the call and writes the audit record to
   `outputs/calls/`.

---

## What the band means

| Band | Score | Reading |
|---|---|---|
| Green | < 35% | Consistent with a real voice |
| Amber | 35–65% | Uncertain — treat with caution |
| Red | ≥ 65% | Likely synthetic — verify by another channel |

These thresholds are **provisional**. They were set on the clean ASVspoof
benchmark and have not been recalibrated on real recordings, which is Lane 1's
work. Do not present them to a jury as tuned values.

SONIX never blocks or ends a call. It raises a signal; a human decides.

---

## Right now: scoring is switched off

Until a trained `head.pt` exists and the server is started with `--ckpt`, the
popup and overlay read **"Monitoring — scoring unavailable"** and no percentage
is shown.

This is deliberate. The server reports a `scoring_available` flag and the
extension honours it, so a mock number can never appear on screen dressed as a
verdict. What you *can* demonstrate today is genuine: tab capture, the consent
gate, windowing, and the silence gate all running live on a real Meet call.

Once the checkpoint lands:

```bash
python -m realtime.server --ckpt outputs/models/head.pt --ws-port 8000 --mode webrtc
```

The band appears on its own. Nothing in the extension needs to change.

---

## How it works

```
Google Meet tab
  │  chrome.tabCapture.getMediaStreamId()      ← needs a user gesture
  ▼
background.js  (service worker — no DOM, no audio)
  │  creates one offscreen document, hands over the stream id
  ▼
offscreen.js
  │  getUserMedia({chromeMediaSource:"tab"})
  │  AudioContext at native rate (usually 48 kHz)
  │    ├─→ ctx.destination   so you still HEAR the meeting
  │    └─→ AudioWorklet      a copy of the samples
  │  Float32 → int16 LE → WebSocket binary frames
  ▼
ws://localhost:8000/ws
  TEXT   {"type":"start_mic_call","sample_rate":48000,"caller":"google-meet"}
  BINARY int16 PCM, mono
  ▼
server → resample 16 kHz → consent gate → 4 s window / 0.5 s hop
       → silence gate → embed → score → broadcast
  ▼
offscreen.js receives {"type":"scores"} → background.js → popup + overlay
```

Two details that are easy to get wrong if you edit this:

- **Tab capture mutes the tab.** `source.connect(ctx.destination)` in
  `offscreen.js` is what re-plays it. Delete that line and the meeting goes
  silent the moment you press Start, which looks exactly like a crash.
- **No resampling happens in the browser.** The context runs at its native rate
  and the server is told what that rate is. Resampling client-side would also
  degrade the playback path above.

---

## Files

| File | Role |
|---|---|
| `manifest.json` | MV3 manifest — permissions, content script, offscreen |
| `background.js` | Service worker: capture lifecycle, state, badge, message routing |
| `offscreen.html` / `offscreen.js` | The only place audio is touched |
| `worklet.js` | AudioWorkletProcessor — posts sample blocks to the main thread |
| `popup.html` / `.css` / `.js` | Start/stop, pairing code, server URL, live band |
| `content.js` / `overlay.css` | The on-page overlay inside Meet |

---

## Troubleshooting

**"Open a Google Meet call in this tab first."**
The active tab is not on `meet.google.com`. The extension refuses to capture
anything else on purpose.

**"Cannot reach the SONIX server."**
The server is not running, or it is on a different port. Check
`http://localhost:8000/api/status` in a browser tab.

**The meeting went silent.**
`source.connect(ctx.destination)` is missing or errored in `offscreen.js`. Open
the offscreen document's console from `chrome://extensions` → SONIX →
**Inspect views: offscreen.html**.

**The overlay never appears.**
The content script only injects on `https://meet.google.com/*`, and only on page
load. Reload the Meet tab after installing the extension.

**Nothing scores, state stays CONSENT_PENDING.**
The pairing code has not been approved. Open the dashboard's Live Calls tab and
press Approve. Codes expire after 120 seconds.

**Windows count rises but no scores appear.**
Expected while `scoring_available` is false. See the section above.

---

## Limitations

- One monitored tab at a time.
- Mixed audio only — no per-speaker attribution.
- Chrome and Edge only. Firefox has no `chrome.tabCapture` equivalent.
- The server must be reachable on localhost; this is not a hosted service.
- Meet's own audio processing (noise suppression, AGC) has already altered the
  audio before it reaches the tab. That is additional domain shift on top of the
  problem the model already has, and it is a fair thing for a jury to press on.

---

## Privacy and consent

Audio leaves the browser only to `localhost` and is never sent anywhere else.
The server holds it in a four-second ring buffer, scores it, and writes an audit
record of the call's events — not the audio — to `outputs/calls/`.

The consent gate is real: `Session.push_audio()` discards every chunk until the
pairing code is approved, and the refusal is recorded in the audit trail. That
is a design commitment in SONIX, not a UI convention, and it is worth saying so
if a jury asks.

None of which removes your obligation to tell people in the call that you are
monitoring it. Recording law varies by jurisdiction and by whether all parties
consent. For the hackathon demo, use your own team.
