# Claude Code prompt — SONIX Google Meet extension

Paste everything between the rules below into Claude Code, from the repo root
(`SIH_sonix`). It is self-contained: it states the protocol, the constraints and
the acceptance criteria, so Claude Code does not have to guess at any of them.

---

Build a Manifest V3 Chrome extension in `extension/` that monitors a Google Meet
call for AI voice-clone impersonation, using the SONIX detection server that
already exists in this repo.

## Context you need

The server is `realtime/server.py`. Read it, plus `realtime/session.py` and
`realtime/miccapture.py`, before writing anything — `miccapture.py` already does
browser capture against this protocol from a plain web page, and the extension
is the same client in a different host.

Protocol, on `ws://localhost:8000/ws`:

- Client sends TEXT `{"type":"start_mic_call","sample_rate":48000,"caller":"google-meet"}`
- Server replies TEXT `{"type":"mic_call_started","call_id":"...","pairing_code":"123456"}`
- Client sends BINARY frames: **int16 little-endian, mono**, at the declared rate.
  The server resamples to 16 kHz itself — do not resample in the browser.
- Server broadcasts TEXT `{"type":"call_state","call_id":"...","state":"listening"}`
  after the pairing code is approved.
- Server broadcasts TEXT `{"type":"scores","data":{"<call_id>":{"window_idx":0,"score":0.32}}}`
- Client sends TEXT `{"type":"end_call","call_id":"..."}` to finish.

`GET http://localhost:8000/api/telemetry` returns a top-level
`scoring_available` boolean. It is false whenever no trained checkpoint is
loaded.

**The consent gate is server-side.** `Session.push_audio()` discards every chunk
until the pairing code is approved through the dashboard. Do not try to work
around it, and do not reimplement it client-side.

## What to build

Capture **the tab's audio** — the other participants — not the user's
microphone. The threat model is an impersonator calling the user.

1. `manifest.json` — MV3. Permissions: `tabCapture`, `offscreen`, `activeTab`,
   `storage`, `scripting`. Host permissions for `http://localhost/*`. Content
   script on `https://meet.google.com/*`. Declare `worklet.js` in
   `web_accessible_resources`. Set `minimum_chrome_version` to 116.

2. `background.js` — the service worker. It owns the capture lifecycle and all
   state, and never touches audio. It calls
   `chrome.tabCapture.getMediaStreamId({targetTabId})`, ensures exactly one
   offscreen document exists via `chrome.offscreen.createDocument` with reason
   `USER_MEDIA`, and relays state to the popup and the content script. Refuse to
   start when the active tab is not on `meet.google.com`. Drive the toolbar
   badge from the current risk band.

3. `offscreen.html` + `offscreen.js` — the only place audio is touched. Take the
   stream id, call `getUserMedia` with
   `{audio:{mandatory:{chromeMediaSource:"tab",chromeMediaSourceId:streamId}}}`,
   run an `AudioContext` at its **native** rate, and open the WebSocket.

4. `worklet.js` — an `AudioWorkletProcessor` that posts each block of samples to
   the main thread and returns `true` so it survives silence.

5. `popup.html` / `popup.css` / `popup.js` — start/stop, the pairing code while
   consent is pending, call id, state, window count, an editable server URL
   persisted in `chrome.storage.local`, and the live band.

6. `content.js` / `overlay.css` — a fixed card in the top-right of the Meet page
   showing the same band, so nobody has to watch the popup during a call.
   Create it lazily, remove it when capture stops.

## Constraints that will bite you

- **A service worker cannot hold an `AudioContext` or a `MediaStream`.** That is
  the entire reason the offscreen document exists. Do not attempt capture in
  `background.js`.
- **`chrome.tabCapture.getMediaStreamId` requires a user gesture.** Call it in
  the handler for the popup's button click, not on a timer or on startup.
- **Tab capture mutes the tab.** You must `source.connect(ctx.destination)` in
  the offscreen document to re-play it, or the meeting goes silent the moment
  capture starts — which presents exactly like a crash.
- **Do not resample client-side.** Declare `ctx.sampleRate` in `start_mic_call`
  and let the server do it. Resampling in the browser would also degrade the
  playback path above.
- Use `AudioWorklet`, not the deprecated `ScriptProcessorNode`.
- Load the worklet with `chrome.runtime.getURL("worklet.js")`.

## Honesty requirement — do not skip this

Read `scoring_available` from `/api/telemetry` and **show no percentage and no
risk band while it is false.** Display "Monitoring — scoring unavailable"
instead. A mock scorer is currently in use; a fabricated number presented as a
verdict is the single worst failure mode this project has, and hiding it must be
enforced by the flag, not by remembering.

## Acceptance criteria

- Loads unpacked in Chrome 116+ with no manifest or console errors.
- Pressing Start on a Meet tab produces a pairing code within a second.
- **Audio sent before approval produces zero windows** — verify against
  `GET /api/telemetry`, where `ringbuffer.windows_emitted` must be 0.
- After approval, `windows_emitted` climbs by roughly two per second (4 s window,
  0.5 s hop).
- **The meeting is still audible throughout.** Check this explicitly.
- Stop writes an audit record to `outputs/calls/`.
- With no `--ckpt`, no percentage appears anywhere in the UI.
- `node --check` passes on every `.js` file.

## Then

Write `extension/README.md` covering install, use, the consent step,
troubleshooting, limitations, and what the bands mean. State plainly that tab
audio is mixed, so a score applies to a window and not to a named participant.

---

## If the extension already exists

It does — this prompt documents how it was built. To extend it, the highest-value
next steps in order:

1. **Reconnect handling.** The WebSocket has no retry. A server restart mid-call
   currently ends the session silently.
2. **A score sparkline in the popup.** The last 60 windows, drawn on a canvas.
   The data already arrives; nothing is kept.
3. **Rolling smoothing and hysteresis.** `background.js` shows the raw
   per-window score, so the band can flicker between Amber and Green. The server's
   own design calls for smoothing plus hysteresis before display.
4. **A capture indicator other participants can see.** Post a one-line chat
   message on start. Right now the overlay is visible only to the operator.
5. **Zoom and Teams.** Both are tab-capturable; only the URL match and the
   overlay anchor differ.
