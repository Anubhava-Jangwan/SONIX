# Lane 4 — Live Audio Ingest & Live Chart

**Owner:** Ken · **Status:** audio path built, scoring gated
**Target:** Stage 2 PoC, 1–2 Sept 2026 · **PS:** SIH26104

Lanes 1–3 (per Anubhav's handbook of 31 Aug) are: fix the model, honest numbers,
a demo that cannot fail. This lane is the fourth: **get real audio into the
pipeline live, and show what it is doing on screen.** It exists because the deck
claims "WebSocket audio streaming" and a live Green/Amber/Red band, and until now
nothing behind that claim was implemented.

---

## 1. What was actually there before this lane

Worth writing down, because the project state file described this module as
"fully functional in mock mode" and it was not:

| Component | Claimed | Actual |
|---|---|---|
| `RingBuffer.push()` | 4s sliding window | `pass` — no-op |
| `RingBuffer.get_emitted_windows()` | emits windows | `return []` |
| `WavFileSource.read()` | reads audio | `return None` |
| `VAD.is_speech()` | silence gate | `return True` always |
| `engine._embed_window()` | wav2vec2 1024-dim | `np.random.randn(1024)` |
| `MockScorer.score()` | deterministic mock | `RandomState(42).rand()` |
| Streamlit Live Calls tab | live call view | hardcoded literals |
| `server.py` consent | approve → LISTENING | approved from the wrong state; **all audio dropped** |
| `--mock` flag | opt into mock | `default=True` — real scoring unreachable |

Consequence: `/api/score-file` returned `windows_scored: 0` for every upload. The
example response in the project reference (68 windows, mean 0.32) was not
reachable from that code.

---

## 2. Scope

**In:** browser microphone capture, sample-rate normalisation, the sliding
window, the silence gate, consent gating on real audio, per-window telemetry, and
the live chart in the dashboard.

**Out:** training anything, the wav2vec2 embedding step, threshold calibration,
Asterisk/SIP. Those belong to Lanes 1–2 or to the wav2vec2 task below.

**Non-negotiable:** the dashboard shows **no risk band while no trained head is
loaded**. A mock number rendered as a verdict is the one thing that could lose the
jury Q&A, so the gate is enforced server-side (`scoring_available`), not by
remembering to hide it.

---

## 3. How live audio connects

```
Browser (Chrome/Edge)  http://localhost:8000/mic
  getUserMedia → AudioContext({sampleRate:16000}) → AudioWorklet "cap"
  Float32 frames → Int16 LE PCM → WebSocket BINARY frames
        │
        ▼
ws://localhost:8000/ws            ← the same socket the dashboard uses
  TEXT  {"type":"start_mic_call"} → creates Session, returns pairing code
  BINARY  raw int16 PCM           → pcm16_to_float32 → resample(sr→16k)
        │
        ▼
Session.push_audio()              ← DROPS everything until consent is approved
  RingBuffer  4.0s window / 0.5s hop (87.5% overlap)
  VAD         25ms frames, RMS + zero-crossing, ≥20% speech to pass
  window_log  every window recorded, passed or dropped
        │
        ▼
ScoringEngine   batches ≤8 windows across calls every 0.5s
  _embed_window()  ← STILL A STUB: returns random noise, no wav2vec2
  MockScorer / head.pt
        │
        ▼
GET /api/telemetry → Streamlit chart
```

Why the browser and not the OS microphone: it needs no native audio library on
Windows, it works from a phone on the same host, and it is the path the deck
already claims. `getUserMedia` requires a secure context — **`http://localhost`
counts, a LAN IP does not.** Demo from the machine running the server.

### New / rewritten files

| File | What it does |
|---|---|
| `realtime/miccapture.py` | **new** — serves the capture page at `/mic` |
| `realtime/ringbuffer.py` | rewritten — real sliding window, never zero-pads a short window |
| `realtime/vad.py` | rewritten — energy + ZCR gate, exposes `last_stats` for the chart |
| `realtime/resample.py` | rewritten — 48k/44.1k/8k → 16k, PCM16 codecs |
| `realtime/source.py` | rewritten — `WavFileSource` actually decodes; new `MicSource` |
| `realtime/session.py` | added `window_log` + `telemetry()` |
| `realtime/server.py` | mic call lifecycle, binary WS ingest, `/api/telemetry`, `/api/approve`, `/api/end-call`, consent fix, `--mock` fix |
| `realtime/live_ui.py` | Live Calls tab rebuilt on real data with two charts |

### New endpoints

```
GET  /mic                      microphone capture page
GET  /api/telemetry?limit=240  per-call windows, VAD stats, scores, scoring_available
POST /api/approve              {"call_id": "..."}   approve pairing
POST /api/end-call             {"call_id": "..."}   end a call
WS   /ws  TEXT  {"type":"start_mic_call","sample_rate":48000,"caller":"..."}
WS   /ws  BINARY  int16 LE PCM mono
```

---

## 4. The chart

Two charts, deliberately **not** on one pair of axes — a risk probability and an
audio level are different measures and sharing a y-axis would misrepresent both.

**Risk timeline** (hidden until a real head is loaded): P(AI voice) against
seconds into the call, 0–100% axis, with Green/Amber/Red bands shaded behind the
line and the two thresholds drawn as labelled dotted rules. One series, so the
line carries no legend. Band colours are reserved status colours and always ship
with a text label — never colour alone.

**Audio path** (always visible): speech-ratio per window, with windows the
silence gate dropped marked as grey ×. This is the chart that makes the demo
honest while scoring is off — it shows the capture, the windowing and the gate
genuinely working on live microphone audio, with nothing to fake.

Thresholds are currently **Amber ≥ 35%, Red ≥ 65%**, carried over from the clean
benchmark and captioned on screen as provisional. Lane 1 owns recalibration; when
it lands, change `AMBER_AT` / `RED_AT` in `live_ui.py`.

---

## 5. Requirements

**Runtime** — `pip install -r requirements.txt` (aiohttp, streamlit, plotly,
requests, scipy and pytest-asyncio were all missing and are now listed).

**Browser** — Chrome or Edge. Firefox does not honour a requested
`AudioContext` sample rate; it still works, but resampling then happens
server-side. Safari is untested.

**Host** — open the capture page as `http://localhost:8000/mic`. Over a LAN IP
the browser refuses microphone access without https.

**Hardware** — none beyond a microphone while scoring is off. Once the wav2vec2
step is implemented, a 4-second window on CPU is roughly a second of compute, so
a GPU is needed for genuine real-time; the batching engine already exists to
amortise that.

**Still blocked on Yugal** — `head.pt`, and the fp16 NaN fix. Note the cached
G.711 eval embeddings in this clone are **100% NaN** and the clean set is 2.5%
NaN, so fp16 is not a theoretical problem.

---

## 6. Phases

| # | Work | State |
|---|---|---|
| 0 | Audit — find what is stub vs real | done |
| 1 | Ring buffer, VAD, resampling | done |
| 2 | Mic page, binary WS ingest, consent fix | done |
| 3 | Telemetry endpoint + live chart | done |
| 4 | **wav2vec2 embedding step** — replace `_embed_window`'s random noise with the real frozen front-end, shared with `extract_embeddings.py` | **next, unblocked** |
| 5 | Wire `head.pt` via `--ckpt`, flip `scoring_available`, verify against fp32 embeddings | blocked on Yugal |
| 6 | Recalibrate Amber/Red on real recordings | blocked on Lane 1 |
| 7 | Replay mode — stream a validated wav through the same path, for a demo with no live-audio risk | not started |

Phase 4 is the one thing on the critical path that nobody is blocked on. Until it
is done, the numbers the engine produces are noise regardless of which head is
loaded — so it must land **before** `head.pt` arrives, not after.

Phase 7 is worth doing even if phases 5–6 land: `WavFileSource` now decodes
properly, so replay is a small piece of work and it is the difference between a
demo that depends on stage luck and one that does not.

---

## 7. How to run it

```bash
pip install -r requirements.txt

# terminal 1 — server
python -m realtime.server --mock --ws-port 8000 --mode webrtc

# terminal 2 — dashboard
streamlit run realtime/live_ui.py

# browser tab — capture (Chrome/Edge, must be localhost)
http://localhost:8000/mic
```

Press **Start capture**, allow the microphone, read the 6-digit pairing code off
the capture page, approve it in the dashboard's Live Calls tab. The audio-path
chart starts moving about four seconds later — the first window has to fill.

Once `head.pt` exists:

```bash
python -m realtime.server --ckpt outputs/models/head.pt --ws-port 8000 --mode webrtc
```

which flips `scoring_available` to true and reveals the risk timeline. No other
change is needed in the dashboard.

---

## 8. Verified

End-to-end against a running server, 14s of synthetic audio sent at 48 kHz with a
5-second silent stretch in the middle:

- audio sent before approval → **0 windows emitted** (consent gate holds)
- 48 kHz → 16 kHz, 14.0s in, 14.0s buffered
- 21 windows emitted from 14s at 4.0s/0.5s — arithmetically correct
- silence gate dropped 5 of 21
- `scoring_available: false` throughout, so the risk band stayed hidden

Not yet covered by `realtime/tests/`. Phase 4 should add cases for the ring
buffer's window arithmetic and the VAD boundary, which are the two places a
silent regression would be expensive.
