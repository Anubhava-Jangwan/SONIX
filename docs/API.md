# SONIX API — integration reference

SONIX is a **security layer**, not an application. It runs as a self-hosted service inside a
bank's, contact centre's or telecom operator's own infrastructure. Their systems call it; call
audio never leaves their network.

The Streamlit dashboard and the Chrome extension are both *clients* of this API. A production
integration is code, not a UI.

```
┌─────────────────────┐
│ Core banking /      │
│ contact centre /    │──── audio stream ────┐
│ VoIP / collab tool  │                      │
└─────────────────────┘                      ▼
         ▲                          ┌──────────────────┐
         │                          │  SONIX service   │
         └──── risk score ──────────│  (self-hosted)   │
              GREEN/AMBER/RED       └──────────────────┘
                                       ▲            ▲
                          ┌────────────┘            └────────────┐
                    Operator console            Chrome extension
                    (reference client)          (reference client)
```

---

## Quick start

```python
from sonix_sdk import SonixClient

sonix = SonixClient("http://sonix.internal:8000")

result = sonix.score_file("call_88213.wav")
if result.band == "RED":
    hold_transaction(reason=f"voice-clone risk {result.mean:.0%}")
```

Three lines to integrate. That is the whole point of the architecture.

---

## Endpoints

### `GET /api/status`
Service health and throughput.
```json
{ "status": "ok", "mode": "voip", "active_calls": 2, "max_calls": 4,
  "engine_stats": { "total_windows_scored": 18412, "warm": true },
  "ws_clients": 1, "timestamp": "2026-09-02T23:41:02" }
```

### `GET /api/models`
Which detector models this service can score with, and whether the front-end is warm.
```json
{ "mock": false, "warm": true, "default": "full_ho",
  "models": [ { "key": "baseline", "label": "Baseline", "exists": true },
              { "key": "full_ho",  "label": "Full + Indic", "exists": true } ] }
```
Multiple models behind one endpoint matters operationally: a new cloning tool appears, we
retrain a 262k-parameter head in minutes and add it here — no redeployment of the 300M
front-end, no downtime.

### `POST /api/score-file`
Score a recording. Multipart form.

| Field | Meaning |
|---|---|
| `file` | audio (wav/mp3/flac/ogg) |
| `model` | optional — detector to use |
| `vad` | `auto` (default) / `strict` / `off` — silence gate |
| `wait` | `1` to block for the full result; omit to stream |

Returns immediately with a `call_id` and the number of windows to expect; poll
`/api/telemetry` as scores land. Use for post-call review, dispute investigation, batch audit
of recorded lines.

### `GET /api/telemetry?call_id=…`
Per-window state for a call in progress: every score recorded so far, ring-buffer stats, and
what the silence gate accepted or dropped. This is what drives a live risk display.

### `WS /ws`
Live streaming. Client opens a socket, sends `{"type":"start_mic_call", "sample_rate":16000}`,
then pushes raw 16-bit PCM frames as binary messages. The server pushes back:
```json
{ "type": "scores", "data": { "<call_id>": { "window_idx": 41, "score": 0.0312 } } }
```
`realtime/miccapture.py` is a working reference implementation.

### `POST /api/approve` · `POST /api/end-call`
Consent approval and call teardown. **A call is not scored until consent is approved** —
see below.

---

## Consent, privacy and audit

The problem statement asks for a privacy and compliance module. This is ours.

- **Consent gate.** Every call enters `CONSENT_PENDING` and is assigned a pairing code. Audio
  is buffered but **not scored** until `/api/approve` is called. This is enforced in the state
  machine, not by convention — `push_audio()` drops frames in any other state.
- **Immutable audit trail.** Every consent event, state change and call boundary is appended
  to a per-call record with a timestamp, written to disk at call end.
- **No audio retention.** Scores and features only. The ring buffer holds at most 4.5 seconds
  of audio in memory and is discarded.
- **Silence gate.** Near-silent windows are dropped before they reach the model at all.

Maps directly onto the PS requirements for *"minimal retention of voice recordings"* and
*"anonymization or feature-only logging"*.

---

## Deployment

**Self-hosted container.** The service runs inside the customer's own infrastructure, so audio
never crosses a network boundary they do not control. One process serves multiple concurrent
calls — the engine batches windows across sessions against a single resident front-end.

**Configuration per workflow.** Amber and Red thresholds are set by the operator and can differ
by risk scenario: stricter for high-value transfers or privileged-access approvals than for
general enquiries.

**Scaling.** The 300M front-end is the memory cost and it is loaded once. Adding a detector
model costs ~1 MB and no extra GPU memory. Horizontal scaling is stateless above the session
layer.

---

## What is not built yet — stated honestly

| PS item | Status |
|---|---|
| gRPC transport | REST + WebSocket today; gRPC is a transport addition, not an architecture change |
| Packaged SDKs (pip/npm/Java) | `sonix_sdk.py` is a working reference client; not yet published |
| Cross-session consistency vs enrolled samples | Not built — needs speaker enrolment |
| Contextual enrichment (call origin, transaction context) | Not built — the risk score is designed as a clean input to a fraud engine |
| SMS / email / in-app alert fan-out | Console alerts only |
| On-device / edge inference | Server-side; edge needs a distilled model |

Presenting these as a scoped roadmap is deliberate. A prototype that claims every line of the
problem statement is a prototype nobody believes.
