# SONIX — Phase 1 Audit

Scope: `realtime/`, `website/`. Every finding below cites a file and line.
Anything I could not prove from the code is labelled POTENTIAL or HARDENING,
never CONFIRMED.

> **Status:** the audit was a read-only pass; Phase 2a/2b has since landed.
> S-1, S-2, S-3, S-4 and S-6 are fixed — see §8 at the end for what changed and
> what deliberately did not. S-5 and S-7 remain open.

Not verified on this machine: the test suite. `pytest` is unavailable in the
audit environment, so the "before" baseline in §7 must be produced on the dev
box before any change lands.

---

## 1. Architecture — the real production flow

```
browser (website/index.html + script.js)
   |  GET  /                     static site            server.py:624
   |  GET  /api/models           head catalogue         server.py:610
   |  GET  /api/status           warm/mock state        server.py:609
   |  GET  /api/telemetry        per-call scores        server.py:611
   |  POST /api/score-file       file upload            server.py:608
   |  POST /api/approve          consent grant          server.py:612
   |  POST /api/end-call         teardown               server.py:613
   |  WS   /ws                   live control + PCM     server.py:607
   |  GET  /mic                  capture page           server.py:614
   v
Session (session.py)        consent state machine; push_audio() drops
                            every chunk until state == LISTENING
   v
ScoringEngine (engine.py)   batches windows, runs off the event loop
   v
checkpoint.py -> demo/score_file.py   (real)   |   mock.py (fallback, capped 0.5)
```

Capture sources feeding the same path: AudioSocket VOIP (`audiosocket.py`,
:5000), browser mic (`/mic` + `/ws` binary frames), Chrome extension, file
upload. All converge on `Session.push_audio()`.

### WebSocket protocol

| Direction | Type | Handler |
|---|---|---|
| in | `approve_pairing` | server.py:208 |
| in | `end_call` | server.py:212 |
| in | `start_mic_call` | server.py:216 |
| in | `ping` | server.py:219 |
| in | BINARY (int16 PCM) | server.py:229 |
| out | `server_state`, `call_state`, `scores`, `pairing_request`, `mic_call_started`, `pong`, `error` | `_broadcast` |

### Code classification

| Class | Files | Evidence |
|---|---|---|
| Production | `realtime/{server,session,engine,checkpoint,frontend,models,ringbuffer,vad,resample,source}.py`, `website/*` | on the request path above |
| Dev/demo only | `realtime/live_ui.py`, `demo/app.py` | server.py:617 comment: "internal/debug only" |
| Test-only | `realtime/tests/`, `realtime/selftest.py`, `realtime/make_dev_head.py` | no production importer |
| **Dead** | `realtime/consent.py` | 1 line, docstring only, no class or function |
| Legacy | `uidemo/SONIX_Suryansh_Demo_UI_v9/`, `src/extract_embeddings.py` | CODE_MAP.md:33, :120 |
| Fallback | `realtime/mock.py` | reachable only when no head resolves |

---

## 2. Security findings

### S-1 — Any client can approve anyone's consent

- **Severity:** CRITICAL
- **File / endpoint:** `realtime/server.py:186` `websocket_handler`, branch at :208
- **Finding:** `/ws` requires no authentication and validates no `Origin`.
  The `approve_pairing` branch takes a `call_id` straight from the message and
  calls `_on_pairing_approved(call_id)` with no check that this socket owns
  that session, and without requiring the 6-digit pairing code at all.
- **Attack scenario:** A user visits any web page while SONIX runs locally.
  That page opens `ws://localhost:8000/ws` — WebSockets are exempt from the
  same-origin policy, so no CORS header stops it. It sends
  `{"type":"approve_pairing","call_id":"mic_20260906T..."}`. Call IDs are
  timestamps (server.py:276), so they are guessable, and `_broadcast` hands
  them out to every connected client anyway.
- **Impact:** The consent gate is the project's central privacy guarantee
  (`CODE_MAP.md:72`). This bypasses it: audio is scored without the operator
  approving. Same shape at :212 for `end_call` — denial of service against any
  live session.
- **Evidence:** server.py:188-190 (`ws.prepare` with no origin check),
  :208-214 (no ownership check on `call_id`).
- **Fix:** Bind sockets to the sessions they created — `self.ws_calls[ws]` already
  exists (:298) and is the natural ownership record. Reject `approve_pairing` /
  `end_call` for any `call_id` not owned by that socket. Add an `Origin`
  allowlist at `ws.prepare()`. Require the pairing code in the approve message.
- **Status:** CONFIRMED. Fix is safe to implement immediately; it touches only the
  two WS branches and leaves the consent state machine untouched.

### S-2 — Pairing codes are not cryptographically random, and unlimited

- **Severity:** HIGH
- **File:** `realtime/pairing.py:8`
- **Finding:** `random.randint(100000, 999999)` uses Mersenne Twister, not a
  CSPRNG. There is no attempt counter and no lockout anywhere in the codebase.
  `expiry_sec` is stored on the manager but `generate()` never records an issue
  time, so the manager itself enforces no expiry — that lives on `Session`.
- **Attack scenario:** 900k-code space with no rate limit is brute-forceable;
  observing a few outputs narrows the PRNG state further.
- **Impact:** Session takeover where the code is the only barrier. Note S-1
  makes this moot today — the WS path does not ask for the code at all.
- **Evidence:** pairing.py, whole file (9 lines). No `secrets` import; no
  rate-limit call site found for pairing anywhere in `realtime/`.
- **Fix:** `secrets.randbelow(900000) + 100000`; track attempts per call_id and
  lock after ~5; record issue time so expiry is enforceable at the manager.
- **Status:** CONFIRMED. Safe immediately.

### S-3 — Telemetry is readable across sessions, and `limit` is unvalidated

- **Severity:** HIGH
- **File / endpoint:** `realtime/server.py:349` `http_telemetry_handler`
- **Finding:** `limit = int(request.query.get("limit", 240))` — `int()` on a
  non-numeric query string raises `ValueError`, returning a 500 with a traceback
  rather than a 400. No bound on the value either. No caller check: any client
  may read any `call_id`'s telemetry.
- **Attack scenario:** `GET /api/telemetry?limit=abc` → 500 + internals.
  `GET /api/telemetry?call_id=<other session>` → another user's scores,
  timestamps and caller identifier.
- **Impact:** Information disclosure across sessions; error-path leakage.
- **Evidence:** server.py:349-352.
- **Fix:** Parse with a helper that clamps to `1..1000` and returns 400 on
  garbage; scope `call_id` to the requesting socket/session.
- **Status:** CONFIRMED. Safe immediately.

### S-4 — Unvalidated `sample_rate` kills the socket

- **Severity:** MEDIUM
- **File:** `realtime/server.py:278` `_start_mic_call`
- **Finding:** `int(data.get("sample_rate") or TARGET_SR)` with no type or range
  check. The enclosing `try` at :205 catches only `json.JSONDecodeError`, so a
  `ValueError` from `int("abc")` escapes the `async for` and tears down the
  connection. A huge or zero value reaches `resample()` (:239).
- **Attack scenario:** `{"type":"start_mic_call","sample_rate":"x"}` → socket
  dies. `"sample_rate": 10_000_000` → oversized resample allocation.
- **Impact:** DoS of the socket; potential memory spike.
- **Evidence:** server.py:278, :237-239; except clause at :226.
- **Fix:** Validate against an allowlist (8000/16000/44100/48000); broaden the
  handler's except to return a typed `error` frame instead of dying.
- **Status:** CONFIRMED. Safe immediately.

### S-5 — Broadcast fans every session's state to every client

- **Severity:** MEDIUM
- **File:** `realtime/server.py:170` `_broadcast`
- **Finding:** Sends to all of `self.ws_clients` with no per-session filtering.
- **Impact:** Any connected socket learns every active `call_id`, caller label
  and score stream — which is also what makes S-1's call_id guessing trivial.
- **Evidence:** server.py:170-184; no recipient filter.
- **Fix:** Route per-session frames only to sockets owning that session; keep
  aggregate `server_state` global.
- **Status:** CONFIRMED. Safe, but touches the dashboard's data flow — needs a
  `live_ui.py` check first (Suryansh's lane).

### S-6 — No Origin/CORS policy on state-changing HTTP endpoints

- **Severity:** MEDIUM
- **File:** `realtime/server.py:606-614`
- **Finding:** No CORS middleware and no origin check on `POST /api/approve`,
  `/api/end-call`, `/api/score-file`. Simple `POST` with
  `multipart/form-data` is not preflighted, so a cross-origin page can submit
  these directly.
- **Impact:** CSRF against consent and teardown, and forced inference load.
- **Evidence:** no `aiohttp_cors` import, no `Origin` read anywhere in the file.
- **Fix:** Require `Origin` to match the served host on state-changing routes.
- **Status:** CONFIRMED (absence of control is directly observable). Safe.

### S-7 — No rate limiting on inference paths

- **Severity:** MEDIUM
- **Finding:** `/api/score-file` and the WS binary path have a size cap
  (`MAX_UPLOAD_BYTES = 256 MB`, server.py:29) and `max_calls` (default 4), but no
  per-IP or per-session request-rate limit.
- **Impact:** Repeated 256 MB uploads saturate CPU/GPU and disk.
- **Evidence:** server.py:29, :606; no limiter in the file.
- **Status:** HARDENING RECOMMENDATION — needs a limit that does not break
  legitimate realtime use, so it should not be bolted on blind.

### Endpoint matrix

| Endpoint | AuthN | AuthZ | Protected? | Direct-call risk |
|---|---|---|---|---|
| `GET /ws` | none | none | no | S-1, S-4 |
| `POST /api/approve` | none | none | no | consent bypass |
| `POST /api/end-call` | none | none | no | DoS live session |
| `GET /api/telemetry` | none | none | no | S-3 cross-session read |
| `POST /api/score-file` | none | n/a | size cap only | S-7 |
| `GET /api/models`, `/api/status` | none | n/a | read-only | INFO — exposes absolute host paths (`resolved_path`) |
| `GET /mic`, `/` | none | n/a | static | — |

**There is no authentication in this application.** A pairing code is a consent
prompt, not authentication — and per S-1 the WS path does not even check it. For
a localhost demo that is a defensible posture; it must not be exposed on
`0.0.0.0` (the current default, server.py:694) on an untrusted network.

---

## 3. Frontend / SPA findings

- `website/` is a **single-document, anchor-navigated page**, not an SPA. There
  are no routes to convert — `#home`, `#results`, `#pipeline` are in-page
  anchors. A router would add machinery for navigation that already costs
  nothing. **Recommendation: do not build a router.** The real SPA-adjacent
  wins here are lifecycle ones, listed next.
- `script.js` is **1036 lines in one file** with no module boundaries — the
  brief's "separation between routing / state / API client / WS client / UI /
  notifications / utilities" is the genuine gap, independent of routing.
- `script.js:293` detects dark mode by string-comparing `--bg` to `"#ffffff"`.
  Any palette edit silently breaks the hero canvas. Load-bearing colour value.
- Threshold constants are duplicated in at least four places — `live_ui.py`
  (`AMBER_AT`/`RED_AT`), `miccapture.py:117` (JS copy, with a comment saying
  they must match), `script.js`, and the extension. `test_miccapture.py` pins
  two of them; nothing pins the rest.

## 4. Threshold reconciliation

| Source | Amber | Red | Authoritative? |
|---|---|---|---|
| `realtime/live_ui.py`, `/mic`, extension | 0.35 | 0.65 | **yes** — CODE_MAP.md:22 |
| `demo/app.py` slider defaults | 0.10 | 0.90 | no — UI slider defaults |
| `demo/test_suryansh.py` | 0.45 | 0.70 | no — test fixture |

The three are not in conflict in the way the brief assumes: only the first is a
production band. Consolidation means one exported constant that `live_ui.py`,
`miccapture.py`, `script.js` and the extension all read, plus a test asserting
they agree.

---

## 5. "0 / N windows scored" — root-caused, two separate defects

Both are fixed. Neither was the warm-up race originally suspected.

**Defect A — the silence gate could sit above the signal.**
`_adaptive_vad_floor` (server.py:381) clamped its lower bound to `0.0006`. For a
clip below roughly -65 dBFS that lands *above* the clip's own speech level, so
every frame fails the energy test. Measured on `testcall.wav` attenuated to that
level: p90 frame RMS `0.000564`, clamped floor `0.0006`, **0/41 windows**. Fixed
by capping the floor at half the measured speech level; after, 41/41. Safe
because the ZCR ceiling rejects dead audio independently of energy (digital
silence `zcr=0.000`, dither and room tone `zcr≈0.50`, all still rejected with
the energy gate fully open).

**Defect B — `engine._head_forward` used `torch` without importing it.**
`realtime/engine.py` imported asyncio, logging, pathlib, typing, numpy and the
model registry — not torch — yet `_head_forward` (line 250) called
`torch.no_grad()`, `torch.from_numpy()`, `torch.softmax()` and `torch.sigmoid()`.
Every real batch raised `NameError`. Mock mode returns at `_score_windows`
line 228, before reaching it, so the entire test suite stayed green while real
scoring had never worked.

**Defect B was invisible because of a third problem:** `ScoringEngine.run()`
caught only `asyncio.CancelledError`, so that `NameError` escaped the `while`
loop and killed the scoring task outright. The server kept serving, sessions
kept accepting audio, and uploads kept reporting `0 / N` with nothing logged.
`run()` now catches per-batch failures, logs them with a traceback, counts them
in `failed_batches`, and continues — so the next failure of this class is
visible in `/api/status` instead of silent.

Diagnosis used `realtime/diagnose_upload.py` (new): it runs
`WavFileSource → RingBuffer → adaptive floor → VAD` against a file with no
server involved. On the user's `output.wav`: 24.64s, 42 windows, **32 passed the
gate** — which ruled out the VAD and pointed at the scorer.

---

## 6. Proposed phase order

Ordered by risk-adjusted value, not by the brief's numbering.

| Phase | Work | Risk |
|---|---|---|
| **2a** | S-1, S-2, S-3, S-4 — socket/session ownership, `secrets`, input validation | low, contained |
| **2b** | Delete `realtime/consent.py` (proven dead); centralise thresholds + add the drift test | low |
| **3** | S-5, S-6 — per-session broadcast, origin checks | medium — touches `live_ui.py` |
| **4** | Split `script.js` into modules; lifecycle cleanup (listeners, timers, single WS) | medium |
| **5** | S-7 rate limiting; upload hardening | needs load measurement first |
| **6** | Legacy removal (`uidemo/`, `src/extract_embeddings.py`) | needs Yugal + Suryansh sign-off |

## 7. What has NOT been done

No code has been modified. No performance numbers are quoted — none were
measured. No test baseline exists yet; run it on the dev box first:

```powershell
python -m pytest realtime/tests -q
```

Lane note (`CLAUDE.md`): `realtime/` UI, `live_ui.py` and `miccapture.py` are
Suryansh's; ML under `src/` is Yugal's. Phases 3, 4 and 6 cross those lanes and
need their sign-off before landing.

---

## 8. Phase 2a / 2b — what landed

| ID | Fix | File |
|---|---|---|
| S-1 | WS `approve_pairing` branch **deleted**, not hardened — no client ever used it (`live_ui.py:715` and `script.js` both POST `/api/approve`). Consent now has exactly one entry point. | `server.py` |
| S-1 | WS `end_call` bound to `self.ws_calls[ws]`; a socket can only end its own call. Every real caller already sent its own id, so no flow changed. | `server.py` |
| S-1/S-6 | `_origin_allowed()` on `/ws`, `/api/approve`, `/api/end-call`, `/api/score-file`. Requests with no `Origin` (curl, Asterisk, tests) still pass; `chrome-extension://` allowed; extra origins via `SONIX_ALLOWED_ORIGINS`. | `server.py` |
| S-2 | `secrets.randbelow` replaces `random.randint`; codes are recorded, expire, are single-use, and burn a 5-attempt budget. `check()` uses `compare_digest`. | `pairing.py` |
| S-3 | `limit` parsed safely (400, not a 500 traceback) and clamped to `MAX_TELEMETRY_LIMIT`. | `server.py` |
| S-4 | `sample_rate` validated against `ALLOWED_SAMPLE_RATES`; the message loop now catches non-JSON errors and replies instead of dying. | `server.py` |
| 2b | `realtime/consent.py` — 35 bytes, docstring only, zero importers repo-wide. **Delete pending** (`rm` was blocked from the audit sandbox). | — |
| 2b | `realtime/thresholds.py` — one definition of 0.35/0.65 plus `band_for()`. | new |
| 2b | `test_thresholds.py` pins the mic page, website and extension against it. | new |
| VAD | `_adaptive_vad_floor` no longer returns a floor above the clip's own speech level — the cause of "0 / N windows scored". | `server.py` |

### Deliberately NOT done

- **Telemetry `call_id` scoping (part of S-3).** `live_ui.py` is an operator
  dashboard that legitimately reads calls it did not create. Scoping per socket
  would break it. Needs a real operator identity first.
- **Ownership binding on `POST /api/approve`.** Same reason — `live_ui.py:715`
  approves third-party calls by design. The origin check is the control that
  fits the actual threat (a page the operator has open), without breaking the
  dashboard.
- **S-5, S-7.** Broadcast scoping touches the dashboard's data flow; rate
  limiting needs load measurement first. Both still open.

### Verification

`realtime/tests/test_adaptive_vad_floor.py` and `test_thresholds.py` are new;
`test_miccapture.py` now imports `realtime.thresholds` instead of `live_ui`'s
sliders. All touched files pass `py_compile`, and 40 assertions covering S-1
through S-6 and the band boundaries were run directly. **`pytest` itself was not
available in the audit environment — run the suite on the dev box before
merging.**
