# 03 — API contract

Base URL `http://localhost:8000`. All responses `application/json`, UTF-8, `orjson`-encoded.

Two rules that govern everything below:

1. **Existing routes are frozen.** `GET /ws`, `POST /api/score-file`, `GET /api/status`,
   `GET /api/models`, `GET /api/telemetry`, `POST /api/approve`, `POST /api/end-call`, `GET /mic`
   and the `website/` static mount at `/` keep their current behaviour, status codes and response
   shapes exactly. `website/script.js`, `extension/` and `realtime/miccapture.py` are live clients.
   `realtime/tests/test_existing_routes.py` exists to catch you breaking one.
2. **Errors are structured.** Never a bare 500, never an HTML error page, never a stack trace in
   a user-facing `message`.

```json
{ "error": { "code": "unknown_model", "message": "No model registered under key 'head_v9'.",
             "detail": { "known_keys": ["baseline", "v3", "full_ho"] } } }
```

Codes: `bad_request` · `unknown_model` · `checkpoint_missing` · `unsupported_audio` ·
`file_too_large` · `job_not_found` · `capacity` · `scoring_unavailable` · `internal`.

---

## `GET /api/dashboard/summary`

One call that fills the landing view. This is what the SPA polls; it must be cheap enough to
poll at 2 s without touching disk or the GPU. Serve it from in-memory state that the engine and
job store already maintain.

```json
{
  "server": { "mode": "webrtc", "uptime_s": 4182, "version": "…git sha…" },
  "engine": {
    "warm": true, "warming": false, "mock": false, "scoring_available": true,
    "device": "cuda", "default_model": "v3", "loaded_models": ["v3", "baseline"],
    "total_windows_scored": 18412, "total_batches": 2611, "failed_batches": 0,
    "avg_windows_per_batch": 7.05, "errors": 0, "last_error": null,
    "max_batch_size": 8, "batch_interval": 0.1
  },
  "calls": { "active": 2, "max": 4, "pending_consent": 1 },
  "jobs": { "queued": 0, "running": 1, "done_24h": 14, "failed_24h": 0 },
  "thresholds": { "amber_at": 0.35, "red_at": 0.65,
                  "note": "provisional, uncalibrated — realtime/thresholds.py" },
  "audio": { "sample_rate": 16000, "window_sec": 4.0, "hop_sec": 0.5 }
}
```

`engine` mirrors `ScoringEngine.get_stats()` plus device and mock state. `thresholds` comes from
`realtime/thresholds.py` — the client must read it from here rather than hardcoding.

`scoring_available` is `false` when no `head*.pt` was found and the server fell back to
`realtime/mock.py`. **The UI's most important job is making that state impossible to miss** —
mock scores are random and capped at 0.5, so a mock demo looks reassuringly green while
detecting nothing.

Note that `scoring_available` and `scoring_synthetic` are already served by the existing
`GET /api/status`, and the `/mic` page gates its verdict and chart on them. Read them from there
for the same semantics rather than deriving a second, subtly different definition here — and do
not remove them from `/api/status` while consolidating.

---

## `POST /api/compare`

Multi-model A/B on one clip, sharing a single embedding pass.

**Request** — `multipart/form-data`

| Field | Required | Value |
|---|---|---|
| `file` | yes | wav / mp3 / flac / ogg, ≤ 50 MB |
| `models` | yes | comma-separated registry keys, 1–4, e.g. `v3,baseline,full_ho` |
| `vad` | no | `auto` (default) · `strict` · `off` |
| `max_windows` | no | cap for very long files; default 600 (5 min at 0.5 s hop) |

**Response `200`**

```json
{
  "job_id": "cmp_20260920T141233_a4f1",
  "filename": "call_88213.wav",
  "duration_s": 20.5,
  "windows": 34,
  "windows_dropped_by_vad": 2,
  "embed_ms": 412,
  "shared_embedding_pass": true,
  "results": {
    "v3": {
      "key": "v3", "label": "SONIX v3 (multilingual)",
      "scores": [0.012, 0.009, 0.031, "…"],
      "mean": 0.041, "median": 0.020, "p95": 0.18, "max": 0.31,
      "band": "Green", "head_ms": 12
    },
    "baseline": { "key": "baseline", "label": "Baseline", "scores": ["…"],
                  "mean": 0.58, "median": 0.61, "p95": 0.93, "max": 0.97,
                  "band": "Amber", "head_ms": 11 }
  },
  "note": "one embedding pass shared across 2 heads"
}
```

`embed_ms` is reported once; `head_ms` per model. The UI shows that split because it is the
evidence for the "a new detector costs ~1 MB and no extra GPU memory" claim.

**Errors**

- Unknown key → `400 unknown_model`, listing known keys.
- Key valid but checkpoint absent on this machine → `400 checkpoint_missing` naming the key and
  expected path. **Never silently substitute another head** — a real number under the wrong
  model's name is the worst failure mode in this system.
- No `head*.pt` at all → `409 scoring_unavailable`.

`band` is computed with `realtime.thresholds.band_for()` on the **mean** score. State that on
screen; a per-window max would read Red on almost any clip given 87.5% overlap.

---

## Jobs

### `POST /api/jobs` → `202`

```json
{ "kind": "batch", "models": ["v3"], "params": { "dir": "sonix_real/sonix_real/fake", "vad": "auto" } }
```

Returns immediately:

```json
{ "job_id": "job_20260920T141233_9c2e", "state": "queued", "position": 0 }
```

`kind` is `score` (one file), `compare` (one file, several heads) or `batch` (a directory or a
list of uploaded files). Live calls always take priority; `07_TESTING.md` has the test that
proves it.

### `GET /api/jobs?limit=25&offset=0&kind=&state=`

```json
{ "total": 132, "limit": 25, "offset": 0,
  "jobs": [ { "job_id": "…", "kind": "batch", "state": "running",
              "created_at": "2026-09-20T14:12:33Z", "started_at": "2026-09-20T14:12:33Z",
              "finished_at": null, "progress": { "done": 41, "total": 120 },
              "models": ["v3"], "error": null } ] }
```

List responses omit `result` and `params` — those can be large. Fetch them per job.

### `GET /api/jobs/{job_id}`

The full record including `result` and, on failure, `error` as the **complete traceback string**.
`CLAUDE.md` rule 2: a paraphrased failure costs more time than it saves. The UI renders it in a
monospace block with a copy button.

### `DELETE /api/jobs/{job_id}`

Cancels a `queued` or `running` job → `200 {"job_id": "…", "state": "cancelled"}`.
A `done` or `failed` job → `409`. Unknown id → `404 job_not_found`.

---

## `GET /api/runs?limit=50&offset=0`

Flattened history for the operator's history table: completed jobs from `outputs/jobs/` joined
with per-call records from `outputs/calls/`, newest first.

```json
{ "total": 340, "runs": [
  { "run_id": "upload_20260920T141233_487396", "source": "call",
    "kind": "upload", "model": "v3", "started_at": "…", "duration_s": 20.5,
    "windows": 34, "mean_score": 0.041, "band": "Green" },
  { "run_id": "job_20260920T141233_9c2e", "source": "job",
    "kind": "batch", "model": "v3", "started_at": "…", "files": 120,
    "mean_score": 0.72, "band": "Red" } ] }
```

Scan `outputs/` once at startup and keep the index in memory; re-scanning per request will be
the slowest endpoint in the app by an order of magnitude within a week of demo runs.

---

## WebSocket `/ws`

**The existing frame shapes do not change.** You are adding frame types, not editing ones that
are already parsed by three clients.

Existing (frozen):

```json
{ "type": "scores", "data": { "<call_id>": { "window_idx": 41, "score": 0.0312 } } }
```

New:

```json
{ "type": "job",    "data": { "job_id": "…", "state": "running",
                              "progress": { "done": 41, "total": 120 } } }
{ "type": "engine", "data": { "warm": true, "active_calls": 2, "queued_jobs": 0 } }
```

Client rules:

- One shared socket for the whole SPA. Not one per component.
- Unknown `type` values are ignored silently — that is how this stays forward-compatible.
- Reconnect with exponential backoff capped at 10 s, and show connection state in the header.
- `engine` frames are a push-based replacement for polling `summary`, not an addition to it.
  If you subscribe to `engine`, drop the poll interval; running both doubles the traffic for
  nothing.

---

## TypeScript types

`dashboard/src/lib/types.ts` mirrors this document exactly — same field names, same optionality.
When you change a response shape, change it in three places in the same commit: the handler,
this file, and `types.ts`. A contract that drifts from its implementation is worse than no
contract, because people trust it.
