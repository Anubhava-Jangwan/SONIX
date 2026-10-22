# Build prompt — SONIX Dashboard (new surface)

Copy everything below the line into a fresh coding-agent session opened at the repo root
(`B:\SIH_sonix`). It is written to be pasted whole. It assumes the agent can read
`CLAUDE.md`, `CODE_MAP.md`, `PROJECT_STATE.txt`, `docs/API.md` and `docs/RESULTS_TABLE.md`
from the repo.

---

## Task

Build a **new** SONIX dashboard: a React SPA in `dashboard/` plus new API endpoints on the
existing realtime server. This is an **additional surface**, not a replacement.

Read `CLAUDE.md`, `CODE_MAP.md` and `docs/API.md` before writing any code. Read
`docs/RESULTS_TABLE.md` before putting any number on screen.

## Specs — read these, they answer most of your questions

The full specification lives in `docs/dashboard/`. Read it in order before Phase 0; this prompt
is the brief, those files are the contract.

| File | Settles |
|---|---|
| `docs/dashboard/README.md` | Index and repo context |
| `docs/dashboard/01_TECH_STACK.md` | Every dependency, pinned, with the banned list and why |
| `docs/dashboard/02_STRUCTURE.md` | Full directory tree; the exact three-line `server.py` diff |
| `docs/dashboard/03_API_CONTRACT.md` | Every endpoint and WS frame, request and response shapes |
| `docs/dashboard/04_PERFORMANCE.md` | **The latency budget and the rules that keep it — read before writing any handler** |
| `docs/dashboard/05_UI_SPEC.md` | Demo and operator mode, screen by screen, with design tokens |
| `docs/dashboard/06_BUILD_ORDER.md` | Six phases, each with a gate you must actually run |
| `docs/dashboard/07_TESTING.md` | Test matrix, perf benchmarks, definition of done |

## Skills — use these, they are installed

**`ponytail`** — invoke it for **every** coding step in this task, backend and frontend. It is
lazy-senior-dev mode: YAGNI, standard library first, no unrequested abstractions. This build has
an obvious failure mode — a job queue that grows a broker, a compare endpoint that grows a
plugin system, a frontend that grows three state libraries — and `ponytail` is the counterweight.
`01_TECH_STACK.md`'s rejected-dependency table is the same instinct written down; when you want
to add something not on the approved list, `ponytail` is the test it has to pass.

**`ponytail-review`** — run it on the full diff at Phase 6, before you report done. Delete what
it finds. Report what you deleted.

**`ui-ux-pro-max`** — invoke it before writing any component, and again before the Phase 6 polish
pass. It carries the palettes, font pairings, chart conventions, UX guidelines and accessibility
rules that `05_UI_SPEC.md` assumes rather than restates. `05_UI_SPEC.md` tells you *what* each
screen must show and must never claim; `ui-ux-pro-max` tells you how to make it look like a
product rather than a student project. Where they conflict, `05_UI_SPEC.md` wins on content and
honesty rules — never let a design suggestion turn "not yet measured" into a nice-looking number.

Related skills in the same family, if the pass calls for them: `ui-ux-pro-max:ui-styling` for
the shadcn/Tailwind component work, and `ui-ux-pro-max:design-system` if the token layer in
`05_UI_SPEC.md` needs extending. Do not pull in the logo, banner or slide-generation skills —
this is an application, not a brand exercise.

### Do not touch

- `website/` — the existing static marketing site. Leave it byte-identical.
- `demo/` and `uidemo/` — Suryansh's Streamlit demo. Read `demo/score_file.py` for the
  contract; do not edit it.
- `realtime/live_ui.py` — the existing Streamlit operator dashboard. Untouched.
- `src/`, root `extract_embeddings.py`, `make_codec.py` — Yugal's and Anubhav's lanes.
- `realtime/thresholds.py` — import from it, never redefine `AMBER_AT` / `RED_AT`.

The only existing file you may modify is `realtime/server.py`, and only to **add** routes and
a static mount. Every existing route must keep its current behaviour and response shape —
`website/script.js`, `extension/` and `realtime/miccapture.py` are live clients of it.

---

## What SONIX is (so the UI tells the truth)

```
4s audio @16kHz mono
  → frozen wav2vec2 XLS-R 300M   (never trained, loaded once, the memory cost)
  → 1024-dim mean-pooled embedding
  → trained MLP head ~300k params (the only trained part; outputs/models/head*.pt)
  → risk score in [0,1] = P(synthetic)
```

Bands come from `realtime/thresholds.py` — Green `<0.35`, Amber `0.35–0.65`, Red `≥0.65`.
They are **provisional and uncalibrated**; the UI must say so wherever it shows a band legend.

The architectural claim the dashboard exists to make visible: the 300M front-end is frozen and
resident, so adding a detector for a new cloning tool costs a ~1 MB head and zero extra GPU
memory. Multi-model compare is the demonstration of that claim, not a gimmick.

The metric is **EER**, always written `"X% on <split>"`, never a bare number.

---

## Part 1 — Backend: extend `realtime/server.py`

The server is aiohttp, launched as
`python -m realtime.server --ws-port 8000 --mode webrtc [--ckpt ...] [--max-batch-size 4]`.
Existing routes (do not change): `GET /ws`, `POST /api/score-file`, `GET /api/status`,
`GET /api/models`, `GET /api/telemetry`, `POST /api/approve`, `POST /api/end-call`,
`GET /mic`, and the static mount of `website/` at `/`.

Reuse the existing machinery — `ScoringEngine` (`realtime/engine.py`) for batching and
`ensure_model(key)` / `model_catalogue()` for heads, `realtime/models.py` `REGISTRY` +
`catalogue()` for the model list, `Session` (`realtime/session.py`) for the consent state
machine. Do not build a second inference path and do not load a second copy of the front-end.

### New endpoints

**`POST /api/compare`** — multi-model A/B on one clip.
Multipart: `file`, `models` (comma-separated registry keys), `vad` (`auto`|`strict`|`off`).
Embed each 4s window **once**, then run every requested head over the same embedding matrix.
This is the whole point — say so in the response payload and surface it in the UI.

```json
{ "job_id": "cmp_...", "windows": 41, "embed_ms": 1840,
  "results": { "v3":      { "label": "SONIX v3 (multilingual)", "scores": [...],
                            "mean": 0.041, "median": 0.020, "max": 0.31,
                            "band": "Green", "head_ms": 12 },
               "baseline":{ "label": "Baseline", "scores": [...], ... } },
  "note": "one embedding pass shared across 2 heads" }
```

Reject unknown keys and keys whose checkpoint is missing with a 400 naming the key — never
silently substitute a different head.

**`GET /api/jobs` · `GET /api/jobs/{job_id}` · `POST /api/jobs` · `DELETE /api/jobs/{job_id}`**
— background job queue and run history.

- `POST /api/jobs` accepts `{ "kind": "score" | "compare" | "batch", ... }`. `batch` takes a
  directory or a list of uploaded files. Returns `202` with a `job_id` immediately.
- Jobs run on the existing engine; cap concurrency so batch work cannot starve live calls
  (respect `--max-calls`). Live sessions take priority.
- Job record: `job_id`, `kind`, `state` (`queued`|`running`|`done`|`failed`|`cancelled`),
  `created_at`, `started_at`, `finished_at`, `progress` (`{done, total}`), `models`, `params`,
  `error` (**full traceback string, not a paraphrase** — team rule 2 in `CLAUDE.md`), `result`.
- Persist one JSON per job under `outputs/jobs/`, same spirit as `outputs/calls/`. Write
  atomically (temp file + rename) so a half-written record never loads. Reload the index on
  startup so history survives a restart.
- `GET /api/jobs` supports `limit`, `offset`, `kind`, `state`.

**`GET /api/runs`** — flattened history for the dashboard's history table: past jobs joined
with the existing per-call records in `outputs/calls/*.json`, newest first, paginated.

**`GET /api/dashboard/summary`** — one call that fills the landing view: engine stats
(`get_stats()`), active call count, model catalogue with `exists` flags, queue depth, thresholds
from `realtime/thresholds.py`, and whether real scoring is available (`scoring_available` —
false when no `head*.pt` is on disk and the server fell back to `mock.py`).

**`WS /ws` additions** — add job progress events alongside the existing `scores` frames:
`{"type": "job", "data": {"job_id": ..., "state": ..., "progress": {...}}}`. Do not alter the
existing `scores` frame shape; `website/script.js` and the extension parse it today.

### Backend constraints

- Never block the aiohttp event loop. Inference goes through the engine's existing
  off-loop execution; file I/O for job records goes through a thread executor.
- No audio retention. Uploaded files are scored and deleted; only scores, window stats and
  metadata persist. The privacy story in `docs/API.md` has to stay literally true.
- The consent gate is real: a live call is not scored until `/api/approve`. Job/batch scoring of
  uploaded recordings is a different path — keep it that way and label it that way in the UI.
- Cap upload size and validate audio type. Return structured JSON errors
  (`{"error": {"code": ..., "message": ...}}`), never a bare 500 page.
- Add pytest tests under `realtime/tests/` mirroring the existing style: compare endpoint shares
  one embedding pass, job lifecycle reaches `done`, cancel works, unknown model key 400s, job
  records survive a simulated restart, and **a regression test asserting every pre-existing
  route still returns its documented shape**.

---

## Part 2 — Frontend: `dashboard/`

**Stack:** React 18 + Vite + TypeScript + Tailwind + shadcn/ui. Recharts for charts.
`dashboard/` is self-contained: its own `package.json`, no changes to the repo-root
`package.json`. `npm run build` emits `dashboard/dist/`, which `realtime/server.py` mounts at
`/dashboard`. In dev, Vite proxies `/api` and `/ws` to `localhost:8000`.

**One dashboard, two modes**, toggled in the header and persisted in `localStorage`:

### Demo mode — for judges, five minutes, no training needed

1. **Hero / live band.** Big current band, current score, the sentence explaining what the
   band means. Fed by the WebSocket mic path (`start_mic_call` → pairing code → `/api/approve`
   → `scores` frames) — the same flow `website/script.js` already implements; read it rather
   than reinventing the handshake.
2. **Pipeline visual.** The eight stages — capture → resample → consent gate → 4s windowing →
   silence gate → frozen embedding → trained head → band + audit — lighting up as audio moves
   through them, driven by real telemetry, not a canned animation. `website/index.html` has an
   SVG of this pipeline worth reading for the stage names and copy.
3. **Model comparison.** Pick a clip (upload or one of the bundled `sonix_real/` pairs), pick
   2–4 heads, hit compare. Show per-model score timelines overlaid on one chart plus a summary
   table (mean, median, max, band, head latency). Show the shared-embedding cost split —
   embedding ms once vs head ms per model — because that is the architecture argument.
4. **Results panel.** Read from `docs/RESULTS_TABLE.md` section A only. Show reproduction
   commands next to each number. Anything in section B renders as "not yet measured" — never
   as a blank or an estimate.
5. **Upload & score.** Drag a wav/mp3/flac/ogg in, watch windows score in, timeline + band.

### Operator mode — what a bank's console would look like

1. **Active calls table** — call_id, state (`CONSENT_PENDING` / `LISTENING` / `ENDED`), current
   band, windows scored, duration, model in use. Consent state is the first-class column.
2. **Consent / pairing panel** — pending codes with their 120s expiry counting down, approve
   and end-call actions.
3. **Audit trail viewer** — per-call events from the session audit record, timestamped, plus a
   clear statement that no audio was retained.
4. **Engine health** — throughput, batch size, warm/cold, queue depth, device, which head is
   loaded, `scoring_available`. When the server is on `mock.py`, the entire UI must be
   unmistakably marked **MOCK — not a real detection** (mock is capped at 0.5 and never Red;
   a demo that looks green because scoring is off is the worst possible failure mode).
5. **Job queue & run history** — queued/running/done/failed with progress, cancel, and a
   history table drilling into any past run's scores. Failed jobs show the **full traceback**.
6. **Thresholds panel** — display `AMBER_AT` / `RED_AT` as served by the API, labelled
   "provisional, uncalibrated". If you expose runtime overrides, make them per-session display
   settings and say plainly that they do not change the pinned production bands.

### Frontend rules

- **Never invent a number.** No placeholder EERs, no lorem metrics, no fabricated call rows.
  If data has not arrived, render an explicit empty state saying what would fill it.
- Band colours and cut-offs come from the API's threshold payload, never hardcoded in TS.
- Model labels and notes come from `/api/models` (`realtime/models.py` REGISTRY) — do not
  retype them. Heads with `exists: false` are shown disabled with "checkpoint not on this
  machine", because `head*.pt` is gitignored and file-transferred.
- Degrade honestly when the server is down: a clear "no SONIX server at localhost:8000" state
  with the launch command, not an infinite spinner and not fake data.
- Accessible and responsive: keyboard-navigable, visible focus, labelled charts, works from
  1280px projector down to a laptop. Dark theme default; the room lights will be down.
- No `localStorage` for anything but the mode toggle and display preferences.

---

## Latency is a requirement, not a nice-to-have

`04_PERFORMANCE.md` is binding. The short version:

- The hop is 500 ms and the front-end is 300M parameters. Those two facts set the floor and
  nothing you write can move them. Everything else must be invisible next to them — **budget
  ≤ 200 ms added latency from window-ready to pixel-changed.**
- Never block the event loop. Inference and file I/O go off it, always.
- Compare embeds **once** and runs every head over the same matrix. Looping `score_file()` per
  model both blows the budget and misrepresents the architecture.
- Cache embeddings by content hash. Re-comparing a clip against another head must be instant,
  because that is precisely what a judge asks for next.
- The live timeline is a `<canvas>` drawn on rAF, not a Recharts tree re-rendered per point.
- `/api/dashboard/summary` is memory-only. No disk, no GPU, on a polled endpoint.

Every budget ships as an assertion in `realtime/tests/test_perf.py`. If one cannot be met,
revise the number in `04_PERFORMANCE.md` with a written reason — do not ship past it quietly,
and do not report "fast" without the measurement behind it.

## Deliverables

1. `dashboard/` — full Vite app, `README.md` with dev and build commands.
2. New endpoints in `realtime/server.py` + any new modules under `realtime/` (e.g.
   `realtime/jobs.py`). Keep new logic in new files; keep the `server.py` diff to route
   registration and handler methods.
3. Tests in `realtime/tests/` for every new endpoint, plus the existing-routes regression test.
4. `docs/API.md` updated with the new endpoints in the existing voice and format — including
   an honest addition to the "What is not built yet" table if anything here lands partial.
5. A short `docs/DASHBOARD.md`: how to run it, what each mode shows, what is real vs mocked.

## Verification before you report done

- `python -m pytest realtime/tests/ -q` passes, including the pre-existing tests.
- Server starts and every old route still answers: `/api/status`, `/api/models`,
  `/api/telemetry`, `/mic`, and `website/` still serves at `/`.
- `cd dashboard && npm run build` succeeds; the built app loads from `/dashboard`.
- Run the dashboard against a server with **no** `head*.pt` present and confirm the mock
  warning is impossible to miss.
- Run a compare across at least two heads that exist on this machine and confirm the embedding
  pass ran once, not once per model.
- Take screenshots of both modes and check them yourself before claiming the UI works.

## Team rules that apply to this task

1. Never invent a number — in code, in copy, or in a status report.
2. If a script or build fails, paste the **full traceback**, not a summary.
3. Don't modify a script you were handed without checking with its owner (`CLAUDE.md` lanes).
4. Cache everything expensive — embedding is the slow step; never re-embed what you already have.
5. Mock before you integrate — build the UI against fixtures so it is not blocked on a head file.
