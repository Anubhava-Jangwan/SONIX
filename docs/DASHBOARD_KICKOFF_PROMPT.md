# Kickoff prompt — paste this into a fresh agent session

Open a new agent session at the repo root (`B:\SIH_sonix`) and paste everything below the line.
It is the ordered runbook. The detail it refers to lives in `docs/dashboard/` — this prompt
sequences the work and sets the stop conditions.

---

You are building the SONIX dashboard. Work through the steps below **in order**. Do not skip
ahead, do not batch phases together, and do not start a phase whose gate you have not run.

## Before anything

Read these, in this order, and do not write code until you have:

1. `CLAUDE.md` — team rules, lanes, machine gotchas
2. `CODE_MAP.md` — where things live, and the duplication traps
3. `PROJECT_STATE.txt` — current merge state, blocking bugs
4. `docs/API.md` — the existing API and the privacy commitments it makes
5. `docs/RESULTS_TABLE.md` — **the only source of numbers in this project**
6. `docs/dashboard/README.md`, then `01` through `07` in order

Then invoke the **`ponytail`** skill. Keep it active for every coding step in this task, backend
and frontend. Invoke **`ui-ux-pro-max`** before you write your first component.

Three rules that override anything you think is a good idea:

- **Never invent a number.** Not in code, not in UI copy, not in a placeholder, not in a commit
  message, not in a progress report to me. If a script has not produced it, it does not exist.
- **Paste full tracebacks**, never a paraphrase, when anything fails.
- **Stay in your lane.** The only pre-existing file you may modify is `realtime/server.py`, and
  only to add route registrations and one static mount. `website/`, `demo/`, `uidemo/`, `src/`
  and `scripts/` are other people's work — read them, do not edit them.

---

## Step 0 — Establish the baseline

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls -la outputs/models/
python -m pytest realtime/tests/ -q
node --version
```

Report back: which `head*.pt` files exist on this machine, whether CUDA is available, and
whether the existing test suite is green.

**Stop and tell me if the suite is already red.** Do not build on a broken baseline, and do not
"fix" a failing test in someone else's lane without asking me first.

---

## Step 1 — Backend skeleton and fixtures

Build, per `docs/dashboard/02_STRUCTURE.md` and `03_API_CONTRACT.md`:

- `realtime/jobs.py` — `Job` dataclass, `JobStore`, one `asyncio.Queue`, one consumer task,
  atomic persistence to `outputs/jobs/`, index rebuilt at startup
- `realtime/serialize.py` — orjson helpers, float coercion, wire downsampling
- `realtime/dashboard_api.py` — handlers for `/api/dashboard/summary`, `/api/jobs*`, `/api/runs`;
  `/api/compare` returns a documented `501` for now
- `realtime/server.py` — the three-line diff only: instantiate `DashboardAPI`, call `attach(app)`,
  mount `dashboard/dist` at `/dashboard` **before** the existing catch-all `website/` mount
- `realtime/tests/fixtures/` — canned summary, job list, compare result, score stream, so the
  frontend is never blocked on a checkpoint

**Gate:** `python -m pytest realtime/tests/ -q` green including the pre-existing tests, and
`realtime/tests/test_existing_routes.py` proving `/api/status`, `/api/models`, `/api/telemetry`,
`/mic` and `/` all return exactly what they returned in Step 0. Report the gate result before
moving on.

---

## Step 2 — Frontend shell

Scaffold `dashboard/` per `01_TECH_STACK.md` — Vite + React 18 + TS + Tailwind + shadcn, its own
`package.json`, do not touch the repo-root one. Add only the shadcn primitives you actually
render.

Build: `Header` with the mode toggle, `MockBanner`, `OfflineState`, the four-state server pill,
`lib/api.ts`, `lib/types.ts`, `lib/bands.ts` (thresholds **from the API**, never hardcoded),
`lib/ws.ts`, `useSummary`, `useServerHealth`, and operator-mode `EngineHealth` + `JobQueue`.

**Gate:** `npm run build` succeeds, the app loads from `/dashboard`, and you have forced and
visually confirmed all four server states — offline, warming, mock, ready. Stop the server and
check you get the launch command, not a spinner. Screenshot each state and look at it.

---

## Step 3 — Live path

`worklets/pcm-worklet.js` (AudioWorklet, not ScriptProcessor — read `extension/worklet.js`),
`useMicCapture`, the WS handshake (`start_mic_call` → pairing code → `/api/approve` → `scores`),
`RiskTimeline` as a **canvas on a rAF loop** (read `website/script.js` first — it already does
this), `LiveBand` with the 4-second first-reading countdown, and operator-mode `ActiveCalls` +
`ConsentPanel` with the 120 s pairing expiry.

**Gate:** speak into the mic, watch the band move, and confirm the latency budget in
`04_PERFORMANCE.md` is met. Then start a call and **do not approve it** — confirm zero scores.
Open three dashboard tabs and confirm the timeline still draws smoothly.

---

## Step 4 — Compare and the embedding cache

`realtime/compare.py` — embed each window **once**, then run every requested head over the same
matrix. Report `embed_ms` once and `head_ms` per model. Then `POST /api/compare` for real, a
content-hash embedding cache under `outputs/cache/embeddings/` (LRU-capped), and demo-mode
`ModelCompare` + `CompareChart` showing the cost split.

**Gate:** `test_compare_embeds_once` and `test_compare_scales_flat` pass — three models must cost
materially less than three times one model. Re-comparing the same clip with a different head is
near-instant. An unknown model key returns a 400 naming the key, and **scores nothing**.

---

## Step 5 — Jobs, history, and the remaining screens

Batch jobs with progress over WS and cancellation. `/api/runs` index built at startup from
`outputs/jobs/` + `outputs/calls/`. Operator: `RunHistory` (virtualized), `AuditTrail`,
`ThresholdPanel`. Demo: `PipelineDiagram`, `ResultsPanel`, `UploadScore`.

`ResultsPanel` reads `docs/RESULTS_TABLE.md` §A only. §B rows render as "not yet measured" with
the owner named — never blank, never a dash, never an estimate.

**Gate:** a 100-file batch runs to completion while a live call scores at full rate — measured,
not assumed. Job records survive a server restart. A deliberately failed job shows its full
traceback in the UI.

---

## Step 6 — Performance, polish, docs

- Write and run `realtime/tests/test_perf.py` — every budget in `04_PERFORMANCE.md` as an
  assertion. Record the real outputs in `docs/dashboard/PERF_RESULTS.md`. **Create that file
  only once you have real numbers; do not pre-fill it with the targets.**
- Bundle analysis, `React.lazy` route-split on the two modes.
- Invoke **`ui-ux-pro-max`** again for the visual pass, then **`ponytail-review`** on the full
  diff. Delete what it finds and tell me what you deleted.
- Accessibility: keyboard, focus rings, contrast, `aria-live="polite"` on the band,
  `prefers-reduced-motion`.
- Update `docs/API.md` with the new endpoints in its existing voice, including an honest addition
  to its "what is not built yet" table for anything that landed partial.
- Write `docs/DASHBOARD.md` — how to run it, what each mode shows, what is real vs mocked.

**Gate:** the ten manual checks in `07_TESTING.md`, run by you, with screenshots you have
actually looked at. Every perf budget either met and recorded, or revised in
`04_PERFORMANCE.md` with a written reason.

---

## How to report

After each step: the gate command you ran, its real output, and anything that surprised you.
One short paragraph, not a summary of what you wrote — I can read the diff.

If a gate fails, stop at that step and tell me. A dashboard that is finished across three
features beats one that is eighty percent done across six.

If you find yourself wanting a dependency that is not on the approved list in `01_TECH_STACK.md`,
stop and ask. That list has a rejected column with reasons, and `ponytail` is the test any
addition has to pass.
