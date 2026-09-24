# 06 — Build order

Six phases. Each has a gate you must actually run before starting the next one. The ordering
exists so that the project is demoable from the end of Phase 2 onward — if time runs out, you
stop at a phase boundary and still have something that works, rather than six half-finished
features.

Team rule 5: **mock before you integrate.** Phase 1 ships fixtures so the frontend is never
blocked on a `head*.pt` that lives on someone else's machine.

---

## Phase 0 — Read and confirm (no code)

Read, in this order: `CLAUDE.md` · `CODE_MAP.md` · `PROJECT_STATE.txt` · `docs/API.md` ·
`docs/RESULTS_TABLE.md` · `realtime/server.py` · `realtime/engine.py` · `realtime/session.py` ·
`realtime/models.py` · `realtime/thresholds.py` · `realtime/ringbuffer.py` ·
`website/script.js` (the mic/WS handshake and the canvas timeline) · `extension/worklet.js`.

Then check the machine:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ls -la outputs/models/           # which head*.pt actually exist here
python -m pytest realtime/tests/ -q
node --version                   # 20.x
```

**Gate:** you can state, without guessing, which heads exist on this machine, whether CUDA is
available, and whether the existing test suite passes *before* you change anything. If the suite
is already red, say so and stop — do not build on top of a broken baseline and do not "fix" a
test in someone else's lane without asking.

---

## Phase 1 — Backend skeleton + fixtures

- `realtime/jobs.py` — `Job`, `JobStore`, queue, atomic persistence, startup index rebuild.
- `realtime/serialize.py` — orjson helpers, float coercion, wire downsampling.
- `realtime/dashboard_api.py` — handlers for `/api/dashboard/summary`, `/api/jobs*`, `/api/runs`.
  `/api/compare` returns a documented `501` for now.
- `realtime/server.py` — the three-line diff from `02_STRUCTURE.md`.
- Fixtures: `realtime/tests/fixtures/` with a canned summary, a job list, a compare result and a
  score stream, so the frontend can build with the server off.

**Gate:** `python -m pytest realtime/tests/ -q` green, including the pre-existing tests.
`GET /api/status`, `/api/models`, `/api/telemetry`, `/mic` and `/` all still return exactly what
they returned in Phase 0 — `test_existing_routes.py` proves it, not your memory.

---

## Phase 2 — Frontend shell

- Vite + React + TS + Tailwind + shadcn scaffold in `dashboard/`, proxy configured.
- Shell: `Header`, mode toggle, `MockBanner`, `OfflineState`, server status pill.
- `lib/api.ts`, `lib/types.ts`, `lib/bands.ts` (thresholds from the API), `lib/ws.ts`.
- `hooks/useSummary.ts`, `hooks/useServerHealth.ts`.
- Operator mode: `EngineHealth` and `JobQueue` wired to the real endpoints.

**Gate:** `npm run build` succeeds; the built app loads from `/dashboard`; all four server states
(offline / warming / mock / ready) render correctly — **force each one and look at it**. Stop the
server and confirm you get the launch command, not a spinner. This is the phase where the app
first becomes demoable.

---

## Phase 3 — Live path

- `worklets/pcm-worklet.js`, `hooks/useMicCapture.ts`.
- WS client: `start_mic_call` → pairing code → `/api/approve` → `scores` frames.
- `charts/RiskTimeline.tsx` — **canvas**, rAF draw loop, store-fed.
- Demo mode: `LiveBand`, the 4-second countdown before the first reading.
- Operator mode: `ActiveCalls`, `ConsentPanel` with the 120 s expiry countdown.

**Gate:** speak into the mic and watch the band move with the latency budget from
`04_PERFORMANCE.md` met. Confirm the consent gate genuinely blocks — audio before approval
produces **no** scores. Open three dashboard tabs at once and confirm the timeline still draws
smoothly.

---

## Phase 4 — Compare and the embedding cache

- `realtime/compare.py` — one shared embedding pass, per-head timings.
- `POST /api/compare` for real.
- Content-hash embedding cache under `outputs/cache/embeddings/`, LRU-capped.
- Demo mode: `ModelCompare`, `CompareChart`, the cost-split line.

**Gate:** `test_compare_embeds_once` and `test_compare_scales_flat` pass. Comparing 3 models
takes materially less than 3× one model. Re-comparing the same clip with a different head is
near-instant (cache hit). An unknown model key returns a 400 that names the key — and never
scores with a different head.

---

## Phase 5 — Jobs, history, and the rest of the UI

- `kind: "batch"` jobs, progress over WS, cancellation.
- `/api/runs` index built at startup from `outputs/jobs/` + `outputs/calls/`.
- Operator: `RunHistory` (virtualized), `AuditTrail`, `ThresholdPanel`.
- Demo: `PipelineDiagram`, `ResultsPanel`, `UploadScore`.

**Gate:** a 100-file batch runs to completion while a live call scores at full rate — measured,
not assumed (`test_live_priority`). Job records survive a server restart. A deliberately failed
job shows its **full traceback** in the UI.

---

## Phase 6 — Polish, perf, docs

- `realtime/tests/test_perf.py` — run it, record real numbers in
  `docs/dashboard/PERF_RESULTS.md`.
- Bundle analysis, route-splitting, `React.lazy` on the two modes.
- Accessibility pass: keyboard, focus, contrast, `aria-live`, `prefers-reduced-motion`.
- Invoke **`ponytail-review`** on the full diff and delete what it finds.
- `docs/API.md` updated with the new endpoints, in its existing voice — including an honest
  addition to its "what is not built yet" table for anything that landed partial.
- `docs/DASHBOARD.md` — how to run it, what each mode shows, what is real vs mocked.

**Gate:** every budget in `04_PERFORMANCE.md` either met and recorded, or revised in that file
with a written reason. Screenshots of both modes taken and inspected. Full suite green.

---

## Working rules for the whole build

- **Commit per phase**, with the gate result in the commit message.
- **Never edit outside the lanes.** `CLAUDE.md` names the owners. The only pre-existing file you
  modify is `realtime/server.py`, and only as specified.
- **Never invent a number** — in code, in copy, in a commit message, or in a status report.
- **Paste full tracebacks.** Not summaries.
- **Cache what is expensive.** Embedding, always.
- If a phase gate fails, fix it before moving on. A dashboard that is 80% done across six
  features is worth less at a demo than one that is finished across three.
