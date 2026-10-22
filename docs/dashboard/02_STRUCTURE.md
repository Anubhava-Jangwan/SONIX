# 02 — File structure

Every file below is either new or an explicitly-scoped edit. Anything not listed stays as it is.

---

## Backend

```
realtime/
├── server.py              ← EDIT (routes + static mount only, see below)
├── jobs.py                ← NEW  job queue, records, persistence
├── compare.py             ← NEW  multi-model scoring on one shared embedding pass
├── dashboard_api.py       ← NEW  all new HTTP handlers, as a class the server mounts
├── serialize.py           ← NEW  orjson helpers + the one place a score array is shaped for the wire
│
│   # untouched, but you will read all of these:
├── engine.py                    ScoringEngine — batching, ensure_model(), model_catalogue(), get_stats()
├── session.py                   Session, CallState, consent gate in push_audio()
├── models.py                    REGISTRY, catalogue(), resolve_ckpt(), key_for_path(), DEFAULT_KEY
├── thresholds.py                AMBER_AT = 0.35, RED_AT = 0.65, band_for()
├── checkpoint.py                StandardisedHead, load_checkpoint()
├── frontend.py                  wav2vec2 wrapper: load(), embed()
├── ringbuffer.py                SAMPLE_RATE 16000, WINDOW_SEC 4.0, HOP_SEC 0.5
├── vad.py                       silence gate
├── mock.py                      MockScorer — random, capped 0.5, never Red
└── tests/
    ├── test_dashboard_api.py    ← NEW
    ├── test_jobs.py             ← NEW
    ├── test_compare.py          ← NEW
    ├── test_existing_routes.py  ← NEW  regression: every pre-existing route keeps its shape
    └── test_perf.py             ← NEW  the budgets in 04_PERFORMANCE.md, as assertions
```

### The `server.py` diff, in full

Three things and nothing else:

1. Instantiate `DashboardAPI(self)` in `SonicServer.__init__`.
2. Register the new routes in `run()`, next to the existing block at ~line 793.
3. Mount `dashboard/dist` at `/dashboard` — **before** the existing `add_static('/', website_dir)`,
   because that catch-all mount will swallow `/dashboard` otherwise.

```python
# in run(), immediately after the existing api routes
self.dashboard_api.attach(app)              # all /api/compare, /api/jobs, /api/runs, /api/dashboard/*

# static mounts, in this order
dist = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
if dist.exists():
    app.router.add_get('/dashboard', _dashboard_index)     # SPA entry
    app.router.add_static('/dashboard/', dist, show_index=False)
else:
    app.router.add_get('/dashboard', _dashboard_not_built)  # helpful page, never a 404 or a 500
# ... then the EXISTING website mount, unchanged
```

Every new handler lives in `dashboard_api.py`. `server.py` should gain route registrations and
one attribute — not handler bodies. Keeping the diff tiny is what makes it safe to merge into a
file three other lanes depend on.

### What each new module owns

**`realtime/jobs.py`**

```python
@dataclass
class Job:
    job_id: str          # "job_20260920T141233_a4f1"
    kind: str            # "score" | "compare" | "batch"
    state: str           # "queued" | "running" | "done" | "failed" | "cancelled"
    created_at: str      # ISO8601
    started_at: str | None
    finished_at: str | None
    progress: dict       # {"done": int, "total": int}
    models: list[str]
    params: dict
    result: dict | None
    error: str | None    # FULL traceback. CLAUDE.md rule 2. Never a summary.

class JobStore:
    async def submit(kind, params) -> Job
    async def get(job_id) -> Job | None
    async def list(limit, offset, kind=None, state=None) -> tuple[list[Job], int]
    async def cancel(job_id) -> bool
    async def _persist(job)          # atomic: write .tmp, os.replace
    def load_index()                 # startup: scan outputs/jobs/, rebuild in-memory index
```

One `asyncio.Queue`, one consumer task, concurrency capped so batch work cannot starve live
calls. Records go to `outputs/jobs/<job_id>.json`, mirroring the existing `outputs/calls/`
convention. Write atomically — a half-written record that loads as valid JSON is worse than one
that fails to load.

**`realtime/compare.py`**

```python
async def compare_models(engine, windows: list[np.ndarray], model_keys: list[str],
                         vad: str = "auto") -> dict:
    """Embed each window ONCE, then run every requested head over the same matrix.

    This is the architecture claim the dashboard exists to demonstrate: the 300M
    front-end is the cost and it is paid once, regardless of how many detectors
    you run. Do not loop score_file() per model -- that re-embeds and both
    destroys the latency budget and misrepresents the design.
    """
```

Returns per-model score arrays plus `embed_ms` (once) and `head_ms` (per model) so the UI can
show the split. Unknown or missing checkpoint keys raise before any work starts.

**`realtime/serialize.py`**

`orjson` dumps, the `numpy.float32 -> float` coercion in one place, and score-array
downsampling for the wire (see `04_PERFORMANCE.md` — a 40-minute call is ~4,800 windows and
you do not ship all of them to a chart that is 900 px wide).

---

## Frontend

```
dashboard/
├── package.json
├── package-lock.json               committed
├── .nvmrc                          20
├── vite.config.ts                  proxy /api + /ws -> localhost:8000
├── tailwind.config.ts
├── tsconfig.json                   strict: true, noUncheckedIndexedAccess: true
├── index.html
├── README.md                       dev + build commands, dependency justifications
└── src/
    ├── main.tsx
    ├── App.tsx                     mode router: demo | operator
    ├── index.css                   tailwind layers + design tokens from 05_UI_SPEC.md
    │
    ├── lib/
    │   ├── api.ts                  typed fetch wrapper, AbortController on every call
    │   ├── ws.ts                    single shared WebSocket, reconnect with backoff
    │   ├── types.ts                 mirrors 03_API_CONTRACT.md exactly
    │   ├── bands.ts                 band from score — reads thresholds FROM THE API, never hardcoded
    │   ├── format.ts                Intl-based number/time/duration formatting
    │   └── utils.ts                 cn() only
    │
    ├── store/
    │   ├── useMode.ts               demo | operator, persisted to localStorage
    │   ├── useLive.ts               WS frames: scores, call state, job progress
    │   └── useCompare.ts            selected models, current comparison result
    │
    ├── hooks/
    │   ├── useSummary.ts            react-query -> /api/dashboard/summary
    │   ├── useJobs.ts               react-query -> /api/jobs, with WS-driven invalidation
    │   ├── useMicCapture.ts         getUserMedia -> AudioWorklet -> 16k PCM -> WS
    │   └── useServerHealth.ts       reachable? warm? mock? scoring_available?
    │
    ├── components/
    │   ├── ui/                      shadcn primitives, copied in — only the ones used
    │   ├── shell/
    │   │   ├── Header.tsx           mode toggle, server status pill, model indicator
    │   │   ├── MockBanner.tsx       the unmissable MOCK warning
    │   │   └── OfflineState.tsx     "no SONIX server at :8000" + the launch command
    │   ├── charts/
    │   │   ├── RiskTimeline.tsx     CANVAS. the live one. not recharts.
    │   │   ├── CompareChart.tsx     recharts, static data, fine
    │   │   └── Sparkline.tsx        canvas, engine throughput
    │   ├── demo/
    │   │   ├── LiveBand.tsx
    │   │   ├── PipelineDiagram.tsx  SVG, stages lit by real telemetry
    │   │   ├── ModelCompare.tsx
    │   │   ├── ResultsPanel.tsx     reads docs/RESULTS_TABLE.md section A only
    │   │   └── UploadScore.tsx
    │   └── operator/
    │       ├── ActiveCalls.tsx
    │       ├── ConsentPanel.tsx     pairing codes + 120s expiry countdown
    │       ├── AuditTrail.tsx
    │       ├── EngineHealth.tsx
    │       ├── JobQueue.tsx
    │       ├── RunHistory.tsx
    │       └── ThresholdPanel.tsx
    │
    ├── worklets/
    │   └── pcm-worklet.js           mic -> float32 -> int16 @16k. AudioWorklet, NOT ScriptProcessor.
    │
    └── data/
        └── results.ts               transcribed from docs/RESULTS_TABLE.md §A, with a comment
                                     pointing at the source file and the reproduce command
```

### Notes on a few of these

- **`lib/bands.ts`** takes thresholds as an argument, sourced from
  `/api/dashboard/summary`. There is exactly one definition of 0.35/0.65 in this repo and it is
  `realtime/thresholds.py`. `realtime/tests/test_thresholds.py` already pins the copies in
  `miccapture.py`, `website/script.js` and the extension. Do not add a fourth copy for the test
  to chase.

- **`worklets/pcm-worklet.js`** — `ScriptProcessorNode` is deprecated and runs on the main
  thread, so it drops samples the moment React renders. `extension/worklet.js` already does this
  correctly; read it.

- **`data/results.ts`** is transcribed by hand from `docs/RESULTS_TABLE.md` because that file is
  Yukti's single source of truth and parsing markdown at runtime is not worth it. Every entry
  carries its reproduce command. Section B entries are present with
  `status: "not-measured"` — they render as "not yet measured", never as a blank, an estimate,
  or a dash that reads like zero.

- **`components/ui/`** is generated by the shadcn CLI. Do not hand-edit those files beyond
  theming; re-running the CLI will overwrite them.

---

## Files you must not create

- Anything under `website/`, `demo/`, `uidemo/`, `src/`, `scripts/`.
- A fourth `score_file.py`. There are already three (`src/`, `demo/`, `uidemo/...`) and
  `CODE_MAP.md` flags which one binds where. Import, do not copy.
- A second `thresholds` definition in TypeScript.
- A `dashboard/server.js` or any Node backend. The Python server is the backend.
