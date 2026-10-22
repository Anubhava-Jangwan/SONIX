# 04 — Performance and latency

Read this before writing a single handler or component. Latency in this system is not something
you profile at the end; it is something you either design in or never get back.

Every number in this file is a **budget to be measured**, not a measured result. `CLAUDE.md`
rule 1 applies to performance claims exactly as it applies to EER: when the benchmark in
`realtime/tests/test_perf.py` has produced a real figure, quote the real figure. Until then
these are targets.

---

## What is physically fixed, and what you control

From `realtime/ringbuffer.py`:

```
SAMPLE_RATE = 16000     WINDOW_SEC = 4.0     HOP_SEC = 0.5
```

A window is 64,000 samples. The hop is 8,000 samples — 87.5% overlap. Consequences you cannot
engineer around:

- **A new score exists at most every 500 ms.** That is the hop. 2 Hz is the ceiling on how often
  the live band can legitimately change. Anything faster on screen is animation, not data, and
  must not be presented as a new reading.
- **The first score of a call cannot arrive before 4 seconds of audio has.** The ring buffer
  never zero-pads to length, deliberately — a half-silent window is exactly the input that
  produces a false "fake" verdict, and `docs/RESULTS_TABLE.md` has the before/after numbers from
  the padding fix that proves it. The UI must show "listening — first reading in ~Ns" with a real
  countdown, not a spinner that looks like a hang.
- **The 300M front-end takes tens of seconds to load.** `ScoringEngine.preload()` already does
  this once at startup in a thread, precisely so the first upload does not look like a crash.
  Never trigger a lazy load from a request path.

What you actually control is everything between "a window is ready" and "a pixel changes":
batching, serialization, the event loop, the network hop, and React.

---

## Latency budget — score to screen

| Stage | Budget | Owner |
|---|---|---|
| Hop — audio accumulates | 500 ms | fixed, not yours |
| Window leaves ring buffer → enters batch | ≤ `batch_interval` (100 ms) | `engine.py`, tune don't rewrite |
| Embed (batched, GPU) | measure; target ≤ 80 ms for a batch of 8 | `frontend.py` |
| Head forward (~300k params) | ≤ 5 ms | negligible by design |
| Serialize + WS broadcast | ≤ 2 ms | `serialize.py` |
| Network (localhost) | ≤ 1 ms | — |
| Client parse → canvas draw | ≤ 16 ms (one frame) | `RiskTimeline.tsx` |
| **Total added latency beyond the hop** | **≤ 200 ms** | |

Target for user-perceived freshness: **score on screen within ~700 ms of the audio that produced
it**, of which 500 ms is the hop and is not recoverable.

### Interaction budgets

| Action | Budget |
|---|---|
| Mode toggle (demo ↔ operator) | < 100 ms, no refetch |
| Any cached view render | < 100 ms |
| `/api/dashboard/summary` round trip | < 50 ms |
| Job submit → `202` | < 100 ms (enqueue only, never do work in the handler) |
| Compare, 20 s clip, 3 models | dominated by one embed pass; **must not scale with model count** |
| First contentful paint, cold | < 1.5 s |
| Route/tab switch | < 100 ms |

---

## Backend rules

**1. Never block the event loop.** One `await` on a synchronous call stalls every live call on
the server. Anything that touches the GPU, the filesystem or a decoder goes through
`run_in_executor` or `aiofiles`. The engine already does this for inference — do not undo it and
do not add a second path that ignores it.

**2. Compare embeds once.** `realtime/compare.py` runs the front-end over the window matrix one
time, then every head over the same embeddings. Looping `score_file()` per model re-embeds,
makes the endpoint scale linearly with model count, and misrepresents the architecture the
dashboard exists to demonstrate. There is a test for this: assert that comparing 3 models calls
the embed path exactly once.

**3. Cache embeddings by content hash.** Team rule 4 — *cache everything expensive*. Embedding is
the slow step everywhere in this repo. Key on `sha256(audio_bytes) + vad_mode`; store under
`outputs/cache/embeddings/`, LRU-capped by total bytes. Re-comparing the same clip against a
different head must be near-instant, because that is exactly what a judge does: run it, see the
result, ask "what about the baseline?". Hitting the cache there is the difference between a
snappy demo and an awkward pause.

**4. `summary` is memory-only.** No `os.listdir`, no JSON parsing, no `stat()` on the polled
endpoint. The job index and the run index are built at startup and mutated in place.

**5. Bound everything.** Max upload size, max `max_windows`, max concurrent jobs, max in-memory
score history per call. An unbounded list is a memory leak with a delay fuse, and the demo
machine will find it during the demo.

**6. Batch broadcasts.** Coalesce score frames on a ~50 ms tick rather than sending one frame per
window per client. At 2 Hz per call this barely matters for one call and matters a lot at
`--max-calls 4` with several dashboard tabs open.

**7. Compress the wire, not the meaning.** Scores are `float32`; round to 4 decimals for
transport. Do not downsample the stored array — downsample only what a 900 px chart renders
(`serialize.py`), and say "showing 900 of 4,812 windows" when you do.

**8. Job work yields.** The job consumer must `await asyncio.sleep(0)` between units so a batch of
800 files cannot monopolise the loop. Live calls take priority over jobs, always, by design and
by test.

---

## Frontend rules

**1. The live timeline is a canvas.** Not Recharts. Recharts rebuilds a React element tree per
data point; under a 2 Hz stream over a long call it drops frames and pins a core.
`website/script.js` already draws this timeline on a `<canvas>` — read it before writing yours.
Recharts is fine for the comparison chart and histograms, which are static once rendered.

**2. Draw on `requestAnimationFrame`, not on message arrival.** WS frames land in a zustand
store; one rAF loop reads the store and draws. Decouple ingest rate from paint rate or a burst
of frames becomes a burst of repaints.

**3. Keep WS data out of React state.** A `setState` per frame re-renders a subtree 2×/second
per call. Frames go to the store; only components that display a *discrete* change (the band
label, the window counter) subscribe, with a selector so they re-render only when their own slice
changes.

**4. Virtualize long lists.** Run history and audit trails grow without limit. Render a window of
rows, not all of them.

**5. Cancel in-flight requests.** Every `fetch` takes an `AbortController`; React Query handles
this if you let it. Switching modes mid-request must not leave a response landing in a
component that has unmounted.

**6. Never poll what the socket pushes.** If `engine` frames are flowing, the summary poll is off.
Two sources of the same truth is how a dashboard starts disagreeing with itself.

**7. Memoize chart transforms.** `useMemo` on the array reshaping, not on the render. Recomputing
a 4,800-point transform per render is the classic way a fast backend ends up looking slow.

**8. Ship a small bundle.** Route-split demo and operator modes with `React.lazy`. Target
< 300 KB gzipped for the initial chunk. Run `npx vite-bundle-visualizer` before you claim it.

---

## Startup and warmth

Cold start is the latency the audience actually sees, because it happens while everyone is
watching.

- `preload()` loads the front-end at startup in a thread. Keep it that way.
- Expose `warming: true` so the UI can show a real progress state — "loading the 300M front-end,
  ~Ns" — instead of an ambiguous spinner.
- Preload the **default head only** (`models.DEFAULT_KEY`, currently `v3`). Others load lazily on
  first use via `ensure_model()`, which is already the engine's behaviour.
- Add a `--warm-models v3,baseline` flag so a demo machine can preload exactly the heads the
  demo will compare, and pay that cost before anyone is in the room.
- Health endpoint distinguishes **reachable** / **warming** / **warm** / **mock**. The UI must
  never show a green "ready" that means "reachable".

---

## Benchmarks — `realtime/tests/test_perf.py`

Assertions, not prose. Run them on the demo machine and record the output in
`docs/dashboard/PERF_RESULTS.md` (create it when you have real numbers; do not pre-fill it).

| Test | Asserts |
|---|---|
| `test_summary_is_memory_only` | no filesystem call during `/api/dashboard/summary`; p95 < 50 ms over 100 calls |
| `test_compare_embeds_once` | embed path called exactly once for 3 models |
| `test_compare_scales_flat` | 3-model compare wall time < 1.5× the 1-model time |
| `test_embedding_cache_hit` | second compare on the same bytes skips embedding entirely |
| `test_job_submit_is_fast` | `POST /api/jobs` returns in < 100 ms with a 500-file batch queued |
| `test_live_priority` | live-call windows score at full rate while a batch job runs |
| `test_no_loop_block` | event-loop lag stays < 50 ms while a batch job runs |
| `test_broadcast_coalescing` | 4 calls × 2 Hz × 3 clients produces ≤ 20 frames/s/client |

If a budget in this file cannot be met, **change the number here and say why** — do not quietly
ship past it and do not report "fast" without the measurement behind it.

---

## The honest caveat

The dominant cost in this system is the frozen 300M front-end, and the hop is 500 ms. No amount
of frontend work moves either. What the rules above buy is that nothing *else* adds to them — the
dashboard should be invisible in the latency budget, and every millisecond the user waits should
be one the physics of the model actually required.
