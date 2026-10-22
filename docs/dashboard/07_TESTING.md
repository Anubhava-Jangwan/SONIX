# 07 — Testing

Match the existing style in `realtime/tests/` — `pytest` + `pytest-asyncio`, both already in
`requirements.txt`. The suite that exists (`test_engine.py`, `test_session.py`,
`test_thresholds.py`, `test_miccapture.py`, and the rest) must stay green throughout. If one of
them goes red, you broke something in someone else's lane; fix your change, do not edit their
test.

---

## Backend

### The regression test that matters most — `test_existing_routes.py`

Three live clients depend on the current API: `website/script.js`, `extension/`, and
`realtime/miccapture.py`. This test asserts, route by route, that nothing you added changed
anything that existed.

| Route | Assert |
|---|---|
| `GET /api/status` | 200; keys `status`, `mode`, `scoring_available`, `scoring_synthetic`, `active_calls`, `max_calls`, `engine_stats`, `ws_clients`, `timestamp`. **`scoring_available` and `scoring_synthetic` are load-bearing** — the `/mic` page gates its entire verdict and chart on them, and they were added because serving them only on `/api/telemetry` left the live chart silently empty. |
| `GET /api/models` | 200; `mock`, `warm`, `warming`, `default`, `models[]` each with `key`, `label`, `exists` |
| `GET /api/telemetry?call_id=…` | 200 for a known call; unchanged shape |
| `POST /api/approve` | unchanged behaviour and status codes |
| `POST /api/end-call` | unchanged |
| `POST /api/score-file` | unchanged multipart contract and response |
| `GET /mic` | 200, HTML, and its inline thresholds still match `realtime/thresholds.py` |
| `GET /` | still serves `website/index.html` |
| `WS /ws` | `{"type":"scores","data":{...}}` frame shape byte-for-byte unchanged |

Adding a key to a response is not automatically safe — `test_thresholds.py` already pins values
across files for exactly this reason. Assert on the keys those three clients read.

### `test_dashboard_api.py`

- `/api/dashboard/summary` returns every documented key; `thresholds` matches
  `realtime.thresholds.AMBER_AT` / `RED_AT` (imported, not retyped).
- `scoring_available` is `false` when the engine is in mock mode and `true` when a real head is
  loaded.
- Errors use the structured envelope with a known `code`, never a bare 500 or an HTML page.
- `/api/runs` pagination: `total` is stable across pages, no duplicate `run_id`s.

### `test_jobs.py`

- Full lifecycle: `queued` → `running` → `done`, with `progress` monotonically increasing.
- Cancel a `running` job → `cancelled`; cancelling a `done` job → 409; unknown id → 404.
- A failing job stores the **complete traceback** in `error` — assert the string contains
  `Traceback (most recent call last)`, not a paraphrase.
- Records are written atomically: no `.tmp` file remains, and killing mid-write never leaves a
  loadable-but-wrong record.
- `load_index()` rebuilds history after a simulated restart.
- Live calls take priority: with a batch job running, live windows still score at full rate.

### `test_compare.py`

- **Embed called exactly once for N models.** The core assertion of the whole feature — patch
  the embed path and count calls.
- Per-model score arrays have identical length and correspond to the same windows.
- `band` derives from `thresholds.band_for()` on the mean, not recomputed locally.
- Unknown key → 400 `unknown_model` listing known keys.
- Key valid, checkpoint absent → 400 `checkpoint_missing`; **and assert no scoring happened**.
  Silently falling back to another head is the failure this test exists to prevent.
- No head at all → 409 `scoring_unavailable`.
- Cache hit: same bytes + same vad → embedding skipped, scores identical.

### `test_perf.py`

The budgets from `04_PERFORMANCE.md`, as assertions. Mark them `@pytest.mark.perf` so CI can skip
them on a machine without a GPU, and run them on the demo machine before demo day.

| Test | Asserts |
|---|---|
| `test_summary_is_memory_only` | no filesystem access during the call; p95 < 50 ms over 100 calls |
| `test_compare_embeds_once` | exactly one embed pass for 3 models |
| `test_compare_scales_flat` | 3-model wall time < 1.5× the 1-model time |
| `test_embedding_cache_hit` | second identical compare skips embedding |
| `test_job_submit_is_fast` | `POST /api/jobs` < 100 ms with 500 files queued |
| `test_live_priority` | live scoring rate unaffected by a running batch |
| `test_no_loop_block` | event-loop lag < 50 ms while a batch runs |
| `test_broadcast_coalescing` | ≤ 20 frames/s/client at 4 calls × 3 clients |

Record real outputs in `docs/dashboard/PERF_RESULTS.md`. Create that file when you have numbers —
**do not pre-fill it with targets**, or the targets will get quoted as results, which is exactly
the failure `CLAUDE.md` rule 1 exists to prevent.

---

## Frontend

Vitest + React Testing Library. Keep it proportionate — this is a dashboard, not a bank ledger.

**Worth testing:**

- `lib/bands.ts` — band boundaries from API-supplied thresholds, including the exact edges
  (0.35 → Amber, 0.65 → Red) and out-of-range inputs.
- `lib/ws.ts` — reconnect backoff, unknown frame types ignored without throwing.
- `MockBanner` renders whenever `scoring_available === false`, and cannot be dismissed.
- `OfflineState` renders on fetch failure and shows the launch command.
- `ResultsPanel` renders §B entries as "not yet measured" — assert the string. This is the guard
  against a future edit quietly turning an unmeasured row into a number.
- `format.ts` — EER always renders with its split name.

**Not worth testing:** shadcn primitives, Tailwind classes, canvas pixel output.

---

## Manual checks before claiming done

Run these yourself and look at the result. Every one of them has a failure mode that no
automated test in this list catches.

1. **Kill the server.** Dashboard shows the offline state with the launch command. No spinner,
   no blank page, no stale data presented as live.
2. **Run with no `head*.pt`.** The purple MOCK banner is unmissable on every screen. Ask someone
   who has not read this spec whether the app is currently detecting anything; if they say yes,
   the banner has failed.
3. **Start a mic call and do not approve it.** Zero scores. The consent gate is the compliance
   story — verify it rather than trusting it.
4. **Speak, and watch the band.** First reading arrives after ~4 s with a real countdown, then
   updates about twice a second.
5. **Compare 3 models on a 20 s clip.** Check the reported `embed_ms` vs `head_ms` split. Then
   compare again with a different model set and confirm the cache made it fast.
6. **Queue a 100-file batch, then make a live call.** The live call must not slow down.
7. **Open three dashboard tabs.** Timeline stays smooth in all three.
8. **Resize to 1280×720** (projector) **and to a 13" laptop.** No horizontal scroll, no clipped
   charts.
9. **Tab through the whole app.** Every control reachable, focus always visible.
10. **Take screenshots of both modes and actually look at them.** Rendering without errors is not
    the same as looking right.

---

## Definition of done

- `python -m pytest realtime/tests/ -q` green, pre-existing tests included.
- `cd dashboard && npm run build` succeeds; app loads from `/dashboard`.
- All ten manual checks pass.
- Perf budgets met and recorded, or revised in `04_PERFORMANCE.md` with a written reason.
- `ponytail-review` run on the full diff and its findings deleted.
- `docs/API.md` and `docs/DASHBOARD.md` updated.
- No invented numbers anywhere in code, copy, or commit messages.
