# SONIX Dashboard — spec set

Everything an agent (or a human) needs to build the new SONIX dashboard: a React SPA in
`dashboard/` plus new endpoints on the existing aiohttp server in `realtime/`.

This is a **new surface**. `website/`, `demo/`, `uidemo/` and `realtime/live_ui.py` are not
touched. The only existing file that gets modified is `realtime/server.py`, and only to add
routes and one static mount.

## Read order

| # | File | What it settles |
|---|---|---|
| 0 | `../DASHBOARD_BUILD_PROMPT.md` | The task itself — paste this into a fresh agent session |
| 1 | [`01_TECH_STACK.md`](01_TECH_STACK.md) | Every dependency, pinned, with the reason and the banned list |
| 2 | [`02_STRUCTURE.md`](02_STRUCTURE.md) | Full directory tree, backend and frontend, file by file |
| 3 | [`03_API_CONTRACT.md`](03_API_CONTRACT.md) | Every endpoint and WS frame, request and response shapes |
| 4 | [`04_PERFORMANCE.md`](04_PERFORMANCE.md) | The latency budget and the rules that keep it — **read before writing any handler** |
| 5 | [`05_UI_SPEC.md`](05_UI_SPEC.md) | Demo mode and operator mode, screen by screen, with design tokens |
| 6 | [`06_BUILD_ORDER.md`](06_BUILD_ORDER.md) | Six phases with an acceptance gate on each |
| 7 | [`07_TESTING.md`](07_TESTING.md) | Test matrix, perf benchmarks, what "done" means |

## Repo context you must read first

These already exist and are authoritative. Do not re-derive what they state.

- `CLAUDE.md` — team rules, lanes, and the machine-specific gotchas (CUDA wheels, dataset
  layout, Git Bash paths). Rule 1 is *never invent a number*; it applies to UI copy too.
- `CODE_MAP.md` — where everything lives, plus the known duplication traps (three copies of
  `score_file.py`, two of `make_codec.py`, the frozen `uidemo/` snapshot).
- `PROJECT_STATE.txt` — current merge state and blocking bugs.
- `docs/API.md` — the existing public API and the privacy/consent story it commits to.
- `docs/RESULTS_TABLE.md` — **the only source of numbers.** Section A is presentable.
  Section B is "not yet measured" and renders as such. Nothing else goes on screen.

## The one-paragraph version

SONIX turns 4-second windows of call audio into a risk score. A frozen wav2vec2 XLS-R 300M
front-end produces a 1024-dim embedding; a ~300k-parameter trained MLP head turns that into
P(synthetic) in [0,1]. Only the head is trained, and there are several of them in
`realtime/models.py`'s `REGISTRY`. The front-end is loaded once and shared, which is why
adding a detector for a newly-released cloning tool costs about a megabyte and no extra GPU
memory. The dashboard exists to make that architecture legible — and to be a working operator
console while it does it.

Bands come from `realtime/thresholds.py`: Green `<0.35`, Amber `0.35–0.65`, Red `≥0.65`.
Provisional and uncalibrated. Never hardcode them anywhere else.
