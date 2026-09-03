# Uncommitted code changes — multi-model checkpoint support

Branch: `anubhav/data-extraction`. Four modified files, none committed yet:

```
 M demo/app.py            184 +/-
 M demo/score_file.py      64 +/-
 M realtime/checkpoint.py   8 +
 M realtime/session.py     31 +/-
```

## Why these changes exist

`outputs/models/` now contains **three** heads — `head.pt`, `head_aug.pt`,
`head_robust.pt` — but the demo UI had exactly two checkpoint paths hardcoded at
the top of `demo/app.py`, so `head_robust.pt` was invisible. The theme of every
change below is: **discover checkpoints instead of hardcoding them**, and carry
the resulting per-model scores through the realtime session.

---

## 1. `demo/score_file.py` — checkpoint discovery

### `_candidate_roots()` (new helper)
Extracted the "walk upward from CWD and from this file's directory" logic that
`resolve_ckpt()` had inline. It yields each ancestor directory once (deduped by
resolved path) so the same search can be reused by more than one function.

### `MODEL_DIRS = ("models", "outputs/models")` (new constant)
`resolve_ckpt()` previously only looked under `outputs/models/<name>`. It now
tries both `models/<name>` and `outputs/models/<name>` at every level of the
upward walk. The `FileNotFoundError` message was updated to name both locations,
so a miss still tells you exactly where it searched.

> Note: only `outputs/models/` exists on this machine right now; `models/` is
> supported as an alternate layout, not a directory anything currently writes to.

### `discover_checkpoints() -> list[dict]` (new)
Scans `models/` and `outputs/models/` under every candidate root and returns one
dict per `.pt` file:

```python
{"id": "head_aug", "label": "Augmented",
 "path": "<absolute>", "filename": "head_aug.pt"}
```

Deduped by resolved absolute path (the same file reached from two roots appears
once) and sorted by label. **It is a pure file scan — it does not import torch or
load weights**, which is what makes it safe to call on every Streamlit rerun.

### `label_for_checkpoint(path) -> str` (new)
Maps a checkpoint stem to a display name: `head` → Baseline, `head_aug` →
Augmented, `head_robust` → Robust. Anything unrecognised falls back to a
title-cased version of the stem with the `head_` prefix stripped, so a future
`head_whatever.pt` gets a sensible name with no code change.

---

## 2. `realtime/checkpoint.py` — pass-through wrapper

Added `discover_checkpoints()`, matching the existing wrapper style in that file:
resolve the underlying module via `_module()`, call through if the function is
present, and return `[]` if it isn't. The `hasattr` guard keeps the realtime side
working against an older `score_file.py` that predates the new function.

---

## 3. `realtime/session.py` — per-model score storage

### New field
```python
self.scores_by_model: Dict[str, Dict[int, Dict]] = {}   # model_id -> {window_idx: entry}
```
alongside the existing `self.scores`, which now explicitly holds the **primary**
model's results.

### `record_score()` signature extended
```python
async def record_score(self, window_idx, score, timestamp=None,
                       model_id="primary", model_label="Primary", primary=True)
```
The three new parameters all default such that **existing callers behave exactly
as before**. The score entry is written into `scores_by_model[model_id]`, and
additionally into `self.scores` only when `primary=True`. The debug log line now
names the model that produced the score.

### `scores_by_model` added to three outputs
`status()`, `telemetry()`, and the JSON written by `save_audit()` all now carry
`scores_by_model` next to `scores`. This is **additive** — the existing `scores`
key is unchanged, so any consumer reading it keeps working.

---

## 4. `demo/app.py` — dynamic tabs, plus cosmetic cleanup

### Functional changes

- **Removed** the hardcoded `BASELINE_CKPT` / `AUGMENTED_CKPT` constants and the
  two `checkpoint_available()` probes.
- **Sidebar** now lists every discovered checkpoint (label + filename), or a
  "no `.pt` checkpoints found" line, instead of two fixed found/missing rows.
- **Tabs are generated from `discover_checkpoints()`** — one tab per checkpoint,
  each with its own Run button keyed `run_<model_id>`. Adding a fourth head to
  `outputs/models/` now surfaces a fourth tab with no edit to this file.
- **`MODEL_DESCRIPTIONS`** dict holds the per-model blurb keyed by model id, with
  a generic fallback for unknown checkpoints. The baseline blurb still carries
  the in-domain eval EER of 1.49%, unchanged from before.
- **The train-`head_aug` command moved** from a "not trained yet" warning branch
  to a retrain hint inside the augmented tab (the checkpoint now exists, so the
  old branch was dead).
- **Each tab shows the resolved absolute checkpoint path** in real mode — useful
  for confirming which file a run actually used.
- **`st.stop()`** if real mode is selected and no checkpoints were found, instead
  of rendering tabs that cannot run.
- **Mock mode** falls back to a single synthetic "Mock" tab.
- The scoring-mode radio defaults to real mode when any checkpoint exists (was:
  when the baseline specifically existed).

### Cosmetic / non-functional changes

- All em-dashes, `·` separators, `✅`/`❌`, `▶`, `🛡️`, `🧪`, and `…` replaced with
  ASCII equivalents; `page_icon` changed from `🛡️` to the string `"SONIX"`.
- `plot_timeline()` internals renamed from `_s`/`_mt`/`_step`/`_leg`/`_t` to
  `spine`/`max_x`/`step`/`legend`/`text`; the smoothing branch and the
  real-vs-mock iterator choice collapsed into conditional expressions.
- Several explanatory comments and docstrings shortened.
- Sidebar/info captions reworded; no threshold defaults changed (amber 0.10,
  red 0.90 as before).

---

## Not done / worth knowing

- **Nothing has been run.** These are source edits only — no EER number, no demo
  launch, and no verification that the three-tab UI renders, is reported here.
- `session.scores_by_model` is **written but not yet read** by any consumer; the
  demo still runs one model per tab-button press rather than scoring several
  heads into one session.
- `models/` support in `MODEL_DIRS` is speculative — no code writes there today.
- The emoji-to-ASCII sweep is unrelated to the multi-model work; if it was meant
  to fix a console/encoding issue, that reason isn't recorded anywhere in the diff.
