# CODE_MAP.md — repo structure for agents

Hand this to any agent working in SONIX so it knows where things live before it
reads a line of code. Pair it with `CLAUDE.md` (gotchas + lanes),
`PROJECT_STATE.txt` (merge state, blocking bugs), `FEATURES.txt` (what runs).
Symbol lists below are AST-extracted, not hand-maintained — treat them as an
index, not a spec.

## What SONIX is

Real-time detection of AI voice-cloning on calls. One pipeline, reused everywhere:

```
4s audio @16kHz mono
  -> frozen wav2vec2 XLS-R 300M  (never trained)
  -> 1024-dim embedding, mean-pooled
  -> trained MLP head ~300k params  (the only trained part; head.pt)
  -> risk score in [0,1] = P(synthetic)
```

Metric is **EER**, always "X% on <split>", never a bare number. Bands:
Green <0.35, Amber 0.35–0.65, Red ≥0.65 (provisional, uncalibrated).

## Directory map

| Path | Purpose | Owner (lane) |
|---|---|---|
| `src/` | Offline benchmark pipeline: verify → extract → train → eval | Yugal (ML) |
| `extract_embeddings.py` (root) | **Canonical** embedding extractor (moved out of `src/` in a merge; `src/extract_embeddings.py` is gone) | Yugal |
| `realtime/` | Live detection server (aiohttp), consent gate, VAD, audit trail, dashboard | Suryansh + Yugal bridge |
| `realtime/tests/` | pytest suite for the live server (session, engine, checkpoint bridge, mic page) | — |
| `demo/` | Streamlit demo UI — score a file, watch it stream. Mock mode works; real mode needs head.pt | Suryansh |
| `uidemo/SONIX_Suryansh_Demo_UI_v9/` | **Older snapshot** of `demo/` — near-duplicate, do not edit, read `demo/` instead | Suryansh |
| `extension/` | Chrome MV3 extension — captures Google Meet **tab** audio, thin client, all detection in Python | Suryansh |
| `website/` | Static marketing/demo site (`index.html` + `script.js` + `styles.css`), talks to the realtime server API | — |
| `scripts/` | IndicVoices extraction/embed experiments (out-of-distribution eval data) | Navya/Akshat |
| `docs/` | Architecture, per-lane handoff notes, results table, PPT reference | all |
| `docs/handoff/scripts/` | One-off probe scripts (`build_probe*.py`, `tier_a.py`) — not part of any pipeline | — |
| `_anubhav_sample/` | 60-file mini ASVspoof dataset in real `--data-root` layout, for smoke tests | Anubhav |
| `outputs/models/` | `head*.pt` checkpoints — **gitignored**, copied by file transfer, not in any branch | Yugal |
| root `*.py` (add_noise, augment_folder, make_*, prep_*, split_*, join_labels, …) | Data-prep / augmentation one-offs (RawBoost, RIR, codec, noise, dataset prep). Standalone `main()` scripts. | Anubhav / Navya |

## The four entry points

### 1. Offline benchmark (produces THE EER number)
```
src/verify_protocol.py        gate — must print PASS, run FIRST
extract_embeddings.py         audio -> cached embedding shards (SLOW, resumable, never re-run from scratch)
src/train.py                  trains MLP head -> outputs/models/head.pt
src/eval.py                   EER
src/make_codec.py             codec-robustness degradation (G.711 etc)
```
`train.py` and `eval.py` share `eer() / load_split() / build_head()` — near-identical, keep them in sync.

### 2. Streamlit demo
```
demo/app.py            UI
demo/windowing.py      wav -> overlapping 4s windows
demo/risk.py           moving_average + hysteresis_bands so the band doesn't flicker
demo/streaming.py      yields scores one at a time (fake live)
demo/score_file.py     integration contract: score_file(wav) -> list[float]
```
Run: `streamlit run demo/app.py` (:8501).

### 3. Live detection server
```
python -m realtime.server --ws-port 8000 --mode webrtc [--ckpt outputs/models/head.pt] [--max-batch-size 4]
streamlit run realtime/live_ui.py     # dashboard
```
Call chain: `server.py` (aiohttp, WS + HTTP API) → `session.py` (consent state machine, audit) → `engine.py` (batches windows, runs model off the event loop) → `checkpoint.py` (loads `demo/score_file.py` by path as the real scorer) / `mock.py` (fallback, capped 0.5, never Red).
Modes: `voip` (AudioSocket TCP :5000), `webrtc` (mic page + extension), `upload`.
Consent gate is real: `Session.push_audio()` discards every chunk until state is LISTENING.

### 4. Browser capture
- `http://localhost:8000/mic` — `realtime/miccapture.py` serves the page, streams 16kHz PCM to `/ws`.
- `extension/` — `background.js` ↔ `offscreen.js` ↔ `worklet.js` capture tab audio; `content.js` draws the overlay; `popup.js` the popup. Load unpacked, Chrome/Edge 116+.

## Key modules (path → job → main symbols)

### src/
- **src/verify_protocol.py** — dataset/protocol gate. `load_protocol`, `_check_split`, `main`.
- **src/extract_embeddings.py** — *(this is the stale `src/` copy; root `extract_embeddings.py` is canonical and has the same symbols plus `--audio-dir` mode)*. `build_manifest`, `_build_manifest_dir`, `_build_manifest_asvspoof`, `_build_manifest_itw`, `load_audio_fixed`, `load_frontend`, `embed_batch`, `save_shard_atomic`, `run`, `_auto_device`, `build_argparser`.
- **src/train.py** / **src/eval.py** — `eer`, `load_split`, `build_head`, `main`.
- **src/score_file.py** — offline scorer. `configure`, `_lazy_load`, `_load_audio`, `_windows`, `score_file`.
- **src/metrics.py** — Yukti's lane. `calculate_eer`, `create_det_curve`, `create_score_histogram`.
- **src/calibration.py** — temperature scaling. `find_best_temperature`, `calibrate_scores`.
- **src/merge_df21.py**, **src/padding_plot.py** — DF21 merge, padding analysis plots.

### realtime/
- **server.py** — `SonicServer`: `websocket_handler`, `http_*_handler` (approve/end-call/telemetry/models/upload/status), `_on_new_call`, `_on_pairing_approved`, `_on_scores_ready`, `_broadcast`, `_adaptive_vad_floor`, `run`, `main`.
- **session.py** — `Session` + `CallState`/`AuditEntry`/`CallMetadata`: `request_consent`, `push_audio` (consent gate), `get_pending_windows`, `record_score`, `end_call`, `save_audit`, `telemetry`, `to_dict`.
- **engine.py** — `ScoringEngine`: `ensure_model`, `preload`, `model_catalogue`, `_collect_batch`, `_embed_windows`, `_score_windows`, `_head_forward`, `add_session`, `remove_session`, `run`, `get_stats`.
- **checkpoint.py** — `StandardisedHead` (`forward`), `load_checkpoint`. Loads the torch checkpoint **directly** and bakes mu/sd into the module. It does *not* bridge to `demo/score_file.py` any more — there is no `_module()` and no `checkpoint_available` here; that one lives in `demo/score_file.py:121` for the Streamlit demo.
- **frontend.py** — wav2vec2 wrapper. `load`, `embed`.
- **models.py** — checkpoint resolution. `resolve_ckpt`, `key_for_path`, `catalogue`.
- **ringbuffer.py** — `RingBuffer` (capacity 64000 @ 16kHz): `push`, `get_emitted_windows`, `reset`, `stats`.
- **vad.py** — `VAD`: `_zcr`, `is_speech`, `stats`. Silence gate.
- **pairing.py** — `PairingCodeManager`: `generate` (6-digit, 120s expiry).
- **resample.py** — `resample`, `to_mono`, `pcm16_to_float32`, `float32_to_pcm16`.
- **source.py** — `SourceAdapter` / `WavFileSource` / `MicSource`.
- **audiosocket.py** — `AudioSocketServer` (Asterisk VOIP, port 5000).
- **mock.py** — `MockScorer.score` — random, capped 0.5.
- **live_ui.py** — Streamlit dashboard. `band_for`, `hysteresis_bands`, `risk_chart`, `audio_path_chart`, `upload_chart`, `render_upload_result`, `run_upload_stream`, plus `AMBER_AT`/`RED_AT` threshold constants.
- **miccapture.py** — `mic_page_handler` (serves `/mic`; JS thresholds must match `live_ui.py` or `test_miccapture.py` fails).
- **make_dev_head.py** — writes an untrained `head_dev.pt` stamped `synthetic=True` for plumbing tests.
- **selftest.py** — acceptance test + latency benchmark for a checkpoint.

### demo/
- **score_file.py** — `score_file`, `score_stream`, `configure`, `resolve_ckpt`, `checkpoint_available`, `_load_frontend`, `_load_head`, `_score_windows`, `dev_eer`, `set_vad`/`get_vad`.
- **windowing.py** — `load_audio_mono_16k`, `make_windows`, `audio_duration_seconds`.
- **risk.py** — `band_from_score`, `moving_average`, `hysteresis_bands`, `process_scores`.
- **streaming.py** — `iter_windows`, `demo_score_stream`.
- **model_adapter.py** — `yugal_score_stream` (binds real mode).
- **app.py** — `run_analysis`, `plot_timeline`, `fake_score_stream`, `real_score_stream`, `band_info`.

## Known duplication / traps

- **`score_file.py` exists 3×**: `src/`, `demo/`, `uidemo/.../`. `realtime/checkpoint.py` loads the **`demo/`** one by path. `src/` version is the offline path. Which one "real mode" officially binds to is unsettled — check before assuming.
- **`make_codec.py` exists 2×**: repo root and `src/`. Same 3 functions (`have_ffmpeg`, `transcode_g711`, `main`).
- **`extract_embeddings.py`**: root copy is canonical; `src/` copy is stale but still on disk. CLAUDE.md/README still describe the old `src/` layout.
- **Band thresholds**: `realtime/thresholds.py` is now the single source (`0.35/0.65`), pinned across `miccapture.py`, `website/script.js` and the extension by `realtime/tests/test_thresholds.py`. The other numbers are *not* competing definitions: `demo/app.py` and `realtime/live_ui.py:185` expose Streamlit **sliders** an operator moves at runtime (defaults `0.10/0.90`), and `demo/test_suryansh.py` `0.45/0.70` is a test fixture. Do not "reconcile" those into the production bands.
- **`uidemo/SONIX_Suryansh_Demo_UI_v9/`** is a frozen older copy of `demo/` — 11 near-identical files. Ignore unless doing UI archaeology.
- **`demo/test_suryansh.py`** collects zero pytest tests (asserts run at import). Run it directly: `python demo/test_suryansh.py`.
- **No `head.pt` in any branch** — gitignored, file-transfer only. Server falls back to mock and reports `scoring_available=False`.
- **`fp16 NaN bug`** (PROJECT_STATE.txt §3): `extract_embeddings.py` `model.half()` on CUDA → all-NaN embeddings, passes the shape-only self-check silently. Not yet fixed.

## Lanes — do not edit across them

| Owner | Owns | Ask before touching |
|---|---|---|
| Yugal | all ML: `src/train.py`, model arch, `head.pt`, `extract_embeddings.py` | anything under `src/` that scores or trains |
| Anubhav | eval-split extraction, codec-robustness runs, root data-prep scripts | `make_codec.py`, `extract_embeddings.py --split eval` |
| Suryansh | `demo/`, `realtime/` UI, windowing/smoothing/hysteresis, `score_file` contract | `demo/*`, `realtime/live_ui.py`, `realtime/miccapture.py` |
| Navya / Akshat | `scripts/` IndicVoices, In-the-Wild extraction, TTS demo audio | `scripts/*` |
| Yukti | `src/metrics.py`, EER validation, score plots, results table | `src/metrics.py` |

## Prompting other agents efficiently

- Name the **entry point** ("the offline pipeline", "the realtime server", "the Streamlit demo") — each is a self-contained file chain above.
- Point at the **module**, not the repo: e.g. "consent gate → `realtime/session.py:push_audio`".
- State which **`score_file.py`** you mean by path.
- Give the agent `CLAUDE.md` + `PROJECT_STATE.txt` + this file. Don't make it rediscover the fp16 bug, the missing dataset, or the missing `head.pt`.
- Never ask for a number a script hasn't produced. Recorded result: **1.49% EER on ASVspoof2019 LA eval** (teammate's machine, `docs/handoff/SONIX_REFERENCE.md`).
