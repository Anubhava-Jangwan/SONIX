# CLAUDE.md

Project memory for Claude Code sessions in this repo. Full architecture and setup
docs live in `README.md` and `docs/00_PROJECT_CONTEXT.md` — read those for context,
this file is for facts and gotchas to apply every session.

## What this project is

SONIX (SIH26104) — real-time detection of AI voice-cloning on phone calls. Frozen
wav2vec2 XLS-R (300M) turns 4s audio windows into 1024-dim embeddings; a small
trained MLP head (~300k params) scores them. Only the head is trained. Metric is
**EER**, not accuracy — report it as "X% on <split name>", never a bare number.

## Team & ownership — don't touch someone else's lane

- **Yugal** — all ML code: `train.py`, model architecture, `head.pt`. Ping him
  before changing anything under `src/` that isn't your own task's script.
- **Anubhav** — eval-split extraction (`extract_embeddings.py --split eval`),
  codec-robustness runs (`make_codec.py`), opening pitch.
- **Suryansh** — demo UI (Streamlit), windowing/smoothing/hysteresis, integration
  via `score_file(wav) -> list[float]`.
- **Navya / Akshat** — In-the-Wild extraction / demo audio generation (TTS clips).
- **Yukti** — `metrics.py`, EER validation, score-distribution plots, results table.

## Hard rules (from the team's own handoff docs)

1. **Never invent a number.** If a script hasn't produced a real result yet, say so.
2. **If a script fails, paste the full traceback**, not a paraphrase or summary.
3. **Don't modify a script you were handed** (e.g. `extract_embeddings.py`,
   `make_codec.py`) without checking with Yugal — other people's runs and Yugal's
   scoring step depend on its exact output format.
4. **Cache everything expensive.** Embedding extraction is the slow step; never
   suggest re-extracting when a resumable rerun would pick up where it left off.
5. **Mock before you integrate** (UI work) — don't block on a teammate's output.

## Path/environment gotchas — verified against this exact machine

- **PyTorch + CUDA on Python 3.13**: `cu121` has **no wheels for Python 3.13** —
  install fails with "Could not find a version that satisfies the requirement
  torch". Use `cu124` or `cu126` instead:
  `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126`
- **`--data-root` must point at the folder that directly contains**
  `ASVspoof2019_LA_eval/`, `ASVspoof2019_LA_train/`, `ASVspoof2019_LA_dev/`, and
  `ASVspoof2019_LA_cm_protocols/`. The official `LA.zip` sometimes extracts into a
  doubled `LA/LA/...` structure — flatten it, don't just pass a deeper path and
  hope. Current canonical location on this repo: `data/asvspoof19_la/`.
- **Verify dataset completeness before any long run**, not after:
  `find data/asvspoof19_la/ASVspoof2019_LA_eval/flac -name "*.flac" | wc -l`
  should be **71933**. A partial pendrive copy or interrupted `LA.zip` download
  can silently leave this far short (seen: 21844/71933, i.e. 69% missing) — this
  surfaces downstream as mass `! missing source` / `! SKIP` errors in
  `make_codec.py` or `extract_embeddings.py`, not as an upfront failure.
- **Shell is Git Bash (MINGW64) on Windows.** Use `find`, not `dir`. `ls -la`
  works. Paths like `/b/SIH_sonix` map to `B:\SIH_sonix`.
- **`winget install`** requires closing and reopening the terminal afterward for
  PATH to pick up new tools (e.g. ffmpeg) — a fresh terminal, not just re-`cd`.
- **The official `LA.zip` is ~7.12 GB.** If a downloaded copy is far smaller
  (e.g. tens of MB) and `unzip -l` reports "End-of-central-directory signature not
  found," the download didn't finish — don't attempt to extract it; check the
  browser's download status and re-verify the file size before retrying.

## Run order (see README.md for full detail)

```bash
python src/verify_protocol.py                 # gate — must print PASS, always run first
python extract_embeddings.py --split eval --batch 4 --limit 50   # smoke test
python extract_embeddings.py --split eval --batch 4              # real run, resumable
python src/eval.py --split eval                # THE number
```

## Sanity-checking EER

- low single digits → working
- ~40% → bug, almost always protocol parse or a flipped label → re-run the gate
- exactly 0% → eval leaked into training
- In-the-Wild / codec-degraded EER dramatically worse than clean eval → **expected**,
  this is the project's headline generalization-gap finding, not a bug to hide.
