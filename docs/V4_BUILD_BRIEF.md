# SONIX v4 — build brief

**SIH26104 · Team SONIX · UPES Dehradun · branch `yugal/ml-pipeline`**
Written 2026-09-20. Supersedes earlier v4 plans. Read this fully before writing code.

Companion docs: `docs/PROBE_RESULTS.md` (the measurements this brief rests on),
`docs/slides/sonix_v4_pipeline.svg` (the flow), `CLAUDE.md` (environment gotchas),
`FEATURES.txt`, `PROJECT_STATE.txt`.

---

## 0. What SONIX is

Real-time detection of AI-cloned voices during a live call. Scores a 16 kHz mono stream
continuously, surfaces Green / Amber / Red, **never blocks a call — flags for a human.**
The problem statement demands: real-time (not post-hoc), diverse Indian accents and dialects,
REST/gRPC APIs and SDKs, and a reusable security layer other systems call.

Current production path: frozen `facebook/wav2vec2-xls-r-300m` → 1024-dim → 262,657-param MLP
head → sigmoid → 5-window moving average → 3-of-5 hysteresis → band. 4.0 s window, 0.5 s hop.
Seven trained heads in `realtime/models.py`, `DEFAULT_KEY = "v3"`.

---

## 1. Why v4 exists — the measurement

Run 2026-09-19 by `probe_shortcut.py` and `probe_transfer.py`, both in the repo root.
Full detail and caveats in `docs/PROBE_RESULTS.md`. These numbers are 🔒 LOCKED.

Representation tested: `last_hidden_state` → `mean(dim=1)`
(`src/extract_embeddings.py:244-246`, `realtime/frontend.py:70-71`).

| Probe | Setup | Result | Chance |
|---|---|---|---|
| A — language | 12-way, **within IndicSynth only** (all spoof, all Indic, one generator) | **99.33%** | 8.33% |
| B — corpus | IndicVoices vs ASVspoof, **bonafide rows only**, split by recording id | **100.00%** | 82.54% |
| C — transfer | spoof probe trained on English, tested on Indic | **AUC 0.5196** | 0.50 |
| C — after INLP | same, after projecting out 20 language/corpus directions | **AUC 0.5011** | 0.50 |

In-domain AUC was 0.9992 throughout and never moved.

**Read these correctly.** It is not a threshold problem — AUC 0.52 is not a well-ranked
detector at a bad operating point, it is no ranking, so Case C is eliminated. It is not
*only* a language shortcut either: if language were masking a transferable signal, projecting
language out would reveal it. Twenty directions removed and cross-domain AUC drifts *toward*
0.50. **There is no transferable synthesis signal in the vector to unmask.** The 0.9992
in-domain figure is ASVspoof corpus fingerprint — documented generator pipelines, channel,
recording conditions. Sun et al. (2026) call this "benchmark overfitting, where models exploit
quirks or attack fingerprints in the data set that fail to carry over."

**Mechanism.** XLS-R runs at ~50 fps, so a 4 s window is ~200 frames averaged into one vector.
Stationary properties — language, speaker, channel, SNR — survive averaging intact. Vocoder
artifacts are transient (phase discontinuities at frame boundaries, local spectral
over-smoothing, micro-jitter) and average into noise. Mean-pooling is a low-pass filter that
preserves every confound and destroys the evidence. `last_hidden_state` compounds it by being
the most content-abstracted layer in the stack.

**Consequence.** Group-balanced loss, OC-Softmax, DANN, MLAAD, LibriSeVoc all operate on the
vector. None can restore information pooling already removed. **Fix the representation first.**
Data composition work is necessary and strictly insufficient.

---

## 2. Hard rules — violating any of these invalidates the run

1. **`outputs/embeddings_indicvoices_ho` never enters training.** A naive glob of
   `outputs/embeddings*` swallows it. 52,402 − 5,050 = 47,352 is how you know.
2. **Augmentation is symmetric.** Identical conditioning distribution on bonafide and spoof,
   and across every corpus. Condition one side or one corpus and the effect becomes the label.
   This is exactly how `embeddings_rawboost` silently became a marker for "this row is ASVspoof."
3. **Recording-level holdout, never row-level.** `<id>_chunk_N` siblings must not straddle
   splits. Bird & Lotfi (2023) get 99.3% with row-level CV on 62 minutes of audio; that number
   is an artifact and we do not reproduce the method.
4. **Never fabricate a number.** Anything not measured is `[[PLACEHOLDER]]` with a named owner.
   A blank in front of a jury is survivable; a wrong number is not.
5. **EER, not accuracy.** Benchmark data is ~90% spoof. Always report as "X% on <split>".
6. **Score is 0–1, higher means the model thinks it is FAKE.**
7. **Never train on the held-out sets** in §5.
8. **Do not fork `src/extract_embeddings.py`.** It runs unattended on three machines. Import
   from it and override, as `src/extract_embeddings_v2.py` already does.

---

## 3. What already exists

**Written and working:**
- `probe_shortcut.py` — probes A and B
- `probe_transfer.py` — probe C + INLP
- `src/extract_embeddings_v2.py` — all hidden states, learned layer choice, mean+std pooling,
  `meta.json` sidecar. Wraps the original. **Written, not yet run.**
- `prep_mlaad.py` — MLAAD Indic fetcher, documents the ~45 h Indic slice. **Never run.**
- `make_codec.py` — G.711 µ-law only, needs AMR-WB and Opus
- `augment_folder.py`, `make_rawboost.py`, `add_noise.py`
- `src/train.py` — BCE + `pos_weight`. No group or language-aware sampling anywhere.
- `src/eval.py`, `src/metrics.py`, `src/calibration.py`, `src/merge_df21.py`,
  `src/verify_protocol.py`
- `realtime/` — full serving stack, 7 REST + WS, 7-head registry
- `diagnose_pairs.py`, `verify_head.py`, `realtime/selftest.py` — **all three never run**

**Known bugs, already fixed** (see `PROJECT_STATE.txt` §10): missing `import torch` in
`engine.py`; `_trim_pending()` subtracting an int from `None`; `live_ui` reporting stalled runs
as complete; front-end load walking HF pull requests. Every upload verdict collected before
3 September is void.

**Environment:** `activate.bat` does not work — call `.venv\Scripts\python.exe` directly.
Set `HF_HUB_OFFLINE=1` before any server run. Repo lives under OneDrive, which syncs `.git/`
and causes stale `index.lock` — moving it off OneDrive is recommended. Windows checkout is
CRLF, repo stores LF; committing from a Linux shell without `-c core.autocrlf=true` rewrites
the tree. GPUs: 1× RTX 3050, 2× RTX 4060.

---

## 4. Datasets

**Critical constraint: embeddings cannot be re-pooled.** Every existing `.npy` root is frozen
in the representation proven broken above. Only raw audio has forward value.

### Have raw audio
`data/asvspoof19_la/` (train/dev/eval + protocols) · `data/indic_spoof/` + `_aug/` (800 + 800
MMS-TTS wavs) · `data/asvspoof5/raw/` (never prepped) · `sonix_real/` (39 clips)

### Embeddings only — must re-acquire audio or they are lost
| Asset | Rows | Action |
|---|---|---|
| IndicVoices train | 37,032 | re-download from AI4Bharat; not under `data/` anywhere |
| IndicVoices holdout | 5,050 | same |
| IndicSynth | 16,800 | wavs are on Akshat's machine only |

### Derived — regenerate in S3, do not re-download
`embeddings_g711`, `embeddings_rawboost`, `embeddings_rirmusan_*` are ASVspoof with an effect
applied. Regenerate across **every** corpus this time.

### Download
IndicVoices · IndicSynth (from Akshat) · **MLAAD Indic** (`hi,bn,ta,mr,kn,ml,ur`, CC BY-NC 4.0 —
fine for SIH and a paper, not for a commercial claim) · **In-the-Wild** (`data/in_the_wild/` has
only a README) · **LibriSeVoc** (6 vocoders + 13,201 English real + the multitask label) ·
**Common Voice Indic** (~15k, CC0) · **LibriSpeech train-clean-100** (~10k) ·
**DEEP-VOICE** (Kaggle, RVC family)

### Generate
XTTS-v2 and OpenVoice v2 clones in Indic languages, **speakers disjoint from `sonix_real/`**
so that set stays a valid test.

### Target composition (~68k bonafide / 86k spoof)
| Bonafide | Rows | | Spoof | Rows |
|---|---|---|---|---|
| IndicVoices | 37,032 | | LibriSeVoc | ~25,000 |
| Common Voice Indic | ~15,000 | | MLAAD Indic | ~20,000 |
| LibriSeVoc real | 13,201 | | IndicSynth | 16,800 |
| ASVspoof19 | 2,580 | | ASVspoof19 **subsampled from 91,200** | ~15,000 |
| | | | zero-shot clones | ~8,000 |
| | | | MMS-TTS | 1,600 |

Smaller than today's 173,752 rows and better composed. Today's split gives
P(spoof | English) = 89.8% vs P(spoof | Indic) = 48.7% — a 41-point language prior. The target
brings English to ~16k real / 40k spoof and group balancing closes the rest.

---

## 5. Held out — never trained on, at any stage

`outputs/embeddings_indicvoices_ho` (5,050) · `sonix_real/` · **In-the-Wild** (the honest
cross-dataset number) · **DEEP-VOICE / RVC** (leave-one-family-out) · ASVspoof19 LA eval
(71,237) · ASVspoof 5 eval · ASVspoof21 DF (report with "75% key coverage") · WaveFake
(comparability only)

---

## 6. The build

### S0 — Diagnose (blocks everything, ~1 hour)
Run `diagnose_pairs.py --dir sonix_real\sonix_real`, then again with
`--extra-fake data\indic_spoof`. Run `verify_head.py` to settle or retire the contested
1.4937% headline. Run `python -m realtime.selftest --ckpt outputs\models\head_v3.pt` for the
real-time factor — the number the PS title demands, which does not exist yet.

### S1 — Manifest
One row per clip: `path, recording_id, corpus, language, generator_family, channel, label,
split`. Group-balanced sampling, per-family reporting, leave-one-family-out and leak checking
all depend on this. Today the metadata lives only in folder names.
**Exit:** `verify_protocol.py` passes; no `recording_id` in two splits.

### S2 — Acquire
Everything in §4. Parallelise; IndicVoices and IndicSynth are the critical path.

### S3 — Condition
Extend `make_codec.py` to **AMR-WB** and **Opus** (ffmpeg + opencore-amr) alongside G.711.
One augmentation pass per source sampling from {none, G.711, AMR-WB, Opus} × {RIR, MUSAN,
pink noise, RawBoost}. **One conditioned copy per corpus, not parallel roots** — the old four
parallel ASVspoof roots are what created the 4× duplication.
**Exit:** assert per-class augmentation distributions match, per corpus. Fail the build if not.

### S4 — Extract ★ the change everything rests on
`src/extract_embeddings_v2.py` is written. Two phases:

1. **Sweep.** Small sample, all layers, mean+std. ASVspoof audio is already on disk so this
   needs no download:
   ```
   .venv\Scripts\python.exe src\extract_embeddings_v2.py --split train --limit 50 --batch 4 --layers all --out outputs\emb_v2_sweep
   ```
   Expect shape `(50, 51200)` = 25 layers × 2048. Then `--limit 3000`, and the Indic fake side
   via `--audio-dir data\indic_spoof`.
2. **Layer probe.** Re-run probes A, B and C **per layer** on the sweep output. Find the layer
   where language decodability is lowest and cross-domain spoof AUC is highest. That single
   result decides the v4 front-end config.
3. **Commit.** Full extraction with `--layers <winner>`.

All 25 layers at meanstd is 51,200 dims ≈ 102 KB/clip in float16 — fine for a few thousand
clips, not for 126k. Sweep first, then narrow.

Optional after the sweep: keep the full frame sequence and put **AASIST** on it.
AASIST, RawNet2 and OC-Softmax are already declared as next stages in
`docs/handoff/SONIX_REFERENCE.md` §10 — they are promised, not optional.

### S5 — Compose
Subsample ASVspoof spoof 91,200 → ~15,000, stratified by attack id. Assemble §4's table.
**Exit:** no group cell below 10k; no recording straddling splits; `_ho` absent.

### S6 — Train
Three changes to `src/train.py`, in order:
1. **Group-balanced sampling** — groups `{language} × {generator_family} × {label}`,
   weights `N / (n_groups × n_g)`. Replaces the global `pos_weight`, which is precisely what
   leaves the language prior intact.
2. **OC-Softmax** replacing BCE — one-class, built for unseen attacks.
3. **Multitask head** — `[spoof/bona] + [generator family ID]`. Family labels are free for
   LibriSeVoc (6 vocoders), MLAAD (model id), IndicSynth, MMS-TTS, the zero-shot clones, and
   ASVspoof attack ids. Vocoder artifacts are language-independent, so forcing the model to
   name the generator makes "it's Hindi" an invalid answer. In Sun et al. Table 8 the
   vocoder-aware multitask system has the best open-set EER in the table (4.54%).
4. **DANN / gradient reversal on a language head** — only if S0 returns Case A.

Wire `wandb` or `mlflow` here. Nine heads exist and nobody can reconstruct what any of them
saw; that ambiguity is why `head.pt` was lost to a merge.

### S7 — Select
Early-stop and select on **worst-group EER**, not mean. Selecting on the mean lets the
Indic-fake cell collapse to zero contribution unnoticed — that is how `head_v3` happened.
Dev EER is a wiring check and belongs in an appendix, never a headline.

### S8 — Evaluate, open-set only
`sonix_real/` matched pairs (AUC, EER, median real/fake) · **In-the-Wild** (the headline) ·
DEEP-VOICE leave-one-family-out · IndicVoices holdout (per-4s-chunk false-alarm rate) ·
ASVspoof19 eval · DF21 with its caveat · a per-codec × language × family grid.
**Baselines required:** AASIST and RawNet2 on the same Indic test set. Without them a paper is
a desk reject and a jury will ask.

**Expectation.** Sun et al. Table 8 puts open-set EER at 4.5–13.5%, with Wav2Vec2-XLS-R — the
same front-end — at 13.48%. Landing in that band is success. Do not chase 0.24%.

### S9 — Integrate
Register `v4` as `DEFAULT_KEY` in `realtime/models.py`. Keep the other heads loaded **for
comparison only** — that is the swap-the-judge story, not a production ensemble. Fit
per-domain thresholds from S8 distributions and state openly that they are domain-conditional.
Re-run `selftest.py` for the final RTF. Do the cost-per-call-minute arithmetic.

---

## 7. The gate — `verify_v4.py`

Build this once. It blocks S9 and fails loudly on any of:

```
 1. holdout leak       embeddings_indicvoices_ho absent from the train manifest
 2. recording leak     no recording_id appears in two splits
 3. aug symmetry       per-class conditioning distributions match, per corpus
 4. group floor        every group cell >= 10k rows
 5. language probe     accuracy DROPPED vs the v3 baseline (99.33%)
 6. corpus probe       accuracy DROPPED vs the v3 baseline (100.00%)
 7. transfer probe     cross-domain AUC ROSE vs the v3 baseline (0.5196)
 8. SNR slope          score-vs-SNR on held-out bonafide is flat
 9. padding regression 1s/2s/3s/66s reproduce the locked values
10. sonix_real gate    Indian-subset AUC > 0.90
11. open-set floor     In-the-Wild EER reported, within 4.5-15%
12. family LOO         DEEP-VOICE degradation measured, not skipped
13. latency            RTF from selftest.py, PASS against the 0.5 s hop budget
```

Checks 5, 6 and 7 are what prove the shortcuts are dead. They are also measurements nobody in
the reference literature has made, which is why they carry the paper.

---

## 8. New failure modes to guard against

- **LibriSeVoc becomes a corpus shortcut.** Its 13,201 real clips are all LibriSpeech — one
  channel. Must receive identical S3 conditioning, and must be included in the corpus probe.
- **The multitask head learns corpus, not vocoder.** If each generator family appears in
  exactly one corpus, "which vocoder" and "which corpus" are the same question. Check the
  confusion matrix; a perfectly block-diagonal-by-corpus result means it is cheating.
- **Group balancing starves a real cell.** Weighting a 200-row cell up to parity overfits those
  200 rows. Enforce the 10k floor or merge into a coarser group.
- **Worst-group selection flattens everything downward.** Report worst-group *and* mean. If the
  mean collapses, back off to a weighted combination.

---

## 9. How to work on this

Casual and direct. Concise over elaborate. Honest over accommodating.

For anything being built here: **explain the pipeline and the steps first, then code in blocks
with explanation — not a finished file dumped in one go.**

Never fabricate a result to make a slide look better. If something is unverified, say so and
mark it `[[PLACEHOLDER]]`.

Do not say: "99% accurate" · "detects any AI voice" · any latency figure until measured ·
"we have a production API" (say "a working service interface") · any number not locked in
`docs/PROBE_RESULTS.md` or `PROJECT_STATE.txt`.

---

## 10. Start here

1. Run S0's three commands. They are one hour and may change the plan.
2. Run the S4 sweep smoke test — it needs no download.
3. Write the per-layer probe that reads the sweep output and picks the front-end layer.

Everything after step 3 is mechanical.
