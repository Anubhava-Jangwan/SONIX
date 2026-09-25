# GPU handoff — confirm the "clean background = fake" shortcut

**SONIX / SIH26104 · branch `yugal/ml-pipeline` · 2 Sept 2026**

You have a CUDA GPU; this box does not. Everything below runs on CPU too (that is
how the numbers in §2 were produced) — it is just ~5–20× slower. Please run §4,
paste the four tables from §5 back, and attach the two output files. **Read-only
scoring — no training, no re-extraction, do not touch the demo checkpoint.**

---

## 1. What we're testing

The trained head `outputs/models/head.pt` (frozen wav2vec2-XLS-R front-end +
~300k-param MLP, mean-pooled 4 s windows) appears to score audio by **background
noise level**, not by synthesis artefacts:

- clean-background recording → scored **fake**
- same recording with audible room tone → scored **real**
- an AI clone that already has a noise floor → sails through as **real**

§6 of `SHORTCUT_INVESTIGATION.md` established this on two clip pairs (numbers in
§2). This handoff runs the **Tier A confirmation battery** — dose-response curves,
a no-speech control, and a correlation across the whole clip bank — which turns
"looks like it on two pairs" into a measured effect. GPU just makes it quick and
lets you widen the sweeps.

## 2. What is already measured (CPU, this branch — cross-check against these)

`diagnose.py` speech-window mean, VAD off. Higher = more "fake". Threshold ≈ 0.5.

**Pair 1 — `real_03` (clean laptop-mic genuine, English) / `fake_03` (downloaded clone)**

| clip | `head.pt` | reads as |
|---|---|---|
| real_03 clean | 0.916 | fake ❌ |
| real_03 + pink noise 30 dB SNR | 0.180 | real |
| real_03 + pink noise 20 dB SNR | 0.000 | real |
| fake_03 clean | 0.874 | fake ✅ |
| fake_03 + noise 30 dB | 0.189 | real ❌ |
| fake_03 + noise 20 dB | 0.000 | real ❌ |
| real_03 clipped to 9 s (no noise) | 0.842 | fake ❌ |

**Pair 2 — Mann Ki Baat broadcast (genuine, Hindi) / `output.wav` (AI clone of it)**

| clip | `head.pt` | reads as |
|---|---|---|
| Mann Ki Baat clean (30 s) | 0.002 | real ✅ |
| + noise 30 dB | 0.000 | real ✅ |
| + noise 20 dB | 0.000 | real ✅ |
| **output.wav (AI Modi) clean** | **0.131** | **real ❌ (missed deepfake)** |
| output.wav + noise 30 dB | 0.006 | real ❌ |
| output.wav + noise 20 dB | 0.000 | real ❌ |
| Mann Ki Baat clipped to 9 s | 0.000 | real ✅ |

**Reading (`SHORTCUT_INVESTIGATION.md` §7, row 2):** real and fake collapse toward
"real" together under noise; clip length barely moves the score (0.842 vs 0.916,
0.000 vs 0.002). The score tracks background level, not whether a human or a
vocoder produced the voice. `head_aug.pt` shows the same pattern and is slightly
worse on the clean genuine clip. `head.pt` stored **dev EER = 0.16 %** — a
near-zero EER on a task this hard is itself a shortcut fingerprint.

Your GPU run should reproduce these within ~1e-3 (fp32 CUDA == CPU, per
`PROJECT_STATE.txt` §3). If they differ wildly, stop — see §6.

## 3. What you need

**Hardware/driver:** any CUDA GPU (4 GB+ is plenty; the front-end is 300 M params,
fp32 inference).

**Repo:** branch `yugal/ml-pipeline` of `github.com/Anubhava-Jangwan/SONIX`.
Scripts are at `docs/handoff/scripts/` (`build_probe.py`, `build_probe2.py`,
`tier_a.py`) — also pasted in Appendix A if you only got this file.

**Python env** (`data/` and `outputs/` are git-ignored, so is every `.venv*`):

```bash
python -m venv .venv-gpu
. .venv-gpu/Scripts/activate            # Windows Git Bash;  .venv-gpu/bin/activate on Linux
# CUDA build of torch — cu124 or cu126. NOTE cu121 has NO Python 3.13 wheels.
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install transformers soundfile numpy
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

**Model:** `facebook/wav2vec2-xls-r-300m` auto-downloads from HF on first run
(~1.2 GB). Or copy `~/.cache/huggingface/hub/models--facebook--wav2vec2-xls-r-300m`
from this box.

**Files to get from us out-of-band** — git-ignored, a `git pull` will NOT bring them:

| path | size | needed for |
|---|---|---|
| `outputs/models/head.pt` | 1.04 MB | **everything (required)** |
| `outputs/models/head_aug.pt` | 1.04 MB | optional — re-run Tier A with `--ckpt` to compare |
| `outputs/models/head_robust.pt` | 1.04 MB | optional |
| `sonix_real/sonix_real/fake/output.wav` | 2.3 MB | Pair 2 (AI Modi clone) |
| `sonix_real/sonix_real/real/जगल…137th edition of Mann Ki Baat…2026.wav` | 19 MB | Pair 2 (genuine broadcast) |

`real_01..10.wav`, `fake_01..10.wav`, `clips_manifest.csv` **are** in git — you
already have them after a pull. If the two Pair-2 files are inconvenient to
transfer, skip them: `build_probe2.py` and the Pair-2 rows will just be omitted,
and Pair 1 + A2 + A3-on-the-numbered-bank still give the full confirmation.

## 4. Run

From the repo root, with the env active:

```bash
# 1. build the probe clips (16 kHz mono; needs no ffmpeg)
python docs/handoff/scripts/build_probe.py  --repo-root .
python docs/handoff/scripts/build_probe2.py --repo-root .     # skip if no Pair-2 files

# 2. the confirmation battery — auto-uses the GPU
python docs/handoff/scripts/tier_a.py --repo-root . 2>&1 | tee docs/handoff/scripts/tier_a_gpu.log

# 3. OPTIONAL — same battery through the augmented head
python docs/handoff/scripts/tier_a.py --repo-root . --ckpt outputs/models/head_aug.pt \
    2>&1 | tee docs/handoff/scripts/tier_a_gpu_aug.log
```

`tier_a.py` loads the front-end once, prints `front-end device = cuda`, then
scores everything. Expect a few minutes on GPU. It writes
`docs/handoff/scripts/tier_a_result.json`.

## 5. Send back

1. `docs/handoff/scripts/tier_a_result.json` (and `_aug` if you ran it)
2. `docs/handoff/scripts/tier_a_gpu.log`
3. the four tables below, filled from the run

### A1 — SNR dose-response  (score per clip at each SNR; last column = Spearman(score, SNR))

| clip | clean | 40 dB | 35 dB | 30 dB | 25 dB | 20 dB | 15 dB | 10 dB | ρ(score,SNR) |
|---|---|---|---|---|---|---|---|---|---|
| real_03 | | | | | | | | | |
| fake_03 | | | | | | | | | |
| output.wav | | | | | | | | | |

### A2 — pure noise, NO speech

| input | RMS dBFS | score |
|---|---|---|
| digital silence | ~−100 | |
| pink noise | −60 | |
| pink noise | −50 | |
| pink noise | −40 | |
| pink noise | −30 | |
| pink noise | −20 | |
| pink noise | −15 | |
| **Spearman(score, level)** | | |

### A3 — score vs SNR across the 22-clip bank

| correlation of `head.pt` score with | r |
|---|---|
| measured SNR (Pearson) | |
| measured SNR (Spearman) | |
| clip duration (Pearson) | |
| speech loudness dBFS (Pearson) | |
| true label fake=1 (Pearson) | |

### A4 — background-only (speech frames stripped, room tone scored)

| clip | bg seconds | score |
|---|---|---|
| (rows the script prints; many clips have too little non-speech — note which) | | |

## 6. How to read it

**Confirms the shortcut (score is a background-level readout):**

- **A1** ρ(score, SNR) ≈ **+0.8 … +1.0** for every clip, real and fake alike —
  cleaner ⇒ faker, monotonically. A single clip's verdict flips from fake to real
  purely by adding noise.
- **A2** digital silence / quiet pink noise → score near **1** (fake); loud pink
  noise → score near **0** (real); Spearman(score, level) ≈ **−0.8 … −1.0**. The
  model returns a verdict on audio **with no voice in it at all** — decisive.
- **A3** |r(score, SNR)| **high** (≈ 0.7+) while |r(score, duration)| and
  |r(score, loudness)| stay **low**. That isolates the cue as *noise level*, not
  clip length (`SHORTCUT_INVESTIGATION.md` §4.2) or gain.
- **A4** genuine-background and fake-background score **similarly** (and not near
  0) — voice content is not required for the model to commit.

**Would argue against it / point elsewhere:**

- A1 curves flat → noise is not the cue; check the A3 duration correlation instead.
- A2 flat or random → the boundary needs speech present; weaker "noise meter" claim.
- A3 r(score, SNR) low but r(score, duration) high → it's silence *duration*
  (Müller et al. 2021), not level — different fix, do not buy noise datasets.

**Sanity:** the script prints `dev_eer(stored) = 0.00159…`. GPU A1 "clean" scores
should match §2 (`real_03` ≈ 0.92, `fake_03` ≈ 0.87, `output.wav` ≈ 0.13) within
~1e-3. Large disagreement ⇒ suspect an fp16/NaN path — confirm no `.half()` is
applied (the demo pipeline doesn't; `extract_embeddings.py` does and is unrelated
here) and that `--ckpt` resolved to the real `head.pt`.

## 7. Optional — Tier B (only if you have the ASVspoof2019 LA **train** split)

This box does not have it (`PROJECT_STATE.txt`: dataset 69 % absent). If you do,
this is the root-cause check — does the cue exist in the training data itself:

1. From `ASVspoof2019.LA.cm.train.trn.txt`, sample ~400 bonafide + ~400 spoof.
2. Per clip measure: noise floor (RMS of quietest 10 % of 1024-sample frames),
   SNR, and leading+trailing silence duration (frames < −45 dBFS).
3. Report mean ± sd per class and the AUC of "noise floor alone predicts
   bonafide". If bonafide is systematically noisier or longer-silenced than
   spoof, the head was rewarded for learning background — hypothesis confirmed at
   the source. (No model needed; pure signal stats.)

## 8. Do NOT

- retrain the head or re-extract embeddings (hours on the 4060; defer past the
  PoC — `SHORTCUT_INVESTIGATION.md` §11)
- modify `extract_embeddings.py`, `make_codec.py`, `make_rawboost.py`,
  `add_noise.py`, `score_file.py`, `train.py` (other people's runs depend on the
  exact output format — `CLAUDE.md` rule 3)
- swap or overwrite `outputs/models/head.pt` — the demo depends on it unchanged

## Appendix A — scripts

If you only received this file, save these under `docs/handoff/scripts/`. They are
already on branch `yugal/ml-pipeline`.

> `build_probe.py`, `build_probe2.py`, `tier_a.py` — see `docs/handoff/scripts/`
> in the branch. (Paste on request; omitted here to keep this file readable.)
