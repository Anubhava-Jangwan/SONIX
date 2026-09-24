# SONIX v4 — Anubhav's tasks

Updated 2026-09-24. Pull `yugal/ml-pipeline` first. Full team status: `docs/V4_DATA_STATUS.md`.

---

## Why this matters

We measured on 19 Sept that our old embeddings encode **language at 99.33%** and **corpus at 100%**,
but carry no fake-detection signal that transfers: a probe trained on English scores AUC 0.9992
in-domain and **0.5196 on Indic** — chance. Every old embedding is unusable for v4, so we need
**raw audio** for everything.

On the matched real/clone set, the current default head `v3` scores **AUC 0.509** (chance) and
`robust_v2` scores 0.764. But that's on **9 matched pairs**. Every Indic claim we have rests on
nine pairs, which is why the `sonix_real/` scale-up is your most important task.

---

## The rule

**Send raw audio, not embeddings.** Yugal extracts everything on the 4060 with one set of flags,
so nothing comes back in the wrong format. Your GTX 1650 is fine for everything below — none of
it needs extraction.

---

## 1. Common Voice Indic — do this first, Akshat is waiting on it

- Languages: `hi, bn, ta, mr, pa, ur`
- Target: ~15,000 clips
- Source: the official route is now **commonvoice.mozilla.org → Mozilla Data Collective**. Older
  versions are on HuggingFace (e.g. `mozilla-foundation/common_voice_17_0`). Check which is
  current before downloading.
- **Verify the licence on the release itself.** Don't take our docs' word for "CC0" in a paper.

**Why:** crowd-recorded on cheap phones and laptops in random rooms. That fixes a measured bias:
genuine clips that passed our detector sat at 34.6 / 37.7 dB SNR, genuine clips that failed sat
at 39.2–66.8 dB. The model partly learned "clean recording = fake". Channel variance on the real
side is the fix.

**Deliverables:**
1. The raw audio
2. A manifest with columns:
   `path, recording_id, corpus, language, generator_family, channel, label, split`
3. **`cv_train_speakers.txt`** and **`cv_eval_speakers.txt`** — speakers split by `client_id`

**Send `cv_train_speakers.txt` to Akshat before he generates a single clone.** He uses those
speakers as reference voices. If his clone speakers overlap our held-out speakers, `sonix_real/`
stops being held out and we lose the only test that has ever caught a real failure.

---

## 2. DEEP-VOICE — held out

- Kaggle: `kaggle datasets download -d birdy654/deep-voice-deepfake-voice-recognition`
- Needs `~/.kaggle/kaggle.json`
- RVC voice conversion — a family we have zero training data for. It's our **leave-one-family-out**
  test: train on everything else, measure how much we degrade on a family we've never seen.
- **Never trained on.** Put it somewhere no training glob can reach — `outputs/embeddings*` already
  proved a naive glob swallows a holdout set.

---

## 3. `sonix_real/` scale-up — volunteers

**Decision: volunteers, not more public-figure clips.** We can't publish cloned Indian politicians
(personality rights + copyright on the source recordings). The existing 9 public-figure pairs stay
as an internal demo set, never released.

**Target: 200+ matched pairs.** 20–25 volunteers is enough, not 50:
20–25 people × 3 utterances × 3 cloning tools ≈ 180–225 pairs.

**Recording protocol — vary the channel on purpose.** Each person records the same utterances on:
- a phone mic
- a laptop mic
- over an actual call

**Languages:** Hindi, Punjabi, Bengali, Tamil, Telugu, Marathi — as many as volunteers cover.

**Before recording anyone:**
- One-page consent form covering (a) recording their voice, (b) cloning it with named tools,
  (c) releasing both under a stated licence
- Check whether UPES requires ethics review for human-subject data

**Cloning on the 1650:** OpenVoice v2 runs fine. If XTTS-v2 OOMs, run it on CPU — slow but fine
for a couple hundred clips. Log which tool made each clone (`generator_family`).

**Constraint:** your volunteers must not overlap Akshat's clone reference speakers.

Keep using your `build_manifest.py`:
- `--legacy-ok` writes `unknown` for the old 41 clips — drop the flag once renaming is done
- `--check-disjoint` exits 2 on speaker overlap — Yugal is wiring it into `verify_v4.py` as a gate
- `LANGS` / `CHANNELS` stay closed sets so a typo fails the build instead of creating a one-clip group

---

## 4. WaveFake — optional, eval only

Zenodo. Comparability with published numbers. Never trained on. Lowest priority.

---

## Dropped

**LibriSpeech.** LibriSeVoc's 13,201 real clips already *are* LibriSpeech utterances. Adding it
would duplicate one corpus and one channel, which strengthens a corpus shortcut instead of
weakening it.

## Don't touch

**`diagnose_pairs.py`** — you were right about the `group_of()` prefix bug at line 117. Yugal is
patching it to read `generator_family` from your manifest.

---

## Order

1. Common Voice → speaker split → **send train list to Akshat**
2. Consent form + ethics check → start recruiting volunteers
3. DEEP-VOICE
4. Volunteer recording + cloning
5. WaveFake if time
