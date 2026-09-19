# SONIX v4 — team task briefs

Written 2026-09-20 by Yugal. Forward the relevant section to each person.
Full detail: `docs/V4_BUILD_BRIEF.md`, `docs/DOWNLOADS.md`, `docs/PROBE_RESULTS.md`.

---

## Shared context (both of you need this)

We measured something on 19 Sept that changes the plan.

Our embeddings — frozen XLS-R, last hidden layer, mean-pooled to 1024 dims — encode
**language at 99.33%** (12-way, chance 8.33%) and **corpus at 100%**. Meanwhile a spoof
detector trained on English ASVspoof scores **AUC 0.9992 in-domain and 0.5196 on Indic**.
0.50 is chance. We then projected out 20 language directions and it did not recover —
so language wasn't *hiding* a good signal. There is no transferable fake-detection
signal in those vectors at all.

Cause: XLS-R produces ~200 frames per 4-second window and we averaged all of them into
one vector. Language, speaker, channel and SNR are steady across a clip so they survive
averaging perfectly. Vocoder artifacts are split-second events — they get averaged into
noise. We were feeding the classifier the confounds and throwing away the evidence.

**The consequence for you: every embedding we have is unusable for v4.** You cannot
re-pool an embedding — the information is already gone. We need the **raw audio** back
for everything, and we're re-extracting with new pooling.

That's why the asks below are about audio files, not embeddings.

---

## AKSHAT — you are the #1 blocker on the whole project

### 1. IndicSynth wavs — URGENT, today if possible

We have 16,800 IndicSynth embeddings and **zero IndicSynth audio**. Only the `.npy`
files were ever shared. Those embeddings are now dead weight, which means the entire
Indic voice-conversion spoof family is currently missing from v4.

Nothing downstream can start without this. It's not on the internet — it's on your machine.

**What we need:** the 16,800 source `.wav` files, with the original folder structure and
filenames intact (`indicsynth_<Language>_<n>` — the language tag matters, we use it for
group-balanced training).

**How:** external drive is fastest. Otherwise Google Drive / WeTransfer in language-sized
chunks. If the generation script is reproducible on your machine, sending the script +
source voices also works — tell me which is easier.

If any part is missing or was deleted, say so **now** rather than later. We'd rather drop
IndicSynth and hold it out as a test set than discover a gap at training time.

### 2. In-the-Wild — download it

`data/in_the_wild/` on my machine contains a README and nothing else. Müller et al. 2022,
~38 hours (20.8 h real / 17.2 h fake), real internet deepfakes from unseen tools. On Zenodo.

This is our **only honest cross-dataset number**. Every EER we've reported so far is
in-domain and means nothing outside the corpus it came from. This is the one that goes
in the paper.

Keep it strictly held out. It never touches training.

### 3. Zero-shot clones — the family that actually beats us (start after #1)

`sonix_real/` shows our models fail on clones made with free consumer tools. We've trained
on TTS (MMS-TTS) and voice conversion (IndicSynth) but on **zero zero-shot cloning**. It's
a third artifact family and it's the one real attackers use.

Generate ~8,000 clips across Indic languages using **XTTS-v2** and **OpenVoice v2** (both
free, both run locally, both support Hindi).

**Critical constraint:** the speakers you clone must **not** appear in `sonix_real/`. If
they overlap, `sonix_real/` stops being a valid held-out test and we lose our only
decisive evaluation. Use reference voices from IndicVoices or Common Voice once Navya's
download lands.

Log which tool made each clip — we're adding a generator-ID output to the model, so those
labels are training data, not just bookkeeping.

---

## NAVYA — critical path plus the biggest new dataset

### 1. IndicVoices audio — URGENT, this is the other hard blocker

We have 37,032 train + 5,050 holdout IndicVoices embeddings and **no audio on the machine**.
`scripts/extract_indicvoices.py` reads an `audio_filepath` manifest pointing somewhere
that no longer exists locally.

IndicVoices is our **only source of genuine Indian speech**. With no Indic real audio there
is no Indic real side to learn from, so v4 cannot be trained at all — not badly, not
partially.

**What we need:** re-download from AI4Bharat (check HuggingFace `ai4bharat/IndicVoices`
and the AI4Bharat site for the current release). Same 12 languages our existing embeddings
cover. It's large — start it before you do anything else today.

**Also please check the licence terms for redistribution.** We may publish a paper and
release a corpus built on it, and we need to know what we're allowed to do before we
build on it, not after.

### 2. MLAAD Indic slice — the biggest quality win available

`prep_mlaad.py` is already written, sitting in the repo root, and has never been run.

Right now our Indic spoof side is one MMS-TTS voice per language plus one VC system. That's
a monoculture — the model learns *those two synthesisers*, not synthesis. MLAAD is **~140
TTS models across 51 languages**, and its Indic slice is roughly 45 hours: Hindi 15.1 h,
Bangla 14.1 h, Tamil 5.0 h, Kannada 4.2 h, Marathi 3.0 h, Urdu 2.1 h, Malayalam 1.8 h.

```
pip install -U huggingface_hub
python prep_mlaad.py --list
python prep_mlaad.py --langs hi,bn,ta,mr,kn,ml,ur --out data/mlaad_indic
```

The full repo is 183 GB — the script pulls only the language folders you name, so don't
skip the `--langs` flag.

**Licence note:** MLAAD is CC BY-NC 4.0. Fine for SIH and for a paper, **not** for anything
commercial. Flag this if it ever goes near a product slide.

### 3. MUSAN + RIRS_NOISES — check before downloading

`outputs/embeddings_rirmusan_*` exists, so these were on a machine at some point. If you
still have them, say so and skip. If not: MUSAN is openslr.org/17 (~11 GB), RIRS_NOISES is
openslr.org/28 (~1 GB).

Important change from last time: we're applying augmentation to **every corpus and both
classes**, not just ASVspoof. Applying it to one corpus only is how RawBoost quietly became
a marker for "this row is ASVspoof" — a second shortcut we built ourselves without noticing.

### 4. DF21 + ASVspoof 5 — eval only, after the above

Both stay strictly held out. ASVspoof 5 is the current field benchmark; training on it
would burn the comparability that's its entire value. DF21 numbers always carry the
"75% key coverage" caveat.

---

## Why the urgency

The model rebuild is coded and validated — the new extractor works, shapes confirmed.
The layer sweep is running now. After that the build is mechanical.

But it cannot produce a model from data that isn't on the disk. Right now the critical
path runs through **Akshat's hard drive** and **Navya's download**, not through any GPU.

Everything else — training changes, evaluation, thresholds, the serving integration —
is waiting on those two items.
