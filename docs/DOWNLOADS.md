# SONIX v4 — download sheet

Written 2026-09-20. Companion to `docs/V4_BUILD_BRIEF.md` §4.
Split this across the team. Tier 1 blocks everything; tier 3 can land any time.

**Disk:** ~307 GB free on the repo volume at time of writing. Raw audio below is
roughly 120–180 GB, and S3 conditioning roughly doubles the audio footprint before
you delete intermediates. **Move the repo off OneDrive first** or sync will fight
every large write and convert files to cloud-only stubs mid-training.

**Why raw audio and not embeddings:** embeddings cannot be re-pooled. Every existing
`.npy` root is frozen in the representation `docs/PROBE_RESULTS.md` proved carries no
transferable signal. Only audio has forward value.

---

## TIER 1 — v4 cannot be trained without these

| # | Item | Owner | Why |
|---|---|---|---|
| 1 | **IndicSynth wavs** (16,800) | **Akshat** | We hold embeddings only. Not on the internet — they exist on Akshat's machine. Without them we lose the entire Indic VC spoof family. |
| 2 | **IndicVoices audio** | **Yugal** | We hold 37,032 train + 5,050 holdout embeddings and no audio. This is our *only* Indic bonafide. Zero Indic real = no v4. |

**#1 is a message, not a download. Send it first.**

IndicVoices is from AI4Bharat. Check the HuggingFace hub (`ai4bharat/IndicVoices`) and
the AI4Bharat site for the current release and its licence terms — confirm redistribution
rights before we publish anything built on it. Pull the same 12 languages the existing
embeddings cover.

---

## TIER 2 — the datasets that make v4 different from v3

| # | Item | Owner | Size (approx) | Notes |
|---|---|---|---|---|
| 3 | **MLAAD — Indic slice** | Navya | ~45 h of a 183 GB repo | `prep_mlaad.py` is already written and never run. **CC BY-NC 4.0** — fine for SIH and a paper, NOT for a commercial claim. Say so on any slide. |
| 4 | **In-the-Wild** | Navya | ~38 h (20.8 real / 17.2 fake) | `data/in_the_wild/` currently holds a README and nothing else. This is our only honest cross-dataset number. Müller et al. 2022; on Zenodo / deepfake-total. |
| 5 | **LibriSeVoc** | Anubhav | ~92k clips (13,201 real + 79,206 synthetic) | 6 vocoders: WaveNet, WaveRNN, MelGAN, ParallelWaveGAN, WaveGrad, DiffWave. Gives us the multitask vocoder-ID label AND 13,201 English bonafide. Best open-set EER in Sun et al. Table 8 came from training on this. GitHub, Sun et al. 2023. |

```bat
REM #3 — MLAAD Indic (7 languages, not the full 183 GB)
pip install -U huggingface_hub
.venv\Scripts\python.exe prep_mlaad.py --list
.venv\Scripts\python.exe prep_mlaad.py --langs hi,bn,ta,mr,kn,ml,ur --out data\mlaad_indic
```

---

## TIER 3 — bonafide diversity and held-out test sets

| # | Item | Owner | Size (approx) | Fixes |
|---|---|---|---|---|
| 6 | **Common Voice — Indic** (hi, bn, ta, mr, pa, ur) | Anubhav | pick ~15k clips | **CC0.** Phone mics, varied SNR, thousands of speakers. This is the fix for the SNR shortcut (genuine passing at 34–38 dB, failing at 39–67 dB). |
| 7 | **LibriSpeech train-clean-100** | Anubhav | ~6.3 GB | openslr.org/12. English bonafide in volume, so "English ⇒ fake" stops being learnable. Currently 2,580 English real vs 91,200 English fake. |
| 8 | **DEEP-VOICE** | Navya | small | Kaggle, `birdy654/deep-voice-deepfake-voice-recognition`. RVC family — held out entirely for leave-one-family-out. From Bird & Lotfi 2023. |
| 9 | **WaveFake** | Navya | ~105–118k clips | Zenodo. **Eval only** — its vocoders overlap LibriSeVoc's heavily, so it's worth more as a comparability number (appears in Sun et al. Table 8) than as training rows. |

---

## TIER 4 — conditioning material (check before downloading, we may already have it)

`outputs/embeddings_rirmusan_*` exists, so these were on the machine at some point.
Verify before re-pulling.

| # | Item | Size | Source |
|---|---|---|---|
| 10 | **MUSAN** (noise, music, speech) | ~11 GB | openslr.org/17 |
| 11 | **RIRS_NOISES** (room impulse responses) | ~1 GB | openslr.org/28 |

---

## TIER 5 — generators (models, not datasets — you run these to MAKE spoof)

Family 3 (zero-shot cloning) and family 4 (retrieval VC) are the two we have **zero**
training data for, and family 3 is what `sonix_real/` proves beats us today.

| # | Tool | Use |
|---|---|---|
| 12 | **XTTS-v2** (Coqui) | zero-shot cloning, supports Hindi |
| 13 | **OpenVoice v2** | zero-shot cloning, free |
| 14 | **RVC** | retrieval VC — the consumer family, same as DEEP-VOICE |

Generate ~8,000 clips in Indic languages. **Speakers must be disjoint from `sonix_real/`**
or that set stops being a valid held-out test. Reference voices can come from IndicVoices
or Common Voice once #2 / #6 land.

---

## Accounts and keys — get these before the downloads start

| Need | For |
|---|---|
| HuggingFace token (`huggingface-cli login`) | MLAAD, IndicVoices, Common Voice |
| Kaggle API key (`~/.kaggle/kaggle.json`) | DEEP-VOICE |
| ASVspoof 5 registration | only if we use it beyond the raw copy already in `data/asvspoof5/raw/` |

---

## Explicitly NOT downloading

Each of these adds a domain we would then have to balance, costs extraction GPU-hours,
and touches none of the three diagnosed failures (language, generator gap, SNR).

**Codecfake** (revisit only if codec-LM clones show up in error analysis) · **FoR** ·
**SpoofCeleb** · **PartialSpoof / Half-Truth** · **FakeAVCeleb** (request form) ·
**ADD 2022/2023** (restricted licensing) · **Vaani, Shrutilipi, Kathbath, SPRING-INX,
FLEURS, IndicTTS, MLS** — Common Voice already gives channel and speaker diversity;
one extra bonafide corpus, not seven. Revisit only if the false-alarm rate stays high
after S8.

**ASVspoof 5** stays **eval-only**. It is the current field benchmark, so training on it
burns the comparability that is its entire value, and it would need subsampling to ~15k
anyway to avoid re-creating the imbalance we are fixing.

---

## Order of operations

1. Message Akshat (#1). Start IndicVoices (#2). Move the repo off OneDrive.
2. Keys and accounts.
3. Tier 2 in parallel — three people, three datasets.
4. Tier 3 while tier 2 runs.
5. Tier 5 once #2 or #6 provides reference voices.

Nothing past the layer-probe decision can proceed until #1 and #2 are on disk.
