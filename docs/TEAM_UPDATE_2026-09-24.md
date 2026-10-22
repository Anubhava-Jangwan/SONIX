# SONIX v4 — team update, 24 Sept

Full status table: `docs/V4_DATA_STATUS.md`. Pull `yugal/ml-pipeline` first.

## The one rule for everyone

**Send raw audio, not embeddings.** I extract everything on the 4060 with one set of flags.

If you absolutely must send embeddings, use exactly:

```
src/extract_embeddings_v2.py --layers 5,6,24 --pool meanstd
```

(6,144 dims, `meta.json` says `"pool": "meanstd"`). Anything else can't be co-trained, and
embeddings can't be re-pooled — a wrong-format embedding is a dead file.

Where we are: I have 3 of the 10 v4 training datasets, all English except 1,600 MMS-TTS
clips. **Zero Indic real speech and zero Indic voice-conversion spoof.** v4 cannot train on
Indian audio until IndicVoices (Navya) and IndicSynth (Akshat) arrive as audio.

---

## AKSHAT

LibriSeVoc v2 looks good, and separating `ln` from `hidden_states[24]` was a sharp catch —
those really are different tensors. Your layer probe came back at AUC 1.0 on nearly every
layer though, which means in-domain LibriSeVoc is too easy to rank layers. I'll run it
cross-dataset instead.

**1. IndicSynth WAVS — top priority.**
What arrived is embeddings again: 100 × 1024 per shard, v1 pooling. There's no IndicSynth
audio anywhere on the E: drive. The whole Indic voice-conversion family is still missing from
v4. Need the 16,800 source `.wav` files, original filenames (`indicsynth_<Language>_<n>`).
External drive or Drive/WeTransfer in chunks. If any of it's gone, tell me now and we'll
hold IndicSynth out as a test set instead.

**2. Re-run LibriSeVoc** with `--layers 5,6,24 --pool meanstd`, or just send me the audio.
Yours is `pool=mean` / 26,624 dims; mine is `meanstd` / 6,144. Can't co-train them.
Also send the **vocoder label mapping** — the filenames in `files.txt` are LibriSpeech IDs
with no vocoder in them, so I need the folder→vocoder manifest for the generator-ID head.

**3. Skip In-the-Wild** — I've downloaded and unzipped it.

**4. Zero-shot clones — after Anubhav sends `cv_train_speakers.txt`.**
~8,000 Indic clips with XTTS-v2 + OpenVoice v2. Reference speakers only from that train list —
**never anyone in `sonix_real/`**. Log which tool made each clip; that's training data.

---

## NAVYA

**1. IndicVoices audio — top priority, start today.**
We have 37,032 train + 5,050 holdout embeddings and zero audio. It's our only Indic real speech.
No audio = v4 can't train on Indian speech at all. Re-download from AI4Bharat and check the
redistribution licence before we build a paper on it.

**2. MLAAD Indic.** 183 GB is the whole repo — the Indic slice is ~45 h (~5–10 GB).

```
.venv\Scripts\python.exe -m pip install -U huggingface_hub
.venv\Scripts\python.exe prep_mlaad.py --list
.venv\Scripts\python.exe prep_mlaad.py --langs hi,bn,ta --out data/mlaad_indic
```

If that lands at a sane size, add `mr,kn,ml,ur`. Gotcha: `huggingface_hub` 1.32 renamed
`huggingface-cli` to `hf`. If the script shells out to the old name, use `.venv\Scripts\hf.exe`
or `from huggingface_hub import snapshot_download`. MLAAD is **CC BY-NC 4.0** — fine for SIH
and a paper, not for a commercial claim.

**3. MUSAN + RIRS_NOISES raw files.** We only have old `rirmusan` embeddings. If you still have
the audio, send it; otherwise openslr.org/17 (MUSAN, ~11 GB) and openslr.org/28 (RIRS, ~1 GB).

**4. DF21 / ASVspoof 5** — later, eval only.

---

## ANUBHAV

Your tasks are download + manifest work — the 1650 is fine. **Don't extract embeddings**; send
me raw audio and I'll extract on the 4060 so nothing comes back in the wrong format.

**1. Common Voice Indic — first, because Akshat is waiting on it.**
hi, bn, ta, mr, pa, ur, ~15k clips. The official route is now commonvoice.mozilla.org →
Mozilla Data Collective; older versions are on HuggingFace. Check which is current and verify
the licence on the release itself. Deliverable is the audio **plus** `cv_train_speakers.txt` /
`cv_eval_speakers.txt` split by `client_id`. **Send Akshat the train list before he clones anything.**

**2. DEEP-VOICE** — Kaggle `birdy654/deep-voice-deepfake-voice-recognition`. Held out, never
trained on. Put it somewhere no training glob can reach.

**3. `sonix_real/` scale-up — volunteers.** Nine matched pairs is all we have. Plan: 20–25
consenting volunteers × 3 utterances × 3 cloning tools ≈ 180–225 pairs. Record each person on
phone mic, laptop mic, and over a real call — the channel variance is the point. One-page
consent form (record, clone with named tools, release under a stated licence), and check
whether UPES needs ethics review first. For cloning, OpenVoice v2 runs on your 1650; if XTTS-v2
OOMs, run it on CPU — fine for a couple hundred clips. Speakers must not overlap Akshat's.

**4. WaveFake** — optional, eval only.

**Dropped: LibriSpeech.** LibriSeVoc's 13,201 real clips already *are* LibriSpeech — adding it
would duplicate one corpus and one channel.

Leave `diagnose_pairs.py` alone — I'm patching `group_of()` to read `generator_family` from your
manifest.
