# SONIX v4 — data status

Last updated 2026-09-24. Owner of this file: Yugal.
Scope: only the datasets that go **into v4** or are used to **evaluate v4**.

---

## The format rule — read first

**Raw audio is preferred.** Yugal extracts everything on the RTX 4060 with one set of flags.

If you must send embeddings, they must be extracted with **exactly**:

```
src/extract_embeddings_v2.py --layers 5,6,24 --pool meanstd
```

→ 6,144 dims, and `meta.json` must say `"pool": "meanstd"`.

Anything else cannot be co-trained. Embeddings cannot be re-pooled — a wrong-format
embedding is a dead file, and the only fix is the audio.

---

## v4 training data

| # | Dataset | Rows | Type | From | Status |
|---|---|---|---|---|---|
| 1 | ASVspoof19 LA bonafide | 2,580 | real · English | on disk | ✅ v2 extracted |
| 2 | ASVspoof19 LA spoof (subsampled) | ~15,000 | fake · English | on disk | ✅ v2 extracted |
| 3 | MMS-TTS Indic (`indic_spoof` + `_aug`) | 1,600 | fake · Indic TTS | on disk | ✅ audio — v2 extraction queued |
| 4 | **IndicVoices** | 37,032 | real · Indic | **Navya** | ❌ only v1 embeddings, no audio |
| 5 | **MLAAD Indic** | ~20,000 | fake · Indic TTS, ~140 models | **Navya** | ❌ not received |
| 6 | **IndicSynth** | 16,800 | fake · Indic voice conversion | **Akshat** | ❌ only v1 embeddings, no audio |
| 7 | **LibriSeVoc real** | 13,201 | real · English | **Akshat** | ⚠️ received as `pool=mean`, needs re-run |
| 8 | **LibriSeVoc spoof** | ~25,000 | fake · 6 vocoders | **Akshat** | ⚠️ received as `pool=mean`, needs re-run |
| 9 | **Zero-shot clones** (XTTS-v2, OpenVoice v2) | ~8,000 | fake · Indic cloning | **Akshat** (generated) | ❌ not started — blocked on #10 |
| 10 | **Common Voice Indic** | ~15,000 | real · Indic phone audio | **Anubhav** | ❌ not received |

**Have: 3 of 10.** ~19k of ~154k planned rows, all English except 1,600 MMS-TTS clips.

**Hard blockers:** #4 IndicVoices and #6 IndicSynth. Without them v4 has no Indic real
speech and no Indic voice-conversion spoof — it cannot be trained on Indian audio at all.

### Conditioning material

| Dataset | From | Status |
|---|---|---|
| MUSAN + RIRS_NOISES (raw noise/RIR files) | **Navya** | ❌ only old v1 `rirmusan` embeddings exist |

Applied symmetrically to **every corpus and both classes** — RawBoost, G.711, AMR-WB, Opus,
RIR, MUSAN.

---

## v4 evaluation data — held out, never trained on

| Dataset | From | Status |
|---|---|---|
| In-the-Wild | Yugal | ✅ audio unzipped — v2 extraction running |
| ASVspoof19 LA eval | on disk | ✅ v2 extraction running |
| `sonix_real/` (9 matched pairs) | Anubhav | ✅ audio + manifest — v2 extraction queued |
| `sonix_real/` scale-up to 200+ pairs | Anubhav | ❌ not started |
| DEEP-VOICE (RVC — leave-one-family-out) | Anubhav | ❌ not received |
| WaveFake (comparability only) | Anubhav | ❌ optional |
| ASVspoof 5 eval | on disk | ⏭️ raw, never prepped — later |

---

## Dependencies — the order things must happen in

```
Anubhav: Common Voice → cv_train_speakers.txt ──► Akshat: zero-shot clones
Navya:   IndicVoices audio ──────────────────────► Akshat: zero-shot clones (alt. refs)
Navya + Akshat: IndicVoices + IndicSynth audio ──► Yugal: v2 extraction ──► train head_v4
```

Clone reference speakers must **never** appear in `sonix_real/`, or the only decisive test
set stops being held out.

---

## Deprioritised

**LibriSpeech train-clean-100.** LibriSeVoc's 13,201 real clips *are* LibriSpeech utterances.
Adding LibriSpeech on top duplicates one corpus and one channel, which strengthens a corpus
shortcut rather than weakening it. Revisit only if the LibriSeVoc re-run falls through.

---

## Yugal's own queue

1. ✅ ASVspoof19 train — 254 shards
2. 🔄 chain running: ASVspoof19 dev → eval → In-the-Wild
3. ⏭️ MMS-TTS + `sonix_real/` v2 extraction (small, after the chain)
4. ⏭️ Commit the `src/train.py` v4 rewrite (group-balanced loss, OC-Softmax, generator-ID head,
   worst-group selection)
5. ⏭️ Cross-dataset layer experiment: train on LibriSeVoc block *i*, test on ASVspoof eval —
   settles the layer choice both saturated probes couldn't
6. ⏭️ Train `head_v4` once blockers #4 and #6 clear
7. ⏭️ Beat `robust_v2` (AUC 0.764 on matched pairs) at S8 to earn `DEFAULT_KEY`

## How to check

```
dir /b outputs\embeddings_v2\train\*.npy | find /c /v "labels"     (254 = done)
dir /b outputs\embeddings_v2\dev\*.npy   | find /c /v "labels"     (249 = done)
dir /b outputs\embeddings_v2\eval\*.npy  | find /c /v "labels"     (713 = done)
dir outputs\models\head_v4*
git status --short src\train.py
```
